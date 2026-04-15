#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

from glm5_client import GLM5Client


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def next_run_id(today: str) -> str:
    prefix = f"RUN-{today}-"
    ensure_dir(RESULTS_DIR)
    nums: list[int] = []
    for p in RESULTS_DIR.glob(f"{prefix}*"):
        if not p.is_dir():
            continue
        m = re.match(rf"{re.escape(prefix)}(\d{{3}})$", p.name)
        if m:
            nums.append(int(m.group(1)))
    seq = max(nums) + 1 if nums else 1
    return f"{prefix}{seq:03d}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_graph(path: Path):
    return torch.load(path, weights_only=False)


def parse_sample_id(path_str: str) -> str:
    stem = Path(path_str).stem
    if stem.startswith("safe_") or stem.startswith("vul_"):
        return f"sample_{stem.split('_', 1)[1]}"
    return stem


def graph_sample_id(data: Any, fallback: Path) -> str:
    sid = getattr(data, "sample_id", None)
    if isinstance(sid, str) and sid:
        return sid
    return parse_sample_id(str(fallback))


class GNNClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        codebert_fusion: bool = False,
    ):
        super().__init__()
        self.codebert_fusion = codebert_fusion
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        head_in = hidden_dim + (in_dim if codebert_fusion else 0)
        self.head = nn.Linear(head_in, 2)

    def _pool_codebert(self, data: Any) -> torch.Tensor:
        x = data.x
        batch = data.batch
        src = getattr(data, "x_source", None)
        if isinstance(src, torch.Tensor) and src.numel() == x.shape[0]:
            mask = (src == 1).to(x.dtype).unsqueeze(1)
            summed = global_add_pool(x * mask, batch)
            counts = global_add_pool(mask, batch).clamp_min(1.0)
            return summed / counts
        return global_mean_pool(x, batch)

    def encode(self, data: Any) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        return global_mean_pool(x, batch)

    def forward(self, data: Any) -> torch.Tensor:
        emb = self.encode(data)
        if self.codebert_fusion:
            emb = torch.cat([emb, self._pool_codebert(data)], dim=1)
        return self.head(emb)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def confidence_bins(y_true: Any, y_pred: Any, y_prob: Any) -> dict[str, Any]:
    conf = np.maximum(y_prob, 1.0 - y_prob)
    wrong = y_true != y_pred
    bins = [("high", 0.8, 1.01), ("medium", 0.6, 0.8), ("low", 0.0, 0.6)]
    out: dict[str, Any] = {}
    for name, lo, hi in bins:
        m = (conf >= lo) & (conf < hi)
        n = int(m.sum())
        w = int((wrong & m).sum())
        out[name] = {
            "n": n,
            "wrong": w,
            "error_rate": float(w / n) if n else 0.0,
            "avg_conf": float(conf[m].mean()) if n else 0.0,
        }
    out["high_conf_wrong_indices"] = np.where((conf >= 0.8) & wrong)[0].tolist()[:100]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GLM-5 hybrid correction with RAG")
    parser.add_argument("--base-run-id", type=str, default="RUN-20260225-013")
    parser.add_argument("--model", type=str, default="glm-5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--route-conf", type=float, default=0.65)
    parser.add_argument("--accept-conf", type=float, default=0.70)
    parser.add_argument("--accept-conf-override", type=float, default=0.80)
    parser.add_argument("--max-calls", type=int, default=80)
    args = parser.parse_args()

    set_seed(args.seed)
    base_run = ROOT / "results" / args.base_run_id
    if not base_run.exists():
        raise FileNotFoundError(f"base run not found: {base_run}")

    gnn_eval = load_json(base_run / "gnn_only_eval.json")
    split = load_json(ROOT / "data" / "split.json")
    test_paths = list(split.get("test", []))
    if args.max_test > 0 and len(test_paths) > args.max_test:
        idx = list(range(len(test_paths)))
        random.Random(args.seed).shuffle(idx)
        test_paths = [test_paths[i] for i in sorted(idx[: args.max_test])]

    test_graphs = [load_graph(ROOT / p) for p in test_paths]
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False, num_workers=0)
    test_sids = [graph_sample_id(g, ROOT / p) for g, p in zip(test_graphs, test_paths)]
    sid_to_code = {}
    sid_to_label = {}
    for sid, g in zip(test_sids, test_graphs):
        snippets = list(getattr(g, "code_snippets", []) or [])
        sid_to_code[sid] = (
            "\n".join([str(x) for x in snippets[:5]])[:3000] or "/* code unavailable */"
        )
        sid_to_label[sid] = int(g.y.view(-1)[0].item())

    ckpt_path = ROOT / gnn_eval["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GNNClassifier(
        in_dim=int(gnn_eval["model"]["in_dim"]),
        hidden_dim=int(gnn_eval["model"]["hidden_dim"]),
        dropout=float(gnn_eval["model"]["dropout"]),
        codebert_fusion=bool(gnn_eval["model"].get("codebert_fusion", False)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    base_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy().tolist()
            preds = torch.argmax(logits, dim=1).numpy().tolist()
            ys = batch.y.view(-1).numpy().tolist()
            sids = list(getattr(batch, "sample_id"))
            for sid, y, pred, p in zip(sids, ys, preds, probs):
                base_rows.append(
                    {
                        "sample_id": str(sid),
                        "y_true": int(y),
                        "base_pred": int(pred),
                        "base_prob": float(p),
                        "base_conf": float(max(p, 1.0 - p)),
                    }
                )

    emb_dir = base_run / "embeddings" / "gnn"
    train_vecs = np.load(emb_dir / "vectors.npy")
    query_vecs = np.load(emb_dir / "query_vectors.npy")
    train_meta = [
        json.loads(x)
        for x in (emb_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    query_meta = [
        json.loads(x)
        for x in (emb_dir / "query_metadata.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    query_sid_to_vec = {
        str(m["sample_id"]): query_vecs[i] for i, m in enumerate(query_meta)
    }

    train_norm = train_vecs / (
        np.linalg.norm(train_vecs, axis=1, keepdims=True) + 1e-12
    )
    route_candidates: list[dict[str, Any]] = []
    for row in base_rows:
        sid = row["sample_id"]
        qv = query_sid_to_vec.get(sid)
        if qv is None:
            continue
        qn = qv / (np.linalg.norm(qv) + 1e-12)
        sim = train_norm @ qn
        top_idx = np.argsort(-sim)[:10]
        top_labels = [int(train_meta[i]["label"]) for i in top_idx]
        rel_mean = float(np.mean(top_labels))
        retrieval_pred = 1 if rel_mean >= 0.5 else 0
        uncertain = row["base_conf"] < args.route_conf
        disagree = retrieval_pred != row["base_pred"]
        score = (1.0 - row["base_conf"]) + (0.2 if disagree else 0.0)
        route_candidates.append(
            {
                "sample_id": sid,
                "retrieval_mean": rel_mean,
                "retrieval_pred": retrieval_pred,
                "top_idx": top_idx.tolist(),
                "top_labels": top_labels,
                "uncertain": uncertain,
                "disagree": disagree,
                "route_score": score,
            }
        )

    route_candidates.sort(key=lambda x: x["route_score"], reverse=True)
    routed = [x for x in route_candidates if x["uncertain"] or x["disagree"]][
        : args.max_calls
    ]
    routed_sid = {x["sample_id"] for x in routed}

    client = GLM5Client(model=args.model)
    glm_calls: list[dict[str, Any]] = []
    sid_to_glm: dict[str, dict[str, Any]] = {}

    for r in routed:
        sid = r["sample_id"]
        context = []
        for i in r["top_idx"][:5]:
            m = train_meta[i]
            context.append(
                f"neighbor sample_id={m['sample_id']}, label={m['label']}, path={m['path']}"
            )
        anchor = next(x for x in base_rows if x["sample_id"] == sid)
        out = client.classify_with_rag(
            sample_id=sid,
            code=sid_to_code.get(sid, "/* code unavailable */"),
            anchor_prediction=int(anchor["base_pred"]),
            anchor_confidence=float(anchor["base_conf"]),
            retrieved_context=context,
        )
        sid_to_glm[sid] = out
        glm_calls.append(
            {
                "sample_id": sid,
                "request": {"retrieved_context": context[:3]},
                "response": out,
            }
        )

    corrected_rows: list[dict[str, Any]] = []
    for row in base_rows:
        sid = row["sample_id"]
        final = int(row["base_pred"])
        applied = False
        reason = "baseline"
        glm = sid_to_glm.get(sid)
        if glm:
            decision = str(glm.get("decision", "UNKNOWN"))
            conf = float(glm.get("confidence", 0.0))
            cand = 1 if decision == "VULNERABLE" else 0 if decision == "SAFE" else final
            override = cand != final
            needed = args.accept_conf_override if override else args.accept_conf
            if (
                bool(glm.get("parse_ok", False))
                and decision in {"VULNERABLE", "SAFE"}
                and conf >= needed
            ):
                final = cand
                applied = True
                reason = "glm_accepted"
            else:
                reason = "glm_rejected"
        corrected_rows.append(
            {
                **row,
                "final_pred": int(final),
                "glm_called": sid in routed_sid,
                "glm_applied": applied,
                "final_reason": reason,
            }
        )

    y_true = np.asarray([r["y_true"] for r in corrected_rows], dtype=int)
    y_base = np.asarray([r["base_pred"] for r in corrected_rows], dtype=int)
    y_final = np.asarray([r["final_pred"] for r in corrected_rows], dtype=int)
    base_prob = np.asarray([r["base_prob"] for r in corrected_rows], dtype=float)

    base_metrics = compute_metrics(y_true, y_base)
    final_metrics = compute_metrics(y_true, y_final)
    conf_report_base = confidence_bins(y_true, y_base, base_prob)
    conf_report_final = confidence_bins(y_true, y_final, base_prob)

    table = [
        {
            "setting": "GNN_baseline",
            "precision": base_metrics["precision"],
            "recall": base_metrics["recall"],
            "f1": base_metrics["f1"],
            "accuracy": base_metrics["accuracy"],
        },
        {
            "setting": "GNN_plus_GLM5_RAG",
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1": final_metrics["f1"],
            "accuracy": final_metrics["accuracy"],
        },
    ]

    run_id = next_run_id(datetime.now().strftime("%Y%m%d"))
    run_dir = RESULTS_DIR / run_id
    ensure_dir(run_dir)
    report = {
        "run_id": run_id,
        "experiment_id": "E-003",
        "base_run_id": args.base_run_id,
        "glm_model": args.model,
        "glm_endpoint": GLM5Client.ENDPOINT,
        "route_policy": {
            "route_conf": args.route_conf,
            "route_condition": "uncertain_or_retrieval_disagree",
            "max_calls": args.max_calls,
            "accept_conf": args.accept_conf,
            "accept_conf_override": args.accept_conf_override,
        },
        "coverage": {
            "total_test": int(len(corrected_rows)),
            "glm_called": int(sum(1 for r in corrected_rows if r["glm_called"])),
            "glm_applied": int(sum(1 for r in corrected_rows if r["glm_applied"])),
            "glm_call_rate": float(
                np.mean([1.0 if r["glm_called"] else 0.0 for r in corrected_rows])
            ),
            "glm_apply_rate": float(
                np.mean([1.0 if r["glm_applied"] else 0.0 for r in corrected_rows])
            ),
        },
        "metrics_table": table,
        "delta": {
            "precision": final_metrics["precision"] - base_metrics["precision"],
            "recall": final_metrics["recall"] - base_metrics["recall"],
            "f1": final_metrics["f1"] - base_metrics["f1"],
            "accuracy": final_metrics["accuracy"] - base_metrics["accuracy"],
        },
        "confidence_analysis": {
            "baseline": conf_report_base,
            "corrected": conf_report_final,
        },
        "errors_analysis": {
            "high_conf_wrong_before": conf_report_base["high_conf_wrong_indices"],
            "high_conf_wrong_after": conf_report_final["high_conf_wrong_indices"],
        },
        "glm_calls_preview": glm_calls[:50],
        "spec_refs": [
            "SPEC:docs/spec/workflows.yaml#commands[4] (ID:CMD-005)",
            "SPEC:docs/spec/traceability.yaml#experiments[2] (ID:E-003)",
            "SPEC:docs/spec/quality.yaml#requirements[4] (ID:R-005)",
        ],
    }

    save_json(run_dir / "hybrid_eval_glm5.json", report)
    save_json(run_dir / "hybrid_eval_glm5_rows.json", corrected_rows)

    md = (
        "| Setting | Precision | Recall | F1 | Accuracy |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| GNN_baseline | {base_metrics['precision']:.4f} | {base_metrics['recall']:.4f} | {base_metrics['f1']:.4f} | {base_metrics['accuracy']:.4f} |\n"
        f"| GNN_plus_GLM5_RAG | {final_metrics['precision']:.4f} | {final_metrics['recall']:.4f} | {final_metrics['f1']:.4f} | {final_metrics['accuracy']:.4f} |\n"
    )
    (run_dir / "hybrid_eval_glm5_table.md").write_text(md, encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "output": str(run_dir / "hybrid_eval_glm5.json"),
                "table": table,
                "coverage": report["coverage"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
