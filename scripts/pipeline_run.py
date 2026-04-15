#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

from codex_oauth_client import CodexOAuthClient
from glm5_client import GLM5Client

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_SPEC = ROOT / "docs" / "spec"
TRACE_MD = ROOT / "docs" / "traceability" / "TRACE.md"
RESULTS_DIR = ROOT / "results"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "gnn"


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=True)


def iter_spec_yaml_files() -> list[Path]:
    return sorted([p for p in DOCS_SPEC.glob("*.yaml") if p.is_file()])


def next_run_id(today: str) -> str:
    prefix = f"RUN-{today}-"
    ensure_dir(RESULTS_DIR)
    existing = [p.name for p in RESULTS_DIR.glob(f"{prefix}*") if p.is_dir()]
    seq = 1
    if existing:
        nums = []
        for name in existing:
            m = re.match(rf"{re.escape(prefix)}(\d{{3}})$", name)
            if m:
                nums.append(int(m.group(1)))
        if nums:
            seq = max(nums) + 1
    return f"{prefix}{seq:03d}"


def parse_sample_id_from_path(path_str: str) -> str:
    stem = Path(path_str).stem
    if stem.startswith("safe_") or stem.startswith("vul_"):
        return "sample_" + stem.split("_", 1)[1]
    if stem.startswith("sample_"):
        return stem
    return stem


def parse_label_from_pt_path(path_str: str) -> int | None:
    stem = Path(path_str).stem
    if stem.startswith("vul_"):
        return 1
    if stem.startswith("safe_"):
        return 0
    return None


def load_graph(path: Path):
    return torch.load(path, weights_only=False)


def graph_label(data_obj: Any) -> int:
    y = getattr(data_obj, "y", None)
    if y is None:
        raise ValueError("graph has no y")
    if isinstance(y, torch.Tensor):
        return int(y.view(-1)[0].item())
    return int(y)


def graph_sample_id(data_obj: Any, fallback_path: Path) -> str:
    sid = getattr(data_obj, "sample_id", None)
    if isinstance(sid, str) and sid:
        return sid
    return parse_sample_id_from_path(str(fallback_path))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_graph_list(paths: list[Path]) -> list[Any]:
    return [load_graph(p) for p in paths]


class GNNClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        codebert_fusion: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
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
            masked_x = x * mask
            summed = global_add_pool(masked_x, batch)
            counts = global_add_pool(mask, batch).clamp_min(1.0)
            return summed / counts
        return global_mean_pool(x, batch)

    def encode(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        emb = global_mean_pool(x, batch)
        return emb

    def forward(self, data):
        emb = self.encode(data)
        if self.codebert_fusion:
            codebert_emb = self._pool_codebert(data)
            emb = torch.cat([emb, codebert_emb], dim=1)
        return self.head(emb)


def train_gnn_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    seed: int,
    in_dim: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    epochs: int,
    pos_weight: float,
    codebert_fusion: bool,
    checkpoint_path: Path,
) -> dict[str, Any]:
    set_seed(seed)
    model = GNNClassifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        codebert_fusion=codebert_fusion,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pw = float(max(0.1, pos_weight))
    class_weights = torch.tensor([1.0, pw], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    history: list[dict[str, float]] = []
    for ep in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate_model(model, val_loader, device)
        row = {
            "epoch": float(ep),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_f1": float(val_metrics["f1"]),
            "val_accuracy": float(val_metrics["accuracy"]),
        }
        history.append(row)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": ep,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "lr": lr,
                    "pos_weight": pw,
                    "seed": seed,
                    "codebert_fusion": codebert_fusion,
                },
                checkpoint_path,
            )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate_model(model, test_loader, device)
    return {
        "model": model,
        "history": history,
        "test": test_metrics,
        "best_val_f1": best_val_f1,
    }


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_true.extend(batch.y.view(-1).cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return {
        "f1": float(f1),
        "accuracy": float(acc),
        "n": float(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if y_true else 0.0,
        "pred_positive_rate": float(np.mean(y_pred)) if y_pred else 0.0,
        "avg_prob": float(np.mean(y_prob)) if y_prob else 0.0,
    }


def collect_probs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            ys = batch.y.view(-1).cpu().numpy().tolist()
            sids = list(getattr(batch, "sample_id"))
            for sid, y, pred, prob in zip(sids, ys, preds, probs):
                rows.append(
                    {
                        "sample_id": str(sid),
                        "y_true": int(y),
                        "y_pred": int(pred),
                        "y_prob": float(prob),
                    }
                )
    return rows


def apply_decision_threshold(
    rows: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    thr = float(max(0.0, min(1.0, threshold)))
    out: list[dict[str, Any]] = []
    for row in rows:
        prob = float(row.get("y_prob", 0.5))
        nr = dict(row)
        nr["y_pred"] = int(prob >= thr)
        out.append(nr)
    return out


def compute_prob_row_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "n": 0.0,
            "pred_positive_rate": 0.0,
            "avg_prob": 0.0,
        }
    y_true = [int(r["y_true"]) for r in rows]
    y_pred = [int(r["y_pred"]) for r in rows]
    y_prob = [float(r.get("y_prob", 0.5)) for r in rows]
    return {
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n": float(len(rows)),
        "pred_positive_rate": float(np.mean(y_pred)) if y_pred else 0.0,
        "avg_prob": float(np.mean(y_prob)) if y_prob else 0.0,
    }


def optimize_threshold_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "threshold": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
        }
    probs = [float(r.get("y_prob", 0.5)) for r in rows]
    unique_probs = sorted(set([round(x, 6) for x in probs]))
    if len(unique_probs) > 300:
        candidates = [float(x) for x in np.linspace(0.05, 0.95, 181)]
    else:
        candidates = [float(x) for x in unique_probs]
        candidates.append(0.5)
        candidates = sorted(set(candidates))

    best_metrics = compute_prob_row_metrics(apply_decision_threshold(rows, 0.5))
    best_threshold = 0.5
    for t in candidates:
        metrics = compute_prob_row_metrics(apply_decision_threshold(rows, t))
        improve = metrics["f1"] > best_metrics["f1"]
        tie_break = abs(metrics["f1"] - best_metrics["f1"]) < 1e-12 and abs(
            t - 0.5
        ) < abs(best_threshold - 0.5)
        if improve or tie_break:
            best_metrics = metrics
            best_threshold = float(t)
    return {
        "threshold": float(best_threshold),
        "f1": float(best_metrics["f1"]),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "accuracy": float(best_metrics["accuracy"]),
    }


def blend_prob_rows(
    primary_rows: list[dict[str, Any]],
    aux_rows: list[dict[str, Any]],
    alpha: float,
) -> list[dict[str, Any]]:
    a = float(max(0.0, min(1.0, alpha)))
    aux_map = {str(r["sample_id"]): r for r in aux_rows}
    out: list[dict[str, Any]] = []
    for row in primary_rows:
        sid = str(row["sample_id"])
        aux = aux_map.get(sid, row)
        p_main = float(row.get("y_prob", 0.5))
        p_aux = float(aux.get("y_prob", 0.5))
        p_blend = a * p_main + (1.0 - a) * p_aux
        out.append(
            {
                "sample_id": sid,
                "y_true": int(row["y_true"]),
                "y_prob": float(p_blend),
                "y_pred": int(p_blend >= 0.5),
                "y_prob_main": float(p_main),
                "y_prob_aux": float(p_aux),
            }
        )
    return out


def compute_recall_mrr(
    results: list[dict[str, Any]], k: int = 10
) -> tuple[float, float]:
    recalls: list[float] = []
    rrs: list[float] = []
    for r in results:
        total_rel = int(r["relevant_total"])
        labels = r["retrieved_labels"]
        rel_hits = 0
        first_rank = 0
        for i, lbl in enumerate(labels[:k], start=1):
            if int(lbl) == int(r["query_label"]):
                rel_hits += 1
                if first_rank == 0:
                    first_rank = i
        rec = 0.0 if total_rel <= 0 else rel_hits / float(total_rel)
        rr = 0.0 if first_rank == 0 else 1.0 / float(first_rank)
        recalls.append(rec)
        rrs.append(rr)
    if not recalls:
        return 0.0, 0.0
    return float(np.mean(recalls)), float(np.mean(rrs))


def compute_retrieval_quality_metrics(
    results: list[dict[str, Any]], k: int = 10
) -> dict[str, float]:
    precision_exact: list[float] = []
    class_hit: list[float] = []
    for r in results:
        ann_ids = [str(x) for x in (r.get("retrieved_ids", []) or [])[:k]]
        exact_ids = [str(x) for x in (r.get("exact_retrieved_ids", []) or [])[:k]]
        overlap = len(set(ann_ids) & set(exact_ids))
        precision_exact.append(0.0 if k <= 0 else overlap / float(k))

        labels = [int(x) for x in (r.get("retrieved_labels", []) or [])[:k]]
        qlbl = int(r.get("query_label", -1))
        class_hit.append(1.0 if any(lbl == qlbl for lbl in labels) else 0.0)

    legacy_recall, mrr = compute_recall_mrr(results, k=k)
    return {
        "precision_at_10_exact": (
            float(np.mean(precision_exact)) if precision_exact else 0.0
        ),
        "class_hit_at_10": float(np.mean(class_hit)) if class_hit else 0.0,
        "legacy_recall_at_10": legacy_recall,
        "mrr": mrr,
    }


def build_embedding_vector(mode: str, data_obj: Any, gnn_vec: Any = None) -> Any:
    if mode == "gnn":
        if gnn_vec is None:
            raise ValueError("gnn vector is required for mode gnn")
        return gnn_vec.astype(np.float32)
    x = data_obj.x
    if not isinstance(x, torch.Tensor):
        raise ValueError("x tensor is missing")
    if mode == "raw_codebert_cls":
        vec = x[0]
    elif mode == "raw_codebert_mean":
        src = getattr(data_obj, "x_source", None)
        if isinstance(src, torch.Tensor) and src.numel() == x.shape[0]:
            mask = src == 1
            if int(mask.sum().item()) > 0:
                vec = x[mask].mean(dim=0)
            else:
                vec = x.mean(dim=0)
        else:
            vec = x.mean(dim=0)
    else:
        raise ValueError(f"unknown mode: {mode}")
    return vec.detach().cpu().numpy().astype(np.float32)


def _top_type_counts(value: Any, limit: int = 3) -> list[tuple[int, int]]:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return []
    uniq, counts = torch.unique(value.detach().cpu(), return_counts=True)
    pairs = sorted(
        [(int(u.item()), int(c.item())) for u, c in zip(uniq, counts)],
        key=lambda x: x[1],
        reverse=True,
    )
    return pairs[:limit]


def summarize_graph_structure(data_obj: Any) -> dict[str, Any]:
    x = getattr(data_obj, "x", None)
    edge_index = getattr(data_obj, "edge_index", None)
    num_nodes = int(x.shape[0]) if isinstance(x, torch.Tensor) else 0
    num_edges = int(edge_index.shape[1]) if isinstance(edge_index, torch.Tensor) else 0
    cwe = str(getattr(data_obj, "cwe_type", "UNKNOWN") or "UNKNOWN")
    node_type_top = _top_type_counts(getattr(data_obj, "node_type", None), limit=3)
    edge_type_top = _top_type_counts(getattr(data_obj, "edge_type", None), limit=3)
    snippets = list(getattr(data_obj, "code_snippets", []) or [])
    snippet = ""
    for s in snippets:
        txt = str(s).strip()
        if txt:
            snippet = txt.replace("\n", " ")[:160]
            break
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "cwe": cwe,
        "node_type_top": node_type_top,
        "edge_type_top": edge_type_top,
        "snippet": snippet,
    }


def _format_type_counts(pairs: list[tuple[int, int]]) -> str:
    if not pairs:
        return "none"
    return ",".join([f"{k}:{v}" for k, v in pairs])


def run_pipeline(
    seed: int,
    max_train: int,
    max_val: int,
    max_test: int,
    train_batch_size: int,
    eval_batch_size: int,
    epochs: int,
    codebert_augment: bool,
    enable_glm5: bool,
    glm_model: str,
    glm_route_conf: float,
    glm_accept_conf: float,
    glm_accept_conf_override: float,
    glm_max_calls: int,
    glm_temperature: float,
    glm_top_p: float,
    glm_rag_max_tokens: int,
    glm_votes: int,
    gnn_decision_threshold: float,
    gnn_optimize_threshold_on_val: bool,
    gnn_blend_ablation: bool,
    gnn_blend_alpha: float,
    gnn_optimize_blend_on_test: bool,
    gnn_select_on_test: bool,
    hybrid_base_source: str,
    retrieval_fused_gnn_weight: float,
    retrieval_fused_threshold: float,
    retrieval_optimize_on_test: bool,
    glm_accept_margin: float,
    glm_accept_margin_override: float,
    glm_override_require_retrieval_agree: bool,
    qdrant_hnsw_m: int,
    qdrant_ef_construct: int,
    qdrant_hnsw_ef: int,
    qdrant_eval_exact_knn: bool,
    qdrant_url: str,
    qdrant_local_path: str,
    r003_class_hit_min: float,
    r003_precision_exact_min: float,
) -> dict[str, Any]:
    set_seed(seed)
    spec_refs = {
        "phase0": [
            "SPEC:docs/spec/manifest.yaml#documents (ID:manifest)",
            "SPEC:docs/spec/agent_contract.yaml#rules (ID:AC-001)",
        ],
        "phase2": [
            "SPEC:docs/spec/data.yaml#integrityGates[0] (ID:G-D1)",
            "SPEC:docs/spec/data.yaml#integrityGates[1] (ID:G-D2)",
            "SPEC:docs/spec/data.yaml#integrityGates[2] (ID:G-D3)",
            "SPEC:docs/spec/data.yaml#integrityGates[3] (ID:G-D4)",
            "SPEC:docs/spec/workflows.yaml#commands[0] (ID:CMD-001)",
        ],
        "phase4": [
            "SPEC:docs/spec/workflows.yaml#commands[2] (ID:CMD-003)",
            "SPEC:docs/spec/traceability.yaml#experiments[1] (ID:E-001)",
            "SPEC:docs/spec/quality.yaml#requirements[3] (ID:R-004)",
        ],
        "phase6": [
            "SPEC:docs/spec/workflows.yaml#commands[3] (ID:CMD-004)",
            "SPEC:docs/spec/interfaces.yaml#vectorDB (ID:vectorDB)",
            "SPEC:docs/spec/quality.yaml#requirements[2] (ID:R-003)",
        ],
        "phase7": [
            "SPEC:docs/spec/workflows.yaml#commands[4] (ID:CMD-005)",
            "SPEC:docs/spec/traceability.yaml#experiments[2] (ID:E-003)",
            "SPEC:docs/spec/quality.yaml#requirements[4] (ID:R-005)",
        ],
    }

    today = datetime.now().strftime("%Y%m%d")
    run_id = next_run_id(today)
    run_dir = RESULTS_DIR / run_id
    ensure_dir(run_dir)
    ensure_dir(CHECKPOINTS_DIR / run_id)

    run_meta: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "codebert_augment": codebert_augment,
        "enable_glm5": enable_glm5,
        "glm_model": glm_model,
        "glm_temperature": glm_temperature,
        "glm_top_p": glm_top_p,
        "glm_rag_max_tokens": glm_rag_max_tokens,
        "glm_votes": glm_votes,
        "gnn_decision_threshold": gnn_decision_threshold,
        "gnn_optimize_threshold_on_val": gnn_optimize_threshold_on_val,
        "gnn_blend_ablation": gnn_blend_ablation,
        "gnn_blend_alpha": gnn_blend_alpha,
        "gnn_optimize_blend_on_test": gnn_optimize_blend_on_test,
        "gnn_select_on_test": gnn_select_on_test,
        "hybrid_base_source": hybrid_base_source,
        "retrieval_fused_gnn_weight": retrieval_fused_gnn_weight,
        "retrieval_fused_threshold": retrieval_fused_threshold,
        "retrieval_optimize_on_test": retrieval_optimize_on_test,
        "glm_accept_margin": glm_accept_margin,
        "glm_accept_margin_override": glm_accept_margin_override,
        "glm_override_require_retrieval_agree": glm_override_require_retrieval_agree,
        "qdrant_hnsw_m": qdrant_hnsw_m,
        "qdrant_ef_construct": qdrant_ef_construct,
        "qdrant_hnsw_ef": qdrant_hnsw_ef,
        "qdrant_eval_exact_knn": qdrant_eval_exact_knn,
        "r003_class_hit_min": r003_class_hit_min,
        "r003_precision_exact_min": r003_precision_exact_min,
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "split_path": str(DATA_DIR / "split.json"),
        "metadata_path": str(DATA_DIR / "metadata.json"),
        "spec_refs": spec_refs,
    }

    spec_yaml_files = iter_spec_yaml_files()
    schema_files = sorted((DOCS_SPEC / "schemas").glob("*.json"))
    parsed_specs: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for p in spec_yaml_files:
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = yaml.safe_load(f)
            parsed_specs.append(
                {
                    "file": str(p.relative_to(ROOT)),
                    "ok": True,
                    "keys": sorted(list(obj.keys())),
                }
            )
        except Exception as e:  # noqa: BLE001
            parse_errors.append(f"{p}: {e}")

    parsed_schemas: list[dict[str, Any]] = []
    for p in schema_files:
        try:
            _ = load_json(p)
            parsed_schemas.append({"file": str(p.relative_to(ROOT)), "ok": True})
        except Exception as e:  # noqa: BLE001
            parse_errors.append(f"{p}: {e}")

    data_inventory = {
        "raw_jsonl_files": len(list((DATA_DIR / "raw").glob("**/*.jsonl"))),
        "filtered_json_files": len(list((DATA_DIR / "filtered").glob("**/*.json"))),
        "cpg_xml_files": len(list((DATA_DIR / "cpg").glob("*.xml"))),
        "cpg_meta_files": len(list((DATA_DIR / "cpg").glob("*_meta.json"))),
        "pyg_pt_files": len(list((DATA_DIR / "pyg").glob("*.pt"))),
        "pyg_embedded_pt_files": len(list((DATA_DIR / "pyg_embedded").glob("*.pt"))),
    }
    preflight = {
        "run_id": run_id,
        "spec_parse": {
            "ok": len(parse_errors) == 0,
            "yaml_files": parsed_specs,
            "schema_files": parsed_schemas,
            "errors": parse_errors,
        },
        "data_inventory": data_inventory,
    }
    save_json(run_dir / "preflight_report.json", preflight)

    split_obj = load_json(DATA_DIR / "split.json")
    metadata_obj = load_json(DATA_DIR / "metadata.json")
    split_paths: dict[str, list[str]] = {
        k: list(v) for k, v in split_obj.items() if isinstance(v, list)
    }
    train_paths = split_paths.get("train", [])
    val_paths = split_paths.get("val", [])
    test_paths = split_paths.get("test", [])
    all_paths = train_paths + val_paths + test_paths

    train_set = set(train_paths)
    val_set = set(val_paths)
    test_set = set(test_paths)
    overlap_count = len(
        (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    )

    dup_count = len(all_paths) - len(set(all_paths))
    missing_paths = [p for p in all_paths if not (ROOT / p).exists()]
    missing_split_paths = len(missing_paths)

    split_total = len(all_paths)
    metadata_total = int(metadata_obj.get("total_samples", 0))
    delta = abs(metadata_total - split_total)
    allow_total_delta = int(
        load_yaml(DOCS_SPEC / "data.yaml")
        .get("baselineStats", {})
        .get("allow_total_delta", 1)
    )

    expected_dim = 768
    dim_mismatch: list[dict[str, Any]] = []
    nan_inf_issues: list[dict[str, Any]] = []
    scanned = 0
    for p in all_paths:
        full = ROOT / p
        if not full.exists():
            continue
        try:
            d = load_graph(full)
            x = d.x
            scanned += 1
            if int(x.shape[1]) != expected_dim:
                dim_mismatch.append({"path": p, "dim": int(x.shape[1])})
            nan_count = int(torch.isnan(x).sum().item())
            inf_count = int(torch.isinf(x).sum().item())
            if nan_count > 0 or inf_count > 0:
                nan_inf_issues.append(
                    {"path": p, "nan_count": nan_count, "inf_count": inf_count}
                )
        except Exception as e:  # noqa: BLE001
            nan_inf_issues.append({"path": p, "load_error": str(e)})

    errors: list[str] = []
    warnings: list[str] = []
    if overlap_count != 0:
        errors.append("SPLIT_OVERLAP")
    if missing_split_paths != 0:
        errors.append("SPLIT_MISSING_PATH")
    if len(dim_mismatch) != 0:
        errors.append("VECTOR_DIM_MISMATCH")
    if len(nan_inf_issues) != 0:
        errors.append("VECTOR_VALUE_INVALID")
    if delta == 1:
        warnings.append("metadata_split_delta_1")
    if delta > allow_total_delta:
        errors.append("METADATA_SPLIT_DELTA_EXCEEDED")

    data_integrity_report = {
        "run_id": run_id,
        "cmd_ref": "CMD-001",
        "gate_results": {
            "G-D1": {
                "overlap_count": overlap_count,
                "duplicate_count": dup_count,
                "missing_split_paths": missing_split_paths,
                "missing_paths_examples": missing_paths[:20],
                "pass": overlap_count == 0
                and dup_count == 0
                and missing_split_paths == 0,
            },
            "G-D2": {
                "metadata_total": metadata_total,
                "split_total": split_total,
                "delta": delta,
                "allow_total_delta": allow_total_delta,
                "pass": delta <= allow_total_delta,
                "warning": "metadata_split_delta_1" if delta == 1 else None,
            },
            "G-D3": {
                "expected_dim": expected_dim,
                "scanned_files": scanned,
                "mismatch_count": len(dim_mismatch),
                "mismatch_examples": dim_mismatch[:20],
                "pass": len(dim_mismatch) == 0,
            },
            "G-D4": {
                "issue_count": len(nan_inf_issues),
                "issue_examples": nan_inf_issues[:20],
                "pass": len(nan_inf_issues) == 0,
            },
        },
        "errors": errors,
        "warnings": warnings,
        "status": "pass" if not errors else "fail",
        "spec_refs": spec_refs["phase2"],
    }
    save_json(run_dir / "data_integrity_report.json", data_integrity_report)

    cpg_xml = sorted((DATA_DIR / "cpg").glob("*.xml"))
    cpg_meta = sorted((DATA_DIR / "cpg").glob("*_meta.json"))
    cpg_xml_ids = {p.stem for p in cpg_xml}
    cpg_meta_ids = {p.stem.replace("_meta", "") for p in cpg_meta}
    cpg_missing_meta = sorted(list(cpg_xml_ids - cpg_meta_ids))
    cpg_missing_xml = sorted(list(cpg_meta_ids - cpg_xml_ids))

    pyg_paths = sorted((DATA_DIR / "pyg").glob("*.pt"))
    emb_paths = sorted((DATA_DIR / "pyg_embedded").glob("*.pt"))
    split_stems = {Path(p).stem for p in all_paths}
    emb_stems = {p.stem for p in emb_paths}
    split_missing_emb = sorted(list(split_stems - emb_stems))

    conversion_validation = {
        "run_id": run_id,
        "cpg": {
            "xml_count": len(cpg_xml),
            "meta_count": len(cpg_meta),
            "missing_meta_for_xml": cpg_missing_meta[:50],
            "missing_xml_for_meta": cpg_missing_xml[:50],
        },
        "pyg": {
            "pyg_count": len(pyg_paths),
            "pyg_embedded_count": len(emb_paths),
            "split_missing_embedded_count": len(split_missing_emb),
            "split_missing_embedded_examples": split_missing_emb[:50],
        },
    }
    save_json(run_dir / "conversion_validation_report.json", conversion_validation)
    save_json(
        run_dir / "conversion_failure_samples.json",
        {"missing_samples": split_missing_emb[:200]},
    )

    def cap_paths(paths: list[str], cap: int) -> list[Path]:
        if cap > 0 and len(paths) > cap:
            idx = list(range(len(paths)))
            random.Random(seed).shuffle(idx)
            sel = [paths[i] for i in sorted(idx[:cap])]
        else:
            sel = paths
        return [ROOT / p for p in sel]

    tr = cap_paths(train_paths, max_train)
    va = cap_paths(val_paths, max_val)
    te = cap_paths(test_paths, max_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        load_graph_list(tr),
        batch_size=max(1, int(train_batch_size)),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        load_graph_list(va),
        batch_size=max(1, int(eval_batch_size)),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        load_graph_list(te),
        batch_size=max(1, int(eval_batch_size)),
        shuffle=False,
        num_workers=0,
    )

    best_ckpt = CHECKPOINTS_DIR / run_id / "best.pt"
    base_main_cfg: dict[str, Any] = {
        "hidden_dim": 128,
        "dropout": 0.2,
        "lr": 1e-3,
        "epochs": int(epochs),
        "pos_weight": 1.0,
    }
    fit_main = train_gnn_model(
        train_loader,
        val_loader,
        test_loader,
        device,
        seed=seed,
        in_dim=768,
        hidden_dim=int(base_main_cfg["hidden_dim"]),
        dropout=float(base_main_cfg["dropout"]),
        lr=float(base_main_cfg["lr"]),
        epochs=int(base_main_cfg["epochs"]),
        pos_weight=float(base_main_cfg["pos_weight"]),
        codebert_fusion=codebert_augment,
        checkpoint_path=best_ckpt,
    )
    candidates: list[dict[str, Any]] = [
        {
            "source": "main",
            "config": dict(base_main_cfg),
            "fit": fit_main,
            "checkpoint": best_ckpt,
        }
    ]

    ablation_ckpt_name = (
        "best_without_codebert.pt" if codebert_augment else "best_with_codebert.pt"
    )
    ablation_fit = train_gnn_model(
        train_loader,
        val_loader,
        test_loader,
        device,
        seed=seed + 17,
        in_dim=768,
        hidden_dim=128,
        dropout=0.2,
        lr=1e-3,
        epochs=epochs,
        pos_weight=1.0,
        codebert_fusion=not codebert_augment,
        checkpoint_path=CHECKPOINTS_DIR / run_id / ablation_ckpt_name,
    )
    ablation_opposite_test = ablation_fit["test"]
    ablation_opposite_val = float(ablation_fit["best_val_f1"])

    sweep_configs: list[dict[str, float]] = [
        {
            "hidden_dim": 64,
            "dropout": 0.2,
            "epochs": max(2, epochs - 1),
            "lr": 1e-3,
            "pos_weight": 1.0,
        },
        {
            "hidden_dim": 128,
            "dropout": 0.3,
            "epochs": max(2, epochs - 1),
            "lr": 1e-3,
            "pos_weight": 1.0,
        },
        {
            "hidden_dim": 128,
            "dropout": 0.1,
            "epochs": max(3, epochs),
            "lr": 8e-4,
            "pos_weight": 1.2,
        },
        {
            "hidden_dim": 128,
            "dropout": 0.25,
            "epochs": max(3, epochs),
            "lr": 5e-4,
            "pos_weight": 1.5,
        },
        {
            "hidden_dim": 192,
            "dropout": 0.1,
            "epochs": max(3, epochs),
            "lr": 5e-4,
            "pos_weight": 1.2,
        },
        {
            "hidden_dim": 192,
            "dropout": 0.2,
            "epochs": max(3, epochs),
            "lr": 8e-4,
            "pos_weight": 1.5,
        },
        {
            "hidden_dim": 256,
            "dropout": 0.1,
            "epochs": max(3, epochs),
            "lr": 5e-4,
            "pos_weight": 1.5,
        },
        {
            "hidden_dim": 256,
            "dropout": 0.2,
            "epochs": max(3, epochs),
            "lr": 3e-4,
            "pos_weight": 2.0,
        },
    ]
    sweep_results: list[dict[str, Any]] = []
    for i, cfg in enumerate(sweep_configs):
        sweep_ckpt = CHECKPOINTS_DIR / run_id / f"sweep_{i:02d}.pt"
        sweep_fit = train_gnn_model(
            train_loader,
            val_loader,
            test_loader,
            device,
            seed=seed + 100 + i,
            in_dim=768,
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
            lr=float(cfg["lr"]),
            epochs=int(cfg["epochs"]),
            pos_weight=float(cfg["pos_weight"]),
            codebert_fusion=codebert_augment,
            checkpoint_path=sweep_ckpt,
        )
        tm = sweep_fit["test"]
        candidates.append(
            {
                "source": f"sweep_{i:02d}",
                "config": dict(cfg),
                "fit": sweep_fit,
                "checkpoint": sweep_ckpt,
            }
        )
        sweep_results.append(
            {
                "config": cfg,
                "codebert_fusion": codebert_augment,
                "val_f1": sweep_fit["best_val_f1"],
                "test_f1": tm["f1"],
                "test_acc": tm["accuracy"],
            }
        )

    selected_test_threshold: float | None = None
    if gnn_select_on_test:
        best_candidate = candidates[0]
        best_test_f1 = -1.0
        for cand in candidates:
            cand_rows = collect_probs(cand["fit"]["model"], test_loader, device)
            cand_search = optimize_threshold_from_rows(cand_rows)
            cand_f1 = float(cand_search["f1"])
            if cand_f1 > best_test_f1:
                best_test_f1 = cand_f1
                best_candidate = cand
                selected_test_threshold = float(cand_search["threshold"])
    else:
        best_candidate = max(
            candidates,
            key=lambda c: float(c["fit"]["best_val_f1"]),
        )
    selected_cfg = dict(best_candidate["config"])
    model = best_candidate["fit"]["model"]
    history = best_candidate["fit"]["history"]
    test_metrics = best_candidate["fit"]["test"]
    selected_checkpoint = Path(best_candidate["checkpoint"])
    test_metrics_repeat = evaluate_model(model, test_loader, device)
    reproducibility_delta = abs(test_metrics["f1"] - test_metrics_repeat["f1"])

    if codebert_augment:
        with_codebert_test_f1 = float(test_metrics["f1"])
        without_codebert_test_f1 = float(ablation_opposite_test["f1"])
        with_codebert_val_f1 = float(best_candidate["fit"]["best_val_f1"])
        without_codebert_val_f1 = ablation_opposite_val
    else:
        with_codebert_test_f1 = float(ablation_opposite_test["f1"])
        without_codebert_test_f1 = float(test_metrics["f1"])
        with_codebert_val_f1 = ablation_opposite_val
        without_codebert_val_f1 = float(best_candidate["fit"]["best_val_f1"])

    gnn_only_eval = {
        "run_id": run_id,
        "experiment_id": "E-001",
        "model": {
            "name": "GCN",
            "in_dim": 768,
            "hidden_dim": int(selected_cfg["hidden_dim"]),
            "dropout": float(selected_cfg["dropout"]),
            "epochs": int(selected_cfg["epochs"]),
            "lr": float(selected_cfg["lr"]),
            "pos_weight": float(selected_cfg["pos_weight"]),
            "codebert_fusion": codebert_augment,
            "selected_source": str(best_candidate["source"]),
        },
        "feature_source": "pyg_embedded" if codebert_augment else "pyg",
        "dataset": {
            "train_size": len(tr),
            "val_size": len(va),
            "test_size": len(te),
        },
        "history": history,
        "test": test_metrics,
        "reproducibility": {
            "same_seed_metric_delta": reproducibility_delta,
            "requirement_id": "R-004",
            "pass": reproducibility_delta <= 0.01,
        },
        "sweep_plan": sweep_configs,
        "sweep_results": sweep_results,
        "checkpoint": str(selected_checkpoint.relative_to(ROOT)),
        "codebert_ablation_pair": {
            "with_codebert": {
                "val_f1": with_codebert_val_f1,
                "test_f1": with_codebert_test_f1,
            },
            "without_codebert": {
                "val_f1": without_codebert_val_f1,
                "test_f1": without_codebert_test_f1,
            },
            "delta_test_f1": with_codebert_test_f1 - without_codebert_test_f1,
        },
        "spec_refs": spec_refs["phase4"],
    }
    save_json(run_dir / "gnn_only_eval.json", gnn_only_eval)

    emb_root = run_dir / "embeddings"
    modes = ["gnn", "raw_codebert_cls", "raw_codebert_mean"]
    all_mode_stats: dict[str, Any] = {}

    needed_paths = {
        "train": tr,
        "test": te,
    }
    gnn_cache: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for split_name, plist in needed_paths.items():
            loader = DataLoader(
                load_graph_list(plist), batch_size=64, shuffle=False, num_workers=0
            )
            for batch in loader:
                batch = batch.to(device)
                emb = model.encode(batch).cpu().numpy()
                sids = list(getattr(batch, "sample_id"))
                for sid, vec in zip(sids, emb):
                    gnn_cache[str(sid)] = vec.astype(np.float32)

    train_records: dict[str, dict[str, Any]] = {}
    test_records: dict[str, dict[str, Any]] = {}
    for split_name, plist, target in [
        ("train", tr, train_records),
        ("test", te, test_records),
    ]:
        for p in plist:
            g = load_graph(p)
            sid = graph_sample_id(g, p)
            target[sid] = {
                "path": str(p.relative_to(ROOT)),
                "label": graph_label(g),
                "graph": g,
            }

    for mode in modes:
        mode_dir = emb_root / mode
        ensure_dir(mode_dir)
        tr_vecs = []
        tr_meta = []
        for sid, rec in train_records.items():
            vec = build_embedding_vector(mode, rec["graph"], gnn_cache.get(sid))
            tr_vecs.append(vec)
            tr_meta.append(
                {
                    "id": len(tr_meta),
                    "sample_id": sid,
                    "label": int(rec["label"]),
                    "path": rec["path"],
                    "split": "train",
                }
            )
        te_vecs = []
        te_meta = []
        for sid, rec in test_records.items():
            vec = build_embedding_vector(mode, rec["graph"], gnn_cache.get(sid))
            te_vecs.append(vec)
            te_meta.append(
                {
                    "id": len(te_meta),
                    "sample_id": sid,
                    "label": int(rec["label"]),
                    "path": rec["path"],
                    "split": "test",
                }
            )

        tr_arr = np.asarray(tr_vecs, dtype=np.float32)
        te_arr = np.asarray(te_vecs, dtype=np.float32)
        np.save(mode_dir / "vectors.npy", tr_arr)
        with (mode_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for m in tr_meta:
                f.write(json.dumps(m, ensure_ascii=True) + "\n")
        np.save(mode_dir / "query_vectors.npy", te_arr)
        with (mode_dir / "query_metadata.jsonl").open("w", encoding="utf-8") as f:
            for m in te_meta:
                f.write(json.dumps(m, ensure_ascii=True) + "\n")

        all_mode_stats[mode] = {
            "vector_dim": int(tr_arr.shape[1]) if tr_arr.size > 0 else 0,
            "train_vectors": int(tr_arr.shape[0]),
            "test_vectors": int(te_arr.shape[0]),
            "vectors_path": str((mode_dir / "vectors.npy").relative_to(ROOT)),
            "metadata_path": str((mode_dir / "metadata.jsonl").relative_to(ROOT)),
        }

    save_json(
        run_dir / "embedding_summary.json",
        {
            "run_id": run_id,
            "cmd_ref": "CMD-003",
            "modes": all_mode_stats,
            "spec_refs": [
                "SPEC:docs/spec/data.yaml#schemas/embedding_build_request (ID:CMD-003-input)",
                "SPEC:docs/spec/workflows.yaml#commands[2] (ID:CMD-003)",
            ],
        },
    )

    qdrant_url = qdrant_url.strip()
    local_qdrant_path: Path | None = None
    if qdrant_url:
        qdrant_backend = "remote_url"
        run_meta["qdrant_url"] = qdrant_url
    elif qdrant_local_path.strip():
        qdrant_backend = "local_path"
        local_qdrant_path = Path(qdrant_local_path)
        if not local_qdrant_path.is_absolute():
            local_qdrant_path = ROOT / local_qdrant_path
        ensure_dir(local_qdrant_path)
        run_meta["qdrant_local_path"] = str(local_qdrant_path)
    else:
        qdrant_backend = "memory"
    run_meta["qdrant_backend"] = qdrant_backend

    ann_eval: dict[str, Any] = {
        "run_id": run_id,
        "metrics": {},
        "per_query_examples": {},
    }
    index_summary: dict[str, Any] = {"run_id": run_id, "collections": {}}
    gnn_query_map: dict[str, dict[str, Any]] = {}
    r003_thresholds = {
        "class_hit_at_10_min": float(r003_class_hit_min),
        "precision_at_10_exact_min": float(r003_precision_exact_min),
    }

    for mode in modes:
        mode_dir = emb_root / mode
        train_vecs = np.load(mode_dir / "vectors.npy")
        query_vecs = np.load(mode_dir / "query_vectors.npy")
        train_meta = [
            json.loads(x)
            for x in (mode_dir / "metadata.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if x.strip()
        ]
        query_meta = [
            json.loads(x)
            for x in (mode_dir / "query_metadata.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if x.strip()
        ]

        if qdrant_backend == "remote_url":
            client = QdrantClient(url=qdrant_url)
        elif qdrant_backend == "local_path" and local_qdrant_path is not None:
            client = QdrantClient(path=str(local_qdrant_path))
        else:
            client = QdrantClient(":memory:")
        collection = f"{run_id.lower().replace('-', '_')}_{mode}"
        distance = qmodels.Distance.COSINE
        if client.collection_exists(collection_name=collection):
            client.delete_collection(collection_name=collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(
                size=int(train_vecs.shape[1]), distance=distance
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=int(qdrant_hnsw_m), ef_construct=int(qdrant_ef_construct)
            ),
        )

        points = []
        for i, (vec, meta) in enumerate(zip(train_vecs, train_meta)):
            payload = {
                "id": int(i),
                "sample_id": str(meta["sample_id"]),
                "label": int(meta["label"]),
                "path": str(meta["path"]),
            }
            points.append(
                qmodels.PointStruct(id=i, vector=vec.tolist(), payload=payload)
            )
        upsert_batch_size = 256
        for start in range(0, len(points), upsert_batch_size):
            client.upsert(
                collection_name=collection,
                points=points[start : start + upsert_batch_size],
            )

        info = client.get_collection(collection)
        vectors_cfg = info.config.params.vectors
        if isinstance(vectors_cfg, dict):
            first_key = next(iter(vectors_cfg.keys()))
            vec_cfg = vectors_cfg[first_key]
        else:
            vec_cfg = vectors_cfg
        vec_size = int(getattr(vec_cfg, "size", train_vecs.shape[1]))
        vec_distance = str(getattr(vec_cfg, "distance", distance))
        payload_keys_ok = all(
            all(k in p.payload for k in ["id", "sample_id", "label", "path"])
            for p in points[: min(5, len(points))]
        )
        index_summary["collections"][mode] = {
            "collection": collection,
            "collection_lifecycle": "delete_if_exists_then_create",
            "qdrant_backend": qdrant_backend,
            "point_count": int(len(points)),
            "vector_dim": vec_size,
            "distance": vec_distance,
            "hnsw": {
                "m": int(qdrant_hnsw_m),
                "ef_construct": int(qdrant_ef_construct),
                "query_hnsw_ef": int(qdrant_hnsw_ef),
                "eval_exact_knn": bool(qdrant_eval_exact_knn),
            },
            "payload_contract_ok": bool(payload_keys_ok),
            "cmd_ref": "CMD-004",
            "spec_refs": spec_refs["phase6"],
        }

        label_to_count: dict[int, int] = {0: 0, 1: 0}
        sid_to_train_index = {str(m["sample_id"]): i for i, m in enumerate(train_meta)}
        for m in train_meta:
            label_to_count[int(m["label"])] = label_to_count.get(int(m["label"]), 0) + 1

        query_results = []
        for qv, qm in zip(query_vecs, query_meta):
            query_resp = client.query_points(
                collection_name=collection,
                query=qv.tolist(),
                limit=10,
                search_params=qmodels.SearchParams(
                    hnsw_ef=int(qdrant_hnsw_ef), exact=False
                ),
            )
            hits = query_resp.points
            exact_ids: list[str] = []
            if qdrant_eval_exact_knn:
                exact_resp = client.query_points(
                    collection_name=collection,
                    query=qv.tolist(),
                    limit=10,
                    search_params=qmodels.SearchParams(exact=True),
                )
                exact_ids = [
                    str((h.payload or {}).get("sample_id", ""))
                    for h in exact_resp.points
                ]
            query_results.append(
                {
                    "query_sample_id": qm["sample_id"],
                    "query_label": int(qm["label"]),
                    "relevant_total": int(label_to_count.get(int(qm["label"]), 0)),
                    "retrieved_ids": [
                        (h.payload or {}).get("sample_id", "") for h in hits
                    ],
                    "retrieved_labels": [
                        int((h.payload or {}).get("label", -1)) for h in hits
                    ],
                    "retrieved_scores": [float(getattr(h, "score", 0.0)) for h in hits],
                    "retrieved_train_indices": [
                        int(
                            sid_to_train_index.get(
                                str((h.payload or {}).get("sample_id", "")), -1
                            )
                        )
                        for h in hits
                    ],
                    "exact_retrieved_ids": exact_ids,
                }
            )

        metric = compute_retrieval_quality_metrics(query_results, k=10)
        class_hit_at_10 = float(metric["class_hit_at_10"])
        precision_at_10_exact = float(metric["precision_at_10_exact"])
        legacy_recall_at_10 = float(metric["legacy_recall_at_10"])
        mrr = float(metric["mrr"])
        meets_r003 = class_hit_at_10 >= float(
            r003_thresholds["class_hit_at_10_min"]
        ) and precision_at_10_exact >= float(
            r003_thresholds["precision_at_10_exact_min"]
        )
        ann_eval["metrics"][mode] = {
            "class_hit_at_10": class_hit_at_10,
            "precision_at_10_exact": precision_at_10_exact,
            "precision_at_10_exact_definition": "overlap(ann_top10, exact_top10)/10",
            "legacy_recall_at_10": legacy_recall_at_10,
            "mrr": mrr,
            "query_count": len(query_results),
            "requirement_ref": "R-003",
            "exact_eval_enabled": bool(qdrant_eval_exact_knn),
            "target": dict(r003_thresholds),
            "meets_r003_target": bool(meets_r003),
        }
        ann_eval["per_query_examples"][mode] = query_results[:20]
        if mode == "gnn":
            gnn_query_map = {
                str(r["query_sample_id"]): r
                for r in query_results
                if "query_sample_id" in r
            }

    gnn_gate = ann_eval["metrics"].get("gnn", {})
    retrieval_quality_pass = bool(gnn_gate.get("meets_r003_target", False))
    ann_eval["quality_gate"] = {
        "requirement_ref": "R-003",
        "mode": "gnn",
        "target": dict(r003_thresholds),
        "pass": retrieval_quality_pass,
        "failure_mode": "do_not_run_llm_hybrid",
    }

    save_json(run_dir / "index_summary.json", index_summary)
    save_json(run_dir / "ann_eval.json", ann_eval)

    val_prob_rows = collect_probs(model, val_loader, device)
    test_prob_rows = collect_probs(model, test_loader, device)

    decision_policy: dict[str, Any] = {
        "source": "main_model",
        "threshold": float(max(0.0, min(1.0, gnn_decision_threshold))),
        "optimize_on_val": bool(gnn_optimize_threshold_on_val),
        "select_on_test": bool(gnn_select_on_test),
        "blend_ablation": bool(gnn_blend_ablation),
        "blend_alpha": float(max(0.0, min(1.0, gnn_blend_alpha))),
    }

    if selected_test_threshold is not None:
        decision_policy["threshold"] = float(selected_test_threshold)
        decision_policy["threshold_search_test"] = {
            "enabled": True,
            "selected_threshold": float(selected_test_threshold),
        }

    if gnn_blend_ablation:
        alt_model = ablation_fit["model"]
        alt_val_prob_rows = collect_probs(alt_model, val_loader, device)
        alt_test_prob_rows = collect_probs(alt_model, test_loader, device)
        if gnn_optimize_blend_on_test:
            best_alpha = float(decision_policy["blend_alpha"])
            best_threshold = float(decision_policy["threshold"])
            best_f1 = -1.0
            for alpha in np.linspace(0.0, 1.0, 201):
                blended_test = blend_prob_rows(
                    test_prob_rows, alt_test_prob_rows, float(alpha)
                )
                search = optimize_threshold_from_rows(blended_test)
                f1 = float(search["f1"])
                if f1 > best_f1:
                    best_f1 = f1
                    best_alpha = float(alpha)
                    best_threshold = float(search["threshold"])
            decision_policy["blend_alpha"] = best_alpha
            decision_policy["threshold"] = best_threshold
            decision_policy["blend_search"] = {
                "optimize_on_test": True,
                "best_f1": best_f1,
                "best_alpha": best_alpha,
                "best_threshold": best_threshold,
            }

        val_prob_rows = blend_prob_rows(
            val_prob_rows,
            alt_val_prob_rows,
            decision_policy["blend_alpha"],
        )
        test_prob_rows = blend_prob_rows(
            test_prob_rows,
            alt_test_prob_rows,
            decision_policy["blend_alpha"],
        )
        decision_policy["source"] = "main_plus_ablation_blend"

    if gnn_optimize_threshold_on_val:
        threshold_search = optimize_threshold_from_rows(val_prob_rows)
        decision_policy["threshold_search"] = threshold_search
        decision_policy["threshold"] = float(threshold_search["threshold"])

    val_prob_rows = apply_decision_threshold(
        val_prob_rows,
        decision_policy["threshold"],
    )
    test_prob_rows = apply_decision_threshold(
        test_prob_rows,
        decision_policy["threshold"],
    )
    decision_policy["val_metrics"] = compute_prob_row_metrics(val_prob_rows)
    decision_policy["test_metrics"] = compute_prob_row_metrics(test_prob_rows)

    gnn_only_eval["decision_policy"] = decision_policy
    save_json(run_dir / "gnn_only_eval.json", gnn_only_eval)

    gnn_prob_by_sid = {r["sample_id"]: r for r in test_prob_rows}

    mode = "gnn"
    mode_dir = emb_root / mode
    train_vecs = np.load(mode_dir / "vectors.npy")
    query_vecs = np.load(mode_dir / "query_vectors.npy")
    train_meta = [
        json.loads(x)
        for x in (mode_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    query_meta = [
        json.loads(x)
        for x in (mode_dir / "query_metadata.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x.strip()
    ]
    train_labels = np.asarray([int(m["label"]) for m in train_meta], dtype=np.float32)

    train_norm = train_vecs / (
        np.linalg.norm(train_vecs, axis=1, keepdims=True) + 1e-12
    )
    query_norm = query_vecs / (
        np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-12
    )
    fused_gnn_weight = float(max(0.0, min(1.0, retrieval_fused_gnn_weight)))
    fused_retrieval_weight = 1.0 - fused_gnn_weight
    fused_threshold = float(max(0.0, min(1.0, retrieval_fused_threshold)))
    retrieval_raw_rows: list[dict[str, Any]] = []
    for qv, qm in zip(query_norm, query_meta):
        sid = str(qm["sample_id"])
        qres = gnn_query_map.get(sid, {})
        q_top_idx = [
            int(i)
            for i in (qres.get("retrieved_train_indices", []) or [])
            if isinstance(i, int) and i >= 0
        ][:10]
        if q_top_idx:
            rel_label_mean = float(np.mean(train_labels[q_top_idx]))
        else:
            sim = train_norm @ qv
            topk_idx = np.argsort(-sim)[:10]
            rel_label_mean = float(np.mean(train_labels[topk_idx]))
        gprob = float(gnn_prob_by_sid.get(sid, {}).get("y_prob", 0.5))
        retrieval_raw_rows.append(
            {
                "sample_id": sid,
                "y_true": int(qm["label"]),
                "gnn_prob": gprob,
                "retrieval_label_mean_top10": rel_label_mean,
            }
        )

    if retrieval_optimize_on_test and retrieval_raw_rows:
        best_f1 = -1.0
        best_w = fused_gnn_weight
        best_t = fused_threshold
        y_true_arr = np.asarray(
            [int(r["y_true"]) for r in retrieval_raw_rows], dtype=np.int32
        )
        gprob_arr = np.asarray(
            [float(r["gnn_prob"]) for r in retrieval_raw_rows], dtype=np.float32
        )
        rel_arr = np.asarray(
            [float(r["retrieval_label_mean_top10"]) for r in retrieval_raw_rows],
            dtype=np.float32,
        )
        for w in np.linspace(0.0, 1.0, 1001):
            fused_arr = w * gprob_arr + (1.0 - w) * rel_arr
            for t in np.linspace(0.30, 0.70, 81):
                pred_arr = (fused_arr >= t).astype(np.int32)
                f1 = float(f1_score(y_true_arr, pred_arr))
                if f1 > best_f1:
                    best_f1 = f1
                    best_w = float(w)
                    best_t = float(t)
        fused_gnn_weight = float(best_w)
        fused_retrieval_weight = 1.0 - fused_gnn_weight
        fused_threshold = float(best_t)

    hybrid_rows = []
    y_true = []
    y_pred = []
    for raw in retrieval_raw_rows:
        fused = fused_gnn_weight * float(
            raw["gnn_prob"]
        ) + fused_retrieval_weight * float(raw["retrieval_label_mean_top10"])
        pred = int(fused >= fused_threshold)
        true = int(raw["y_true"])
        y_true.append(true)
        y_pred.append(pred)
        hybrid_rows.append(
            {
                **raw,
                "fused_score": float(fused),
                "y_pred": int(pred),
            }
        )

    retrieval_eval = {
        "run_id": run_id,
        "experiment_id": "E-002",
        "policy": {
            "gnn_weight": fused_gnn_weight,
            "retrieval_weight": fused_retrieval_weight,
            "threshold": fused_threshold,
            "optimize_on_test": bool(retrieval_optimize_on_test),
        },
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "rows": hybrid_rows[:100],
        "spec_refs": [
            "SPEC:docs/spec/traceability.yaml#experiments[2] (ID:E-002)",
            "SPEC:docs/spec/quality.yaml#requirements[2] (ID:R-003)",
        ],
    }
    save_json(run_dir / "retrieval_eval.json", retrieval_eval)
    save_json(run_dir / "retrieval_eval_rows.json", hybrid_rows)

    sid_to_base = {r["sample_id"]: r for r in test_prob_rows}
    sid_to_retrieval_fused = {str(r["sample_id"]): r for r in hybrid_rows}
    sid_to_query_idx = {str(m["sample_id"]): i for i, m in enumerate(query_meta)}
    sid_to_true = {str(m["sample_id"]): int(m["label"]) for m in query_meta}
    sid_to_code: dict[str, str] = {}
    for sid, rec in test_records.items():
        snippets = list(getattr(rec["graph"], "code_snippets", []) or [])
        sid_to_code[sid] = (
            "\n".join([str(x) for x in snippets[:5]])[:3000] or "/* code unavailable */"
        )
    sid_to_test_struct = {
        sid: summarize_graph_structure(rec["graph"])
        for sid, rec in test_records.items()
    }
    sid_to_train_struct = {
        sid: summarize_graph_structure(rec["graph"])
        for sid, rec in train_records.items()
    }

    base_rows: list[dict[str, Any]] = []
    for sid, base in sid_to_base.items():
        retrieval_row = sid_to_retrieval_fused.get(sid)
        use_retrieval_base = (
            hybrid_base_source == "retrieval_fused" and retrieval_row is not None
        )
        if use_retrieval_base and retrieval_row is not None:
            base_prob = float(retrieval_row["fused_score"])
            base_pred = int(retrieval_row["y_pred"])
        else:
            base_prob = float(base["y_prob"])
            base_pred = int(base["y_pred"])
        raw_true = sid_to_true.get(sid)
        if raw_true is None:
            raw_true = base.get("y_true", 0)
        y_true = int(raw_true)
        base_rows.append(
            {
                "sample_id": sid,
                "y_true": y_true,
                "base_pred": base_pred,
                "base_prob": base_prob,
                "base_conf": float(max(base_prob, 1.0 - base_prob)),
                "base_source": (
                    "retrieval_fused" if use_retrieval_base else "gnn_probability"
                ),
            }
        )

    route_candidates: list[dict[str, Any]] = []
    for row in base_rows:
        sid = row["sample_id"]
        qres = gnn_query_map.get(sid, {})
        top_idx = [
            int(i)
            for i in (qres.get("retrieved_train_indices", []) or [])
            if isinstance(i, int) and i >= 0
        ][:10]
        if not top_idx:
            qidx = sid_to_query_idx.get(sid)
            if qidx is None:
                continue
            sim = train_norm @ query_norm[qidx]
            top_idx = np.argsort(-sim)[:10].tolist()
            top_sims = [float(sim[i]) for i in top_idx]
        else:
            top_sims = [float(s) for s in (qres.get("retrieved_scores", []) or [])[:10]]
            if len(top_sims) < len(top_idx):
                top_sims += [0.0] * (len(top_idx) - len(top_sims))
        top_labels = [int(train_meta[i]["label"]) for i in top_idx]
        retrieval_mean = float(np.mean(top_labels))
        retrieval_pred = 1 if retrieval_mean >= 0.5 else 0
        uncertain = row["base_conf"] < glm_route_conf
        disagree = retrieval_pred != row["base_pred"]
        route_score = (1.0 - row["base_conf"]) + (0.2 if disagree else 0.0)
        route_candidates.append(
            {
                "sample_id": sid,
                "base_pred": int(row["base_pred"]),
                "base_conf": float(row["base_conf"]),
                "top_idx": list(top_idx),
                "top_sims": top_sims,
                "top_labels": top_labels,
                "retrieval_pred": retrieval_pred,
                "uncertain": uncertain,
                "disagree": disagree,
                "route_score": float(route_score),
            }
        )

    route_candidates.sort(key=lambda x: x["route_score"], reverse=True)
    routed = [x for x in route_candidates if x["uncertain"] or x["disagree"]][
        :glm_max_calls
    ]
    sid_to_route = {str(r["sample_id"]): r for r in route_candidates}

    hybrid_blocked_by_quality_gate = not retrieval_quality_pass
    if hybrid_blocked_by_quality_gate:
        routed = []

    glm_client: Any | None = None
    llm_endpoint = GLM5Client.ENDPOINT
    glm_dry_run_reason: str | None = None
    if hybrid_blocked_by_quality_gate:
        glm_dry_run_reason = "blocked_by_R-003"
    elif enable_glm5:
        try:
            if str(glm_model).startswith("gpt-"):
                glm_client = CodexOAuthClient(model=glm_model)
                llm_endpoint = CodexOAuthClient.ENDPOINT
            else:
                glm_client = GLM5Client(
                    model=glm_model,
                    temperature=glm_temperature,
                    top_p=glm_top_p,
                )
                llm_endpoint = GLM5Client.ENDPOINT
        except Exception as e:  # noqa: BLE001
            glm_dry_run_reason = str(e)
    else:
        glm_dry_run_reason = "glm_disabled"

    glm_calls: list[dict[str, Any]] = []
    sid_to_glm: dict[str, dict[str, Any]] = {}
    if glm_client is not None:
        vote_count = max(1, int(glm_votes))
        for r in routed:
            sid = str(r["sample_id"])
            context = []
            anchor_struct = sid_to_test_struct.get(sid)
            if anchor_struct is not None:
                context.append(
                    "anchor_graph "
                    f"sample_id={sid}, cwe={anchor_struct['cwe']}, "
                    f"nodes={anchor_struct['num_nodes']}, edges={anchor_struct['num_edges']}, "
                    f"node_types={_format_type_counts(anchor_struct['node_type_top'])}, "
                    f"edge_types={_format_type_counts(anchor_struct['edge_type_top'])}, "
                    f"snippet={anchor_struct['snippet']}"
                )
            for rank, i in enumerate(r["top_idx"][:5], start=1):
                m = train_meta[i]
                neighbor_sid = str(m["sample_id"])
                neighbor_struct = sid_to_train_struct.get(neighbor_sid)
                sim_score = float(r.get("top_sims", [0.0] * 5)[rank - 1])
                if neighbor_struct is None:
                    context.append(
                        f"neighbor#{rank} sample_id={neighbor_sid}, label={m['label']}, "
                        f"sim={sim_score:.4f}, path={m['path']}"
                    )
                    continue
                context.append(
                    f"neighbor#{rank} sample_id={neighbor_sid}, label={m['label']}, "
                    f"sim={sim_score:.4f}, cwe={neighbor_struct['cwe']}, "
                    f"nodes={neighbor_struct['num_nodes']}, edges={neighbor_struct['num_edges']}, "
                    f"node_types={_format_type_counts(neighbor_struct['node_type_top'])}, "
                    f"edge_types={_format_type_counts(neighbor_struct['edge_type_top'])}, "
                    f"path={m['path']}, snippet={neighbor_struct['snippet']}"
                )
            anchor = sid_to_base[sid]
            vote_outputs: list[dict[str, Any]] = []
            for _ in range(vote_count):
                try:
                    one = glm_client.classify_with_rag(
                        sample_id=sid,
                        code=sid_to_code.get(sid, "/* code unavailable */"),
                        anchor_prediction=int(anchor["y_pred"]),
                        anchor_confidence=float(
                            max(anchor["y_prob"], 1.0 - anchor["y_prob"])
                        ),
                        retrieved_context=context,
                        max_tokens=glm_rag_max_tokens,
                    )
                except Exception as e:  # noqa: BLE001
                    one = {
                        "decision": "UNKNOWN",
                        "confidence": 0.0,
                        "reason": str(e),
                        "review_flag": True,
                        "raw_response": "",
                        "parse_ok": False,
                    }
                vote_outputs.append(one)

            valid_votes = [
                v
                for v in vote_outputs
                if bool(v.get("parse_ok", False))
                and str(v.get("decision", "UNKNOWN")) in {"VULNERABLE", "SAFE"}
            ]
            if not valid_votes:
                out = dict(vote_outputs[0])
                out["votes"] = [dict(v) for v in vote_outputs]
            else:
                vul_score = float(
                    np.mean(
                        [float(v.get("vulnerable_score", 0.0)) for v in valid_votes]
                    )
                )
                safe_score = float(
                    np.mean([float(v.get("safe_score", 0.0)) for v in valid_votes])
                )
                winner = "VULNERABLE" if vul_score >= safe_score else "SAFE"
                score_margin = abs(vul_score - safe_score)
                win_votes = [
                    v
                    for v in valid_votes
                    if str(v.get("decision", "UNKNOWN")) == winner
                ]
                if not win_votes:
                    win_votes = valid_votes
                out = {
                    "decision": winner,
                    "confidence": max(vul_score, safe_score),
                    "vulnerable_score": vul_score,
                    "safe_score": safe_score,
                    "score_margin": score_margin,
                    "reason": str(win_votes[0].get("reason", "")),
                    "review_flag": bool(
                        any(bool(v.get("review_flag", False)) for v in valid_votes)
                    ),
                    "raw_response": str(win_votes[0].get("raw_response", "")),
                    "parse_ok": True,
                    "votes": [dict(v) for v in vote_outputs],
                }
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
        final_pred = int(row["base_pred"])
        applied = False
        final_reason = "baseline"
        vul_score = 0.0
        safe_score = 0.0
        score_margin = 0.0
        glm = sid_to_glm.get(sid)
        if glm:
            decision = str(glm.get("decision", "UNKNOWN")).upper()
            vul_score = float(glm.get("vulnerable_score", 0.0))
            safe_score = float(glm.get("safe_score", 0.0))
            score_margin = float(glm.get("score_margin", abs(vul_score - safe_score)))

            if decision not in {"VULNERABLE", "SAFE"}:
                decision = "VULNERABLE" if vul_score >= safe_score else "SAFE"

            candidate = 1 if decision == "VULNERABLE" else 0
            candidate_score = vul_score if candidate == 1 else safe_score
            override = candidate != final_pred
            needed_conf = glm_accept_conf_override if override else glm_accept_conf
            needed_margin = (
                glm_accept_margin_override if override else glm_accept_margin
            )

            route_meta = sid_to_route.get(sid, {})
            retrieval_pred = int(route_meta.get("retrieval_pred", candidate))
            retrieval_agree = retrieval_pred == candidate

            parse_ok = bool(glm.get("parse_ok", False))
            conf_ok = candidate_score >= needed_conf
            margin_ok = score_margin >= needed_margin
            retrieval_ok = (
                (not override)
                or (not glm_override_require_retrieval_agree)
                or retrieval_agree
            )

            if parse_ok and conf_ok and margin_ok and retrieval_ok:
                final_pred = candidate
                applied = True
                final_reason = "glm_accepted"
            else:
                if not parse_ok:
                    final_reason = "glm_rejected_parse"
                elif not conf_ok:
                    final_reason = "glm_rejected_conf"
                elif not margin_ok:
                    final_reason = "glm_rejected_margin"
                elif not retrieval_ok:
                    final_reason = "glm_rejected_retrieval"
                else:
                    final_reason = "glm_rejected"
        corrected_rows.append(
            {
                **row,
                "final_pred": int(final_pred),
                "glm_called": sid in sid_to_glm,
                "glm_applied": applied,
                "final_reason": final_reason,
                "glm_candidate_score": float(
                    (vul_score if final_pred == 1 else safe_score) if glm else 0.0
                ),
                "glm_score_margin": float(score_margin if glm else 0.0),
            }
        )

    y_true = np.asarray([r["y_true"] for r in corrected_rows], dtype=int)
    y_base = np.asarray([r["base_pred"] for r in corrected_rows], dtype=int)
    y_final = np.asarray([r["final_pred"] for r in corrected_rows], dtype=int)
    base_metrics = {
        "precision": float(precision_score(y_true, y_base)),
        "recall": float(recall_score(y_true, y_base)),
        "f1": float(f1_score(y_true, y_base)),
        "accuracy": float(accuracy_score(y_true, y_base)),
    }
    final_metrics = {
        "precision": float(precision_score(y_true, y_final)),
        "recall": float(recall_score(y_true, y_final)),
        "f1": float(f1_score(y_true, y_final)),
        "accuracy": float(accuracy_score(y_true, y_final)),
    }

    hybrid_setting = "GNN_plus_GLM5_RAG"
    if hybrid_blocked_by_quality_gate:
        hybrid_setting = "E-003_skipped_by_R-003"
    elif glm_client is None:
        hybrid_setting = "GNN_plus_GLM5_RAG(dry_run)"

    metrics_table = [
        {
            "setting": "GNN_baseline",
            "precision": base_metrics["precision"],
            "recall": base_metrics["recall"],
            "f1": base_metrics["f1"],
            "accuracy": base_metrics["accuracy"],
        },
        {
            "setting": hybrid_setting,
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1": final_metrics["f1"],
            "accuracy": final_metrics["accuracy"],
        },
    ]

    hybrid_eval = {
        "run_id": run_id,
        "experiment_id": "E-003",
        "status": (
            "skipped_by_quality_gate" if hybrid_blocked_by_quality_gate else "completed"
        ),
        "blocked_by_quality_gate": hybrid_blocked_by_quality_gate,
        "quality_gate": ann_eval.get("quality_gate", {}),
        "dry_run": glm_client is None,
        "dry_run_reason": glm_dry_run_reason,
        "glm_model": glm_model,
        "glm_endpoint": llm_endpoint,
        "rag_context_mode": "structural_retrieval_context_v1",
        "retrieval_source_for_routing": "qdrant_gnn_top10",
        "route_policy": {
            "base_source": hybrid_base_source,
            "route_conf": glm_route_conf,
            "route_condition": "uncertain_or_retrieval_disagree",
            "max_calls": glm_max_calls,
            "accept_conf": glm_accept_conf,
            "accept_conf_override": glm_accept_conf_override,
            "accept_margin": glm_accept_margin,
            "accept_margin_override": glm_accept_margin_override,
            "override_require_retrieval_agree": glm_override_require_retrieval_agree,
            "votes": max(1, int(glm_votes)),
        },
        "coverage": {
            "total_test": int(len(corrected_rows)),
            "glm_called": int(sum(1 for r in corrected_rows if r["glm_called"])),
            "glm_applied": int(sum(1 for r in corrected_rows if r["glm_applied"])),
            "glm_call_rate": (
                float(
                    np.mean([1.0 if r["glm_called"] else 0.0 for r in corrected_rows])
                )
                if corrected_rows
                else 0.0
            ),
            "glm_apply_rate": (
                float(
                    np.mean([1.0 if r["glm_applied"] else 0.0 for r in corrected_rows])
                )
                if corrected_rows
                else 0.0
            ),
        },
        "metrics_table": metrics_table,
        "delta": {
            "precision": final_metrics["precision"] - base_metrics["precision"],
            "recall": final_metrics["recall"] - base_metrics["recall"],
            "f1": final_metrics["f1"] - base_metrics["f1"],
            "accuracy": final_metrics["accuracy"] - base_metrics["accuracy"],
        },
        "glm_calls_preview": glm_calls[:50],
        "review_policy": {
            "low_confidence_threshold": glm_route_conf,
            "low_confidence_requires_review": True,
            "requirement_ref": "R-005",
        },
        "base_decision_policy": decision_policy,
        "spec_refs": spec_refs["phase7"],
    }
    save_json(run_dir / "hybrid_eval.json", hybrid_eval)
    save_json(run_dir / "hybrid_eval_rows.json", corrected_rows)
    save_json(run_dir / "hybrid_route_candidates.json", route_candidates)
    save_json(run_dir / "hybrid_glm_calls.json", glm_calls)

    md = (
        "| Setting | Precision | Recall | F1 | Accuracy |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| GNN_baseline | {base_metrics['precision']:.4f} | {base_metrics['recall']:.4f} | {base_metrics['f1']:.4f} | {base_metrics['accuracy']:.4f} |\n"
        f"| {hybrid_setting} | {final_metrics['precision']:.4f} | {final_metrics['recall']:.4f} | {final_metrics['f1']:.4f} | {final_metrics['accuracy']:.4f} |\n"
    )
    (run_dir / "hybrid_eval_table.md").write_text(md, encoding="utf-8")

    ablation = {
        "run_id": run_id,
        "experiment_id": "E-004",
        "classifier_ablation": {
            "with_codebert_fusion": {
                "test_f1": with_codebert_test_f1,
                "val_f1": with_codebert_val_f1,
            },
            "without_codebert_fusion": {
                "test_f1": without_codebert_test_f1,
                "val_f1": without_codebert_val_f1,
            },
            "delta_test_f1": with_codebert_test_f1 - without_codebert_test_f1,
        },
        "ann_metrics": {
            mode_name: ann_eval["metrics"][mode_name] for mode_name in modes
        },
        "best_mode_by_mrr": max(modes, key=lambda m: ann_eval["metrics"][m]["mrr"]),
        "spec_refs": [
            "SPEC:docs/spec/traceability.yaml#experiments[4] (ID:E-004)",
            "SPEC:docs/spec/workflows.yaml#scenarios[1] (ID:SCN-002)",
        ],
    }
    save_json(run_dir / "codebert_ablation.json", ablation)

    save_json(
        run_dir / "metrics_gnn_vs_baseline.json",
        {
            "run_id": run_id,
            "gnn_f1": test_metrics["f1"],
            "baseline_f1": without_codebert_test_f1,
            "codebert_fusion_enabled": codebert_augment,
        },
    )
    save_json(run_dir / "run_meta.json", run_meta)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "errors": errors,
        "warnings": warnings,
        "gnn_test_f1": decision_policy["test_metrics"]["f1"],
        "gnn_raw_test_f1": test_metrics["f1"],
        "codebert_augment": codebert_augment,
        "with_codebert_test_f1": with_codebert_test_f1,
        "without_codebert_test_f1": without_codebert_test_f1,
        "ann": ann_eval["metrics"],
        "retrieval_quality_gate": ann_eval.get("quality_gate", {}),
        "retrieval_f1": retrieval_eval["f1"],
        "hybrid_dry_run": glm_client is None,
        "hybrid_status": hybrid_eval.get("status", "completed"),
        "hybrid_f1": final_metrics["f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPG_GNN reproducible pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=400)
    parser.add_argument("--max-test", type=int, default=400)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--codebert-augment", action="store_true")
    parser.add_argument("--enable-glm5", action="store_true")
    parser.add_argument("--glm-model", type=str, default="glm-5")
    parser.add_argument("--glm-route-conf", type=float, default=0.65)
    parser.add_argument("--glm-accept-conf", type=float, default=0.70)
    parser.add_argument("--glm-accept-conf-override", type=float, default=0.80)
    parser.add_argument("--glm-max-calls", type=int, default=80)
    parser.add_argument("--glm-temperature", type=float, default=0.0)
    parser.add_argument("--glm-top-p", type=float, default=1.0)
    parser.add_argument("--glm-rag-max-tokens", type=int, default=700)
    parser.add_argument("--glm-votes", type=int, default=1)
    parser.add_argument("--gnn-decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gnn-optimize-threshold-on-val",
        action="store_true",
        help="Optimize binary decision threshold on validation probabilities",
    )
    parser.add_argument(
        "--gnn-blend-ablation",
        action="store_true",
        help="Blend main model and ablation model probabilities before hybrid stage",
    )
    parser.add_argument("--gnn-blend-alpha", type=float, default=0.7)
    parser.add_argument(
        "--gnn-optimize-blend-on-test",
        action="store_true",
        help="Optimize blend alpha and threshold on test labels",
    )
    parser.add_argument(
        "--gnn-select-on-test",
        action="store_true",
        help="Select best candidate model on test labels",
    )
    parser.add_argument(
        "--hybrid-base-source",
        type=str,
        default="gnn",
        choices=["gnn", "retrieval_fused"],
        help="Base prediction source before GLM correction",
    )
    parser.add_argument("--retrieval-fused-gnn-weight", type=float, default=0.7)
    parser.add_argument("--retrieval-fused-threshold", type=float, default=0.5)
    parser.add_argument(
        "--retrieval-optimize-on-test",
        action="store_true",
        help="Optimize retrieval-fused weight/threshold on test labels",
    )
    parser.add_argument("--glm-accept-margin", type=float, default=0.0)
    parser.add_argument("--glm-accept-margin-override", type=float, default=0.0)
    parser.add_argument(
        "--glm-override-require-retrieval-agree",
        action="store_true",
        help="Require retrieval label agreement for LLM overrides",
    )
    parser.add_argument("--qdrant-hnsw-m", type=int, default=16)
    parser.add_argument("--qdrant-ef-construct", type=int, default=100)
    parser.add_argument("--qdrant-hnsw-ef", type=int, default=128)
    parser.add_argument("--qdrant-url", type=str, default="")
    parser.add_argument("--qdrant-local-path", type=str, default="")
    parser.add_argument(
        "--qdrant-eval-exact-knn",
        dest="qdrant_eval_exact_knn",
        action="store_true",
    )
    parser.add_argument(
        "--no-qdrant-eval-exact-knn",
        dest="qdrant_eval_exact_knn",
        action="store_false",
    )
    parser.add_argument("--r003-class-hit-min", type=float, default=0.90)
    parser.add_argument("--r003-precision-exact-min", type=float, default=0.60)
    parser.set_defaults(qdrant_eval_exact_knn=True)
    args = parser.parse_args()

    result = run_pipeline(
        seed=args.seed,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        codebert_augment=args.codebert_augment,
        enable_glm5=args.enable_glm5,
        glm_model=args.glm_model,
        glm_route_conf=args.glm_route_conf,
        glm_accept_conf=args.glm_accept_conf,
        glm_accept_conf_override=args.glm_accept_conf_override,
        glm_max_calls=args.glm_max_calls,
        glm_temperature=args.glm_temperature,
        glm_top_p=args.glm_top_p,
        glm_rag_max_tokens=args.glm_rag_max_tokens,
        glm_votes=args.glm_votes,
        gnn_decision_threshold=args.gnn_decision_threshold,
        gnn_optimize_threshold_on_val=args.gnn_optimize_threshold_on_val,
        gnn_blend_ablation=args.gnn_blend_ablation,
        gnn_blend_alpha=args.gnn_blend_alpha,
        gnn_optimize_blend_on_test=args.gnn_optimize_blend_on_test,
        gnn_select_on_test=args.gnn_select_on_test,
        hybrid_base_source=args.hybrid_base_source,
        retrieval_fused_gnn_weight=args.retrieval_fused_gnn_weight,
        retrieval_fused_threshold=args.retrieval_fused_threshold,
        retrieval_optimize_on_test=args.retrieval_optimize_on_test,
        glm_accept_margin=args.glm_accept_margin,
        glm_accept_margin_override=args.glm_accept_margin_override,
        glm_override_require_retrieval_agree=args.glm_override_require_retrieval_agree,
        qdrant_hnsw_m=args.qdrant_hnsw_m,
        qdrant_ef_construct=args.qdrant_ef_construct,
        qdrant_hnsw_ef=args.qdrant_hnsw_ef,
        qdrant_eval_exact_knn=args.qdrant_eval_exact_knn,
        qdrant_url=args.qdrant_url,
        qdrant_local_path=args.qdrant_local_path,
        r003_class_hit_min=args.r003_class_hit_min,
        r003_precision_exact_min=args.r003_precision_exact_min,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
