#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from codex_oauth_client import CodexOAuthClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH = ROOT / "data" / "split.json"
DEFAULT_DEVIGN_GLOB = "devign_test.jsonl"


@dataclass
class DevignSample:
    sample_id: str
    label: int
    code: str
    project: str
    commit_id: str
    idx: int
    source_file: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def extract_graph_label(graph_obj: Any) -> int:
    y = getattr(graph_obj, "y", None)
    if y is None:
        return 0
    if hasattr(y, "view"):
        return int(y.view(-1)[0].item())
    return int(y)


def extract_first_snippet(graph_obj: Any) -> str:
    snippets = getattr(graph_obj, "code_snippets", None)
    if isinstance(snippets, list):
        for s in snippets:
            if isinstance(s, str) and s.strip():
                return s.strip()
    return ""


def load_internal_rag_pool(split_path: Path) -> tuple[dict[int, list[str]], float]:
    split_obj = load_json(split_path)
    train_paths = [Path(p) for p in split_obj.get("train", [])]

    by_label: dict[int, list[str]] = {0: [], 1: []}
    labels: list[int] = []

    for rel in train_paths:
        abs_path = ROOT / rel
        if not abs_path.exists():
            continue
        graph_obj = torch.load(abs_path, weights_only=False)
        label = extract_graph_label(graph_obj)
        label = 1 if label == 1 else 0
        snippet = extract_first_snippet(graph_obj)
        if not snippet:
            snippet = f"path={rel}"
        by_label[label].append(snippet[:500])
        labels.append(label)

    prior = float(sum(labels) / len(labels)) if labels else 0.5
    return by_label, prior


def has_function_like_body(code: str) -> bool:
    text = str(code or "").strip()
    if len(text) < 80:
        return False
    if "{" not in text or "}" not in text or "(" not in text or ")" not in text:
        return False

    pattern = re.compile(
        r"(?m)^\s*(?!if\b|for\b|while\b|switch\b|catch\b|else\b)"
        r"(?:template\s*<[^>]+>\s*)?"
        r"(?:[A-Za-z_][\w:<>\[\]\s\*&~,]*?\s+)?"
        r"[A-Za-z_~][\w:<>~]*\s*\([^;{}]*\)\s*"
        r"(?:const\s*)?(?:noexcept(?:\s*\([^)]*\))?\s*)?"
        r"(?:->\s*[\w:<>\s\*&]+)?\s*\{"
    )
    if pattern.search(text):
        return True

    fallback_pattern = re.compile(
        r"(?m)^\s*(?!if\b|for\b|while\b|switch\b|catch\b|else\b)"
        r"[A-Za-z_~][\w:<>~]*\s*\([^;{}]*\)\s*$\n\s*\{"
    )
    return bool(fallback_pattern.search(text))


def load_devign_samples(
    *,
    input_files: list[Path],
    max_records: int,
    max_code_chars: int,
    require_function_body: bool,
) -> tuple[list[DevignSample], dict[str, int]]:
    samples: list[DevignSample] = []
    stats: dict[str, int] = defaultdict(int)
    seen = 0

    for input_file in input_files:
        with input_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["json_decode_error"] += 1
                    continue

                seen += 1
                if max_records > 0 and seen > max_records:
                    return samples, dict(stats)

                raw_label = record.get("target")
                try:
                    label = int(raw_label)
                except (TypeError, ValueError):
                    stats["invalid_label"] += 1
                    continue
                label = 1 if label == 1 else 0

                code = str(record.get("func") or "")[:max_code_chars]
                if not code.strip():
                    stats["empty_code"] += 1
                    continue

                if require_function_body and not has_function_like_body(code):
                    stats["without_function_body"] += 1
                    continue

                project = str(record.get("project") or "unknown")
                commit_id = str(record.get("commit_id") or "unknown")
                idx_val = record.get("idx")
                try:
                    idx = int(idx_val)
                except (TypeError, ValueError):
                    idx = -1

                sample_id = f"{project}:{commit_id}:{idx}:{line_no}"
                samples.append(
                    DevignSample(
                        sample_id=sample_id,
                        label=label,
                        code=code,
                        project=project,
                        commit_id=commit_id,
                        idx=idx,
                        source_file=f"{input_file.name}:{line_no}",
                    )
                )
                stats["added"] += 1

    return samples, dict(stats)


def select_balanced_subset(
    *, samples: list[DevignSample], sample_per_label: int, seed: int
) -> tuple[list[DevignSample], dict[str, int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[DevignSample]] = {0: [], 1: []}
    for s in samples:
        by_label[s.label].append(s)

    k = min(sample_per_label, len(by_label[0]), len(by_label[1]))
    if k <= 0:
        return [], {
            "safe_available": len(by_label[0]),
            "vul_available": len(by_label[1]),
            "sampled_per_label": 0,
            "sampled_total": 0,
        }

    selected = rng.sample(by_label[0], k) + rng.sample(by_label[1], k)
    rng.shuffle(selected)
    stats = {
        "safe_available": len(by_label[0]),
        "vul_available": len(by_label[1]),
        "sampled_per_label": k,
        "sampled_total": 2 * k,
    }
    return selected, stats


def build_rag_context(
    *, internal_pool: dict[int, list[str]], anchor_pred: int
) -> list[str]:
    context: list[str] = ["anchor_graph external_sample, dataset=Devign"]
    same = internal_pool.get(anchor_pred, [])
    opp = internal_pool.get(1 - anchor_pred, [])

    if same:
        context.append(
            f"neighbor#1 label={anchor_pred}, source=internal_train, snippet={{ {same[0][:240]} }}"
        )
    if opp:
        context.append(
            f"neighbor#2 label={1 - anchor_pred}, source=internal_train, snippet={{ {opp[0][:240]} }}"
        )
    if len(context) < 3:
        context.append("neighbor#fallback label=unknown, source=none")
    return context


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "n": 0.0,
        }

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float((2 * precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = float((tp + tn) / n)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "n": float(n),
    }


def evaluate_devign(
    *,
    selected: list[DevignSample],
    model: str,
    internal_pool: dict[int, list[str]],
    prior: float,
    timeout: int,
) -> dict[str, Any]:
    client = CodexOAuthClient(model=model, timeout=timeout, max_retries=0)

    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    parse_ok_count = 0

    for i, s in enumerate(selected, start=1):
        anchor_pred = 1 if prior >= 0.5 else 0
        anchor_conf = max(prior, 1.0 - prior)
        context = build_rag_context(
            internal_pool=internal_pool, anchor_pred=anchor_pred
        )

        try:
            out = client.classify_with_rag(
                sample_id=f"devign_{i:04d}",
                code=s.code,
                anchor_prediction=anchor_pred,
                anchor_confidence=anchor_conf,
                retrieved_context=context,
                max_tokens=700,
            )
        except Exception as e:  # noqa: BLE001
            out = {
                "decision": "UNKNOWN",
                "confidence": float(anchor_conf),
                "parse_ok": False,
                "reason": f"request_error: {e}",
                "response_model": client.model,
            }

        parse_ok = bool(out.get("parse_ok", False))
        if parse_ok:
            parse_ok_count += 1

        decision = str(out.get("decision", "UNKNOWN")).upper()
        if decision == "VULNERABLE":
            pred = 1
        elif decision == "SAFE":
            pred = 0
        else:
            pred = anchor_pred

        y_true.append(int(s.label))
        y_pred.append(int(pred))

        rows.append(
            {
                "sample_id": f"devign_{i:04d}",
                "origin_sample_id": s.sample_id,
                "project": s.project,
                "commit_id": s.commit_id,
                "idx": s.idx,
                "source_file": s.source_file,
                "y_true": int(s.label),
                "anchor_pred": int(anchor_pred),
                "anchor_confidence": float(anchor_conf),
                "decision": decision,
                "pred": int(pred),
                "confidence": float(out.get("confidence", 0.0)),
                "parse_ok": parse_ok,
                "reason": str(out.get("reason", "")),
                "response_model": str(out.get("response_model", "")),
            }
        )

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["parse_ok_rate"] = float(parse_ok_count / len(rows)) if rows else 0.0

    by_project: dict[str, dict[str, float]] = {}
    for project in sorted(set(r["project"] for r in rows)):
        p_rows = [r for r in rows if r["project"] == project]
        p_true = [int(r["y_true"]) for r in p_rows]
        p_pred = [int(r["pred"]) for r in p_rows]
        by_project[project] = compute_binary_metrics(p_true, p_pred)

    return {
        "rows": rows,
        "metrics": metrics,
        "by_project": by_project,
    }


def build_report_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Devign Generalization Evaluation")
    lines.append("")
    lines.append("## Overall")
    m = result["eval"]["metrics"]
    lines.append(f"- n: {int(m['n'])}")
    lines.append(f"- precision: {m['precision']:.4f}")
    lines.append(f"- recall: {m['recall']:.4f}")
    lines.append(f"- f1: {m['f1']:.4f}")
    lines.append(f"- accuracy: {m['accuracy']:.4f}")
    lines.append(f"- parse_ok_rate: {m['parse_ok_rate']:.4f}")
    lines.append("")
    lines.append("## By Project")
    for project, stat in result["eval"]["by_project"].items():
        lines.append(
            f"- {project}: n={int(stat['n'])}, f1={stat['f1']:.4f}, "
            f"precision={stat['precision']:.4f}, recall={stat['recall']:.4f}, accuracy={stat['accuracy']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Devign no-training generalization eval"
    )
    parser.add_argument(
        "--devign-dir", type=Path, default=ROOT / "data" / "raw" / "devign"
    )
    parser.add_argument("--devign-glob", type=str, default=DEFAULT_DEVIGN_GLOB)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--model", type=str, default="gpt-5.3-codex")
    parser.add_argument("--sample-per-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--max-code-chars", type=int, default=5000)
    parser.add_argument("--require-function-body", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_files = sorted(args.devign_dir.glob(args.devign_glob))
    if not input_files:
        raise FileNotFoundError(
            f"No Devign files found under {args.devign_dir} with glob {args.devign_glob}"
        )

    internal_pool, prior = load_internal_rag_pool(args.split_path)
    devign_samples, load_stats = load_devign_samples(
        input_files=input_files,
        max_records=max(0, int(args.max_records)),
        max_code_chars=max(500, int(args.max_code_chars)),
        require_function_body=bool(args.require_function_body),
    )

    selected, sampled_stats = select_balanced_subset(
        samples=devign_samples,
        sample_per_label=max(1, int(args.sample_per_label)),
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError("No selected samples. Check input data and filters.")

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_id = f"DEVIGN-GEN-{timestamp}"
    out_dir = ROOT / "results" / run_id
    ensure_dir(out_dir)

    eval_result = evaluate_devign(
        selected=selected,
        model=args.model,
        internal_pool=internal_pool,
        prior=prior,
        timeout=args.timeout,
    )

    result = {
        "run_id": run_id,
        "devign_dir": str(args.devign_dir),
        "devign_glob": args.devign_glob,
        "input_files": [str(p) for p in input_files],
        "split_path": str(args.split_path),
        "model": args.model,
        "sample_per_label": int(args.sample_per_label),
        "seed": int(args.seed),
        "max_records": int(args.max_records),
        "max_code_chars": int(args.max_code_chars),
        "require_function_body": bool(args.require_function_body),
        "internal_prior_vul": float(prior),
        "devign_load_stats": load_stats,
        "sampled_stats": sampled_stats,
        "eval": eval_result,
    }

    save_json(out_dir / "devign_generalization_eval.json", result)
    (out_dir / "devign_generalization_eval.md").write_text(
        build_report_markdown(result), encoding="utf-8"
    )

    summary = {
        "run_id": run_id,
        "model": args.model,
        "n": int(eval_result["metrics"]["n"]),
        "f1": float(eval_result["metrics"]["f1"]),
        "precision": float(eval_result["metrics"]["precision"]),
        "recall": float(eval_result["metrics"]["recall"]),
        "accuracy": float(eval_result["metrics"]["accuracy"]),
        "parse_ok_rate": float(eval_result["metrics"]["parse_ok_rate"]),
        "result_path": str(out_dir),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
