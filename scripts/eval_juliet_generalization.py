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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from codex_oauth_client import CodexOAuthClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JULIET_ROOT = ROOT / "data" / "raw" / "juliet" / "juliet_v1.3" / "C"
DEFAULT_SPLIT_PATH = ROOT / "data" / "split.json"


@dataclass
class JulietSample:
    cwe: str
    label: int
    file_name: str
    file_path: Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def normalize_cwe(raw: Any) -> str:
    text = str(raw or "UNKNOWN").strip()
    if not text:
        return "UNKNOWN"
    m = re.match(r"CWE-?(\d+)", text)
    if m:
        return f"CWE-{m.group(1)}"
    return text


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


def load_internal_rag_pool(
    split_path: Path,
) -> tuple[dict[str, dict[int, list[str]]], dict[str, float]]:
    split_obj = load_json(split_path)
    train_paths = [Path(p) for p in split_obj.get("train", [])]

    by_cwe: dict[str, dict[int, list[str]]] = defaultdict(lambda: {0: [], 1: []})
    prior_count: dict[str, list[int]] = defaultdict(list)

    for rel in train_paths:
        abs_path = ROOT / rel
        if not abs_path.exists():
            continue
        graph_obj = torch.load(abs_path, weights_only=False)
        cwe = normalize_cwe(getattr(graph_obj, "cwe_type", "UNKNOWN"))
        label = extract_graph_label(graph_obj)
        snippet = extract_first_snippet(graph_obj)
        if not snippet:
            snippet = f"path={rel}"
        by_cwe[cwe][label].append(snippet[:500])
        prior_count[cwe].append(label)

    priors: dict[str, float] = {}
    for cwe, arr in prior_count.items():
        if arr:
            priors[cwe] = float(sum(arr) / len(arr))
    return by_cwe, priors


def build_juliet_file_index(juliet_root: Path) -> dict[str, list[Path]]:
    testcases_root = juliet_root / "testcases"
    index: dict[str, list[Path]] = defaultdict(list)
    for p in testcases_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".c", ".cpp", ".cc", ".cxx"}:
            continue
        index[p.name].append(p)
    return index


def load_juliet_manifest_samples(juliet_root: Path) -> list[JulietSample]:
    import xml.etree.ElementTree as ET

    manifest_path = juliet_root / "manifest.xml"
    root = ET.parse(manifest_path).getroot()
    file_index = build_juliet_file_index(juliet_root)

    dedup: dict[tuple[str, str], JulietSample] = {}
    for testcase in root.findall("testcase"):
        for f in testcase.findall("file"):
            raw_path = str(f.get("path") or "").strip()
            file_name = Path(raw_path).name
            m = re.match(r"CWE(\d+)_", file_name)
            if not m:
                continue
            cwe = f"CWE-{m.group(1)}"
            label = 1 if f.find("flaw") is not None else 0
            candidates = file_index.get(file_name, [])
            if not candidates:
                continue
            key = (file_name, cwe)
            if key not in dedup:
                dedup[key] = JulietSample(
                    cwe=cwe,
                    label=label,
                    file_name=file_name,
                    file_path=candidates[0],
                )
            elif label == 1:
                dedup[key].label = 1
    return list(dedup.values())


def select_balanced_subset(
    *,
    samples: list[JulietSample],
    cwe_filter: set[str],
    sample_per_cwe_label: int,
    seed: int,
) -> tuple[list[JulietSample], dict[str, dict[str, int]]]:
    rng = random.Random(seed)
    by_cwe: dict[str, dict[int, list[JulietSample]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for s in samples:
        if s.cwe not in cwe_filter:
            continue
        by_cwe[s.cwe][s.label].append(s)

    selected: list[JulietSample] = []
    stats: dict[str, dict[str, int]] = {}
    for cwe in sorted(by_cwe):
        safe_list = by_cwe[cwe][0]
        vul_list = by_cwe[cwe][1]
        k = min(sample_per_cwe_label, len(safe_list), len(vul_list))
        if k <= 0:
            continue
        selected_safe = rng.sample(safe_list, k)
        selected_vul = rng.sample(vul_list, k)
        selected.extend(selected_safe)
        selected.extend(selected_vul)
        stats[cwe] = {
            "safe_available": len(safe_list),
            "vul_available": len(vul_list),
            "sampled_per_label": k,
            "sampled_total": 2 * k,
        }
    rng.shuffle(selected)
    return selected, stats


def build_rag_context(
    *,
    cwe: str,
    internal_pool: dict[str, dict[int, list[str]]],
    anchor_pred: int,
) -> list[str]:
    pool = internal_pool.get(cwe)
    if pool is None:
        pool = {0: [], 1: []}

    context: list[str] = [
        f"anchor_graph external_sample, cwe={cwe}, note=Juliet filtered overlap evaluation"
    ]

    same = pool.get(anchor_pred, [])
    opp = pool.get(1 - anchor_pred, [])
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


def infer_family_prefix(file_name: str) -> str | None:
    stem = Path(file_name).stem
    m = re.match(r"(.+_\d+)[a-z]$", stem)
    if m:
        return m.group(1)
    m = re.match(r"(.+_\d+)_(?:bad|good|goodG2B|goodB2G)$", stem)
    if m:
        return m.group(1)
    return None


def build_augmented_code(
    file_path: Path,
    *,
    max_chars: int = 6000,
    max_related: int = 3,
) -> tuple[str, list[str]]:
    target_rel = str(file_path.relative_to(ROOT))
    chunk_list: list[str] = []
    used_paths: list[str] = [target_rel]

    target_text = file_path.read_text(encoding="utf-8", errors="ignore")
    chunk_list.append(f"/* target_file: {target_rel} */\n{target_text}")

    family = infer_family_prefix(file_path.name)
    if family:
        siblings = sorted(
            p
            for p in file_path.parent.glob(f"{family}*")
            if p.is_file()
            and p.suffix.lower() in {".c", ".cpp", ".cc", ".cxx"}
            and p != file_path
        )
        for p in siblings[:max_related]:
            rel = str(p.relative_to(ROOT))
            used_paths.append(rel)
            text = p.read_text(encoding="utf-8", errors="ignore")
            chunk_list.append(f"/* related_file: {rel} */\n{text}")

    code = "\n\n".join(chunk_list)
    return code[:max_chars], used_paths


def evaluate_juliet(
    *,
    selected: list[JulietSample],
    model: str,
    internal_pool: dict[str, dict[int, list[str]]],
    priors: dict[str, float],
    timeout: int,
    use_related_files: bool,
) -> dict[str, Any]:
    client = CodexOAuthClient(model=model, timeout=timeout)

    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    parse_ok_count = 0

    for i, s in enumerate(selected, start=1):
        p_vul = priors.get(s.cwe, 0.5)
        anchor_pred = 1 if p_vul >= 0.5 else 0
        anchor_conf = max(p_vul, 1.0 - p_vul)
        context = build_rag_context(
            cwe=s.cwe, internal_pool=internal_pool, anchor_pred=anchor_pred
        )
        if use_related_files:
            code, used_files = build_augmented_code(s.file_path)
        else:
            code = s.file_path.read_text(encoding="utf-8", errors="ignore")[:3500]
            used_files = [str(s.file_path.relative_to(ROOT))]

        out = client.classify_with_rag(
            sample_id=f"juliet_{i:04d}",
            code=code,
            anchor_prediction=anchor_pred,
            anchor_confidence=anchor_conf,
            retrieved_context=context,
            max_tokens=700,
        )

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
                "sample_id": f"juliet_{i:04d}",
                "cwe": s.cwe,
                "file_name": s.file_name,
                "file_path": str(s.file_path.relative_to(ROOT)),
                "used_files": used_files,
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

    precision = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))

    by_cwe: dict[str, dict[str, float]] = {}
    for cwe in sorted(set(r["cwe"] for r in rows)):
        cwe_rows = [r for r in rows if r["cwe"] == cwe]
        cwe_true = [int(r["y_true"]) for r in cwe_rows]
        cwe_pred = [int(r["pred"]) for r in cwe_rows]
        by_cwe[cwe] = {
            "n": float(len(cwe_rows)),
            "precision": float(precision_score(cwe_true, cwe_pred)),
            "recall": float(recall_score(cwe_true, cwe_pred)),
            "f1": float(f1_score(cwe_true, cwe_pred)),
            "accuracy": float(accuracy_score(cwe_true, cwe_pred)),
        }

    return {
        "rows": rows,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc,
            "n": float(len(rows)),
            "parse_ok_rate": float(parse_ok_count / len(rows)) if rows else 0.0,
        },
        "by_cwe": by_cwe,
    }


def build_report_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Juliet Generalization Evaluation")
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
    lines.append("## CWE Subset")
    for cwe, stat in result["eval"]["by_cwe"].items():
        lines.append(
            f"- {cwe}: n={int(stat['n'])}, f1={stat['f1']:.4f}, "
            f"precision={stat['precision']:.4f}, recall={stat['recall']:.4f}, accuracy={stat['accuracy']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Juliet 1.3 CWE-overlap generalization evaluation"
    )
    parser.add_argument("--juliet-root", type=Path, default=DEFAULT_JULIET_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--model", type=str, default="gpt-5.3-codex")
    parser.add_argument("--sample-per-cwe-label", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--use-related-files", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    internal_pool, priors = load_internal_rag_pool(args.split_path)
    internal_cwes = {c for c in priors if c.startswith("CWE-")}

    juliet_samples = load_juliet_manifest_samples(args.juliet_root)
    juliet_cwes = {s.cwe for s in juliet_samples}
    overlap_cwes = internal_cwes & juliet_cwes

    selected, sampled_stats = select_balanced_subset(
        samples=juliet_samples,
        cwe_filter=overlap_cwes,
        sample_per_cwe_label=max(1, int(args.sample_per_cwe_label)),
        seed=args.seed,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_id = f"JULIET-GEN-{timestamp}"
    out_dir = ROOT / "results" / run_id
    ensure_dir(out_dir)

    eval_result = evaluate_juliet(
        selected=selected,
        model=args.model,
        internal_pool=internal_pool,
        priors=priors,
        timeout=args.timeout,
        use_related_files=bool(args.use_related_files),
    )

    result = {
        "run_id": run_id,
        "juliet_root": str(args.juliet_root),
        "split_path": str(args.split_path),
        "model": args.model,
        "sample_per_cwe_label": int(args.sample_per_cwe_label),
        "seed": int(args.seed),
        "use_related_files": bool(args.use_related_files),
        "internal_cwe_count": int(len(internal_cwes)),
        "juliet_cwe_count": int(len(juliet_cwes)),
        "overlap_cwes": sorted(overlap_cwes),
        "sampled_cwe_stats": sampled_stats,
        "eval": eval_result,
    }

    save_json(out_dir / "juliet_generalization_eval.json", result)
    (out_dir / "juliet_generalization_eval.md").write_text(
        build_report_markdown(result), encoding="utf-8"
    )

    summary = {
        "run_id": run_id,
        "model": args.model,
        "overlap_cwes": sorted(overlap_cwes),
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
