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
DEFAULT_SPLIT_PATH = ROOT / "data" / "split.json"
DEFAULT_CVEFIXES_GLOB = "training_set_part*.jsonl"


@dataclass
class CVEFixesSample:
    sample_id: str
    cwe: str
    label: int
    code: str
    cve_id: str
    file_change_id: str
    programming_language: str
    source_file: str


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


def extract_cwe_from_record(record: dict[str, Any]) -> str:
    cwe_info = record.get("cwe_info")
    if isinstance(cwe_info, list):
        for item in cwe_info:
            if not isinstance(item, dict):
                continue
            cwe_id = item.get("cwe_id")
            cwe = normalize_cwe(cwe_id)
            if cwe.startswith("CWE-"):
                return cwe
    return "UNKNOWN"


def is_c_family_language(lang: str) -> bool:
    t = str(lang or "").strip().lower()
    return t in {"c", "c++", "c/c++", "objective-c", "objective-c++"}


def build_code_from_file_change(fc: dict[str, Any], label: int, max_chars: int) -> str:
    if label == 1:
        code = str(fc.get("code_before") or "")
    else:
        code = str(fc.get("code_after") or "")
    return code[:max_chars]


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


def load_cvefixes_samples(
    *,
    input_files: list[Path],
    max_cve_records: int,
    max_code_chars: int,
    require_function_body: bool,
) -> tuple[list[CVEFixesSample], dict[str, int]]:
    samples: list[CVEFixesSample] = []
    stats: dict[str, int] = defaultdict(int)

    cve_seen = 0
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

                cve_seen += 1
                if max_cve_records > 0 and cve_seen > max_cve_records:
                    return samples, dict(stats)

                cwe = extract_cwe_from_record(record)
                if not cwe.startswith("CWE-"):
                    stats["unknown_cwe"] += 1
                    continue

                cve_id = str(record.get("cve_id") or "UNKNOWN-CVE")
                fixes_info = record.get("fixes_info")
                if not isinstance(fixes_info, list):
                    stats["missing_fixes_info"] += 1
                    continue

                for fix in fixes_info:
                    if not isinstance(fix, dict):
                        continue
                    commit_details = fix.get("commit_details")
                    if not isinstance(commit_details, dict):
                        continue
                    file_changes = commit_details.get("file_changes")
                    if not isinstance(file_changes, list):
                        continue

                    for fc in file_changes:
                        if not isinstance(fc, dict):
                            continue

                        lang = str(fc.get("programming_language") or "")
                        if not is_c_family_language(lang):
                            stats["non_c_family_skipped"] += 1
                            continue

                        file_change_id = str(fc.get("file_change_id") or "")
                        if not file_change_id:
                            stats["missing_file_change_id"] += 1
                            continue

                        code_before = str(fc.get("code_before") or "")
                        code_after = str(fc.get("code_after") or "")
                        if not code_before.strip() and not code_after.strip():
                            stats["empty_code_pair"] += 1
                            continue

                        if code_before.strip():
                            before_code = build_code_from_file_change(
                                fc, label=1, max_chars=max_code_chars
                            )
                            before_ok = (
                                not require_function_body
                            ) or has_function_like_body(before_code)
                            if not before_ok:
                                stats["before_without_function_body"] += 1
                            else:
                                samples.append(
                                    CVEFixesSample(
                                        sample_id=f"{cve_id}:{file_change_id}:before",
                                        cwe=cwe,
                                        label=1,
                                        code=before_code,
                                        cve_id=cve_id,
                                        file_change_id=file_change_id,
                                        programming_language=lang,
                                        source_file=f"{input_file.name}:{line_no}",
                                    )
                                )
                                stats["added_vulnerable"] += 1

                        if code_after.strip():
                            after_code = build_code_from_file_change(
                                fc, label=0, max_chars=max_code_chars
                            )
                            after_ok = (
                                not require_function_body
                            ) or has_function_like_body(after_code)
                            if not after_ok:
                                stats["after_without_function_body"] += 1
                            else:
                                samples.append(
                                    CVEFixesSample(
                                        sample_id=f"{cve_id}:{file_change_id}:after",
                                        cwe=cwe,
                                        label=0,
                                        code=after_code,
                                        cve_id=cve_id,
                                        file_change_id=file_change_id,
                                        programming_language=lang,
                                        source_file=f"{input_file.name}:{line_no}",
                                    )
                                )
                                stats["added_safe"] += 1

    return samples, dict(stats)


def select_balanced_subset(
    *,
    samples: list[CVEFixesSample],
    cwe_filter: set[str],
    sample_per_cwe_label: int,
    seed: int,
) -> tuple[list[CVEFixesSample], dict[str, dict[str, int]]]:
    rng = random.Random(seed)
    by_cwe: dict[str, dict[int, list[CVEFixesSample]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for s in samples:
        if s.cwe not in cwe_filter:
            continue
        by_cwe[s.cwe][s.label].append(s)

    selected: list[CVEFixesSample] = []
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
        f"anchor_graph external_sample, cwe={cwe}, note=CVEFixes filtered overlap evaluation"
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


def evaluate_cvefixes(
    *,
    selected: list[CVEFixesSample],
    model: str,
    internal_pool: dict[str, dict[int, list[str]]],
    priors: dict[str, float],
    timeout: int,
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

        out = client.classify_with_rag(
            sample_id=f"cvefixes_{i:04d}",
            code=s.code,
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
                "sample_id": f"cvefixes_{i:04d}",
                "origin_sample_id": s.sample_id,
                "cwe": s.cwe,
                "cve_id": s.cve_id,
                "file_change_id": s.file_change_id,
                "programming_language": s.programming_language,
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
    lines.append("# CVEFixes Generalization Evaluation")
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
        description="CVEFixes CWE-overlap generalization evaluation"
    )
    parser.add_argument(
        "--cvefixes-dir", type=Path, default=ROOT / "data" / "raw" / "cvefixes"
    )
    parser.add_argument("--cvefixes-glob", type=str, default=DEFAULT_CVEFIXES_GLOB)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--model", type=str, default="gpt-5.3-codex")
    parser.add_argument("--sample-per-cwe-label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-cve-records", type=int, default=4000)
    parser.add_argument("--max-code-chars", type=int, default=5000)
    parser.add_argument("--max-overlap-cwes", type=int, default=0)
    parser.add_argument("--require-function-body", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_files = sorted(args.cvefixes_dir.glob(args.cvefixes_glob))
    if not input_files:
        raise FileNotFoundError(
            f"No input files found under {args.cvefixes_dir} with glob {args.cvefixes_glob}"
        )

    internal_pool, priors = load_internal_rag_pool(args.split_path)
    internal_cwes = {c for c in priors if c.startswith("CWE-")}

    cvefixes_samples, load_stats = load_cvefixes_samples(
        input_files=input_files,
        max_cve_records=max(0, int(args.max_cve_records)),
        max_code_chars=max(500, int(args.max_code_chars)),
        require_function_body=bool(args.require_function_body),
    )
    cvefixes_cwes = {s.cwe for s in cvefixes_samples}
    overlap_cwes = internal_cwes & cvefixes_cwes
    if int(args.max_overlap_cwes) > 0:
        overlap_cwes = set(sorted(overlap_cwes)[: int(args.max_overlap_cwes)])

    selected, sampled_stats = select_balanced_subset(
        samples=cvefixes_samples,
        cwe_filter=overlap_cwes,
        sample_per_cwe_label=max(1, int(args.sample_per_cwe_label)),
        seed=args.seed,
    )

    if not selected:
        raise RuntimeError(
            "No selected samples after overlap filtering. "
            "Increase --max-cve-records or verify input schema."
        )

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_id = f"CVEFIXES-GEN-{timestamp}"
    out_dir = ROOT / "results" / run_id
    ensure_dir(out_dir)

    eval_result = evaluate_cvefixes(
        selected=selected,
        model=args.model,
        internal_pool=internal_pool,
        priors=priors,
        timeout=args.timeout,
    )

    result = {
        "run_id": run_id,
        "cvefixes_dir": str(args.cvefixes_dir),
        "cvefixes_glob": args.cvefixes_glob,
        "input_files": [str(p) for p in input_files],
        "split_path": str(args.split_path),
        "model": args.model,
        "sample_per_cwe_label": int(args.sample_per_cwe_label),
        "seed": int(args.seed),
        "max_cve_records": int(args.max_cve_records),
        "max_code_chars": int(args.max_code_chars),
        "max_overlap_cwes": int(args.max_overlap_cwes),
        "require_function_body": bool(args.require_function_body),
        "internal_cwe_count": int(len(internal_cwes)),
        "cvefixes_cwe_count": int(len(cvefixes_cwes)),
        "overlap_cwes": sorted(overlap_cwes),
        "cvefixes_load_stats": load_stats,
        "sampled_cwe_stats": sampled_stats,
        "eval": eval_result,
    }

    save_json(out_dir / "cvefixes_generalization_eval.json", result)
    (out_dir / "cvefixes_generalization_eval.md").write_text(
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
