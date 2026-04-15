#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

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
    nums = []
    for p in RESULTS_DIR.glob(f"{prefix}*"):
        if not p.is_dir():
            continue
        m = re.match(rf"{re.escape(prefix)}(\d{{3}})$", p.name)
        if m:
            nums.append(int(m.group(1)))
    seq = max(nums) + 1 if nums else 1
    return f"{prefix}{seq:03d}"


def parse_sample_id_from_path(path_str: str) -> str:
    stem = Path(path_str).stem
    if stem.startswith("safe_") or stem.startswith("vul_"):
        return "sample_" + stem.split("_", 1)[1]
    return stem


def parse_label_from_path(path_str: str) -> int | None:
    stem = Path(path_str).stem
    if stem.startswith("safe_"):
        return 0
    if stem.startswith("vul_"):
        return 1
    return None


def robust_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "q1": 0.0, "q3": 0.0, "iqr": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
    }


def detect_iqr_outliers(values: list[int]) -> int:
    if len(values) < 8:
        return 0
    arr = np.asarray(values, dtype=np.float64)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return int(((arr < low) | (arr > high)).sum())


def evaluate_graph(path: Path, expected_dim: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "load_ok": False,
        "sample_id": parse_sample_id_from_path(str(path)),
        "x_dim_ok": False,
        "finite_ok": False,
        "y_ok": False,
        "shape_ok": False,
        "n_nodes": 0,
        "n_edges": 0,
        "label_from_file": parse_label_from_path(str(path)),
        "label_from_graph": None,
        "errors": [],
    }
    if not path.exists():
        out["errors"].append("MISSING_PATH")
        return out
    try:
        data = torch.load(path, weights_only=False)
        out["load_ok"] = True
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"LOAD_FAILED:{e}")
        return out

    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    y = getattr(data, "y", None)
    node_type = getattr(data, "node_type", None)
    edge_type = getattr(data, "edge_type", None)
    sid = getattr(data, "sample_id", None)
    if isinstance(sid, str) and sid:
        out["sample_id"] = sid

    if isinstance(x, torch.Tensor) and x.ndim == 2:
        out["n_nodes"] = int(x.shape[0])
        out["x_dim_ok"] = int(x.shape[1]) == expected_dim
        out["finite_ok"] = bool(torch.isfinite(x).all().item())
    else:
        out["errors"].append("X_INVALID")

    if isinstance(y, torch.Tensor) and y.numel() >= 1:
        label = int(y.view(-1)[0].item())
        out["label_from_graph"] = label
        out["y_ok"] = label in (0, 1)
    else:
        out["errors"].append("Y_INVALID")

    if (
        isinstance(edge_index, torch.Tensor)
        and edge_index.ndim == 2
        and edge_index.shape[0] == 2
    ):
        out["n_edges"] = int(edge_index.shape[1])
        if out["n_nodes"] > 0 and out["n_edges"] >= 0:
            min_idx = int(edge_index.min().item()) if edge_index.numel() > 0 else 0
            max_idx = int(edge_index.max().item()) if edge_index.numel() > 0 else 0
            edge_range_ok = (
                min_idx >= 0 and max_idx < out["n_nodes"]
                if edge_index.numel() > 0
                else True
            )
        else:
            edge_range_ok = False
    else:
        edge_range_ok = False
        out["errors"].append("EDGE_INDEX_INVALID")

    node_type_ok = (
        isinstance(node_type, torch.Tensor)
        and node_type.ndim == 1
        and int(node_type.shape[0]) == out["n_nodes"]
    )
    edge_type_ok = (
        isinstance(edge_type, torch.Tensor)
        and edge_type.ndim == 1
        and int(edge_type.shape[0]) == out["n_edges"]
    )
    out["shape_ok"] = bool(edge_range_ok and node_type_ok and edge_type_ok)
    if not node_type_ok:
        out["errors"].append("NODE_TYPE_SHAPE_INVALID")
    if not edge_type_ok:
        out["errors"].append("EDGE_TYPE_SHAPE_INVALID")

    file_label = out["label_from_file"]
    graph_label = out["label_from_graph"]
    out["label_consistency_ok"] = (
        file_label is None or graph_label is None or file_label == graph_label
    )
    if not out["label_consistency_ok"]:
        out["errors"].append("LABEL_MISMATCH")

    out["mandatory_ok"] = bool(
        out["load_ok"]
        and out["x_dim_ok"]
        and out["finite_ok"]
        and out["y_ok"]
        and out["shape_ok"]
        and out["label_consistency_ok"]
    )
    return out


def split_records(records: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    valid = [
        r for r in records if r["mandatory_ok"] and r["label_from_graph"] in (0, 1)
    ]
    labels = [int(r["label_from_graph"]) for r in valid]
    paths = [str(r["path"]) for r in valid]
    if len(valid) < 10:
        return {"train": [], "val": [], "test": []}

    try:
        tr_val_paths, te_paths, tr_val_labels, _ = train_test_split(
            paths,
            labels,
            test_size=0.15,
            random_state=seed,
            stratify=labels,
        )
        val_ratio = 0.15 / 0.85
        tr_paths, va_paths, _, _ = train_test_split(
            tr_val_paths,
            tr_val_labels,
            test_size=val_ratio,
            random_state=seed,
            stratify=tr_val_labels,
        )
    except Exception:
        idx = list(range(len(paths)))
        random.Random(seed).shuffle(idx)
        n = len(idx)
        n_te = int(round(n * 0.15))
        n_va = int(round(n * 0.15))
        te_idx = idx[:n_te]
        va_idx = idx[n_te : n_te + n_va]
        tr_idx = idx[n_te + n_va :]
        tr_paths = [paths[i] for i in tr_idx]
        va_paths = [paths[i] for i in va_idx]
        te_paths = [paths[i] for i in te_idx]

    return {"train": tr_paths, "val": va_paths, "test": te_paths}


def run(
    split_path: Path,
    metadata_path: Path,
    expected_dim: int,
    quality_threshold: float,
    sample_limit: int,
    rebuild_if_poor: bool,
    rebuild_split_path: Path,
    seed: int,
) -> dict[str, Any]:
    split_obj = load_json(split_path)
    metadata_obj = load_json(metadata_path)
    split_paths: dict[str, list[str]] = {
        k: list(v) for k, v in split_obj.items() if isinstance(v, list)
    }
    all_paths = (
        split_paths.get("train", [])
        + split_paths.get("val", [])
        + split_paths.get("test", [])
    )
    if sample_limit > 0 and len(all_paths) > sample_limit:
        rng = random.Random(seed)
        idx = list(range(len(all_paths)))
        rng.shuffle(idx)
        keep = set(idx[:sample_limit])
        limited = [all_paths[i] for i in range(len(all_paths)) if i in keep]
        all_paths = limited

    train_set = set(split_paths.get("train", []))
    val_set = set(split_paths.get("val", []))
    test_set = set(split_paths.get("test", []))
    overlap_count = len(
        (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    )
    duplicate_count = len(all_paths) - len(set(all_paths))
    missing_paths = [p for p in all_paths if not (ROOT / p).exists()]
    metadata_total = int(metadata_obj.get("total_samples", 0))
    split_total = (
        len(split_paths.get("train", []))
        + len(split_paths.get("val", []))
        + len(split_paths.get("test", []))
    )
    total_delta = abs(metadata_total - split_total)

    records = [evaluate_graph(ROOT / p, expected_dim=expected_dim) for p in all_paths]
    counter_errors = Counter()
    node_counts: list[int] = []
    edge_counts: list[int] = []
    split_label_count: dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    cpg_xml_missing = 0
    cpg_meta_missing = 0
    for r in records:
        for err in r["errors"]:
            counter_errors[err] += 1
        if r["n_nodes"] > 0:
            node_counts.append(int(r["n_nodes"]))
        if r["n_edges"] >= 0:
            edge_counts.append(int(r["n_edges"]))

        sid = r["sample_id"]
        if sid.startswith("sample_"):
            xml_path = ROOT / "data" / "cpg" / f"{sid}.xml"
            meta_path = ROOT / "data" / "cpg" / f"{sid}_meta.json"
            if not xml_path.exists():
                cpg_xml_missing += 1
            if not meta_path.exists():
                cpg_meta_missing += 1

    label_of_path: dict[str, int] = {}
    for r in records:
        if r["label_from_graph"] in (0, 1):
            label_of_path[r["path"]] = int(r["label_from_graph"])
        elif r["label_from_file"] in (0, 1):
            label_of_path[r["path"]] = int(r["label_from_file"])

    for split_name in ["train", "val", "test"]:
        for p in split_paths.get(split_name, []):
            if p in label_of_path:
                split_label_count[split_name][label_of_path[p]] += 1

    def pos_rate(c: Counter[int]) -> float:
        total = c[0] + c[1]
        return float(c[1] / total) if total > 0 else 0.0

    train_pos = pos_rate(split_label_count["train"])
    val_pos = pos_rate(split_label_count["val"])
    test_pos = pos_rate(split_label_count["test"])
    split_pos_delta = max(
        abs(train_pos - val_pos), abs(train_pos - test_pos), abs(val_pos - test_pos)
    )

    mandatory_fail = (
        overlap_count > 0
        or duplicate_count > 0
        or len(missing_paths) > 0
        or counter_errors["X_INVALID"] > 0
        or counter_errors["EDGE_INDEX_INVALID"] > 0
        or counter_errors["Y_INVALID"] > 0
        or counter_errors["LABEL_MISMATCH"] > 0
        or any("LOAD_FAILED" in k for k in counter_errors)
    )

    dim_mismatch_count = sum(1 for r in records if r["load_ok"] and not r["x_dim_ok"])
    non_finite_count = sum(1 for r in records if r["load_ok"] and not r["finite_ok"])
    shape_invalid_count = sum(1 for r in records if r["load_ok"] and not r["shape_ok"])
    mandatory_ok_count = sum(1 for r in records if r["mandatory_ok"])

    score = 100.0
    score -= min(25.0, overlap_count * 5.0)
    score -= min(25.0, duplicate_count * 0.5)
    score -= min(25.0, len(missing_paths) * 0.5)
    score -= min(20.0, dim_mismatch_count * 1.0)
    score -= min(20.0, non_finite_count * 1.0)
    score -= min(20.0, shape_invalid_count * 1.0)
    score -= min(10.0, abs(total_delta - 1.0) * 5.0) if total_delta > 1 else 0.0
    score -= min(10.0, split_pos_delta * 100.0)
    cpg_missing_ratio = 0.0
    if records:
        cpg_missing_ratio = float(
            (cpg_xml_missing + cpg_meta_missing) / (2.0 * len(records))
        )
    score -= min(8.0, cpg_missing_ratio * 100.0)
    node_outliers = detect_iqr_outliers(node_counts)
    edge_outliers = detect_iqr_outliers(edge_counts)
    if records:
        outlier_ratio = float((node_outliers + edge_outliers) / (2.0 * len(records)))
    else:
        outlier_ratio = 0.0
    score -= min(7.0, outlier_ratio * 100.0)
    score = max(0.0, min(100.0, score))

    if score >= 90:
        grade = "excellent"
    elif score >= 75:
        grade = "good"
    elif score >= 60:
        grade = "fair"
    else:
        grade = "poor"

    poor_quality = mandatory_fail or score < quality_threshold

    rebuild_result: dict[str, Any] = {
        "triggered": False,
        "reason": "quality_ok",
        "split_output": str(rebuild_split_path),
        "new_split_counts": {"train": 0, "val": 0, "test": 0},
    }
    if rebuild_if_poor and poor_quality:
        rebuilt = split_records(records, seed=seed)
        save_json(rebuild_split_path, rebuilt)
        rebuild_result = {
            "triggered": True,
            "reason": "quality_below_threshold",
            "split_output": str(rebuild_split_path),
            "new_split_counts": {
                "train": len(rebuilt.get("train", [])),
                "val": len(rebuilt.get("val", [])),
                "test": len(rebuilt.get("test", [])),
            },
            "dropped_count": len(records) - sum(len(v) for v in rebuilt.values()),
        }

    return {
        "timestamp_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "summary": {
            "records_scanned": len(records),
            "mandatory_ok_count": mandatory_ok_count,
            "score": score,
            "grade": grade,
            "quality_threshold": quality_threshold,
            "poor_quality": poor_quality,
        },
        "data_gate_alignment": {
            "G-D1": {
                "overlap_count": overlap_count,
                "duplicate_count": duplicate_count,
                "missing_split_paths": len(missing_paths),
                "pass": overlap_count == 0
                and duplicate_count == 0
                and len(missing_paths) == 0,
            },
            "G-D2": {
                "metadata_total": metadata_total,
                "split_total": split_total,
                "delta": total_delta,
                "pass": total_delta <= 1,
            },
            "G-D3": {
                "expected_dim": expected_dim,
                "dim_mismatch_count": dim_mismatch_count,
                "pass": dim_mismatch_count == 0,
            },
            "G-D4": {
                "non_finite_count": non_finite_count,
                "pass": non_finite_count == 0,
            },
        },
        "advanced_quality": {
            "shape_invalid_count": shape_invalid_count,
            "label_mismatch_count": int(counter_errors["LABEL_MISMATCH"]),
            "split_positive_rate": {
                "train": train_pos,
                "val": val_pos,
                "test": test_pos,
                "max_delta": split_pos_delta,
            },
            "cpg_alignment": {
                "missing_xml_count": cpg_xml_missing,
                "missing_meta_count": cpg_meta_missing,
                "missing_ratio": cpg_missing_ratio,
            },
            "node_stats": robust_stats(node_counts),
            "edge_stats": robust_stats(edge_counts),
            "node_outlier_count": node_outliers,
            "edge_outlier_count": edge_outliers,
        },
        "error_breakdown": dict(counter_errors),
        "rebuild": rebuild_result,
        "spec_refs": [
            "SPEC:docs/spec/workflows.yaml#commands[0] (ID:CMD-001)",
            "SPEC:docs/spec/data.yaml#integrityGates[0] (ID:G-D1)",
            "SPEC:docs/spec/data.yaml#integrityGates[1] (ID:G-D2)",
            "SPEC:docs/spec/data.yaml#integrityGates[2] (ID:G-D3)",
            "SPEC:docs/spec/data.yaml#integrityGates[3] (ID:G-D4)",
            "SPEC:docs/spec/quality.yaml#requirements[0] (ID:R-001)",
            "SPEC:docs/spec/quality.yaml#requirements[1] (ID:R-002)",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PyG quality and optionally rebuild split"
    )
    parser.add_argument("--split-path", type=Path, default=ROOT / "data" / "split.json")
    parser.add_argument(
        "--metadata-path", type=Path, default=ROOT / "data" / "metadata.json"
    )
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument("--quality-threshold", type=float, default=75.0)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--rebuild-if-poor", action="store_true")
    parser.add_argument(
        "--rebuild-split-path", type=Path, default=ROOT / "data" / "split_rebuilt.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.output is None:
        run_id = next_run_id(datetime.now().strftime("%Y%m%d"))
        run_dir = ROOT / "results" / run_id
        ensure_dir(run_dir)
        output = run_dir / "pyg_quality_report.json"
    else:
        output = args.output
        ensure_dir(output.parent)

    report = run(
        split_path=args.split_path,
        metadata_path=args.metadata_path,
        expected_dim=args.expected_dim,
        quality_threshold=args.quality_threshold,
        sample_limit=args.sample_limit,
        rebuild_if_poor=args.rebuild_if_poor,
        rebuild_split_path=args.rebuild_split_path,
        seed=args.seed,
    )
    save_json(output, report)
    print(
        json.dumps(
            {
                "report_path": str(output),
                "score": report["summary"]["score"],
                "grade": report["summary"]["grade"],
                "poor_quality": report["summary"]["poor_quality"],
                "rebuild_triggered": report["rebuild"]["triggered"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
