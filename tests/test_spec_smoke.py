from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_spec_files_parse() -> None:
    spec_dir = ROOT / "docs" / "spec"
    files = sorted(spec_dir.glob("*.yaml"))
    assert files
    for f in files:
        with f.open("r", encoding="utf-8") as fh:
            obj = yaml.safe_load(fh)
        assert isinstance(obj, dict)
        assert "specVersion" in obj


def test_latest_run_artifacts_exist() -> None:
    results_dir = ROOT / "results"
    runs = sorted([p for p in results_dir.glob("RUN-*") if p.is_dir()])
    assert runs
    required = [
        "data_integrity_report.json",
        "gnn_only_eval.json",
        "embedding_summary.json",
        "index_summary.json",
        "ann_eval.json",
        "retrieval_eval.json",
        "hybrid_eval.json",
        "codebert_ablation.json",
        "run_meta.json",
    ]
    matched = []
    for run in runs:
        if all((run / rel).exists() for rel in required):
            matched.append(run)
    assert matched, "no full pipeline run artifacts found"
