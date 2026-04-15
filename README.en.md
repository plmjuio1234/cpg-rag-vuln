# CPG-GNN

[English](README.en.md) | [한국어](README.ko.md) | [Home](README.md)

> A Python experiment repository for vulnerability detection using code property graphs, graph neural networks, retrieval-style evidence, and LLM-assisted evaluation workflows.

## Overview

This repository contains the working code for a graph-based vulnerability detection pipeline built around PyTorch, PyTorch Geometric, and LLM-assisted evaluators. The codebase mixes model training, hybrid evaluation, dataset-specific generalization scripts, and local experiment bookkeeping.

The current public snapshot is intentionally trimmed. Large artifacts such as datasets, checkpoints, and generated results are not committed, but their directories are preserved so the local workspace structure remains stable.

## What is in this repository

- **Model and evaluation pipeline** in `scripts/pipeline_run.py`
- **Dataset-specific evaluators** for Devign, Juliet, and CVEFixes in `scripts/`
- **Hybrid GLM/GNN evaluation flow** in `scripts/run_glm5_hybrid_eval.py`
- **OAuth-based GPT utility** in `gpt.py`
- **Basic smoke test scaffold** in `tests/test_spec_smoke.py`

## Repository layout

```text
.
├── gpt.py
├── requirements.txt
├── scripts/
│   ├── pipeline_run.py
│   ├── eval_devign_generalization.py
│   ├── eval_juliet_generalization.py
│   ├── eval_cvefixes_generalization.py
│   ├── run_glm5_hybrid_eval.py
│   ├── tune_gnn_plateau.py
│   ├── evaluate_pyg_quality.py
│   ├── generate_paper_figures.py
│   └── ...
├── tests/
├── data/
├── checkpoints/
└── results/
```

## Environment and dependencies

The repository currently declares the following core packages in `requirements.txt`:

- `torch`
- `torch-geometric`
- `transformers`
- `qdrant-client`
- `numpy`, `pandas`, `scikit-learn`
- `pytest`, `ruff`, `black`
- `pyyaml`, `jsonschema`

Typical setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main workflows

### 1. End-to-end graph / hybrid pipeline

`scripts/pipeline_run.py` is the largest orchestration script in the repository. It defines:

- run ID generation under `results/`
- graph loading from `data/`
- GNN model definition and training
- checkpoint writing under `checkpoints/gnn`
- hybrid evaluation support using external clients such as `CodexOAuthClient` and `GLM5Client`

### 2. Dataset-specific generalization evaluation

The repository includes dedicated evaluation scripts for multiple benchmark families:

- `scripts/eval_devign_generalization.py`
- `scripts/eval_juliet_generalization.py`
- `scripts/eval_cvefixes_generalization.py`

These scripts load dataset-specific samples, normalize labels and metadata, and prepare evaluation records for LLM-assisted or hybrid scoring flows.

### 3. Hybrid GLM5 evaluation

`scripts/run_glm5_hybrid_eval.py` contains another evaluation path that combines graph embeddings, GNN outputs, and GLM-based judging logic while writing run artifacts under `results/`.

### 4. Utility and tuning scripts

Other scripts support auxiliary workflows such as:

- hyperparameter tuning (`scripts/tune_gnn_plateau.py`)
- quality checks over PyG artifacts (`scripts/evaluate_pyg_quality.py`)
- figure/report generation helpers (`scripts/generate_paper_figures.py`, `scripts/render_paper_figures_png.py`)

## Data and generated artifacts

The repository keeps these directories in version control as placeholders only:

- `data/`
- `checkpoints/`
- `results/`

They currently use `.gitkeep` so the folder structure exists in GitHub while large local files remain ignored. In practice:

- put raw or processed datasets under `data/`
- store trained model checkpoints under `checkpoints/`
- write run outputs, metrics, and experiment reports under `results/`

## Validation and current caveats

There is a smoke test file at `tests/test_spec_smoke.py`, but it still expects `docs/spec` and `docs/traceability` assets that are no longer included in the current trimmed repository snapshot. That means the code tree and the remaining test assumptions are not fully reconciled yet.

Related caveats:

- some orchestration code in `scripts/pipeline_run.py` still references removed documentation paths
- local experiment outputs may exist on a developer machine but are not committed to the repository
- README examples in this snapshot should be treated as repository guidance, not as a claim that every legacy path is immediately runnable without adjustment

## Suggested starting points

If you are exploring the repository for the first time, this order is the most useful:

1. `requirements.txt` — dependency surface
2. `scripts/pipeline_run.py` — main orchestration flow
3. `scripts/eval_juliet_generalization.py` — dataset-specific evaluation pattern
4. `scripts/eval_devign_generalization.py` and `scripts/eval_cvefixes_generalization.py` — parallel variants
5. `gpt.py` — external GPT/OAuth utility

## Example commands

These are repository-level examples based on file presence, not guaranteed one-click demo commands:

```bash
python scripts/pipeline_run.py --help
python scripts/eval_juliet_generalization.py --help
python scripts/eval_devign_generalization.py --help
python scripts/eval_cvefixes_generalization.py --help
python scripts/run_glm5_hybrid_eval.py --help
pytest tests/test_spec_smoke.py
```

## Status

This repository is best understood as an actively edited experiment workspace rather than a polished benchmark release. The main code paths are present, but some documentation-era dependencies were removed during repository cleanup.

If you want to turn it into a cleaner public release, the next practical steps are:

1. reconcile `tests/` with the current trimmed repository layout
2. remove or update legacy `docs/spec` references in scripts
3. add a reproducible minimal dataset/example flow
4. document required environment variables for external model clients
