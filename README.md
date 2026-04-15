# CPG-GNN · Bilingual README

> Code Property Graph + GNN + retrieval/LLM evaluation workspace for vulnerability detection experiments.

## Choose your language

- [English](README.en.md)
- [한국어](README.ko.md)

## Quick snapshot

- Python-based experiment repository centered on graph-based vulnerability detection and hybrid evaluation flows.
- Main code lives in `scripts/`, `gpt.py`, `tests/`, and `requirements.txt`.
- `data/`, `checkpoints/`, and `results/` are kept as tracked empty directories with `.gitkeep` so local artifacts can be organized without committing large files.

## Repo at a glance

```text
.
├── gpt.py
├── requirements.txt
├── scripts/
├── tests/
├── data/
├── checkpoints/
└── results/
```

## Notes

- This root page is only a landing page. Full project documentation is available in the language-specific READMEs above.
- Some legacy scripts and tests still reference removed `docs/spec` and `docs/traceability` assets, so the current trimmed repository should be treated as a runnable code snapshot rather than a fully reconciled release package.
