# Spec Index

Canonical index of machine-readable spec artifacts and stable IDs.

## Documents

- `docs/spec/manifest.yaml`
- `docs/spec/agent_contract.yaml`
- `docs/spec/glossary.yaml`
- `docs/spec/domain.yaml`
- `docs/spec/workflows.yaml`
- `docs/spec/interfaces.yaml`
- `docs/spec/data.yaml`
- `docs/spec/quality.yaml`
- `docs/spec/traceability.yaml`

## Stable IDs

### Context IDs
- `BC-1` Data Ingestion
- `BC-2` Graph Construction
- `BC-3` Embedding and Retrieval
- `BC-4` Detection and Verification
- `BC-5` Evaluation and Governance

### Invariant IDs
- `INV-001` split overlap == 0
- `INV-002` vector dim == 768
- `INV-003` no NaN/Inf in vectors

### Command IDs
- `CMD-001` ValidateDataBaseline
- `CMD-002` BuildPyGFromCPG
- `CMD-003` BuildEmbeddings
- `CMD-004` IndexVectorsToQdrant
- `CMD-005` RunHybridInference

### Requirement IDs
- `R-001` integrity: split_overlap == 0
- `R-002` vector_contract: vector_dim == 768
- `R-003` retrieval_quality: class_hit_at_10 >= 0.90 and precision_at_10_exact >= 0.60
- `R-004` reproducibility: same_seed_metric_delta <= 0.01
- `R-005` safety: low_confidence_requires_review == true

### Experiment IDs
- `E-000` Data Integrity Gate
- `E-001` GNN only baseline
- `E-002` GNN + Retrieval
- `E-003` GNN + Retrieval + LLM
- `E-004` CodeBERT augmentation ablation

### Claim IDs
- `C-001` CPG structure contributes to detection performance
- `C-002` embedding retrieval contributes to verification quality
- `C-003` LLM correction can improve performance conditionally
- `C-004` pipeline must be reproducible
