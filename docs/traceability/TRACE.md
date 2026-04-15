# Traceability Ledger

This ledger maps spec IDs to implementation targets and verification evidence.

## Mapping Table

| Claim | Requirement | Experiment | Run Artifact | Code Target | Status |
|---|---|---|---|---|---|
| C-001 | R-001 | E-001, E-002 | `results/RUN-20260227-007/metrics_gnn_vs_baseline.json` | `scripts/pipeline_run.py` | in_progress |
| C-002 | R-002, R-003 | E-002 | `results/RUN-20260227-007/ann_eval.json`; `results/RUN-20260227-007/retrieval_eval.json`; `results/hnsw_sweep_external_aligned_20260225_225712.md`; `results/r003_threshold_sensitivity_aligned_20260225_225726.md` | `scripts/pipeline_run.py` | in_progress |
| C-003 | R-003, R-005 | E-003 | `results/RUN-20260225-020/hybrid_eval.json`; `results/RUN-20260225-015/hybrid_eval_glm5.json`; `results/RUN-20260226-001/hybrid_eval.json`; `results/RUN-20260227-004/hybrid_eval.json`; `results/RUN-20260227-008/hybrid_eval.json`; `results/RUN-20260227-009/hybrid_eval.json`; `results/RUN-20260227-010/hybrid_eval.json`; `results/glm_hparam_tuning_20260227_044136.md`; `results/glm_hparam_exhaustive_offline_20260227_055049.md`; `results/glm_hparam_final_report_20260227.md` | `scripts/pipeline_run.py`, `scripts/run_glm5_hybrid_eval.py` | in_progress |
| C-004 | R-004 | E-001, E-002, E-003, E-004 | `results/RUN-20260227-007/run_meta.json` | `scripts/pipeline_run.py` | in_progress |

## Update Rules

- Any new ID in `docs/spec/*.yaml` must be added here.
- Any completed run must update `Status` and concrete artifact path.
- PR notes must include `SPEC:<file>#<path> (ID:<id>)` references.
