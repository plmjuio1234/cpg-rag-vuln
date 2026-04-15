# AGENTS.md

Project operating contract for OpenCode agents.

## 0) Communication Language

- User-facing responses MUST be in Korean by default.
- Use English only when the user explicitly requests English.

## 1) Mission

Build and verify a reproducible vulnerability-detection pipeline from:

`CPG -> PyG -> GNN -> (optional) CodeBERT augmentation -> VectorDB retrieval -> LLM verification`

Current repository state is data+paper+specs first. Implementation must follow specs in `docs/spec`.

## 2) Source of Truth (Priority Order)

Agents MUST read and follow files in this exact order:

1. `docs/spec/manifest.yaml`
2. `docs/spec/agent_contract.yaml`
3. `docs/spec/INDEX.md`
4. `docs/spec/domain.yaml`
5. `docs/spec/workflows.yaml`
6. `docs/spec/interfaces.yaml`
7. `docs/spec/data.yaml`
8. `docs/spec/quality.yaml`
9. `docs/spec/traceability.yaml`
10. `docs/spec/glossary.yaml`
11. `docs/traceability/TRACE.md`

Reference-only docs (non-authoritative):

- `docs/legacy/`

If prose and spec conflict, specs win.

## 3) Required Working Method

### 3.1 Spec-first execution

- Do not implement from memory or prose.
- Every action maps to a `CMD-*`, `INV-*`, `R-*`, or `E-*` item.
- Any new behavior requires spec update before code change.

### 3.2 ID discipline

- Use stable IDs exactly as defined in `docs/spec/manifest.yaml`.
- No free-form aliases in code, docs, or outputs.
- New IDs must be appended, never repurposed.

### 3.3 Reference protocol

When implementing a task, agents must produce this mapping in notes/PR text:

- `Spec`: exact file + key path (example: `docs/spec/workflows.yaml#commands[CMD-003]`)
- `Code`: exact files changed
- `Validation`: exact commands run + results
- `Traceability`: `C-* -> R-* -> E-* -> RUN-*`

### 3.4 Spec citation format

Use this exact format in implementation notes and PR text:

- `SPEC:<file>#<path> (ID:<id>)`
- Example: `SPEC:docs/spec/workflows.yaml#commands[CMD-003] (ID:CMD-003)`

## 4) Execution Order (Default)

Unless explicitly overridden by user, execute in this order:

1. `ValidateDataBaseline`
2. `BuildPyGFromCPG`
3. `BuildEmbeddings`
4. `IndexVectorsToQdrant`
5. `RunHybridInference`

Blocking failures are defined in `docs/spec/agent_contract.yaml`.

## 5) Coding Standards

Default language/runtime until changed by project config:

- Python 3.11+
- PyTorch/PyG for graph learning
- Transformers for CodeBERT
- Qdrant for vector index

Code style requirements:

- ASCII-only source by default.
- Type hints required for public functions.
- Small pure functions; avoid hidden global state.
- Explicit error messages with stable error codes where applicable.
- No `as any`, `@ts-ignore`, silent exception swallowing, or TODO placeholders in required logic.

Python formatting/linting (when toolchain is introduced):

- formatter: `black`
- linter: `ruff`
- tests: `pytest`

## 6) Verification Protocol

Minimum verification before claiming completion:

1. Spec consistency check (all required spec files exist and parse)
2. Data gate pass using `G-D1..G-D4`
3. Stage-specific checks (shape/dim, retrieval metrics, run metadata)
4. Reproducibility evidence (seed/split/version in run artifact)

No evidence, no completion.

## 6.1 Merge blockers

Reject changes when any of these are true:

- Code change without `CMD-*`, `R-*`, or `INV-*` reference
- Referenced ID missing in `docs/spec/INDEX.md`
- `docs/traceability/TRACE.md` not updated for new/changed IDs
- Validation command output missing or failing

## 7) Change Control

- Any schema-affecting change must bump `specVersion`.
- Any new requirement must add:
  - `R-*` in `docs/spec/quality.yaml` or relevant spec
  - at least one scenario in `docs/spec/workflows.yaml`
  - trace link in `docs/spec/traceability.yaml`
  - summary row in `docs/traceability/TRACE.md`

## 8) Anti-Patterns (Forbidden)

- Implementing features not represented in `docs/spec`.
- Leaving unresolved references between spec files.
- Mixing legacy prose with authoritative requirements.
- Marking tasks done without commands/results.

## 9) External References

For standards and implementation conventions:

- OpenAPI 3.1: https://spec.openapis.org/oas/v3.1.1.html
- JSON Schema 2020-12: https://json-schema.org/draft/2020-12
- AsyncAPI: https://www.asyncapi.com/docs/reference/specification/latest
- Structurizr DSL: https://docs.structurizr.com/dsl
- DDD Bounded Context: https://martinfowler.com/bliki/BoundedContext.html
- DDD Ubiquitous Language: https://martinfowler.com/bliki/UbiquitousLanguage.html
