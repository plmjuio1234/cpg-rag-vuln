# Traceability Matrix

> Status: ARCHIVE
> Authoritative: NO
> Superseded-By: `docs/spec/traceability.yaml`, `docs/traceability/TRACE.md`

논문 주장(Claim) -> 요구사항(Requirement) -> 실험(Experiment) -> 실행 산출물(Run Artifact) 연결표.

| Claim ID | Claim (요약) | Requirement ID | Experiment ID | Artifact (예정) | 현재 상태 |
|---|---|---|---|---|---|
| C-001 | CPG 구조정보가 텍스트-only 대비 성능에 기여한다 | R-001 (CPG->PyG 계약 고정) | E-001, E-002 | `results/RUN-*/metrics_gnn_vs_baseline.json` | pending |
| C-002 | GNN 임베딩 기반 유사도 검색이 검증 품질을 높인다 | R-002 (VectorDB 계약 + Recall@k 측정) | E-002 | `results/RUN-*/retrieval_eval.json` | pending |
| C-003 | LLM 보정은 조건부로 성능 개선 가능하다 | R-003 (LLM 검증 실험 프로토콜) | E-003 | `results/RUN-*/hybrid_eval.json` | pending |
| C-004 | 파이프라인은 재현 가능해야 한다 | R-004 (seed/split/version 고정) | E-001~E-004 | `results/RUN-*/run_meta.json` | in_progress |
| C-005 | 데이터 누수 없는 분할이 필수다 | R-005 (overlap 0, missing 0) | E-000 (data gate) | `results/RUN-*/data_integrity_report.json` | in_progress |

## Requirement 정의

- R-001: CPG->PyG 변환 규칙 및 필수 텐서 키 정의
- R-002: VectorDB 스키마(dim/distance/payload)와 검색 지표 측정 파이프 확정
- R-003: LLM 검증 입력(Anchor + Retrieved evidence) 및 평가 기준 확정
- R-004: 실행별 seed/split/hash/log 기록 강제
- R-005: split 무결성(중복/누수/누락) 자동 점검

## Experiment 템플릿

- E-000: Data Integrity Gate
- E-001: GNN only baseline
- E-002: GNN + Retrieval
- E-003: GNN + Retrieval + LLM
- E-004: CodeBERT augmentation ablation

## 업데이트 규칙

- claim 추가/수정 시, 반드시 Requirement와 Experiment를 함께 갱신
- 실험 실행 후 `RUN-*`와 아티팩트 경로를 표에 즉시 연결
