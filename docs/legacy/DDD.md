# DDD (Domain-Driven Design)

> Status: ARCHIVE
> Authoritative: NO
> Superseded-By: `docs/spec/domain.yaml`, `docs/spec/glossary.yaml`

## 1) 목적

CPG 기반 취약점 탐지 파이프라인(예정: CPG -> PyG -> GNN -> CodeBERT -> VectorDB -> LLM 검증)의 도메인 경계를 명확히 분리하여, 구현 전 혼선을 제거한다.

## 2) Ubiquitous Language (공통 언어)

- `Raw Sample`: 원본 함수 단위 샘플(jsonl 레코드)
- `Filtered Sample`: 취약/안전 라벨이 확정된 샘플(json)
- `CPG Artifact`: 코드에서 추출된 CPG xml + meta 쌍
- `PyG Graph`: 학습/추론 가능한 그래프 텐서(pt)
- `Embedded Graph`: 노드/그래프 임베딩이 포함된 PyG 산출물(pt)
- `Anchor Prediction`: GNN 1차 판정 결과
- `Retrieval Candidate`: 벡터 유사도 검색으로 얻은 근접 사례
- `Final Verdict`: LLM 보정 후 최종 판정

## 3) Bounded Context

### BC-1: Data Ingestion Context
- 책임: `data/raw/*` 입력 보전, 라이선스/출처/라벨 원천 관리
- 산출: 정규화된 샘플 집합

### BC-2: Graph Construction Context
- 책임: Raw/Filtered -> CPG -> PyG 변환 규칙
- 산출: `data/cpg`, `data/pyg`

### BC-3: Embedding & Retrieval Context
- 책임: GNN/CodeBERT 임베딩 생성, 벡터 인덱싱, 검색 계약
- 산출: `data/pyg_embedded`, VectorDB 컬렉션

### BC-4: Detection & Verification Context
- 책임: Anchor 예측, Retrieval 조건부 증거, LLM 최종 판정
- 산출: 샘플 단위 예측/설명/근거

### BC-5: Evaluation & Governance Context
- 책임: split 고정, 실험 추적, 성능/재현성/리스크 관리
- 산출: 실험 보고서, 추적성 매트릭스, 의사결정 기록

## 4) Context Map (텍스트)

- BC-1 -> BC-2: 샘플/라벨 계약 전달
- BC-2 -> BC-3: 그래프 구조/피처 계약 전달
- BC-3 -> BC-4: 임베딩/검색 결과 전달
- BC-4 -> BC-5: 예측/근거/메트릭 전달
- BC-5 -> 전 BC: 거버넌스 규칙(분할, 재현성, 품질 게이트) 피드백

## 5) 도메인 불변조건 (Invariant)

- `INV-001`: train/val/test 간 샘플 중복은 없어야 한다.
- `INV-002`: PyG 임베딩 차원은 파이프라인 전체에서 일치해야 한다(현재 768 가정).
- `INV-003`: split에 있는 파일 경로는 모두 실제 존재해야 한다.
- `INV-004`: 평가는 고정 split과 고정 seed 정책을 따라야 한다.
- `INV-005`: 논문 주장(C-*)은 반드시 실험(E-*)과 산출물(RUN-*)로 연결되어야 한다.

## 6) 현재 관찰 기반 리스크

- `RISK-001`: `data/metadata.json` 총 7918 vs `data/split.json` 총 7917 (delta=1)
- `RISK-002`: `data/pyg/conversion_stats.json`의 success/skipped와 전체 샘플 수 관계가 즉시 직관적이지 않음
- `RISK-003`: `data/pyg_embedded/embedding_stats.json`의 `total_files=50`와 실제 파일 수 스케일 불일치 가능성
- `RISK-004`: 현재 코드베이스 부재로 파이프라인 계약이 문서로 먼저 고정되어야 함

## 7) 구현 전 승인 기준

- DDD 용어집/경계에 대해 팀 합의 완료
- 각 BC의 입력/출력 계약이 SDD로 연결됨
- 모든 핵심 claim이 traceability 매트릭스에 등록됨
