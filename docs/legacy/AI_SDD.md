# AI-SDD (AI System Design Document)

> Status: ARCHIVE
> Authoritative: NO
> Superseded-By: `docs/spec/data.yaml`, `docs/spec/quality.yaml`, `docs/spec/traceability.yaml`

본 문서는 CPG 기반 취약점 탐지 파이프라인의 AI 설계 기준을 정의한다.
현재는 코드보다 데이터/논문이 선행 상태이므로, 구현 전 검증 가능한 계약을 우선 고정한다.

## 1) System Scope

- 목표 태스크: 함수 단위 취약점 이진 분류 (SAFE / VULNERABLE)
- 계획 파이프라인: CPG -> PyG -> GNN Anchor -> (옵션) CodeBERT 보강 -> VectorDB 검색 -> LLM 검증
- 현재 보유 자산: 논문 PDF + `data/` 전체 아티팩트

## 2) Data Documentation (Datasheet 스타일)

### 2.1 데이터 출처
- `data/raw/primevul/*.jsonl`
- `data/raw/devign/*.jsonl`
- `data/raw/cvefixes/` (현재 비어 있음)

### 2.2 현재 분포 (실측)
- `data/metadata.json`: vulnerable 3959, safe 3959, total 7918
- `data/filtered/vulnerable`: 3959
- `data/filtered/safe`: 3959
- `data/split.json`: train 5541, val 1188, test 1188 (총 7917)
- 주의: metadata 총합과 split 총합 delta=1

### 2.3 데이터 품질 리스크
- split/metadata 불일치 1건
- `conversion_stats.json`, `embedding_stats.json`와 실제 파일 수의 해석 불일치 가능성
- `raw/cvefixes` 공백으로 인한 멀티소스 실험 지연

## 3) Model Lifecycle

### 3.1 모델군
- 그래프 모델: Heterogeneous GNN (논문 기준)
- 텍스트 모델: CodeBERT 임베딩 보강 (설계 후보)
- 추론 보강: LLM 검증기 (RAG 증거 기반)

### 3.2 버전/실험 추적 계약
- 모델 버전: `M-YYYYMMDD-###`
- 실험 ID: `E-###`
- 실행 ID: `RUN-YYYYMMDD-###`
- 필수 로그: seed, split 버전, 주요 하이퍼파라미터, 데이터 해시/카운트, metric

## 4) Evaluation Protocol

### 4.1 핵심 메트릭
- 분류: Accuracy, Precision, Recall, F1
- 검색: Recall@k, MRR
- 보조: confusion matrix, 클래스별 성능

### 4.2 필수 비교 실험
- E-001: GNN only
- E-002: GNN + Retrieval
- E-003: GNN + Retrieval + LLM
- E-004: CodeBERT 보강 유/무 비교

### 4.3 통계 및 재현성
- 고정 split + 고정 seed 재실행
- 최소 3회 반복 실행 후 평균/분산 기록

## 5) Safety / Risk

- false negative가 보안 리스크에 직접 연결됨
- LLM 과신 방지: retrieval 근거 없는 판정 금지 정책 고려
- 데이터 누수 및 split 오염은 실험 무효 조건

## 6) MLOps / Reproducibility

구현 시작 시 다음을 즉시 활성화:
- 환경 고정(venv + pinned requirements)
- 실험 추적 저장소(런 메타데이터)
- 데이터 버전 태깅(적어도 파일 카운트 + checksum)

## 7) Go / No-Go Gate (구현 전)

- G-01: split overlap 0, missing path 0
- G-02: PyG/Embedded 샘플 스키마(x/edge_index/node_type/edge_type/y) 확인
- G-03: 임베딩 차원 계약(768) 일치
- G-04: retrieval 오프라인 지표(Recall@k/MRR) 측정 파이프 확정

## References

- Model Cards: https://arxiv.org/abs/1810.03993
- Datasheets for Datasets: https://arxiv.org/abs/1803.09010
- NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework
- Hugging Face installation: https://huggingface.co/docs/transformers/en/installation
- PyG installation: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
- Qdrant installation: https://qdrant.tech/documentation/guides/installation/
