# SDD (Software Design Description)

> Status: ARCHIVE
> Authoritative: NO
> Superseded-By: `docs/spec/workflows.yaml`, `docs/spec/interfaces.yaml`, `docs/spec/data.yaml`, `docs/spec/quality.yaml`

본 문서는 IEEE 1016 및 arc42 관점으로, 현재 `CPG_GNN`의 "구현 전" 시스템 설계 계약을 정의한다.

## 1. Introduction & Goals

- 목표: CPG 구조정보와 LLM 검증을 결합한 취약점 탐지 파이프라인의 재현 가능한 구축
- 품질 우선순위:
  1) 재현성
  2) 추적성
  3) 실험 비교 가능성
  4) 확장성(새 데이터셋/새 임베더)

## 2. Constraints

- 현재 저장소에는 실행 코드가 없고 데이터/논문 중심 상태
- 데이터 우선 문서화 후 코드 생성 방식 채택
- split 불일치 1건은 known issue로 기록하고, 초기 구현에서 허용 범위 정책으로 통제

## 3. Context & Scope

- In-scope:
  - `data/raw`, `data/filtered`, `data/cpg`, `data/pyg`, `data/pyg_embedded`, `data/split.json`
  - 논문 기반 CPG->GNN->RAG->LLM 파이프라인 재구현
- Out-of-scope (현재 단계):
  - 운영 배포 인프라/온라인 서비스

## 4. Solution Strategy

- Strategy-1: Docs-first 계약 고정 (DDD/SDD/AI-SDD/Traceability)
- Strategy-2: 데이터 무결성 게이트를 코드 작성 전 자동화
- Strategy-3: 모델/리트리벌/LLM 결합은 모듈 계약 기반으로 단계별 검증

## 5. Building Block View

### B1 Data Validator
- 입력: `metadata.json`, `split.json`, `data/pyg_embedded/*.pt`
- 출력: 무결성 보고서(JSON/Markdown)
- 책임: split 중복/누수/missing/shape/NaN-Inf 검증

### B2 Graph Pipeline
- 입력: raw/filtered/cpg
- 출력: `data/pyg/*.pt`
- 책임: CPG->PyG 변환 규칙, 실패 케이스 로깅

### B3 Embedding Pipeline
- 입력: `data/pyg/*.pt`
- 출력: `data/pyg_embedded/*.pt` 또는 별도 임베딩 매트릭스
- 책임: GNN/CodeBERT 임베딩 생성 모드 관리

### B4 Vector Indexer
- 입력: 임베딩 벡터 + payload
- 출력: Qdrant 컬렉션
- 책임: dim/distance/payload contract 고정

### B5 Evaluator
- 입력: split, 예측 결과, 검색 결과
- 출력: F1/Precision/Recall/Accuracy + Retrieval Recall@k + 에러 분석

## 6. Runtime View (핵심 시나리오)

### S1. Data Gate
1) split 로드 -> 2) 중복/누수 검사 -> 3) 파일 존재 확인 -> 4) 임베딩 스키마 검사 -> 5) 리포트 생성

### S2. End-to-End Training
1) PyG 로드 -> 2) GNN 학습 -> 3) checkpoint 저장 -> 4) 임베딩 생성

### S3. Retrieval + Verification
1) 벡터 인덱싱 -> 2) Top-k 검색 -> 3) LLM 검증 프롬프트 구성 -> 4) 최종 판정 기록

## 7. Deployment View (현재 최소)

- 로컬 단일 노드 기준
- Qdrant Docker 또는 로컬 인스턴스
- Python venv 환경에서 PyTorch/PyG/Transformers/Qdrant-client 사용

## 8. Cross-cutting Concepts

- 버전 정책: 데이터셋/모델/인덱스/실험 런에 ID 부여
- 로그 정책: 실행 커맨드, seed, split 버전, 주요 하이퍼파라미터 저장
- 실패 정책: hard-fail 조건과 warn 조건 분리

## 9. Architectural Decisions (초안)

- A-001: 문서 우선 접근
- A-002: split delta=1은 초기 허용, 단 근거 기록 필수
- A-003: VectorDB distance는 실험 단위로 고정 관리

## 10. Quality Requirements

- QR-001 재현성: 동일 split/seed에서 metric 변동 허용 범위 정의
- QR-002 무결성: split overlap 0, missing path 0
- QR-003 추적성: claim-to-run 연결율 100%

## 11. Risks & Technical Debt

- 코드 부재 상태에서 문서->구현 간 오차 가능성
- stats 파일과 실파일 수치 괴리 가능성
- 외부 데이터(raw/cvefixes) 공백으로 인한 다중 데이터셋 실험 지연

## 12. Glossary

- CPG, PyG, GNN Anchor, Retrieval Candidate, Claim, Requirement, Experiment, Run

## References

- IEEE 1016: https://ieeexplore.ieee.org/document/7439308
- arc42: https://arc42.org/overview
