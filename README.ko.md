# CPG-GNN

[English](README.en.md) | [한국어](README.ko.md) | [Home](README.md)

> 코드 속성 그래프, 그래프 신경망, 검색형 근거 활용, LLM 보조 평가 흐름을 조합해 취약점 탐지 실험을 수행하는 Python 저장소입니다.

## 개요

이 저장소는 PyTorch, PyTorch Geometric, 그리고 LLM 기반 평가기를 중심으로 구성된 그래프 기반 취약점 탐지 실험 코드 모음입니다. 모델 학습, 하이브리드 평가, 데이터셋별 일반화 실험, 로컬 실험 산출물 관리 코드가 함께 들어 있습니다.

현재 공개 스냅샷은 의도적으로 가볍게 정리된 상태입니다. 데이터셋, 체크포인트, 생성 결과물 같은 대용량 자산은 커밋하지 않았지만, 로컬 작업 구조를 유지하기 위해 관련 폴더는 비어 있는 상태로 보존했습니다.

## 이 저장소에 들어 있는 것

- `scripts/pipeline_run.py` 의 **주요 모델/평가 파이프라인**
- `scripts/` 의 **Devign / Juliet / CVEFixes 데이터셋별 평가 스크립트**
- `scripts/run_glm5_hybrid_eval.py` 의 **GLM 기반 하이브리드 평가 흐름**
- `gpt.py` 의 **OAuth 기반 GPT 호출 유틸리티**
- `tests/test_spec_smoke.py` 의 **기본 스모크 테스트 틀**

## 저장소 구조

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

## 환경과 의존성

현재 `requirements.txt` 기준 핵심 패키지는 다음과 같습니다.

- `torch`
- `torch-geometric`
- `transformers`
- `qdrant-client`
- `numpy`, `pandas`, `scikit-learn`
- `pytest`, `ruff`, `black`
- `pyyaml`, `jsonschema`

기본 환경 구성 예시는 아래와 같습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 주요 워크플로우

### 1. 전체 그래프 / 하이브리드 파이프라인

`scripts/pipeline_run.py` 는 이 저장소에서 가장 큰 오케스트레이션 스크립트입니다. 이 파일 안에는 다음 흐름이 포함되어 있습니다.

- `results/` 아래 실행 ID 생성
- `data/` 에서 그래프 로드
- GNN 모델 정의 및 학습
- `checkpoints/gnn` 아래 체크포인트 저장
- `CodexOAuthClient`, `GLM5Client` 같은 외부 클라이언트를 활용한 하이브리드 평가 지원

### 2. 데이터셋별 일반화 평가

여러 벤치마크 계열에 대해 전용 평가 스크립트를 두고 있습니다.

- `scripts/eval_devign_generalization.py`
- `scripts/eval_juliet_generalization.py`
- `scripts/eval_cvefixes_generalization.py`

이 스크립트들은 데이터셋별 샘플을 불러오고, 라벨/메타데이터를 정규화한 뒤, LLM 보조 평가 또는 하이브리드 점수화 흐름에 맞는 입력을 준비합니다.

### 3. GLM5 기반 하이브리드 평가

`scripts/run_glm5_hybrid_eval.py` 는 그래프 임베딩, GNN 출력, GLM 기반 판정 로직을 결합하는 별도 평가 경로를 담고 있으며, 실행 산출물은 `results/` 아래에 기록되도록 되어 있습니다.

### 4. 유틸리티 및 튜닝 스크립트

그 외에도 아래 같은 보조 작업 스크립트가 있습니다.

- 하이퍼파라미터 튜닝: `scripts/tune_gnn_plateau.py`
- PyG 산출물 품질 점검: `scripts/evaluate_pyg_quality.py`
- 그림/리포트 생성 보조: `scripts/generate_paper_figures.py`, `scripts/render_paper_figures_png.py`

## 데이터와 생성 산출물 디렉터리

다음 디렉터리는 현재 **placeholder 용도**로만 Git에 남겨 두었습니다.

- `data/`
- `checkpoints/`
- `results/`

세 폴더에는 `.gitkeep` 만 추적되고 실제 대용량 파일은 무시됩니다. 실무적으로는 아래처럼 사용하면 됩니다.

- 원시/전처리 데이터는 `data/`
- 학습된 모델 가중치는 `checkpoints/`
- 실행 결과, 메트릭, 실험 리포트는 `results/`

## 검증 상태와 현재 주의점

`tests/test_spec_smoke.py` 가 존재하지만, 이 테스트는 현재 저장소에 포함되지 않은 `docs/spec`, `docs/traceability` 자산을 여전히 전제로 하고 있습니다. 즉, 지금 저장소는 코드와 테스트 전제가 완전히 다시 맞춰진 상태는 아닙니다.

관련해서 알아둘 점은 다음과 같습니다.

- `scripts/pipeline_run.py` 일부도 제거된 문서 경로를 아직 참조합니다.
- 개발자 로컬 환경에는 실험 결과가 존재할 수 있지만, 그 결과물은 저장소에 커밋되지 않습니다.
- 이 README 의 예시는 저장소 안내용이며, 모든 예전 경로가 즉시 실행 가능하다는 보장은 아닙니다.

## 처음 볼 때 추천하는 읽는 순서

처음 저장소를 볼 때는 아래 순서가 가장 이해하기 쉽습니다.

1. `requirements.txt` — 의존성 범위 파악
2. `scripts/pipeline_run.py` — 전체 오케스트레이션 흐름 확인
3. `scripts/eval_juliet_generalization.py` — 데이터셋별 평가 패턴 확인
4. `scripts/eval_devign_generalization.py`, `scripts/eval_cvefixes_generalization.py` — 병렬 변형 흐름 확인
5. `gpt.py` — 외부 GPT/OAuth 유틸리티 확인

## 예시 명령

아래 명령은 **파일이 실제로 존재한다는 기준의 저장소 안내 예시**이며, 즉시 완전 재현이 보장되는 데모 명령은 아닙니다.

```bash
python scripts/pipeline_run.py --help
python scripts/eval_juliet_generalization.py --help
python scripts/eval_devign_generalization.py --help
python scripts/eval_cvefixes_generalization.py --help
python scripts/run_glm5_hybrid_eval.py --help
pytest tests/test_spec_smoke.py
```

## 현재 상태

이 저장소는 완전히 다듬어진 벤치마크 배포본이라기보다, 실험 중심으로 계속 손보는 워크스페이스에 가깝습니다. 핵심 코드 경로는 남아 있지만, 저장소 정리 과정에서 문서 자산 기반의 일부 전제는 제거된 상태입니다.

좀 더 깔끔한 공개용 저장소로 만들려면 다음 단계가 실용적입니다.

1. `tests/` 를 현재 축소된 저장소 구조에 맞게 정리
2. 스크립트 안의 예전 `docs/spec` 참조 제거 또는 갱신
3. 최소 재현용 예제 데이터/실행 흐름 추가
4. 외부 모델 클라이언트에 필요한 환경변수 정리
