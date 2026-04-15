# Data Baseline (2026-02-25)

> Status: ARCHIVE
> Authoritative: NO
> Superseded-By: `docs/spec/data.yaml`

본 문서는 현재 `CPG_GNN/data`의 실측 상태를 기록한다. (구현 전 기준선)

## 1) 디렉터리 인벤토리

- `data/raw/primevul`: 3 files (`train/valid/test` jsonl)
- `data/raw/devign`: 3 files (`train/validation/test` jsonl)
- `data/raw/cvefixes`: 0 files
- `data/filtered/vulnerable`: 3959 json
- `data/filtered/safe`: 3959 json
- `data/cpg`: 15834 files (xml/meta pair)
- `data/pyg`: 7918 entries (`conversion_stats.json` + `.pt`)
- `data/pyg_embedded`: 7919 entries (`embedding_stats.json`, `embedding_checkpoint.json` + `.pt`)
- `data/splits`: empty

## 2) Split / Metadata 정합성

- `data/split.json`
  - train: 5541
  - val: 1188
  - test: 1188
  - total: 7917
- 중복: 0
- train/val/test 교차누수: 0
- split 경로 missing: 0
- `data/metadata.json` total_samples: 7918
- delta(meta - split): 1

정책 제안:
- 현재 단계에서는 `delta<=1`을 warning으로 허용하고, `delta>1`은 hard fail.

## 3) CPG/PyG/Embedded 관찰

- `data/pyg/conversion_stats.json`
  - success: 5193
  - failed: 0
  - vulnerable: 3958
  - safe: 3959
  - skipped: 2724
- `data/pyg_embedded/embedding_stats.json`
  - total_files: 50
  - processed_files: 7885
  - skipped_files: 32
  - failed_files: 0

해석 주의:
- embedding_stats의 `total_files=50`은 실제 파일 스케일과 직접 대응되지 않아 보임.
- 구현 시작 시 stats 생성 스크립트 계약을 먼저 명확히 해야 함.

## 4) Embedded 샘플 스키마 점검 (표본)

표본에서 확인된 키:
- `x`, `edge_index`, `node_type`, `edge_type`, `y`, `sample_id`, `cwe_type`, `code_snippets`

표본 `x` shape 예:
- `(337, 768)`, `(1518, 768)`, `(61, 768)`, `(491, 768)`, `(78, 768)`

## 5) 즉시 실행해야 할 데이터 게이트

- G-D1: split overlap/missing/duplicate 자동 검증
- G-D2: embedded 파일 로드 가능성 검사
- G-D3: shape/dim(768) 검사
- G-D4: NaN/Inf 검사
- G-D5: stats 파일 의미 일관성 검증

## 6) Evidence Paths

- `data/metadata.json`
- `data/split.json`
- `data/pyg/conversion_stats.json`
- `data/pyg_embedded/embedding_stats.json`
- `data/pyg_embedded/embedding_checkpoint.json`
- `data/filtered/vulnerable/sample_00050.json`
- `data/raw/primevul/primevul_train.jsonl`
