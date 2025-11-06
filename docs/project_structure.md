# Project Structure Overview

이 문서는 현재 작업 디렉터리 내 주요 폴더와 파일의 역할을 요약합니다.

## 최상위 디렉터리

- `README.md` — 저장소 소개 및 기본 사용법.
- `requirements.txt` — 필수 파이썬 의존성 목록.
- `experiments/` — 실험 로그 및 메타데이터 저장 위치.
- `docs/` — 추가 문서. (본 파일 포함)
- `configs/` — 모델/학습 설정 YAML 모음.
- `scripts/` — 학습, 추론, 유틸 스크립트.
- `src/` — 데이터 로더, 모델 팩토리, 학습 엔진 등 핵심 코드.
- `data/` — 데이터 관련 자산(원본 CSV, 제외 목록 등).
- `outputs/` — 학습 산출물(체크포인트, 로그, 제출 파일 등).
- `reports/` — 실험 요약, 온도보정 JSON 등 보조 자료.
- `submissions/` — 최종 제출 CSV (현재 `.gitignore` 로 제외 권장).

## 주요 하위 디렉터리

- `configs/`
  - `baseline*.yaml` — Swin-L(ms_in22k) 기준 실험 설정.
  - `dit_*.yaml` — DiT 백본 실험용 설정.

- `scripts/`
  - `run_experiment.py` — 공통 학습 실행 스크립트.
  - `create_submission.py` — 체크포인트 기반 제출 파일 생성.
  - `apply_temperature_scaling.py` — 검증 로짓 기반 온도 보정 후 제출 생성.
  - `archive_run.py` — 런 디렉터리 메타데이터 보관.
  - 기타: 특수 증강, 평가, 앙상블 등 유틸.

- `src/data/`
  - `dataset.py` — 이미지 로딩/전처리 데이터셋 정의.
  - `datamodule.py` — Stratified K-Fold 분할, DataLoader 생성.
  - `sampler.py` — 희소 클래스 가중 샘플러 로직.

- `src/models/`
  - `factory.py` — timm/HuggingFace 모델 인스턴스 생성.

- `src/training/`
  - `engine.py` — 학습 루프, EMA/SWA, 체크포인트 저장.
  - `optim.py` — 옵티마이저/LLRD 파라미터 그룹 정의.
  - `losses.py`, `scheduler.py` — 손실, 스케줄러 생성 유틸.

- `src/utils/`
  - `run_naming.py` — 런/제출 파일 이름 규칙.
  - `metrics.py`, `ema.py`, `seed.py` 등 학습 보조 유틸.

- `experiments/`
  - `experiment_log.csv` — Fold, 설정, 성능 기록.

- `reports/run_history/`
  - `*_ts.json` — 각 런의 Temperature Scaling 결과.

- `data/raw/`
  - `train.csv`, `meta.csv`, `special_augmented_samples.csv` — 데이터 메타.
  - `excluded_samples.csv` — 학습에서 제외할 이미지 목록.

- `outputs/runs/` (Git 추적 제외 권장)
  - 각 실험별 `config.yaml`, `train.log`, `metrics.jsonl`, `checkpoints/best.pth` 등 산출물.

- `outputs/submissions/` (Git 추적 제외 권장)
  - 제출 CSV 및 확률 NPY.

## 기록/로그 위치

- `experiments/experiment_log.csv` — 실험 ID, 사용한 설정 YAML, Fold, CV/EMA F1, 비고.
- `reports/run_history/*.json` — 각 런에 대한 temperature scaling 정보.
- `outputs/runs/*/train.log` — 학습 로그 (에폭별 손실/성능).

이 문서는 흐름 변경 시 함께 업데이트해 주세요.
