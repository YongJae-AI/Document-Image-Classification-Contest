# Document Type Classification Contest Workspace

이 저장소는 업스테이지 문서 타입 분류 경진대회 참가를 위한 코드와 실험 자산을 체계적으로 관리하기 위한 기본 골격을 제공합니다. 데이터 및 베이스라인 자산은 대회에서 제공한 자료만 사용하며, 외부 데이터는 금지되어 있습니다.

## Directory Overview

```
data/                  # 대회 데이터 자산 관리
  ├─ raw/              # 다운로드한 원본(train/test) 데이터
  ├─ interim/          # 전처리 중간 산출물
  ├─ processed/        # 모델 학습용 최종 데이터
  └─ external/         # 외부 데이터 금지 안내용 (항상 비워둠)
notebooks/             # EDA 및 실험 노트북
  ├─ eda/
  └─ experiments/
src/                   # 재현 가능한 파이프라인 코드
  ├─ data/             # 데이터 로더, 전처리 유틸
  ├─ features/         # 피처 엔지니어링 로직
  ├─ models/           # 모델 정의 및 백본 래퍼
  ├─ training/         # 학습 루프, 스케줄러
  ├─ inference/        # 추론 및 제출 생성 스크립트
  └─ utils/            # 공통 유틸리티
configs/               # 실험/모델 설정 파일(YAML 등)
scripts/               # 데이터/베이스라인 다운로드 등 유틸 스크립트
baseline/              # 배포된 베이스라인 코드 (다운로드 후 보관)
experiments/           # 로그, 체크포인트, 실험 메타데이터
  ├─ logs/
  └─ checkpoints/
submissions/           # 제출 파일(.csv 등)
reports/               # 문서화 및 결과 정리
  └─ figures/
docs/                  # 대회 규정, 회의록 등

추가 참고: 베이스라인 제출 스크립트는 `sample_submission.csv` 의 `image_id` 순서를 `assert` 로 검증한 후 제출 파일을 저장합니다. 커스텀 파이프라인에서도 동일 검증을 유지해 제출 사고를 방지하세요. 자세한 요약은 `docs/baseline_notes.md` 참고.
tests/                 # 자동화 테스트
```

## Getting Started

1. **데이터 & 베이스라인 다운로드**
   ```bash
   bash scripts/download_data.sh
   bash scripts/download_baseline.sh
   ```
2. **필수 설정 확인**
   - Git 사용자 정보가 등록되어 있는지 확인합니다.
   - 외부 데이터는 `data/external/` 에 보관하지 않습니다. (항상 비워둡니다.)
   - 환경 준비: `pip install -r requirements.txt`
3. **특이 샘플 보강(선택)**
   ```bash
   python scripts/augment_special_cases.py
   ```
   - `data/raw/excluded_images/` 에 있는 특이 샘플을 복원하고, CLAHE / MotionBlur / ISO Noise / Perspective 증강본을 생성해 `train.csv` 에 등록합니다.
   - 생성 내역은 `data/raw/special_augmented_samples.csv` 로 관리합니다.
4. **실험 기록**
   - 학습/추론 스크립트는 `src/` 에 모듈화합니다.
   - 실험 결과, 하이퍼파라미터, 코멘트는 `experiments/` 아래에 정리합니다.

## Training & Inference Workflow

1. **단일 Fold 학습**
   ```bash
   python scripts/run_experiment.py --config configs/baseline_ema.yaml
   ```
   - 실행 시 `outputs/runs/<타임스탬프>_.../` 가 생성되고, `config.yaml`, `train.log`, `metrics.jsonl`, `checkpoints/best.pth` 가 저장됩니다.
   - EMA/Temperature Scaling 제출 기준은 `configs/baseline_ema.yaml`, SWA 실험 기준은 `configs/baseline_swa.yaml` 을 사용합니다.
   - 러닝 이름 규칙은 `학습시간_모델명_입력이미지크기_증강태그_정규화태그_lsXX_스케줄러` 입니다.
     - `no-reg` 는 mixup/cutmix 를 비활성화한 설정을 의미합니다.
   - `performance` 설정을 통해 TF32, cuDNN benchmark, channels-last 메모리 포맷을 활성화하여 GPU 활용을 최적화했습니다.
   - `training.early_stopping` 에서 patience / min_delta / mode 를 정의할 수 있으며, 지정된 epoch 내에서 개선이 멈추면 조기 종료됩니다.

2. **제출 파일 생성**
   ```bash
   python scripts/create_submission.py --run-dir outputs/runs/<run_dir_name> --leaderboard-score 0.8352 --save-probs
   ```
   - `sample_submission.csv` 과 동일한 ID 순서를 `assert` 로 검증 후 CSV 를 저장합니다.
   - 파일명은 `학습이름_f1_<CV>_result(<LB>).csv` 규칙을 따릅니다.
   - `--save-probs` 옵션을 주면 softmax 확률을 `outputs/logits/` 에 `.npy` 로 보관합니다.

3. **온도 보정 + 제출 (권장)**
   ```bash
   python scripts/apply_temperature_scaling.py --run-dir outputs/runs/<run_dir_name> --temperature-json reports/run_history/<run_dir_name>_ts.json
   ```
   - 검증 세트를 이용해 단일 스칼라 온도를 추정하고, 보정된 확률로 제출 CSV 및 `.npy` 를 생성합니다.
   - 결과 파일은 `<기존파일명>_tscaled.csv` 형태로 `outputs/submissions/` 에 저장되며, 온도 값은 옵션에 따라 JSON으로 보관할 수 있습니다.

4. **소프트맥스 앙상블 (선택)**
   ```bash
   python scripts/ensemble_softmax.py --logits outputs/logits/<run1>.npy outputs/logits/<run2>.npy --tag swin_large_sampler_uniform
   ```
   - 동일 길이의 `.npy` 확률 파일을 평균(혹은 가중 평균) 후 제출 CSV 를 생성합니다.
   - 출력은 `outputs/ensembles/<timestamp>-ensemble_<tag>.csv` 형태입니다.

설정 세부 요약은 `docs/training_notes.md` 를 참고하세요.

### Run Metadata 백업

GPU 서버 장애 등으로 `outputs/` 폴더가 사라져도 실험 이력을 복구할 수 있도록, 학습이 끝난 뒤 아래 스크립트로 핵심 메타데이터를 저장해 주세요.

```bash
python scripts/archive_run.py --run-dir outputs/runs/<run_dir_name> [--label lb0835]
```

- `config.yaml`, `metrics.jsonl`, `eval_metrics.json`, `train.log`(존재 시) 이 `reports/run_history/<run_dir_name>[_label]/` 에 복사됩니다.
- `reports/run_history/` 는 Git 에 의해 추적되므로, 원격 저장소로 push 하면 실험 설정과 성능 기록을 안전하게 보관할 수 있습니다.
- 동일한 폴더명으로 다시 보관하려면 `--force` 옵션을 사용해 덮어쓰세요.

## Competition Notes

- 외부 데이터 사용, 평가 데이터로 학습하는 행위, 상용 OCR/LLM 모델 사용(유료)은 금지되어 있습니다.
- 결과 재현을 위해 모든 실험을 커밋하고 추적 가능한 상태로 유지해야 합니다.
- 제출 횟수는 팀 단위 하루 12회로 제한되며, 자정(한국시간)에 초기화됩니다.

이 레포는 기본 틀만 제공합니다. 실험을 진행하면서 코드와 문서를 지속적으로 정리하고, 모든 변경 사항을 Git으로 관리해 주세요.
