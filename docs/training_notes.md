# Training Notes (Swin-Large Baseline)

- **Backbone**: `swin_large_patch4_window12_384`, 입력 해상도 384px.
- **증강**: `rot60_affine_rrc`
  - `Rotate` ±60°
  - `A.Affine` scale 0.9~1.1, translation 5%
  - `RandomResizedCrop` scale 0.9~1.0, ratio 0.9~1.1
  - Horizontal flip, brightness/contrast(±0.1), Gaussian blur(10%)
- **정규화**: mixup/cutmix 미사용(`no-reg` 태그), label smoothing 0.05.
- **Optimizer/Scheduler**
  - AdamW(lr 2.2e-4, betas 0.9/0.999, weight decay 0.02)
  - Cosine 스케줄 + 3 epoch warmup, 총 18 epoch
- **Sampler**: `WeightedRandomSampler`
  - 소수 클래스 1/13/14, `weights_power=0.85`, replacement=True
- **Batching**: batch 8, workers 8, pin_memory & persistent_workers on, prefetch 2
- **Loss**: CrossEntropy + label smoothing 0.05
- **평가 지표**: Macro F1 (validation), Leaderboard는 F1 기반 채점
- **제출 규칙**: `sample_submission.csv` 의 `ID` 순서와 동일한지 `assert` 후 저장, 파일명에 `f1`(CV) + `result(리더보드)` 표기
- **데이터 정리**: 문제 소지가 있는 샘플들은 `data/raw/excluded_images/` 에 원본을 보관하고, 필요 시 `scripts/augment_special_cases.py` 로 복원 및 증강본을 생성합니다.
  - 기본 제외 목록: `2b1076abe3e4338d.jpg`, `be53872196b3ae1d.jpg`, `f176f6c25b7bd8ae.jpg`, `f0fc4e2f239e236b.jpg`
  - 증강 파이프라인: CLAHE, MotionBlur, ISO Noise, Perspective (밝기/대비 보정 포함).
  - 생성된 샘플과 레이블은 `data/raw/special_augmented_samples.csv` 에 기록됩니다.
- **학습 제어**: `training.early_stopping` 으로 patience/min_delta를 설정하여 F1 개선이 정체되면 자동으로 학습을 멈출 수 있습니다.

> 추후 실험에서는 Fold 편차를 완화하기 위해 seed 다양화, 희소 클래스 증강 조정, focal loss 비교 등을 고려합니다.
