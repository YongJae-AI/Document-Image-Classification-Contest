# Offline Augmentation Plan

- 입력: `data/raw/train.csv`, `data/raw/train/*.jpg`
- 출력: `data/processed/offline_aug/images/*.jpg`, `data/processed/offline_aug/train_offline_aug.csv`
- 설정: `configs/offline_aug.yaml`

## 실행 방법
```
# 디버그(소량 샘플)
python scripts/offline_augment.py --config configs/offline_aug.yaml --train-csv data/raw/train.csv --image-dir data/raw/train --limit 50

# 전체 생성(약 25K 목표)
python scripts/offline_augment.py --config configs/offline_aug.yaml --train-csv data/raw/train.csv --image-dir data/raw/train
```

## 클래스 프로필
- card_like: 5(driver_license), 8(national_id_card), 9(passport)
- non_doc: 2(car_dashboard), 16(vehicle_registration_plate)
- doc: 그 외 문서류

각 프로필은 Albumentations로 설계된 증강 묶음을 사용하며, Test EDA에서 관측한 밝기↑/채도↓/약블러/가벼운 그림자/퍼스펙티브를 반영합니다.

## 학습 설정 예시(오프라인 증강 데이터 사용)
- `configs/swin_in22k_offline.yaml` 참고(이미지 디렉터리/CSV 경로가 오프라인 산출물로 지정됨)

```
python scripts/run_experiment.py --config configs/swin_in22k_offline.yaml --fold 5
python scripts/create_submission.py --run-dir outputs/runs/<run_dir> --leaderboard-score 0.0 --save-probs
python scripts/apply_temperature_scaling.py --run-dir outputs/runs/<run_dir> --leaderboard-score 0.0 --temperature-json reports/run_history/<run_dir>_ts.json --suffix ts
```

## 산출물
- `data/processed/offline_aug/train_offline_aug.csv`: 원본 + 증강 데이터 병합 목록(ID, target, source, profile).
- `data/processed/offline_aug/offline_aug_summary.csv`: 클래스/프로필별 증강 개수 요약.

