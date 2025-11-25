# Inference Pipeline (DRAFT — confirm model list)

본 문서는 최종 제출 CSV를 생성하기까지의 과정을 정리한 문서입니다. 사용 모델/설정 확인 후 본 문서를 업데이트하세요.

## 사용 모델 (확정)
- EfficientNet‑B7 — `tf_efficientnet_b7_ns`, 입력 448px
  - 최신 런: `outputs/runs/20251111-092056_tf_efficientnet_b7_ns_448px_rot90_jitter_affine_rrc_mixup_only_ls0.05_cosine`
- Swin‑Large — `swin_large_patch4_window12_384_in22k`, 입력 384px(고정)
  - 최신 런: `outputs/runs/20251111-110039_swin_large_patch4_window12_384_in22k_384px_rot90_jitter_affine_rrc_no-reg_ls0.05_cosine`
- ConvNeXtV2‑Large — `convnextv2_large`, 입력 384px
  - 최신 런: `outputs/runs/20251111-152233_convnextv2_large.fcmae_ft_in22k_in1k_384px_rot90_jitter_affine_rrc_no-reg_ls0.05_cosine`
- EfficientNetV2‑L — `tf_efficientnetv2_l.in21k_ft_in1k`, 입력 384px
  - 최신 런: `outputs/runs/20251111-175917_tf_efficientnetv2_l.in21k_ft_in1k_384px_rot90_jitter_affine_rrc_mixup_only_ls0.05_cosine`

## 단계 개요
1) 모델별 TTA 추론(비분할/분할, 멀티스케일)
2) 모델 내부 결합(스케일 결합 → 타입 결합)
3) 클래스별 가중 앙상블(JSON 5세트)
4) 제출 CSV 생성

## 1) 모델별 TTA 추론
- 명령 예시(EffNet‑B7):
```
python -u scripts/core/auto_tta_submit.py \
  --run-dir outputs/runs/<run_dir_b7> \
  --scales "384,448,528" \
  --batch-size 2 \
  --angles "0,90,180,270" --hflip-only \
  --suffix effb7_ms_rot4_hflip --apply-ts
```
- 분할(3×3, overlap 0.125) 추론 시:
```
python -u scripts/core/auto_tta_submit.py ... \
  --tile-grid "3x3" --tile-overlap 0.125 \
  --suffix effb7_ms_rot4_hflip_tile33 --save-tile-probs
```
- 산출물: `outputs/submissions/*.csv`, `outputs/logits/*.npy`, (분할 시)`outputs/logits_tiles/*.npz`

## 2) 모델 내부 결합
- 멀티스케일 결합(448 가중↑): (384,448,528) → (0.3,0.4,0.3)
- 타입 결합(분할:비분할=0.6:0.4)
- 실행:
```
python -u scripts/core/build_final_probs.py --tile-weight 0.6 --non-weight 0.4 --apply-ts
```
- 산출물: `outputs/logits/*_<alias>_final.npy`

## 3) 클래스별 가중 앙상블
- 가중 세트(5종): 기본/보수적/부스트 등
- 실행:
```
python -u scripts/core/posttrain_pipeline.py --do-ensemble \
  --weights-dir reports/summary/<weights_dir_timestamp> \
  2>&1 | tee logs/ensemble_only.out
```
- 실제 사용 가중 폴더(예시): `reports/summary/20251111-085420`
- 산출물: `outputs/submissions/perclass_ensemble_*.csv` (5개)

## 4) 제출 및 관리
- 최종 제출 후보 CSV를 보관하고, 실험/리더보드 점수 메모를 `docs/experiments_log.md`에 갱신합니다.
- 산출물 폴더 정책: `outputs/submission` 단일 폴더 사용, `outputs/submissions`는 심볼릭 링크.

---
(작성자 메모) 본 문서는 모델 확정/경로 확인 후 갱신해주세요.
