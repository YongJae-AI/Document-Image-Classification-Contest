# Document Type Classification — Reproducible Pipeline

문서 이미지 분류 대회에서 사용한 최종 파이프라인을 정리했습니다. 핵심 백본은 EfficientNet‑B7(448), Swin‑Large(384), ConvNeXtV2‑Large(384), EfficientNetV2‑L(384)이며, TTA(회전/플립/멀티스케일/타일)와 클래스별 가중 앙상블로 최종 성능을 냈습니다.

## Directory
```
data/           # 데이터 자산(원본/가공)
notebooks/      # 노트북(루트로 모음, eda/ 산출물 일부 유지)
src/            # 데이터/모델/학습/유틸 코드
configs/
  ├─ final/     # 최종 사용 설정(4개 백본)
  └─ templates/ # CV/Full-train 템플릿
scripts/
  ├─ core/      # 학습/추론/결합/앙상블/제출
  ├─ eda/       # 리포트/그래프 유틸
  ├─ gradcam/   # Grad‑CAM 시각화
  └─ tools/     # 정리/리네임/로그/설정 관리
reports/        # 제출/그림/요약 등 결과물
docs/           # 파이프라인/타임라인 문서
logs/           # 날짜별 로그 + logs_index.csv
```

## 핵심 파이프라인 명령
1) 학습
```
python -u scripts/core/run_experiment.py --config configs/final/efficientnet_b7.yaml
```

2) 모델별 추론(TTA/멀티스케일)
```
python -u scripts/core/auto_tta_submit.py \
  --run-dir outputs/runs/<run_dir> \
  --scales "384,448,528" --batch-size 2 \
  --angles "0,90,180,270" --hflip-only \
  --suffix effb7_ms_rot4_hflip --apply-ts
```

3) 모델 내부 결합(멀티스케일 + 분할/비분할)
```
python -u scripts/core/build_final_probs.py --tile-weight 0.6 --non-weight 0.4 --apply-ts
```

4) 클래스별 가중 앙상블(5 JSON 세트)
```
python -u scripts/core/posttrain_pipeline.py --do-ensemble --weights-dir reports/summary/<timestamp_dir>
```

상세 파라미터/흐름은 `docs/inference_pipeline.md` 참고.

## 운영 메모
- `outputs/runs`, `outputs/logits`, `outputs/submission(s)` 는 Git 미추적(대용량 산출물 제외).
- 로그는 `logs/YYYYMMDD/*.log|.out`, 색인은 `logs/logs_index.csv`.
- EDA 캐시(parquet 등)는 notebooks/eda/reports/eda/ 하위에 있으며 최종 재현에는 필수 아님.

