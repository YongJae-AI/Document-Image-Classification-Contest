# Experiments Timeline (요약)

다음 로그는 주요 실험/제출/결정 사항을 시간순으로 요약합니다. 세부 로그는 `logs/YYYYMMDD/*.out|.log` 및 `experiments/experiment_log_consolidated.csv`를 참고하세요.

## 2025-11-07
- Swin‑Large CV 실험(여러 fold): 라벨 스무딩/DropPath/EMA 변형 비교. 최고 F1≈0.92대(fold별 편차).
- 온도보정(TS) 실험 개시: TS≈0.83~0.92 범위 관측.

## 2025-11-08
- ConvNeXtV2‑Large 재개 학습(resume) 후 TTA+TS 제출. F1≈0.99(Val) 관측.
- 오프라인 증강(문서 특화) 파이프라인 정리 및 병합 CSV 생성.

## 2025-11-09
- 멀티스케일(456/528/613/720/800) TTA 중간 제출: 528까지 상승, 613에서 하락 징후.

## 2025-11-10
- EfficientNet‑B7: fold0~4 학습/검증 로그 정리, full-train 준비.

## 2025-11-11
- 4개 백본 full-train 완료: EfficientNet‑B7, Swin‑L(384), ConvNeXtV2‑L, EfficientNetV2‑L.
- Post-train 파이프라인(제출+앙상블) 가동. 클래스별 가중 5세트 제출(conservative 최고).

## 2025-11-12
- 분할(3×3) TTA 및 멀티스케일 결합 정리, 타일 확률 저장.
- per-class incorrect 비교/Grad‑CAM 샘플링(최고/최저 클래스별 3장) 완료.

## 2025-11-13
- TTA 멀티스케일(추론 스케일) vs 리더보드 점수 비교 그래프 작성.
- 리포 정리(파일명 규칙/로그 인덱스/문서화/Grad‑CAM 정리) 완료.

---
비고: 날짜/내용은 요약 수준입니다. 세부 파라미터 및 파일 경로는 각 날짜의 로그와 결과물에 표기되어 있습니다.
