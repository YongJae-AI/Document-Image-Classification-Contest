# Train vs Test EDA Report


## TL;DR
- **laplacian_var**: KS=0.447 (p=0.000), W1=665.594
- **brightness_mean**: KS=0.320 (p=0.000), W1=23.747
- **qr_seal_score**: KS=0.301 (p=0.000), W1=31.605
- **edge_density**: KS=0.243 (p=0.000), W1=0.023
- **file_size_bytes**: KS=0.176 (p=0.000), W1=7573.832

## 바로 적용 액션
- **rotation_offset_deg**: KS=0.049, W1=0.598 → Test 회전 분포가 ±8° 이상 치우치면 Rotate 증강 상향 및 orientation normalize 고려
- **margin_ratio**: KS=0.095, W1=0.056 → Test 여백비가 커지면 CenterCrop 대신 Pad 또는 RRC 비중 조정
- **laplacian_var**: KS=0.447, W1=665.594 → Blur 증가 시 Blur 증강 on + 추론 시 Sharpen 비활성
- **jpeg_blockiness**: KS=0.097, W1=0.558 → 압축 강하면 JPEGCompression 증강 on, Swin late fusion 가중 ↑
- **brightness_mean**: KS=0.320, W1=23.747 → 밝기 편차 크면 ColorJitter 범위 확장
- **aspect_ratio**: KS=0.168, W1=0.114 → Aspect Ratio 차이가 크면 longer-side resize + pad 일관화
- **text_area_ratio**: KS=0.104, W1=0.054 → 텍스트 밀도 감소 시 OCR 가산치 조정 및 레이아웃 모델 비중 확대

## 수치 요약
Feature | KS | KS p-value | W1
--- | --- | --- | ---
width | 0.1678 | 0.0000 | 28.5554
height | 0.1666 | 0.0000 | 28.4483
aspect_ratio | 0.1678 | 0.0000 | 0.1139
file_size_bytes | 0.1762 | 0.0000 | 7573.8316
pixel_count | 0.0214 | 0.7077 | 7.1130
brightness_mean | 0.3196 | 0.0000 | 23.7468
brightness_std | 0.0392 | 0.0766 | 1.6656
laplacian_var | 0.4468 | 0.0000 | 665.5936
colorfulness | 0.1752 | 0.0000 | 4.9104
margin_ratio | 0.0952 | 0.0000 | 0.0564
rotation_offset_deg | 0.0494 | 0.0114 | 0.5981
edge_density | 0.2426 | 0.0000 | 0.0230
text_area_ratio | 0.1042 | 0.0000 | 0.0538
text_component_count | 0.1667 | 0.0000 | 38.3782
qr_seal_score | 0.3014 | 0.0000 | 31.6046
jpeg_blockiness | 0.0971 | 0.0000 | 0.5575
bytes_per_pixel | 0.1761 | 0.0000 | 0.0289

## 주요 플롯
- ![](reports/eda/compare/overlays/width_overlay.png)
- ![](reports/eda/compare/overlays/height_overlay.png)
- ![](reports/eda/compare/overlays/aspect_ratio_overlay.png)
- ![](reports/eda/compare/overlays/file_size_bytes_overlay.png)
- ![](reports/eda/compare/overlays/pixel_count_overlay.png)
- ![](reports/eda/compare/overlays/brightness_mean_overlay.png)
- ![](reports/eda/compare/overlays/brightness_std_overlay.png)
- ![](reports/eda/compare/overlays/laplacian_var_overlay.png)
- ![](reports/eda/compare/overlays/colorfulness_overlay.png)
- ![](reports/eda/compare/overlays/margin_ratio_overlay.png)
- ![](reports/eda/compare/overlays/rotation_offset_deg_overlay.png)
- ![](reports/eda/compare/overlays/edge_density_overlay.png)
- ![](reports/eda/compare/overlays/text_area_ratio_overlay.png)
- ![](reports/eda/compare/overlays/text_component_count_overlay.png)
- ![](reports/eda/compare/overlays/qr_seal_score_overlay.png)
- ![](reports/eda/compare/overlays/jpeg_blockiness_overlay.png)
- ![](reports/eda/compare/overlays/bytes_per_pixel_overlay.png)

## 샘플 보드
- rotation_offset_deg: Train ![](reports/eda/sample_boards/train_rotation_offset_deg_board.png), Test ![](reports/eda/sample_boards/test_rotation_offset_deg_board.png)
- margin_ratio: Train ![](reports/eda/sample_boards/train_margin_ratio_board.png), Test ![](reports/eda/sample_boards/test_margin_ratio_board.png)
- brightness_mean: Train ![](reports/eda/sample_boards/train_brightness_mean_board.png), Test ![](reports/eda/sample_boards/test_brightness_mean_board.png)
- laplacian_var: Train ![](reports/eda/sample_boards/train_laplacian_var_board.png), Test ![](reports/eda/sample_boards/test_laplacian_var_board.png)
- qr_seal_score: Train ![](reports/eda/sample_boards/train_qr_seal_score_board.png), Test ![](reports/eda/sample_boards/test_qr_seal_score_board.png)

## 품질 체크리스트
- 동일 bin/축 적용 여부 재확인
- 분석용 로더가 학습/추론 전처리와 동일한지
- 샘플러 가중치(0.85)가 EDA 통계에 영향을 주지 않는지
- 손상 파일/중복 이미지 탐지 수행
- Fold 분포와 Train 전체 분포 차이 점검

## 추가 실험 제안
- A/B: Test 특이 분포(회전, 여백) 반영한 증강 vs 기존 증강
- A/B: JPEGCompression + Blur 증강 ON/OFF
- A/B: Orientation normalize 후 추론 TTA