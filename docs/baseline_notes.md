# Baseline Code Highlights

- 제출 파일(`submission.csv`)을 생성할 때 `sample_submission.csv` 의 `image_id` 순서와 일치하는지 `assert` 로 검증한 뒤 저장합니다.  
  - 이 체크를 지키면 Kaggle/AIFactory 스타일 리더보드에서 인덱스 불일치로 점수가 0이 되는 사고를 예방할 수 있습니다.
- 추후 커스텀 추론 스크립트를 작성할 때도 동일한 검증 로직을 유지해 제출 안정성을 확보하세요.

