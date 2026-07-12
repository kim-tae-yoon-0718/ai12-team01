# legacy — 모듈화 이전의 초기 실험 노트북 (기록 보존용)

masked pool 도입 전에 실제 실험에 사용한 노트북 원본입니다. 실험 이력 추적을 위해
수정 없이 그대로 보관하며, 이후 실험은 `yejin/notebooks/{kaggle,colab}`의
pipeline 모듈 기반 노트북을 사용합니다.

| 파일 | 내용 |
|---|---|
| `task0_baseline_kaggle.ipynb` | 원본 train 56종 5-fold baseline (RF-DETR medium) |
| `task1_train56_boost_kaggle.ipynb` | train 56종 + 합성 pool1 보강 학습 |
| `task2_test18_boost_kaggle.ipynb` | test 전용 18종 + 합성 pool2로 74종 학습 (masked 이전 버전) |
| `task4_yolo11_5fold_colab.ipynb` | YOLO11m 5-fold (masked 이전, Colab) |

⚠ 이 노트북들은 corrections 스냅샷·경로 등이 작성 당시 기준이라 현재 데이터/브랜치
구성에서 그대로 재실행되지 않을 수 있습니다.
