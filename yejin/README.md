# yejin — 박예진 실험 작업 폴더

masked pool 병합 실험(task2-masked / task3 / task4)과 최종 WBF 앙상블 추론까지의
작업물을 담는 폴더입니다. 실험 노트북들에서 반복 정의되던 코드를 `pipeline/` 패키지로
모듈화했고, 노트북들은 이 모듈을 import해서 사용합니다.

## 폴더 구성

```
yejin/
├── pipeline/                  # 공통 파이프라인 모듈 (노트북에서 import)
│   ├── cloud.py               #   실행 환경(Kaggle/Colab) 감지 + 입력 경로/파일 재귀 탐색
│   ├── corrections.py         #   corrections 스냅샷 하드코딩 + json 저장
│   ├── pools.py               #   합성/masked pool 로드·검증·병합, 74종 라벨 매핑
│   ├── folds.py               #   고정 fold 분할(json) 저장/로드, 그룹 누수 점검, 분포 요약
│   ├── wbf.py                 #   WBF 융합, 예측 라벨 정제, 제출 CSV 생성
│   ├── viz.py                 #   GT/예측 클래스별 crop 갤러리
│   └── yolo.py                #   YOLO 5-fold 학습·COCO->YOLO 변환·앙상블 수집
└── notebooks/
    ├── kaggle/                # Kaggle 실행용 (Save & Run All 전제)
    │   ├── task2_test18_boost_masked_kaggle.ipynb     # RF-DETR medium 5-fold (+masked pool)
    │   ├── task3_fulldata_large_masked_kaggle.ipynb   # RF-DETR large full-data 단일 학습
    │   ├── task4_yolov8_5fold_masked_kaggle.ipynb     # YOLOv8m fold 분담 학습 (fold1~4)
    │   └── ensemble_wbf_inference_kaggle.ipynb        # 체크포인트 Input -> WBF 앙상블 제출
    ├── colab/
    │   └── task4_yolov8_5fold_masked_colab.ipynb      # YOLOv8m fold 분담 학습 (fold0, Drive 백업)
    └── legacy/                # 모듈화 이전의 초기 실험 원본 (기록 보존용, 수정하지 않음)
        ├── task0_baseline_kaggle.ipynb
        ├── task1_train56_boost_kaggle.ipynb
        ├── task2_test18_boost_kaggle.ipynb
        └── task4_yolo11_5fold_colab.ipynb
```

## 노트북에서 모듈을 쓰는 방법

노트북은 실행 시 팀 저장소를 clone한 뒤 `RF_DETR_split_ver`(팀 공통 코드)와
`yejin`(이 폴더)을 모두 import 경로에 추가합니다.

```python
REPO_BRANCH = 'main'   # yejin/* 브랜치 병합 전에 실행하려면 해당 브랜치명으로 변경
!git clone --depth 1 -b {REPO_BRANCH} {REPO_URL} {REPO_DIR}
sys.path.insert(0, os.path.join(REPO_DIR, 'RF_DETR_split_ver'))
sys.path.insert(0, os.path.join(REPO_DIR, 'yejin'))

from pipeline import cloud, pools, folds, wbf, viz       # 필요 모듈만
from pipeline.corrections import save_corrections_snapshot
```

- 각 모듈은 자신이 필요한 외부 패키지만 import합니다 (yolo -> ultralytics,
  wbf -> ensemble-boxes). 노트북 1번 셀의 설치 목록이 그에 맞춰져 있습니다.
- **저장소 공통 코드(`RF_DETR_split_ver`)는 수정하지 않습니다.** 그대로 쓰기 어려운
  부분만 `pipeline/`에 별도 구현했으며, 각 함수 docstring에 그 이유를 남겨두었습니다.

## 실험 설계 요약 (masked 계열)

| 실험 | 모델 | 데이터 | 산출물 |
|---|---|---|---|
| task2-masked | RF-DETR medium × 5-fold | 원본 train + 합성 pool2 + masked pool (74종) | fold별 best 체크포인트 + 제출 CSV |
| task3 | RF-DETR large × 1 (full-data, 15ep) | task2-masked와 동일 (split 없음) | 마지막(best_total) 체크포인트 + 단일모델 CSV |
| task4 | YOLOv8m × 5-fold (cls gain 1.5) | task2-masked와 동일 | fold별 best 체크포인트 |
| ensemble | 위 체크포인트 전부 | test | WBF 융합 제출 CSV (조합/가중치별 파일명) |

핵심 규칙:
- **fold 분할은 `fold_split_masked.json`으로 고정** — 세션/계정/플랫폼(Colab-Kaggle)이
  달라도 fold 구성이 동일해야 fold-matched WBF가 성립합니다 (`pipeline.folds` docstring 참고).
- **masked pool은 전체 파일을 `msk_` 접두어로 리네임** — 원본 train과 파일명 충돌 방지.
  fold 그룹 키 계산 시에는 접두어를 벗겨 원본과 같은 구성 그룹으로 묶습니다.
- **corrections는 구버전 스냅샷으로 고정** (`pipeline/corrections.py` 상단 주의 참고) —
  fold-matched 앙상블 멤버 간 학습 라벨 조건을 일치시키기 위함입니다.
