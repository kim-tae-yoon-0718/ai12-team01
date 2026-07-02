# ai12-team01

코드잇 AI12기 1팀 알까기포머

## 저장소 목적
이 저장소는 경구약제 이미지 객체 검출 프로젝트의 메인 모델 및 실험 코드를 관리하는 공간입니다.

## 협업 규칙
- main 브랜치에 직접 push 금지
- 작업은 개인 브랜치에서 진행
- Pull Request 생성 후 승인받고 merge

## 프로젝트 구조

```
project/
├── RF_DETR_split_ver/    # 모듈 분리한 RF-DETR 파이프라인 (Colab 실행 기준)
│   ├── colab_setup.py    # Colab 진입점: Drive 마운트 + 데이터 준비/복원 오케스트레이션
│   ├── config.yaml       # 데이터 경로 · 모델 · 학습 하이퍼파라미터
│   ├── corrections.json  # annotation 오류 수정 내역 (아래 참고)
│   ├── dataset.py        # 데이터 경로 탐색, 5-fold 분할, COCO 포맷 생성
│   ├── model.py          # RF-DETR 모델 변형 생성 (nano~large, 체크포인트 로드)
│   ├── train.py          # K-Fold 학습 루프 + fold별 결과 리포팅
│   ├── utils.py          # 학습 곡선/리포팅 유틸
│   └── visualize.py      # 예측 수집, mAP 계산, 오답 시각화
├── RF_DETR_origin/       # 모듈 분리 이전 원본 Colab 셀 스크립트 (참고용)
├── docs/                 # 실험 설정 기록 등 문서
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 데이터 준비

데이터는 COCO 포맷 JSON annotation과 PNG 이미지로 구성되어 있습니다.
용량이 크고 팀 공용이라 레포지토리에는 포함하지 않고, **Google Drive 공유 폴더**로 관리하며 Colab에서 바로 마운트해서 사용합니다.

1. 팀 공유 폴더의 `sprint_ai_project1_data`를 본인의 "내 드라이브"에 바로가기로 추가
2. Colab 노트북에서 `RF_DETR_split_ver`를 작업 디렉토리로 두고 아래처럼 실행

```python
from colab_setup import mount_drive, prepare_data, restore_data
from train import load_config, run_kfold

mount_drive()
config = load_config('config.yaml')

# 최초 1회: sprint_ai_project1_data를 자동 탐색해 5-fold 데이터셋 생성 + zip 백업
prepare_data(config)
# 이후 세션(zip 이미 있음): 압축만 복원해 재사용
# restore_data(config)

run_kfold(config)
```

- 데이터 경로는 하드코딩하지 않고, `config.yaml`의 `data.data_root_candidates`(우선 탐색 후보) / `data.search_root`(재귀 검색 루트)를 기준으로 `dataset.find_data_root()`가 자동으로 찾습니다. 팀원마다 Drive 폴더 구조(`1팀 공유 문서` 하위 여부 등)가 달라도 동작합니다.
- zip 백업/복원 위치가 사람마다 다르면 `prepare_data(config, archive_dir=...)` / `restore_data(config, archive_dir=...)`처럼 인자로 직접 지정할 수 있습니다.

---

## Annotation 수정 내역

원본 데이터의 annotation 오류를 수정한 내역이 `corrections.json`에 기록되어 있습니다.  

| 유형 | 건수 |
|---|---|
| bbox 좌표 오류 수정 | 6건 |
| 누락 bbox 추가 | 8건 |

---