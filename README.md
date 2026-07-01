# Pill Detection RT-DETRv2

RT-DETRv2 기반 알약 객체 탐지 학습 노트북입니다. 로컬 Mac 환경 기준으로 `sprint_ai_project1_data/` 데이터를 읽고, 작업 산출물은 `working/` 아래에 생성합니다.

## Files

- `rtdetrv2_x_pill_detection_kaggle.ipynb`: 학습, 검증, 제출 생성 노트북
- `requirements.txt`: 공유용 Python 패키지 목록
- `corrections.json`: annotation 보정 목록
- `sprint_ai_project1_data/`: 다운로드 후 압축 해제된 데이터 폴더

## Environment

현재 노트북은 Jupyter kernel `Python (dataanalysis)`를 사용하도록 설정되어 있습니다.

새 환경에서 재현하려면 Python 3.11 계열을 권장합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name dataanalysis --display-name "Python (dataanalysis)"
```

이미 로컬에 `/Users/pio/.DataAnalysis/bin/python` 커널이 있으면 그대로 사용하면 됩니다.

Antigravity에서 커널이 보이지 않으면 아래를 한 번 실행한 뒤 창을 reload하세요.

```bash
ln -sfn /Users/pio/.DataAnalysis .venv
/Users/pio/.DataAnalysis/bin/python -m ipykernel install --user --name dataanalysis --display-name "Python (dataanalysis)"
```

그 다음 Antigravity에서 interpreter를 `.venv/bin/python`으로 선택하면 됩니다.

## Kaggle Credentials

Kaggle 인증값은 repo에 직접 넣지 않습니다. 로컬에서는 `.env.example`을 `.env`로 복사한 뒤 본인 계정 값을 채우고, `a.ipynb`를 실행하면 `~/.kaggle/kaggle.json`이 생성됩니다.

```bash
cp .env.example .env
```

```text
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

`.env`와 `kaggle.json`은 `.gitignore`에 포함되어 커밋되지 않습니다.

## Data Layout

프로젝트 루트에 아래 구조가 있어야 합니다.

```text
sprint_ai_project1_data/
  train_images/
  train_annotations/
  test_images/
```

노트북은 `train.csv`가 없으면 `train_annotations/**/*.json`에서 학습용 테이블을 자동 생성합니다.

## Run

1. `rtdetrv2_x_pill_detection_kaggle.ipynb`를 엽니다.
2. 커널을 `Python (dataanalysis)`로 선택합니다.
3. 위에서부터 순서대로 실행합니다.

노트북은 실행 중 `working/RT-DETR`에 RT-DETR repo를 clone하고, COCO 변환 데이터는 `working/pill_coco/`에 만듭니다.

## Notes

- `PROCESSOR = "mps"`로 Apple Silicon MPS를 우선 사용합니다. CUDA 머신에서는 `"cuda"`로 바꾸면 됩니다.
- `70/75/90`처럼 촬영각만 다른 near-duplicate 이미지는 같은 그룹으로 묶어 train/val이 섞이지 않게 split합니다.
- 공유 시 대용량 원본 zip과 `working/` 산출물은 보통 제외해도 됩니다.
