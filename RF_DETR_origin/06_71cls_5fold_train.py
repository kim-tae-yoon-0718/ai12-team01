#@title 74클래스 5-fold 학습 (RFDETRMedium, res 576, COCO 가중치 출발)
#@markdown dataset_74 복원본으로 fold0~4를 순차 학습. fold별 best를 Drive에 백업하고, 이미 백업된 fold는 건너뜀(이어하기). 세션마다 [0][1]과 zip 복원 먼저 실행 필요

# === [0] 설치 (매 세션) ==========================================
!pip install -q "rfdetr[train,loggers]"                    # RF-DETR 학습+로깅 의존성

# === [1] 마운트 + 경로 (매 세션) ==================================
from google.colab import drive                              # 코랩-드라이브 연결 도구
drive.mount('/content/drive')                                # 드라이브 마운트

import os, shutil                                            # 경로·복사 도구
PROJ_ROOT = "/content/drive/MyDrive/1팀 공유 문서/ai12-level1-project"  # zip·백업 공통 상위

# === [2] dataset_74 복원 (매 세션) ================================
ZIP = os.path.join(PROJ_ROOT, "dataset_74_5fold.zip")        # 65장 삭제 반영된 최신 zip
print("zip 존재:", os.path.exists(ZIP))                       # True 확인
!cp "$ZIP" /content/                                          # 드라이브 -> 로컬 복사
!unzip -qo /content/dataset_74_5fold.zip -d /content/dataset_74  # 압축 해제(-o=덮어쓰기)
print("복원 fold:", sorted(d for d in os.listdir("/content/dataset_74") if d.startswith("fold")))

# === [3] B~J: 74클래스 5-fold 학습 =================================
import torch                                                  # GPU 도구
from rfdetr import RFDETRMedium                                # 모델 변형: Medium (res 576 기본)

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")  # GPU 확인

# --- 하이퍼파라미터 (C / C-1 / E / J) — 기존 56종 실험값 재사용 ---
LR, LR_ENCODER, WEIGHT_DECAY        = 1e-4, 1.5e-4, 1e-4      # C: 학습률·encoder 차등·L2
LR_SCHEDULER, WARMUP_EPOCHS, LR_MIN = "cosine", 0.0, 0.0      # C-1: 스케줄러·warmup·최저lr
EPOCHS, BATCH_SIZE, GRAD_ACCUM      = 100, 4, 4               # E: L4 + Medium/576 대비 4×4(유효16)
RESOLUTION                          = 576                      # 입력 해상도 (Medium 기본)
ES_PATIENCE, ES_MIN_DELTA           = 10, 0.001               # J: early stopping
MODEL_TAG = "74cls_medium_res576"                              # 실험명 태그 (56종 실험과 구분)

BACKUP = os.path.join(PROJ_ROOT, "outputs")                    # best 백업 위치 (Drive)
os.makedirs(BACKUP, exist_ok=True)

for fi in range(5):                                            # fold 0~4 순차 학습
    exp = f"{MODEL_TAG}_lr1e-4_fold{fi}"                        # D: 실험명
    dst = os.path.join(BACKUP, f"{exp}_best.pth")               # 이 fold의 best 백업 경로
    if os.path.exists(dst):                                     # 이미 끝난 fold면
        print(f"[fold {fi}] 백업 존재 → 건너뜀"); continue       # 이어하기 (런타임 끊김 대비)

    out = f"/content/outputs/{exp}"; os.makedirs(out, exist_ok=True)  # 로컬 출력 폴더
    print(f"\n{'='*50}\n[fold {fi}] 학습 시작\n{'='*50}")

    model = RFDETRMedium(resolution=RESOLUTION)                 # B: COCO 사전학습 가중치에서 새로 시작
    model.train(                                                 # E~J: 학습·검증·best·earlystop 내장
        dataset_dir      = f"/content/dataset_74/fold{fi}",     # A: 74클래스 fold별 데이터
        output_dir       = out,
        epochs           = EPOCHS, batch_size = BATCH_SIZE, grad_accum_steps = GRAD_ACCUM,  # E
        lr = LR, lr_encoder = LR_ENCODER, weight_decay = WEIGHT_DECAY,                       # C
        lr_scheduler = LR_SCHEDULER, warmup_epochs = WARMUP_EPOCHS, lr_min_factor = LR_MIN,  # C-1
        early_stopping = True, early_stopping_patience = ES_PATIENCE,                        # J
        early_stopping_min_delta = ES_MIN_DELTA,                                             # J
        tensorboard = True,                                                                  # D
    )

    src = os.path.join(out, "checkpoint_best_total.pth")        # 학습이 저장한 best
    if os.path.exists(src):
        shutil.copy(src, dst); print(f"[fold {fi}] best 백업 → {dst}")  # Drive로 백업
    del model; torch.cuda.empty_cache()                          # GPU 메모리 정리 (다음 fold 대비)
    print(f"[fold {fi}] 완료")

print("\n▶ 74클래스 5폴드 학습 완료 — best 5개 준비됨(WBF 앙상블용)")