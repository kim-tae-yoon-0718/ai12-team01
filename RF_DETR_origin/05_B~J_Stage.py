#@title [셀 3] B~J 5폴드 학습 + best 백업 + 이어하기
import torch                                             # GPU 도구
from rfdetr import RFDETRSmall                             # 모델 변형: Small (Medium=RFDETRMedium)

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")  # GPU 확인

# --- 하이퍼파라미터 (C / C-1 / E / J) ---
LR, LR_ENCODER, WEIGHT_DECAY        = 1e-4, 1.5e-4, 1e-4  # C: 학습률·encoder차등·L2
LR_SCHEDULER, WARMUP_EPOCHS, LR_MIN = "cosine", 0.0, 0.0  # C-1: 스케줄러·warmup·최저lr
EPOCHS, BATCH_SIZE, GRAD_ACCUM      = 100, 8, 2           # E: epoch·배치·누적(유효16, L4)
ES_PATIENCE, ES_MIN_DELTA           = 10, 0.001          # J: early stopping
MODEL_TAG = "small_res512"                                # 실험명 태그(변형 바꾸면 교체)

BACKUP = os.path.join(PROJ_ROOT, "outputs")              # best 백업(PROJ_ROOT 기준)
os.makedirs(BACKUP, exist_ok=True)

for fi in range(5):                                       # fold 0~4
    exp = f"{MODEL_TAG}_lr1e-4_fold{fi}"                   # D: 실험명(변수 하나만 바꿔 구분)
    dst = os.path.join(BACKUP, f"{exp}_best.pth")
    if os.path.exists(dst):                                # 이미 끝난 fold면
        print(f"[fold {fi}] 백업 존재 → 건너뜀"); continue  # 이어하기

    out = f"/content/outputs/{exp}"; os.makedirs(out, exist_ok=True)
    print(f"\n{'='*50}\n[fold {fi}] 학습 시작\n{'='*50}")

    model = RFDETRSmall()                                  # B: fold마다 새 모델(초기화)
    model.train(                                           # E~J: 학습·검증·best·earlystop·로깅 내장
        dataset_dir      = f"/content/dataset/fold{fi}",   # A: fold별 데이터
        output_dir       = out,
        epochs           = EPOCHS, batch_size = BATCH_SIZE, grad_accum_steps = GRAD_ACCUM,  # E
        lr = LR, lr_encoder = LR_ENCODER, weight_decay = WEIGHT_DECAY,                       # C
        lr_scheduler = LR_SCHEDULER, warmup_epochs = WARMUP_EPOCHS, lr_min_factor = LR_MIN,  # C-1
        early_stopping = True, early_stopping_patience = ES_PATIENCE,                        # J
        early_stopping_min_delta = ES_MIN_DELTA,                                             # J
        tensorboard = True,                                                                  # D
    )

    src = os.path.join(out, "checkpoint_best_total.pth")   # 학습이 저장한 best
    if os.path.exists(src):
        shutil.copy(src, dst); print(f"[fold {fi}] best 백업 → {dst}")
    del model; torch.cuda.empty_cache()                    # GPU 정리(OOM 방지)
    print(f"[fold {fi}] 완료")

print("\n▶ 5폴드 학습 완료 — best 5개 준비됨(앙상블용)")