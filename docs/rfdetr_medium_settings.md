- 변형:       Small → Medium
- resolution: 512 → 576 (Medium 기본, 자동)
나머지 전부 동일 (한 번에 한 변수 원칙 유지)

[모델]
- 변형: RFDETRMedium (34M)
- resolution: 576 (Medium 기본값)
- num_classes: 56 (자동, 1-indexed)

[데이터]
- 분할: StratifiedGroupKFold 5폴드 (구성 114 그룹, seed 42)
- 박스: 773 (패치 후)
- ※ Small과 동일 데이터 재사용 (A단계 재실행 X)

[배치]
- batch_size: 8
- grad_accum_steps: 2  (유효배치 16)
- GPU: L4

[학습률]
- lr: 1e-4
- lr_encoder: 1.5e-4
- weight_decay: 1e-4

[스케줄러]
- lr_scheduler: cosine
- warmup_epochs: 0
- lr_min_factor: 0

[학습]
- epochs: 100 (early stop으로 조기종료)
- early_stopping: True / patience 10 / min_delta 0.001

[증강]
- 기본 증강 off

[best 선정]
- monitor: mAP@[0.50:0.95] (RF-DETR 기본)
- 대회지표 mAP@[0.75:0.95]는 별도 재계산