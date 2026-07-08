# RF-DETR Large v1.8+ 5-Fold Local MPS Notebook Summary

Notebook:

- `train_74_hidden45_mps_rfdetr_large_v18_5fold.ipynb`

Purpose:

- Train RF-DETR Large on the latest 74-class canvas-balanced dataset using local Apple Silicon MPS.
- Build 5 folds from the local train/valid COCO dataset.
- Add class `0` as a categories-only dummy/background placeholder.
- Save fold checkpoints and metrics under local `working/rfdetr_outputs`.
- Do not run final submission postprocessing in this training notebook; run local inference/postprocess after checkpoints are ready.

Core settings:

- `rfdetr>=1.8.0`
- `MODEL_VARIANT = 'large'`
- `DEVICE = 'mps'`
- `PYTORCH_ENABLE_MPS_FALLBACK = 1`
- `NUM_FOLDS = 5`
- `FOLD_INDICES = [1, 2, 3, 4]` for the next local run; fold 0 has already been run
- `RESUME_CHECKPOINT_BY_FOLD[0]` points to the Colab CUDA fold0 `checkpoint_4.ckpt`; folds 1-4 use their own output folders and auto-resume if rerun
- `VALID_INCLUDE_CANVAS = False`
- `EPOCHS = 100`
- `BATCH_SIZE = 1`
- `GRAD_ACCUM_STEPS = 16`
- `EARLY_STOPPING = True`
- `EARLY_STOPPING_PATIENCE = 1`
- `EARLY_STOPPING_MIN_DELTA = 0.001`
- `EARLY_STOPPING_USE_EMA = True`
- `ENABLE_TRAIN_AUGMENTATION = True`
- augmentation: scale `0.85-1.50`, translate `+-0.04`, rotate `+-90`, weak brightness/contrast, no flip

Local paths:

- source dataset: `/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/rfdetr_dataset_74_hidden45_canvas_balanced`
- folded dataset: `/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/rfdetr_dataset_74_hidden45_canvas_balanced_5fold_cls0_mps`
- outputs: `/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/rfdetr_outputs/mps_large_v18_5fold`
- best backups: `/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/rfdetr_outputs/mps_large_v18_5fold/best`
- explicit fold0 resume checkpoint: `/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/rfdetr_outputs/l_5f_resume_checkpoints/checkpoint_4.ckpt`

Resume behavior:

- If `RESUME_CHECKPOINT_BY_FOLD[fold_idx]` is a valid checkpoint path, it is written into that fold's generated YAML as `train.resume`.
- If a fold value is `None` or missing, the wrapper uses `auto_resume=true` to find the latest local `checkpoint_N.ckpt` under `OUTPUT_ROOT / fold_model_tag`.
- For a fresh local-only run, set `RESUME_CHECKPOINT_BY_FOLD = {}` or set the fold entry to `None`.

Early stopping note:

- Early stopping is validation/eval based, not mini-batch based.
- With `eval_interval = 1`, patience counts epochs/evaluations.
- One epoch is roughly 2.3k train images per fold, so `patience = 1` is the closest match to a roughly 2k-image no-improvement window.

Class 0 rule:

- Class `0` is the background/no-object dummy category in COCO `categories` and `label_map_74.json`.
- It must have zero bbox annotations; RF-DETR/DETR learns background through unmatched queries and non-annotated image regions.
- Real pill labels are `1..74`.
- Local postprocess should drop any prediction mapped to category `0`.

Expected output:

- Per-fold output folders under local `OUTPUT_ROOT`.
- Per fold: `metrics.csv`, `checkpoint_best_total.pth`, `checkpoint_best_map75.ckpt`, `map75_summary.json`.
- Dataset mapping files: `label_map_74.json`, `category_mapping.csv`, `_5fold_ready.json`.
