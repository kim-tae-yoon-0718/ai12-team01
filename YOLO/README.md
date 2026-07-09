# YOLO11m OHEM 5-Fold Experiment

This folder contains the YOLO experiment code only.

## Why YOLO

YOLO11m is used here as a fast experiment path, not because RF-DETR is assumed to be incapable of good boxes. A teammate reached a much higher score with the same model family, so the current hypotheses are about data/training setup:

1. background synthesis may be biasing boxes;
2. scale augmentation up to `1.5x` may have encouraged inflated boxes;
3. selecting best checkpoints by `mAP@0.75` instead of `mAP@[0.75:0.95]` may have selected weaker high-IoU checkpoints;
4. validation may not be test-like enough.

YOLO11m is convenient for quickly testing these assumptions, especially with OHEM-style hard background handling.

## Layout

- `notebooks/train_yolo11m_74_5fold_mps_ohem_param_crosscheck.ipynb`: audit and optional local MPS training notebook.
- `tools/convert_coco5fold_to_yolo_with_backgrounds.py`: convert RF-DETR 5-fold COCO data to YOLO format and add empty-label background negatives.
- `tools/audit_yolo_rfdetr_params.py`: compare YOLO settings against RF-DETR data/config.
- `tools/train_yolo_5fold_mps.py`: train YOLO folds and select `best_map75_95.pt`.
- `tools/yolo_ohem_patch.py`: runtime Ultralytics detection-loss OHEM patch.
- `requirements-yolo.txt`: minimal YOLO pilot dependencies.

## Path Model

Code lives under this `YOLO/` folder. Data and outputs should stay outside git under a workspace selected by `DATA_ROOT`.

Common local setup:

```python
import os

os.environ["DATA_ROOT"] = "/path/to/detectionproject"
os.environ["RFDETR_REPO"] = "/path/to/ai12-team01-rfdetr"
```

The notebook and tools use:

- `DATA_ROOT/working/yolo_74_5fold_bg_mps`
- `DATA_ROOT/working/rfdetr_dataset_74_hidden45_canvas_balanced_5fold_cls0_mps`
- `DATA_ROOT/working/yolo_outputs/yolo11m_74_5fold_bg_mps_10ep`

No local machine-specific absolute paths are stored in the committed notebook or tools.

