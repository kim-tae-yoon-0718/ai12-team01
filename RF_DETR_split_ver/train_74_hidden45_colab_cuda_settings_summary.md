# RF-DETR 74-Class Colab CUDA Run Summary

Last updated: 2026-07-06

## Files

- Notebook: `RF_DETR_split_ver/train_74_hidden45_colab_cuda.ipynb`
- Base config read by the notebook: `RF_DETR_split_ver/config_74_hidden45_canvas_balanced.yaml`
- Runtime config written in Colab: `RF_DETR_split_ver/config_74_hidden45_colab_cuda.yaml`

## Dataset

- Drive project root: `/content/drive/MyDrive/ai12-level1-project`
- Dataset archive folder: `ai12-level1-project/dataset_74_hidden45_latest_20260706`
- Default archive: `dataset_74_hidden45_canvas_balanced_train_valid_20260706.tar.gz`
- Extracted dataset path in Colab: `/content/rfdetr_colab/datasets/rfdetr_dataset_74_hidden45_canvas_balanced`
- Test images folder URL: `https://drive.google.com/drive/folders/1ZdGRPB3Xg4-1QKrKKKzzBOy7d2gorfuw`

## Model

- RF-DETR variant: `nano`
- Baseline tag: `nano_74_hidden45_canvas_balanced_colab_cuda`
- Augmented run tag: `nano_74_hidden45_canvas_balanced_colab_cuda_aug_scale150_rot90_v1`
- Active notebook default: augmented run tag

## Training Controls

- Device: `cuda`
- Epochs: `12`
- Batch size: `2`
- Gradient accumulation steps: `8`
- Effective batch size: `16`
- Seed: `42`
- Workers: `2`
- Pin memory: `true`
- AMP dtype: `auto`
- Multi-scale: `false`
- Eval interval: `1`
- Eval max detections: `100`
- Compute validation loss: `true`
- Compute test loss: `false`

## Augmentation

The notebook default is augmentation ON:

```python
ENABLE_TRAIN_AUGMENTATION = True
AUGMENTATION_NAME = "aug_scale150_rot90_v1"
```

Train-only augmentation:

- Scale: `[0.85, 1.50]`
- Translate percent: `[-0.04, 0.04]`
- Rotate: `[-90, 90]` degrees
- Brightness limit: `0.08`
- Contrast limit: `0.08`
- Color augmentation probability: `0.25`
- Affine augmentation probability: `0.35`
- Flip: disabled

To run the baseline without online augmentation:

```python
ENABLE_TRAIN_AUGMENTATION = False
```

## Checkpoint Policy

- Output root: `/content/drive/MyDrive/ai12-level1-project/rfdetr_outputs`
- Active checkpoint folder: `rfdetr_outputs/{MODEL_TAG}`
- Checkpoint interval: `1`
- Auto-resume: `true`
- Resume source: latest `checkpoint_N.ckpt` in the active checkpoint folder unless `resume` is set explicitly
- Best mAP@0.75 checkpoint: `rfdetr_outputs/{MODEL_TAG}/checkpoint_best_map75.ckpt`
- Best checkpoint backups: `rfdetr_outputs/best`

Expected augmented-run checkpoint paths:

```text
/content/drive/MyDrive/ai12-level1-project/rfdetr_outputs/nano_74_hidden45_canvas_balanced_colab_cuda_aug_scale150_rot90_v1/
/content/drive/MyDrive/ai12-level1-project/rfdetr_outputs/best/nano_74_hidden45_canvas_balanced_colab_cuda_aug_scale150_rot90_v1_best_map75.ckpt
```

## Submission Output

Test inference is off by default:

```python
RUN_TEST_INFERENCE = False
```

After training or finalize-only, set it to `True` to create submission CSV:

```text
/content/drive/MyDrive/ai12-level1-project/rfdetr_outputs/{MODEL_TAG}/submissions/{MODEL_TAG}_submission.csv
```

Submission columns:

```text
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_w, bbox_h, score
```

## Recommended Run Flow

1. Open the notebook in Colab with a CUDA runtime.
2. Leave `RUN_DRY_RUN = True` and run setup/config cells once.
3. Set `RUN_FULL_TRAIN = True` and run the training cell.
4. If training already finished but best mAP@0.75 was not copied, set `RUN_FINALIZE_ONLY = True` and run the finalization cell.
5. Set `RUN_TEST_INFERENCE = True` and run the submission cells.
