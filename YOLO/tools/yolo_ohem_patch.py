"""Loss-level OHEM patch for Ultralytics YOLO detection training.

The patch keeps every foreground/assigned anchor and only the hardest
background anchors for classification loss. Box/DFL losses are untouched.

This is intentionally a runtime monkey patch. Ultralytics internals change
across releases, so keeping the patch isolated makes it easier to audit or
disable without modifying site-packages.
"""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


_OHEM_NEGATIVE_RATIO = 0.25
_OHEM_MIN_NEG_PER_IMAGE = 16
_OHEM_MAX_NEG_PER_IMAGE = 2048
_OHEM_BACKGROUND_ONLY_NEG_PER_IMAGE = 32
_OHEM_NEGATIVE_LOSS_WEIGHT = 0.25
_ORIGINAL_TRAINER_SAVE_MODEL = None


class OHEMv8DetectionLoss(v8DetectionLoss):
    """v8/v11 detection loss with hard-negative mining on classification loss.

    This class intentionally lives at module scope. Ultralytics saves model
    checkpoints with pickle, and pickle cannot serialize local classes created
    inside ``enable_yolo_detection_ohem``.
    """

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights

        # OHEM: keep foreground anchors and only hard background anchors.
        anchor_loss = bce_loss.sum(-1)  # (bs, num_anchors)
        pos_mask = target_scores.sum(-1).gt(0)
        keep_mask = pos_mask.clone()
        for image_idx in range(batch_size):
            neg_mask = ~pos_mask[image_idx]
            neg_count = int(neg_mask.sum().item())
            if neg_count <= 0:
                continue
            pos_count = int(pos_mask[image_idx].sum().item())
            if pos_count > 0:
                keep_count = max(_OHEM_MIN_NEG_PER_IMAGE, int(pos_count * _OHEM_NEGATIVE_RATIO))
            else:
                keep_count = _OHEM_BACKGROUND_ONLY_NEG_PER_IMAGE
            keep_count = max(0, min(_OHEM_MAX_NEG_PER_IMAGE, keep_count, neg_count))
            if keep_count <= 0:
                continue
            neg_scores = anchor_loss[image_idx].masked_fill(~neg_mask, -1)
            topk_idx = torch.topk(neg_scores, k=keep_count, largest=True, sorted=False).indices
            keep_mask[image_idx, topk_idx] = True

        pos_loss = (bce_loss * pos_mask.unsqueeze(-1)).sum()
        neg_keep_mask = keep_mask & ~pos_mask
        neg_loss = (bce_loss * neg_keep_mask.unsqueeze(-1)).sum()
        pos_norm = target_scores_sum
        neg_norm = max(neg_keep_mask.sum(), 1)
        # Keep foreground learning on Ultralytics' original normalization,
        # and add a bounded hard-negative term. This avoids empty/background
        # images dominating batches that contain only a few foreground boxes.
        loss[1] = (pos_loss / pos_norm) + _OHEM_NEGATIVE_LOSS_WEIGHT * (neg_loss / neg_norm)

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


def _patch_trainer_save_model() -> None:
    """Strip the runtime OHEM criterion from checkpoints.

    The criterion is only needed while training. Keeping it inside the saved
    model makes checkpoints depend on this helper module at load time, so we
    remove it just for serialization and restore it immediately afterwards.
    """

    global _ORIGINAL_TRAINER_SAVE_MODEL

    from ultralytics.engine.trainer import BaseTrainer

    if _ORIGINAL_TRAINER_SAVE_MODEL is not None:
        return

    _ORIGINAL_TRAINER_SAVE_MODEL = BaseTrainer.save_model

    def save_model_without_ohem_criterion(self):
        stripped = []
        for candidate in (getattr(self, "model", None), getattr(getattr(self, "ema", None), "ema", None)):
            criterion = getattr(candidate, "criterion", None)
            if isinstance(criterion, OHEMv8DetectionLoss):
                stripped.append((candidate, criterion))
                candidate.criterion = None
        try:
            return _ORIGINAL_TRAINER_SAVE_MODEL(self)
        finally:
            for candidate, criterion in stripped:
                candidate.criterion = criterion

    BaseTrainer.save_model = save_model_without_ohem_criterion


def enable_yolo_detection_ohem(
    *,
    negative_ratio: float = 0.25,
    min_neg_per_image: int = 16,
    max_neg_per_image: int = 2048,
    background_only_neg_per_image: int = 32,
    negative_loss_weight: float = 0.25,
) -> dict[str, Any]:
    """Enable OHEM for Ultralytics YOLO detection loss.

    Args:
        negative_ratio: Keep up to this many negative anchors per foreground
            anchor on images that contain labels.
        min_neg_per_image: Minimum hard negatives kept for labeled images.
        max_neg_per_image: Maximum hard negatives kept per image.
        background_only_neg_per_image: Hard negatives kept for empty-label
            background-only images.
        negative_loss_weight: Additional multiplier for selected hard-negative
            classification loss. This keeps background useful but prevents it
            from dominating foreground learning.

    Returns:
        Patch metadata for logging.
    """

    from ultralytics.nn.tasks import DetectionModel

    global _OHEM_NEGATIVE_RATIO
    global _OHEM_MIN_NEG_PER_IMAGE
    global _OHEM_MAX_NEG_PER_IMAGE
    global _OHEM_BACKGROUND_ONLY_NEG_PER_IMAGE
    global _OHEM_NEGATIVE_LOSS_WEIGHT

    _OHEM_NEGATIVE_RATIO = negative_ratio
    _OHEM_MIN_NEG_PER_IMAGE = min_neg_per_image
    _OHEM_MAX_NEG_PER_IMAGE = max_neg_per_image
    _OHEM_BACKGROUND_ONLY_NEG_PER_IMAGE = background_only_neg_per_image
    _OHEM_NEGATIVE_LOSS_WEIGHT = negative_loss_weight

    def init_ohem_criterion(self):
        if getattr(self, "end2end", False):
            raise RuntimeError("OHEM patch currently supports standard YOLO detection heads, not end2end loss.")
        return OHEMv8DetectionLoss(self)

    DetectionModel.init_criterion = init_ohem_criterion
    _patch_trainer_save_model()
    return {
        "enabled": True,
        "loss_class": "OHEMv8DetectionLoss",
        "negative_ratio": negative_ratio,
        "min_neg_per_image": min_neg_per_image,
        "max_neg_per_image": max_neg_per_image,
        "background_only_neg_per_image": background_only_neg_per_image,
        "negative_loss_weight": negative_loss_weight,
    }
