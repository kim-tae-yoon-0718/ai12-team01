#!/usr/bin/env python3
"""Score a test submission against the frozen pseudo ground truth.

This is intentionally isolated from the training pipeline. It reads a
submission CSV and the exported pseudo/manual test annotations, then writes
metrics under working/reports/test_pseudo_gt_scores/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = PROJECT_ROOT / "working/submission.csv"
DEFAULT_GT = PROJECT_ROOT / "working/test_annotations/test_annotations.csv"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "working/reports/pill_class_number_map.csv"
DEFAULT_MODIFIED_MANIFEST = (
    PROJECT_ROOT
    / "working/reports/test_pseudo_gt_eval/modified_only_review/modified_images_manifest.csv"
)
DEFAULT_UNKNOWN_IGNORE = PROJECT_ROOT / "working/test_annotations/test_unknown_ignore_boxes.csv"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "working/reports/test_pseudo_gt_scores"


@dataclass(frozen=True)
class EvalInputs:
    submission: Path
    pseudo_gt: Path
    class_map: Path
    modified_manifest: Path
    unknown_ignore_boxes: Path
    out_dir: Path
    image_filter: str
    iou_threshold: float
    ignore_iou_threshold: float
    disable_unknown_ignore: bool
    freeze_inputs: bool


def parse_args() -> EvalInputs:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a test submission against the frozen pseudo-GT. "
            "This is not hidden leaderboard scoring."
        )
    )
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--pseudo-gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--modified-manifest", type=Path, default=DEFAULT_MODIFIED_MANIFEST)
    parser.add_argument(
        "--unknown-ignore-boxes",
        type=Path,
        default=DEFAULT_UNKNOWN_IGNORE,
        help=(
            "CSV of real test objects whose class is absent from the reference grid. "
            "Predictions overlapping these boxes are excluded from local scoring."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Used only when --out-dir is omitted.",
    )
    parser.add_argument(
        "--image-filter",
        choices=["all", "modified", "reviewed"],
        default="all",
        help=(
            "all: every pseudo-GT image; modified: images touched by correction log; "
            "reviewed: images with at least one reviewed/manual annotation."
        ),
    )
    parser.add_argument("--iou-threshold", type=float, default=0.75)
    parser.add_argument(
        "--ignore-iou-threshold",
        type=float,
        default=0.50,
        help="Prediction/unknown-box IoU at or above this value is ignored before scoring.",
    )
    parser.add_argument(
        "--disable-unknown-ignore",
        action="store_true",
        help="Do not ignore predictions overlapping unknown/open-set test objects.",
    )
    parser.add_argument(
        "--no-freeze-inputs",
        action="store_true",
        help="Do not copy input CSV snapshots into the score output directory.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = args.out_root / f"{args.submission.stem}_{args.image_filter}_{stamp}"

    return EvalInputs(
        submission=args.submission.resolve(),
        pseudo_gt=args.pseudo_gt.resolve(),
        class_map=args.class_map.resolve(),
        modified_manifest=args.modified_manifest.resolve(),
        unknown_ignore_boxes=args.unknown_ignore_boxes.resolve(),
        out_dir=out_dir.resolve(),
        image_filter=args.image_filter,
        iou_threshold=float(args.iou_threshold),
        ignore_iou_threshold=float(args.ignore_iou_threshold),
        disable_unknown_ignore=bool(args.disable_unknown_ignore),
        freeze_inputs=not args.no_freeze_inputs,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_n(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("N"):
        return f"N{int(text[1:]):02d}"
    return f"N{int(float(text)):02d}"


def load_class_map(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    if not path.exists():
        return {}, {}
    class_map = pd.read_csv(path)
    n_to_cat: dict[str, int] = {}
    cat_to_name: dict[int, str] = {}
    for row in class_map.itertuples(index=False):
        n_number = f"N{int(row.class_no):02d}"
        category_id = int(row.category_id)
        n_to_cat[n_number] = category_id
        cat_to_name[category_id] = str(row.name)
    return n_to_cat, cat_to_name


def resolve_category_id(df: pd.DataFrame, n_to_cat: dict[str, int], label: str) -> pd.Series:
    if "category_id" in df.columns:
        return pd.to_numeric(df["category_id"], errors="coerce").astype("Int64")

    for col in ["resolved_n_number", "n_number", "correct_n_number", "predicted_n_number"]:
        if col in df.columns:
            mapped = df[col].map(clean_n).map(n_to_cat)
            if mapped.notna().any():
                return mapped.astype("Int64")

    if "class_no" in df.columns:
        mapped = df["class_no"].map(lambda value: n_to_cat.get(clean_n(value)))
        if mapped.notna().any():
            return mapped.astype("Int64")

    raise ValueError(f"{label} needs category_id, n_number, predicted_n_number, or class_no columns.")


def normalize_boxes(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    required = ["image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")

    out["image_id"] = pd.to_numeric(out["image_id"], errors="coerce").astype("Int64")
    out["category_id"] = pd.to_numeric(out["category_id"], errors="coerce").astype("Int64")
    for col in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "score" not in out.columns:
        out["score"] = 1.0
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(1.0)

    out = out.dropna(subset=["image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]).copy()
    out = out[(out["bbox_w"] > 0) & (out["bbox_h"] > 0)].copy()
    out["image_id"] = out["image_id"].astype(int)
    out["category_id"] = out["category_id"].astype(int)
    return out


def load_ignore_boxes(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
    ignore = pd.read_csv(path)
    missing = [col for col in ["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"] if col not in ignore.columns]
    if missing:
        raise ValueError(f"unknown ignore box file is missing columns: {missing}")
    ignore = ignore.copy()
    ignore["image_id"] = pd.to_numeric(ignore["image_id"], errors="coerce").astype("Int64")
    for col in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
        ignore[col] = pd.to_numeric(ignore[col], errors="coerce")
    ignore = ignore.dropna(subset=["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]).copy()
    ignore = ignore[(ignore["bbox_w"] > 0) & (ignore["bbox_h"] > 0)].copy()
    ignore["image_id"] = ignore["image_id"].astype(int)
    return ignore


def load_inputs(inputs: EvalInputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, str]]:
    if not inputs.submission.exists():
        raise FileNotFoundError(inputs.submission)
    if not inputs.pseudo_gt.exists():
        raise FileNotFoundError(inputs.pseudo_gt)

    n_to_cat, cat_to_name = load_class_map(inputs.class_map)
    pred = pd.read_csv(inputs.submission)
    gt = pd.read_csv(inputs.pseudo_gt)

    pred = pred.copy()
    gt = gt.copy()
    pred["category_id"] = resolve_category_id(pred, n_to_cat, "submission")
    gt["category_id"] = resolve_category_id(gt, n_to_cat, "pseudo-GT")

    pred = normalize_boxes(pred, "submission")
    gt = normalize_boxes(gt, "pseudo-GT")

    if inputs.image_filter == "modified":
        if not inputs.modified_manifest.exists():
            raise FileNotFoundError(inputs.modified_manifest)
        modified = pd.read_csv(inputs.modified_manifest)
        image_ids = set(pd.to_numeric(modified["image_id"], errors="coerce").dropna().astype(int))
        gt = gt[gt["image_id"].isin(image_ids)].copy()
        pred = pred[pred["image_id"].isin(image_ids)].copy()
    elif inputs.image_filter == "reviewed":
        reviewed_mask = pd.Series(False, index=gt.index)
        if "review_status" in gt.columns:
            reviewed_mask |= gt["review_status"].fillna("").astype(str).eq("reviewed")
        if "source" in gt.columns:
            reviewed_mask |= gt["source"].fillna("").astype(str).str.startswith("manual_added")
        image_ids = set(gt.loc[reviewed_mask, "image_id"].astype(int))
        gt = gt[gt["image_id"].isin(image_ids)].copy()
        pred = pred[pred["image_id"].isin(image_ids)].copy()

    for category_id in sorted(set(gt["category_id"]) | set(pred["category_id"])):
        cat_to_name.setdefault(int(category_id), str(category_id))

    ignore = load_ignore_boxes(inputs.unknown_ignore_boxes)
    if inputs.image_filter != "all":
        image_ids = set(gt["image_id"].astype(int)) | set(pred["image_id"].astype(int))
        ignore = ignore[ignore["image_id"].isin(image_ids)].copy()

    return pred, gt, ignore, cat_to_name


def bbox_iou_xywh(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def bbox_str(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return ",".join(f"{float(row[col]):.2f}" for col in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"])


def split_predictions_overlapping_ignore(
    pred: pd.DataFrame,
    ignore: pd.DataFrame,
    iou_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pred.empty or ignore.empty:
        return pred.copy(), pred.iloc[0:0].copy()

    ignore_by_image = {int(image_id): group for image_id, group in ignore.groupby("image_id")}
    keep_rows = []
    ignored_rows = []
    for row in pred.itertuples(index=False):
        pred_box = [row.bbox_x, row.bbox_y, row.bbox_w, row.bbox_h]
        best_iou = 0.0
        for ignore_row in ignore_by_image.get(int(row.image_id), pd.DataFrame()).itertuples(index=False):
            ignore_box = [ignore_row.bbox_x, ignore_row.bbox_y, ignore_row.bbox_w, ignore_row.bbox_h]
            best_iou = max(best_iou, bbox_iou_xywh(pred_box, ignore_box))
        record = row._asdict()
        record["unknown_ignore_iou"] = best_iou
        if best_iou >= iou_threshold:
            ignored_rows.append(record)
        else:
            keep_rows.append(record)

    return pd.DataFrame(keep_rows), pd.DataFrame(ignored_rows)


def greedy_match(pred: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for image_id in sorted(set(gt["image_id"]) | set(pred["image_id"])):
        gt_img = gt[gt["image_id"] == image_id].reset_index(drop=False)
        pred_img = pred[pred["image_id"] == image_id].sort_values("score", ascending=False).reset_index(drop=True)
        used_gt: set[int] = set()

        for pred_rank, pred_row in enumerate(pred_img.itertuples(index=False), 1):
            pred_box = [pred_row.bbox_x, pred_row.bbox_y, pred_row.bbox_w, pred_row.bbox_h]
            best_local_idx: int | None = None
            best_iou = 0.0
            for local_idx, gt_row in gt_img.iterrows():
                if local_idx in used_gt:
                    continue
                gt_box = [gt_row.bbox_x, gt_row.bbox_y, gt_row.bbox_w, gt_row.bbox_h]
                iou = bbox_iou_xywh(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_local_idx = int(local_idx)

            pred_category = int(pred_row.category_id)
            if best_local_idx is not None and best_iou >= iou_threshold:
                gt_row = gt_img.loc[best_local_idx]
                used_gt.add(best_local_idx)
                gt_category = int(gt_row.category_id)
                match_type = "tp" if gt_category == pred_category else "class_error"
                rows.append(
                    {
                        "image_id": int(image_id),
                        "pred_rank": pred_rank,
                        "match_type": match_type,
                        "iou": best_iou,
                        "score": float(pred_row.score),
                        "gt_category_id": gt_category,
                        "pred_category_id": pred_category,
                        "gt_bbox": bbox_str(gt_row),
                        "pred_bbox": ",".join(f"{v:.2f}" for v in pred_box),
                    }
                )
            else:
                rows.append(
                    {
                        "image_id": int(image_id),
                        "pred_rank": pred_rank,
                        "match_type": "fp",
                        "iou": best_iou,
                        "score": float(pred_row.score),
                        "gt_category_id": pd.NA,
                        "pred_category_id": pred_category,
                        "gt_bbox": "",
                        "pred_bbox": ",".join(f"{v:.2f}" for v in pred_box),
                    }
                )

        for local_idx, gt_row in gt_img.iterrows():
            if int(local_idx) in used_gt:
                continue
            rows.append(
                {
                    "image_id": int(image_id),
                    "pred_rank": pd.NA,
                    "match_type": "fn",
                    "iou": 0.0,
                    "score": pd.NA,
                    "gt_category_id": int(gt_row.category_id),
                    "pred_category_id": pd.NA,
                    "gt_bbox": bbox_str(gt_row),
                    "pred_bbox": "",
                }
            )
    return pd.DataFrame(rows)


def build_per_class(matches: pd.DataFrame, gt: pd.DataFrame, pred: pd.DataFrame, cat_to_name: dict[int, str]) -> pd.DataFrame:
    categories = sorted(set(gt["category_id"]) | set(pred["category_id"]))
    rows = []
    for category_id in categories:
        support = int((gt["category_id"] == category_id).sum())
        pred_count = int((pred["category_id"] == category_id).sum())
        tp = int(((matches["match_type"] == "tp") & (matches["gt_category_id"] == category_id)).sum())
        fn = int(((matches["match_type"].isin(["fn", "class_error"])) & (matches["gt_category_id"] == category_id)).sum())
        fp = int(((matches["match_type"].isin(["fp", "class_error"])) & (matches["pred_category_id"] == category_id)).sum())
        precision = tp / (tp + fp) if tp + fp else math.nan
        recall = tp / (tp + fn) if tp + fn else math.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall and not math.isnan(precision + recall) else math.nan
        rows.append(
            {
                "category_id": category_id,
                "drug_name": cat_to_name.get(category_id, str(category_id)),
                "support": support,
                "predictions": pred_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows).sort_values(["support", "category_id"], ascending=[False, True])


def build_confusion(matches: pd.DataFrame, cat_to_name: dict[int, str]) -> pd.DataFrame:
    gt_labels: list[str] = []
    pred_labels: list[str] = []
    for row in matches.itertuples(index=False):
        gt_label = "__false_positive__" if pd.isna(row.gt_category_id) else f"{int(row.gt_category_id)}:{cat_to_name.get(int(row.gt_category_id), '')}"
        pred_label = "__missed__" if pd.isna(row.pred_category_id) else f"{int(row.pred_category_id)}:{cat_to_name.get(int(row.pred_category_id), '')}"
        gt_labels.append(gt_label)
        pred_labels.append(pred_label)
    confusion = pd.crosstab(pd.Series(gt_labels, name="gt"), pd.Series(pred_labels, name="pred"))
    return confusion


def plot_confusion(confusion: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = confusion.to_numpy(dtype=float)
    if matrix.size == 0:
        return
    height = max(8, min(28, 0.32 * len(confusion.index) + 4))
    width = max(8, min(28, 0.32 * len(confusion.columns) + 4))
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Pseudo GT")
    ax.set_xticks(np.arange(len(confusion.columns)))
    ax.set_yticks(np.arange(len(confusion.index)))
    ax.set_xticklabels(confusion.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(confusion.index, fontsize=6)
    ax.set_title("Pseudo-GT confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def coco_eval(pred: pd.DataFrame, gt: pd.DataFrame, cat_to_name: dict[int, str], out_dir: Path) -> pd.DataFrame:
    from contextlib import redirect_stdout
    from io import StringIO

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images = []
    image_meta = gt.groupby("image_id").agg(width=("width", "max"), height=("height", "max")) if {"width", "height"}.issubset(gt.columns) else None
    for image_id in sorted(set(gt["image_id"]) | set(pred["image_id"])):
        width = 0
        height = 0
        if image_meta is not None and image_id in image_meta.index:
            width = int(image_meta.loc[image_id, "width"])
            height = int(image_meta.loc[image_id, "height"])
        images.append({"id": int(image_id), "file_name": f"{int(image_id)}.png", "width": width, "height": height})

    categories = [
        {"id": int(category_id), "name": cat_to_name.get(int(category_id), str(category_id)), "supercategory": "pill"}
        for category_id in sorted(set(gt["category_id"]) | set(pred["category_id"]))
    ]
    annotations = []
    for ann_id, row in enumerate(gt.itertuples(index=False), 1):
        area = float(row.bbox_w) * float(row.bbox_h)
        annotations.append(
            {
                "id": ann_id,
                "image_id": int(row.image_id),
                "category_id": int(row.category_id),
                "bbox": [float(row.bbox_x), float(row.bbox_y), float(row.bbox_w), float(row.bbox_h)],
                "area": area,
                "iscrowd": 0,
            }
        )

    detections = [
        {
            "image_id": int(row.image_id),
            "category_id": int(row.category_id),
            "bbox": [float(row.bbox_x), float(row.bbox_y), float(row.bbox_w), float(row.bbox_h)],
            "score": float(row.score),
        }
        for row in pred.itertuples(index=False)
    ]
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"description": "Frozen pseudo-GT for local test scoring"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_gt.createIndex()
    (out_dir / "pseudo_gt_coco_snapshot.json").write_text(
        json.dumps(coco_gt.dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "submission_coco_detections.json").write_text(
        json.dumps(detections, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not detections or not annotations:
        return pd.DataFrame(
            [{"metric": "AP_50_95", "value": math.nan, "note": "empty detections or annotations"}]
        )

    buffer = StringIO()
    with redirect_stdout(buffer):
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    (out_dir / "coco_eval_stdout.txt").write_text(buffer.getvalue(), encoding="utf-8")

    names = [
        "AP_50_95",
        "AP_50",
        "AP_75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_maxDets_1",
        "AR_maxDets_10",
        "AR_maxDets_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return pd.DataFrame({"metric": names, "value": evaluator.stats.tolist()})


def write_scorecard(
    path: Path,
    inputs: EvalInputs,
    summary: dict[str, object],
    coco_summary: pd.DataFrame,
    per_class: pd.DataFrame,
) -> None:
    metric_map = dict(zip(coco_summary["metric"], coco_summary["value"], strict=False))
    worst = per_class.sort_values(["f1", "support"], ascending=[True, False]).head(12)
    lines = [
        "# Pseudo-GT Test Scorecard",
        "",
        "> This is local pseudo-GT scoring, not hidden leaderboard scoring. Do not train on this pseudo-GT unless you intentionally accept test-set leakage.",
        "",
        "## Inputs",
        f"- submission: `{inputs.submission}`",
        f"- pseudo_gt: `{inputs.pseudo_gt}`",
        f"- unknown_ignore_boxes: `{inputs.unknown_ignore_boxes}`",
        f"- image_filter: `{inputs.image_filter}`",
        f"- iou_threshold: `{inputs.iou_threshold}`",
        f"- ignore_iou_threshold: `{inputs.ignore_iou_threshold}`",
        "",
        "## Summary",
        f"- images: {summary['images']}",
        f"- pseudo_gt_boxes: {summary['gt_boxes']}",
        f"- prediction_boxes_raw: {summary['prediction_boxes_raw']}",
        f"- prediction_boxes_scored: {summary['prediction_boxes_scored']}",
        f"- ignored_unknown_overlap_predictions: {summary['ignored_unknown_overlap_predictions']}",
        f"- AP_50_95: {metric_map.get('AP_50_95', math.nan):.6f}",
        f"- AP_50: {metric_map.get('AP_50', math.nan):.6f}",
        f"- AP_75: {metric_map.get('AP_75', math.nan):.6f}",
        f"- TP/FP/FN/class_error @ IoU {inputs.iou_threshold}: "
        f"{summary['tp']}/{summary['fp']}/{summary['fn']}/{summary['class_error']}",
        "",
        "## Lowest F1 Classes",
        "",
        "| category_id | drug_name | support | tp | fp | fn | precision | recall | f1 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in worst.itertuples(index=False):
        lines.append(
            f"| {row.category_id} | {row.drug_name} | {row.support} | {row.tp} | {row.fp} | {row.fn} | "
            f"{row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    inputs = parse_args()
    inputs.out_dir.mkdir(parents=True, exist_ok=True)

    pred, gt, ignore, cat_to_name = load_inputs(inputs)
    if gt.empty:
        raise ValueError("No pseudo-GT rows remain after filtering.")

    if inputs.disable_unknown_ignore:
        pred_scored = pred.copy()
        ignored_pred = pred.iloc[0:0].copy()
    else:
        pred_scored, ignored_pred = split_predictions_overlapping_ignore(
            pred,
            ignore,
            inputs.ignore_iou_threshold,
        )

    if inputs.freeze_inputs:
        shutil.copy2(inputs.submission, inputs.out_dir / "submission_input_snapshot.csv")
        shutil.copy2(inputs.pseudo_gt, inputs.out_dir / "pseudo_gt_input_snapshot.csv")

    pred.to_csv(inputs.out_dir / "submission_normalized.csv", index=False)
    pred_scored.to_csv(inputs.out_dir / "submission_scored_after_unknown_ignore.csv", index=False)
    ignored_pred.to_csv(inputs.out_dir / "ignored_predictions_unknown_overlap.csv", index=False)
    gt.to_csv(inputs.out_dir / "pseudo_gt_normalized.csv", index=False)
    ignore.to_csv(inputs.out_dir / "unknown_ignore_boxes_used.csv", index=False)

    coco_summary = coco_eval(pred_scored, gt, cat_to_name, inputs.out_dir)
    coco_summary.to_csv(inputs.out_dir / "coco_map_summary.csv", index=False)

    matches = greedy_match(pred_scored, gt, inputs.iou_threshold)
    matches.to_csv(inputs.out_dir / f"matches_iou{int(inputs.iou_threshold * 100):02d}.csv", index=False)

    per_class = build_per_class(matches, gt, pred_scored, cat_to_name)
    per_class.to_csv(inputs.out_dir / f"per_class_iou{int(inputs.iou_threshold * 100):02d}.csv", index=False)

    confusion = build_confusion(matches, cat_to_name)
    confusion.to_csv(inputs.out_dir / f"confusion_matrix_iou{int(inputs.iou_threshold * 100):02d}.csv")
    try:
        plot_confusion(confusion, inputs.out_dir / f"confusion_matrix_iou{int(inputs.iou_threshold * 100):02d}.png")
    except Exception as exc:  # Plot is convenience only; keep scoring usable.
        (inputs.out_dir / "confusion_plot_error.txt").write_text(repr(exc), encoding="utf-8")

    summary = {
        "warning": "Pseudo-GT scoring only; this is not hidden leaderboard truth.",
        "submission": str(inputs.submission),
        "pseudo_gt": str(inputs.pseudo_gt),
        "unknown_ignore_boxes": str(inputs.unknown_ignore_boxes),
        "submission_sha256": sha256_file(inputs.submission),
        "pseudo_gt_sha256": sha256_file(inputs.pseudo_gt),
        "out_dir": str(inputs.out_dir),
        "image_filter": inputs.image_filter,
        "iou_threshold": inputs.iou_threshold,
        "ignore_iou_threshold": inputs.ignore_iou_threshold,
        "disable_unknown_ignore": inputs.disable_unknown_ignore,
        "images": int(len(set(gt["image_id"]) | set(pred["image_id"]))),
        "gt_boxes": int(len(gt)),
        "prediction_boxes_raw": int(len(pred)),
        "prediction_boxes_scored": int(len(pred_scored)),
        "ignored_unknown_overlap_predictions": int(len(ignored_pred)),
        "unknown_ignore_boxes_used": int(len(ignore)),
        "classes_in_gt": int(gt["category_id"].nunique()),
        "classes_in_predictions": int(pred_scored["category_id"].nunique()),
        "tp": int((matches["match_type"] == "tp").sum()),
        "fp": int((matches["match_type"] == "fp").sum()),
        "fn": int((matches["match_type"] == "fn").sum()),
        "class_error": int((matches["match_type"] == "class_error").sum()),
        "files": {
            "coco_map_summary": str(inputs.out_dir / "coco_map_summary.csv"),
            "matches": str(inputs.out_dir / f"matches_iou{int(inputs.iou_threshold * 100):02d}.csv"),
            "per_class": str(inputs.out_dir / f"per_class_iou{int(inputs.iou_threshold * 100):02d}.csv"),
            "confusion_csv": str(inputs.out_dir / f"confusion_matrix_iou{int(inputs.iou_threshold * 100):02d}.csv"),
            "confusion_png": str(inputs.out_dir / f"confusion_matrix_iou{int(inputs.iou_threshold * 100):02d}.png"),
            "scorecard": str(inputs.out_dir / "scorecard.md"),
        },
    }
    (inputs.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_scorecard(inputs.out_dir / "scorecard.md", inputs, summary, coco_summary, per_class)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
