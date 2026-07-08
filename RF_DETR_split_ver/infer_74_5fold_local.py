#!/usr/bin/env python3
"""Run local 5-fold RF-DETR inference and write Kaggle submission CSVs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm


PROJECT_DIR = Path("/Users/pio/Documents/AIENGINEERCOURSE/detectionproject")
REPO_DIR = Path("/Users/pio/Documents/AIENGINEERCOURSE/ai12-team01-rfdetr/RF_DETR_split_ver")
OUTPUT_ROOT = PROJECT_DIR / "working/rfdetr_outputs/mps_large_v18_5fold"
DATASET_DIR = PROJECT_DIR / "working/rfdetr_dataset_74_hidden45_canvas_balanced_5fold_cls0_mps"
TEST_IMG_DIR = PROJECT_DIR / "sprint_ai_project1_data/test_images"
SUBMISSION_DIR = PROJECT_DIR / "working/submissions"

MODEL_VARIANT = "large"
MODEL_TAG_BASE = "large_74_hidden45_canvas_balanced_5fold_cls0_local_mps_rfdetr_v18plus_aug_scale150_rot90_v1"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ClassInfo:
    internal_label: int
    category_id: int
    n_number: str
    name: str
    is_placeholder: bool


def numeric_image_id_from_path(path: Path) -> int:
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits)
    raise ValueError(f"Cannot parse numeric image_id from filename: {path.name}")


def list_image_paths(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=numeric_image_id_from_path,
    )


def load_category_mapping(path: Path) -> tuple[dict[int, ClassInfo], dict[int, ClassInfo]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    by_internal: dict[int, ClassInfo] = {}
    by_category: dict[int, ClassInfo] = {}
    for row in df.itertuples(index=False):
        info = ClassInfo(
            internal_label=int(row.rfdetr_internal_label),
            category_id=int(row.category_id),
            n_number=str(row.n_number),
            name=str(row.name),
            is_placeholder=bool(row.is_placeholder),
        )
        by_internal[info.internal_label] = info
        by_category[info.category_id] = info
    return by_internal, by_category


def map_model_label(
    raw_label: int,
    by_internal: dict[int, ClassInfo],
    by_category: dict[int, ClassInfo],
) -> ClassInfo | None:
    label = int(raw_label)
    candidates = []
    if label in by_internal:
        candidates.append(by_internal[label])
    if label in by_category:
        candidates.append(by_category[label])
    # Legacy guard: older non-placeholder datasets sometimes emitted zero-based labels.
    if (label + 1) in by_internal:
        candidates.append(by_internal[label + 1])
    if (label - 1) in by_internal:
        candidates.append(by_internal[label - 1])

    for info in candidates:
        if not info.is_placeholder and info.category_id != 0:
            return info
    return None


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_model(variant: str, checkpoint_path: Path):
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    from model import get_rfdetr_model

    return get_rfdetr_model(variant, checkpoint_path=str(checkpoint_path))


def predict_image(
    model: Any,
    image_path: Path,
    fold_idx: int,
    by_internal: dict[int, ClassInfo],
    by_category: dict[int, ClassInfo],
    threshold: float,
    max_det: int,
) -> list[dict[str, Any]]:
    with Image.open(image_path) as im:
        image = im.convert("RGB")
        img_w, img_h = image.size
        detections = model.predict(image, threshold=threshold)

    boxes = np.asarray(getattr(detections, "xyxy", []), dtype=float)
    if boxes.size == 0:
        return []
    boxes = boxes.reshape(-1, 4)
    scores = np.asarray(getattr(detections, "confidence", np.ones(len(boxes))), dtype=float).reshape(-1)
    labels = np.asarray(getattr(detections, "class_id", np.zeros(len(boxes))), dtype=int).reshape(-1)

    order = np.argsort(-scores)
    if max_det > 0:
        order = order[:max_det]

    rows = []
    image_id = numeric_image_id_from_path(image_path)
    for idx in order:
        info = map_model_label(int(labels[idx]), by_internal, by_category)
        if info is None:
            continue
        x1, y1, x2, y2 = boxes[idx].tolist()
        x1 = max(0.0, min(float(x1), float(img_w - 1)))
        y1 = max(0.0, min(float(y1), float(img_h - 1)))
        x2 = max(0.0, min(float(x2), float(img_w)))
        y2 = max(0.0, min(float(y2), float(img_h)))
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "fold": int(fold_idx),
                "image_id": int(image_id),
                "image_file": image_path.name,
                "image_path": str(image_path),
                "raw_class_id": int(labels[idx]),
                "internal_label": int(info.internal_label),
                "category_id": int(info.category_id),
                "n_number": info.n_number,
                "class_name": info.name,
                "bbox_x": round(x1, 2),
                "bbox_y": round(y1, 2),
                "bbox_w": round(x2 - x1, 2),
                "bbox_h": round(y2 - y1, 2),
                "bbox_x2": round(x2, 2),
                "bbox_y2": round(y2, 2),
                "score": round(float(scores[idx]), 6),
            }
        )
    return rows


def run_fold_inference(
    fold_idx: int,
    checkpoint_path: Path,
    output_dir: Path,
    test_paths: list[Path],
    by_internal: dict[int, ClassInfo],
    by_category: dict[int, ClassInfo],
    threshold: float,
    max_det: int,
    force: bool,
) -> pd.DataFrame:
    raw_path = output_dir / f"raw_predictions_fold{fold_idx}_score{int(threshold * 1000):03d}.csv"
    if raw_path.exists() and not force:
        print(f"[fold {fold_idx}] using cached raw predictions: {raw_path}", flush=True)
        return pd.read_csv(raw_path)

    print(f"[fold {fold_idx}] loading checkpoint: {checkpoint_path}", flush=True)
    started = time.time()
    model = load_model(MODEL_VARIANT, checkpoint_path)
    print(f"[fold {fold_idx}] model loaded in {time.time() - started:.1f}s", flush=True)

    rows: list[dict[str, Any]] = []
    infer_started = time.time()
    for i, image_path in enumerate(tqdm(test_paths, desc=f"fold{fold_idx} inference"), start=1):
        rows.extend(
            predict_image(
                model=model,
                image_path=image_path,
                fold_idx=fold_idx,
                by_internal=by_internal,
                by_category=by_category,
                threshold=threshold,
                max_det=max_det,
            )
        )
        if i == 1 or i % 50 == 0 or i == len(test_paths):
            print(
                f"[fold {fold_idx}] progress {i}/{len(test_paths)} images, rows={len(rows)}, elapsed={time.time() - infer_started:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"[fold {fold_idx}] saved raw predictions: {raw_path}", flush=True)

    del model
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass
    return df


def nms_rows(rows: pd.DataFrame, iou_thr: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows.sort_values("score", ascending=False).to_dict("records"):
        box = np.array([row["bbox_x"], row["bbox_y"], row["bbox_x2"], row["bbox_y2"]], dtype=float)
        if any(iou_xyxy(box, np.array([k["bbox_x"], k["bbox_y"], k["bbox_x2"], k["bbox_y2"]], dtype=float)) >= iou_thr for k in kept):
            continue
        out = dict(row)
        out["cluster_size"] = 1
        out["fold_support"] = 1
        out["postprocess"] = "nms"
        kept.append(out)
    return kept


def wbf_rows(rows: pd.DataFrame, iou_thr: float, num_folds: int) -> list[dict[str, Any]]:
    clusters: list[list[dict[str, Any]]] = []
    reps: list[np.ndarray] = []
    for row in rows.sort_values("score", ascending=False).to_dict("records"):
        box = np.array([row["bbox_x"], row["bbox_y"], row["bbox_x2"], row["bbox_y2"]], dtype=float)
        best_i = -1
        best_iou = 0.0
        for i, rep in enumerate(reps):
            value = iou_xyxy(box, rep)
            if value > best_iou:
                best_i = i
                best_iou = value
        if best_i >= 0 and best_iou >= iou_thr:
            clusters[best_i].append(row)
            cluster_df = pd.DataFrame(clusters[best_i])
            weights = cluster_df["score"].astype(float).to_numpy()
            coords = cluster_df[["bbox_x", "bbox_y", "bbox_x2", "bbox_y2"]].astype(float).to_numpy()
            reps[best_i] = np.average(coords, axis=0, weights=weights)
        else:
            clusters.append([row])
            reps.append(box)

    outputs: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster_df = pd.DataFrame(cluster)
        cat_scores = cluster_df.groupby("category_id")["score"].sum().sort_values(ascending=False)
        winner_cat = int(cat_scores.index[0])
        winner_rows = cluster_df[cluster_df["category_id"] == winner_cat].copy()
        info_row = winner_rows.sort_values("score", ascending=False).iloc[0].to_dict()

        weights = cluster_df["score"].astype(float).to_numpy()
        coords = cluster_df[["bbox_x", "bbox_y", "bbox_x2", "bbox_y2"]].astype(float).to_numpy()
        x1, y1, x2, y2 = np.average(coords, axis=0, weights=weights).tolist()
        max_score = float(winner_rows["score"].max())
        score_sum = float(cat_scores.iloc[0])
        fold_support = int(winner_rows["fold"].nunique())
        # Keep max score for AP ranking while retaining support details in the detail CSV.
        final_score = max_score

        out = dict(info_row)
        out.update(
            {
                "category_id": winner_cat,
                "bbox_x": round(float(x1), 2),
                "bbox_y": round(float(y1), 2),
                "bbox_w": round(float(x2 - x1), 2),
                "bbox_h": round(float(y2 - y1), 2),
                "bbox_x2": round(float(x2), 2),
                "bbox_y2": round(float(y2), 2),
                "score": round(final_score, 6),
                "cluster_size": int(len(cluster_df)),
                "fold_support": fold_support,
                "score_sum": round(score_sum, 6),
                "score_mean_by_fold": round(score_sum / max(1, num_folds), 6),
                "postprocess": "wbf",
            }
        )
        outputs.append(out)
    return outputs


def build_submission(
    raw: pd.DataFrame,
    method: str,
    score_thr: float,
    iou_thr: float,
    num_folds: int,
) -> pd.DataFrame:
    filtered = raw[raw["score"].astype(float) >= score_thr].copy()
    final_rows: list[dict[str, Any]] = []
    for _, rows in filtered.groupby("image_id", sort=True):
        if method == "nms":
            final_rows.extend(nms_rows(rows, iou_thr))
        elif method == "wbf":
            final_rows.extend(wbf_rows(rows, iou_thr, num_folds))
        else:
            raise ValueError(method)

    detailed = pd.DataFrame(final_rows)
    if detailed.empty:
        raise RuntimeError(f"No predictions after {method} postprocess")
    detailed = detailed.sort_values(["image_id", "score"], ascending=[True, False]).reset_index(drop=True)
    detailed.insert(0, "annotation_id", range(1, len(detailed) + 1))
    return detailed


def render_low_confidence_grid(detailed: pd.DataFrame, out_path: Path, count: int = 20) -> None:
    low = detailed.sort_values("score", ascending=True).head(count).copy()
    if low.empty:
        return

    def font(size: int):
        for candidate in [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        return ImageFont.load_default()

    tile_w, tile_h, banner_h = 360, 480, 76
    cols = 4
    tiles = []
    for _, row in low.iterrows():
        with Image.open(row["image_path"]) as im:
            base = im.convert("RGB")
        src_w, src_h = base.size
        scale = min(tile_w / src_w, (tile_h - banner_h) / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = base.resize((new_w, new_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        xoff = (tile_w - new_w) // 2
        yoff = banner_h + (tile_h - banner_h - new_h) // 2
        tile.paste(resized, (xoff, yoff))
        draw = ImageDraw.Draw(tile)
        x1 = xoff + float(row["bbox_x"]) * scale
        y1 = yoff + float(row["bbox_y"]) * scale
        x2 = xoff + (float(row["bbox_x"]) + float(row["bbox_w"])) * scale
        y2 = yoff + (float(row["bbox_y"]) + float(row["bbox_h"])) * scale
        draw.rectangle([x1, y1, x2, y2], outline=(230, 20, 20), width=3)
        draw.rectangle([0, 0, tile_w, banner_h], fill=(20, 24, 31))
        draw.text((10, 8), f"ann {int(row['annotation_id'])} img {int(row['image_id'])} s={float(row['score']):.3f}", fill="white", font=font(17))
        draw.text((10, 38), f"{row['n_number']} cat {int(row['category_id'])} fs={int(row.get('fold_support', 1))}", fill=(220, 230, 255), font=font(15))
        tiles.append(tile)

    rows_n = int(np.ceil(len(tiles) / cols))
    grid = Image.new("RGB", (cols * tile_w, rows_n * tile_h), "white")
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % cols) * tile_w, (i // cols) * tile_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=95)


def save_submission_files(
    detailed: pd.DataFrame,
    output_dir: Path,
    timestamp: str,
    method: str,
    score_thr: float,
    iou_thr: float,
) -> tuple[Path, Path, Path]:
    suffix = f"5fold_{method}_score{int(score_thr * 100):02d}_iou{int(iou_thr * 100):02d}_{timestamp}"
    detailed_path = output_dir / f"submitted_predictions_{suffix}_with_paths.csv"
    submission_path = output_dir / f"submission_{suffix}.csv"
    low_grid_path = output_dir / f"low_confidence_20_{suffix}_grid.png"

    submission = detailed[["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]].copy()
    submission.to_csv(submission_path, index=False)
    detailed.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    render_low_confidence_grid(detailed, low_grid_path, count=20)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = SUBMISSION_DIR / f"submission_5fold_{method}_latest.csv"
    dated_path = SUBMISSION_DIR / submission_path.name
    submission.to_csv(latest_path, index=False)
    submission.to_csv(dated_path, index=False)
    return submission_path, detailed_path, low_grid_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--raw-score-thr", type=float, default=0.05)
    parser.add_argument("--final-score-thr", type=float, default=0.25)
    parser.add_argument("--iou-thr", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["wbf", "nms"], choices=["wbf", "nms"])
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")

    test_paths = list_image_paths(TEST_IMG_DIR)
    if len(test_paths) != 842:
        raise RuntimeError(f"Expected 842 test images, got {len(test_paths)} from {TEST_IMG_DIR}")

    mapping_path = DATASET_DIR / "category_mapping.csv"
    by_internal, by_category = load_category_mapping(mapping_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"inference_5fold_large_map75_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("test images:", TEST_IMG_DIR, len(test_paths), flush=True)
    print("mapping:", mapping_path, flush=True)
    print("output:", output_dir, flush=True)
    print("folds:", args.folds, flush=True)

    raw_parts = []
    checkpoint_manifest = []
    for fold_idx in args.folds:
        tag = f"{MODEL_TAG_BASE}_p{fold_idx:02d}"
        checkpoint_path = OUTPUT_ROOT / tag / "checkpoint_best_map75.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing best map75 checkpoint for fold {fold_idx}: {checkpoint_path}")
        checkpoint_manifest.append({"fold": fold_idx, "checkpoint": str(checkpoint_path), "size": checkpoint_path.stat().st_size})
        raw_parts.append(
            run_fold_inference(
                fold_idx=fold_idx,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                test_paths=test_paths,
                by_internal=by_internal,
                by_category=by_category,
                threshold=args.raw_score_thr,
                max_det=args.max_det,
                force=args.force,
            )
        )

    raw = pd.concat(raw_parts, ignore_index=True)
    raw_path = output_dir / f"raw_predictions_all_folds_score{int(args.raw_score_thr * 1000):03d}.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print("saved combined raw:", raw_path, raw.shape, flush=True)

    manifest: dict[str, Any] = {
        "created_at": timestamp,
        "folds": args.folds,
        "model_variant": MODEL_VARIANT,
        "raw_score_thr": args.raw_score_thr,
        "final_score_thr": args.final_score_thr,
        "iou_thr": args.iou_thr,
        "max_det": args.max_det,
        "test_images": str(TEST_IMG_DIR),
        "mapping": str(mapping_path),
        "checkpoints": checkpoint_manifest,
        "raw_predictions": str(raw_path),
        "outputs": {},
    }

    for method in args.methods:
        detailed = build_submission(
            raw=raw,
            method=method,
            score_thr=args.final_score_thr,
            iou_thr=args.iou_thr,
            num_folds=len(args.folds),
        )
        submission_path, detailed_path, low_grid_path = save_submission_files(
            detailed=detailed,
            output_dir=output_dir,
            timestamp=timestamp,
            method=method,
            score_thr=args.final_score_thr,
            iou_thr=args.iou_thr,
        )
        counts = detailed.groupby("image_id").size()
        print(
            f"[{method}] rows={len(detailed)} images={detailed['image_id'].nunique()} "
            f"per_image_min={counts.min()} max={counts.max()} mean={counts.mean():.2f} "
            f"score_min={detailed['score'].min():.6f}",
            flush=True,
        )
        print(f"[{method}] submission: {submission_path}", flush=True)
        print(f"[{method}] details: {detailed_path}", flush=True)
        print(f"[{method}] low confidence grid: {low_grid_path}", flush=True)
        manifest["outputs"][method] = {
            "submission": str(submission_path),
            "details": str(detailed_path),
            "low_confidence_grid": str(low_grid_path),
            "rows": int(len(detailed)),
            "images": int(detailed["image_id"].nunique()),
            "score_min": float(detailed["score"].min()),
            "per_image_min": int(counts.min()),
            "per_image_max": int(counts.max()),
            "per_image_mean": float(counts.mean()),
        }

    manifest_path = output_dir / "inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest:", manifest_path, flush=True)


if __name__ == "__main__":
    main()
