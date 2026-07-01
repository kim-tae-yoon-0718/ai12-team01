#!/usr/bin/env python3
"""Apply image-based NN-assisted corrections to the test pseudo-GT review CSV."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/test_pseudo_ground_truth_review.csv"
DEFAULT_NN = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/test_crop_nn_suggestions.csv"
DEFAULT_TRAIN_EMB = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/train_crop_resnet18_embeddings.npy"
DEFAULT_TRAIN_META = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/train_crop_resnet18_meta.csv"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "working/reports/pill_class_number_map.csv"
DEFAULT_TEST_IMAGES = PROJECT_ROOT / "sprint_ai_project1_data/test_images"
DEFAULT_OUT_DIR = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--nn-suggestions", type=Path, default=DEFAULT_NN)
    parser.add_argument("--train-embeddings", type=Path, default=DEFAULT_TRAIN_EMB)
    parser.add_argument("--train-meta", type=Path, default=DEFAULT_TRAIN_META)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--single-sim-thr", type=float, default=0.97)
    parser.add_argument("--duplicate-sim-thr", type=float, default=0.94)
    parser.add_argument("--missing-sim-thr", type=float, default=0.965)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def clean_n(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("N"):
        return f"N{int(text[1:]):02d}"
    return f"N{int(float(text)):02d}"


def strong_nn_row(row: pd.Series, sim_thr: float) -> bool:
    predicted = clean_n(row["predicted_n_number"])
    if bool(row.get("nn_matches_pred", False)):
        return False
    if float(row["nn_sim"]) < sim_thr:
        return False
    top2 = clean_n(row.get("top2_n_number"))
    top3 = clean_n(row.get("top3_n_number"))
    pred_in_top3 = predicted in {top2, top3}
    top2_sim = row.get("top2_sim")
    gap = float(row["nn_sim"]) - (float(top2_sim) if not pd.isna(top2_sim) else 0.0)
    return (not pred_in_top3) or gap >= 0.03


def clusters_for_image(rows: pd.DataFrame, threshold: float = 0.85) -> list[list[int]]:
    row_indices = list(rows.index)
    parent = {idx: idx for idx in row_indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pos, idx_a in enumerate(row_indices):
        a = rows.loc[idx_a, ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].astype(float).tolist()
        for idx_b in row_indices[pos + 1 :]:
            b = rows.loc[idx_b, ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].astype(float).tolist()
            if iou_xywh(tuple(a), tuple(b)) >= threshold:
                union(idx_a, idx_b)

    groups: dict[int, list[int]] = {}
    for idx in row_indices:
        groups.setdefault(find(idx), []).append(idx)
    return list(groups.values())


def crop_xywh(image: Image.Image, bbox: tuple[float, float, float, float], pad: float = 0.25) -> Image.Image:
    x, y, w, h = [float(v) for v in bbox]
    width, height = image.size
    x1, x2 = min(x, x + w), max(x, x + w)
    y1, y2 = min(y, y + h), max(y, y + h)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    extra = max(bw, bh) * pad
    left = max(0, int(x1 - extra))
    top = max(0, int(y1 - extra))
    right = min(width, int(x2 + extra))
    bottom = min(height, int(y2 + extra))
    return image.crop((left, top, max(left + 1, right), max(top + 1, bottom))).convert("RGB")


class CropClassifier:
    def __init__(self, train_embeddings: Path, train_meta: Path, class_map: Path) -> None:
        self.train_x = np.load(train_embeddings)
        self.train_meta = pd.read_csv(train_meta)
        ref = pd.read_csv(class_map)
        self.cat_to_n = {int(r.category_id): f"N{int(r.class_no):02d}" for r in ref.itertuples(index=False)}
        self.cat_to_name = {int(r.category_id): str(r.name) for r in ref.itertuples(index=False)}
        weights = ResNet18_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval()

    def classify(self, crop: Image.Image, top_k: int = 3) -> list[dict[str, object]]:
        with torch.inference_mode():
            emb = self.model(self.preprocess(crop).unsqueeze(0)).squeeze(0).numpy()
        emb = emb / np.linalg.norm(emb)
        sims = self.train_x @ emb
        order = np.argsort(-sims)[:30]
        out: list[dict[str, object]] = []
        seen: set[int] = set()
        for idx in order:
            cat = int(self.train_meta.iloc[int(idx)].category_id)
            if cat in seen:
                continue
            seen.add(cat)
            out.append(
                {
                    "category_id": cat,
                    "n_number": self.cat_to_n.get(cat),
                    "name": self.cat_to_name.get(cat),
                    "sim": float(sims[int(idx)]),
                }
            )
            if len(out) >= top_k:
                break
        return out


def detect_components(image_path: Path) -> list[dict[str, object]]:
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate(
        [
            lab[:80, :, :].reshape(-1, 3),
            lab[-80:, :, :].reshape(-1, 3),
            lab[:, :80, :].reshape(-1, 3),
            lab[:, -80:, :].reshape(-1, 3),
        ]
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    mask = (dist > 18).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    count, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    comps: list[dict[str, object]] = []
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 15000 or w < 70 or h < 70:
            continue
        if area > 0.35 * width * height:
            continue
        aspect = max(w / h, h / w)
        touches = int(x <= 2) + int(y <= 2) + int(x + w >= width - 2) + int(y + h >= height - 2)
        if aspect > 5.0:
            continue
        if touches >= 2 and (area < 25000 or aspect > 3.5):
            continue
        comps.append({"bbox": (float(x), float(y), float(w), float(h)), "area": area, "aspect": aspect, "touches": touches})
    return comps


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    review = pd.read_csv(args.review)
    nn = pd.read_csv(args.nn_suggestions)
    nn_by_ann = nn.set_index("annotation_id")
    class_map = pd.read_csv(args.class_map)
    n_to_cat = {f"N{int(r.class_no):02d}": int(r.category_id) for r in class_map.itertuples(index=False)}
    n_to_name = {f"N{int(r.class_no):02d}": str(r.name) for r in class_map.itertuples(index=False)}

    if not args.dry_run:
        backup = args.review.with_suffix(args.review.suffix + ".bak_before_visual_nn")
        if not backup.exists():
            shutil.copy2(args.review, backup)

    log: list[dict[str, object]] = []

    seed_mask = review["source"].fillna("").eq("submission_seed")
    seed = review[seed_mask].copy()

    for image_id, rows in seed.groupby("image_id", sort=True):
        for cluster in clusters_for_image(rows):
            if len(cluster) == 1:
                idx = cluster[0]
                ann = int(review.loc[idx, "annotation_id"])
                if ann not in nn_by_ann.index:
                    continue
                nn_row = nn_by_ann.loc[ann]
                if strong_nn_row(nn_row, args.single_sim_thr):
                    target_n = clean_n(nn_row["nn_n_number"])
                    current_n = clean_n(review.loc[idx, "predicted_n_number"])
                    if target_n and target_n != current_n:
                        review.loc[idx, "correct_n_number"] = target_n
                        review.loc[idx, "review_status"] = "reviewed"
                        review.loc[idx, "review_note"] = (
                            f"visual NN class fix: {current_n} -> {target_n} "
                            f"(sim={float(nn_row['nn_sim']):.3f})"
                        )
                        log.append(
                            {
                                "action": "class_fix",
                                "image_id": int(image_id),
                                "annotation_id": ann,
                                "from_n": current_n,
                                "to_n": target_n,
                                "nn_sim": float(nn_row["nn_sim"]),
                            }
                        )
                continue

            candidates = []
            for idx in cluster:
                ann = int(review.loc[idx, "annotation_id"])
                current_n = clean_n(review.loc[idx, "predicted_n_number"])
                target_n = current_n
                nn_sim = None
                strong = False
                if ann in nn_by_ann.index:
                    nn_row = nn_by_ann.loc[ann]
                    nn_sim = float(nn_row["nn_sim"])
                    if strong_nn_row(nn_row, args.duplicate_sim_thr):
                        target_n = clean_n(nn_row["nn_n_number"])
                        strong = True
                candidates.append((idx, ann, current_n, target_n, nn_sim, strong, float(review.loc[idx, "score"])))

            strong_targets = [c for c in candidates if c[5] and c[3]]
            if strong_targets:
                target_n = Counter([c[3] for c in strong_targets]).most_common(1)[0][0]
            else:
                target_n = max(candidates, key=lambda item: item[6])[2]

            target_rows = [c for c in candidates if c[2] == target_n]
            keep_candidate = max(target_rows or candidates, key=lambda item: item[6])
            keep_idx, keep_ann, keep_current_n, _keep_target_n, keep_sim, _strong, _score = keep_candidate
            if keep_current_n != target_n:
                review.loc[keep_idx, "correct_n_number"] = target_n
            review.loc[keep_idx, "keep"] = 1
            review.loc[keep_idx, "review_status"] = "reviewed"
            review.loc[keep_idx, "review_note"] = (
                f"visual NN duplicate cluster keep as {target_n}"
                + (f" (sim={keep_sim:.3f})" if keep_sim is not None else "")
            )
            log.append(
                {
                    "action": "duplicate_keep",
                    "image_id": int(image_id),
                    "annotation_id": keep_ann,
                    "from_n": keep_current_n,
                    "to_n": target_n,
                    "nn_sim": keep_sim,
                    "cluster_size": len(cluster),
                }
            )

            for idx, ann, current_n, _target, nn_sim, _strong, _score in candidates:
                if idx == keep_idx:
                    continue
                review.loc[idx, "keep"] = 0
                review.loc[idx, "review_status"] = "reviewed"
                review.loc[idx, "review_note"] = (
                    f"visual NN duplicate cluster drop; kept annotation {keep_ann} as {target_n}"
                )
                log.append(
                    {
                        "action": "duplicate_drop",
                        "image_id": int(image_id),
                        "annotation_id": ann,
                        "from_n": current_n,
                        "to_n": None,
                        "nn_sim": nn_sim,
                        "cluster_size": len(cluster),
                    }
                )

    classifier = CropClassifier(args.train_embeddings, args.train_meta, args.class_map)
    existing_manual = review["source"].fillna("").str.startswith("manual_added")
    next_ann = int(pd.to_numeric(review["annotation_id"], errors="coerce").max()) + 1
    missing_candidates: list[dict[str, object]] = []
    rows_to_add: list[dict[str, object]] = []

    for image_path in sorted(args.test_images.glob("*.png"), key=lambda p: int(p.stem)):
        image_id = int(image_path.stem)
        kept = review[
            review["image_id"].astype(int).eq(image_id)
            & review["keep"].astype(int).eq(1)
            & (
                review["source"].fillna("").eq("submission_seed")
                | review["source"].fillna("").str.startswith("manual_added")
            )
        ].copy()
        kept_boxes = [
            tuple(row)
            for row in kept[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].astype(float).itertuples(index=False, name=None)
        ]
        image = Image.open(image_path).convert("RGB")
        for comp in detect_components(image_path):
            bbox = tuple(float(v) for v in comp["bbox"])
            best_iou = max([iou_xywh(bbox, b) for b in kept_boxes] or [0.0])
            if best_iou >= 0.15:
                continue
            crop = crop_xywh(image, bbox, pad=0.25)
            top = classifier.classify(crop, top_k=3)
            if not top:
                continue
            best = top[0]
            candidate = {
                "image_id": image_id,
                "bbox_x": round(bbox[0], 2),
                "bbox_y": round(bbox[1], 2),
                "bbox_w": round(bbox[2], 2),
                "bbox_h": round(bbox[3], 2),
                "best_iou_existing": round(best_iou, 4),
                "nn_n_number": best["n_number"],
                "nn_category_id": best["category_id"],
                "nn_name": best["name"],
                "nn_sim": best["sim"],
                "top2_n_number": top[1]["n_number"] if len(top) > 1 else None,
                "top2_sim": top[1]["sim"] if len(top) > 1 else None,
            }
            missing_candidates.append(candidate)

            duplicate_manual = False
            for _idx, row in review[existing_manual & review["image_id"].astype(int).eq(image_id)].iterrows():
                existing_box = tuple(float(row[c]) for c in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"])
                if iou_xywh(bbox, existing_box) >= 0.3:
                    duplicate_manual = True
                    break
            if duplicate_manual or float(best["sim"]) < args.missing_sim_thr:
                continue

            n_number = str(best["n_number"])
            rows_to_add.append(
                {
                    "keep": 1,
                    "review_status": "reviewed",
                    "review_note": f"visual NN auto add unmatched component as {n_number} (sim={float(best['sim']):.3f})",
                    "source": "manual_added_visual_nn",
                    "annotation_id": next_ann,
                    "image_id": image_id,
                    "candidate_rank": int(kept["candidate_rank"].max()) + 1 if not kept.empty else 1,
                    "predicted_n_number": pd.NA,
                    "correct_n_number": n_number,
                    "category_id": int(best["category_id"]),
                    "predicted_drug_name": str(best["name"]),
                    "name": str(best["name"]),
                    "name_en": pd.NA,
                    "bbox_x": round(bbox[0], 2),
                    "bbox_y": round(bbox[1], 2),
                    "bbox_w": round(bbox[2], 2),
                    "bbox_h": round(bbox[3], 2),
                    "score": round(float(best["sim"]), 6),
                    "company": pd.NA,
                    "shape": pd.NA,
                    "color": pd.NA,
                    "print_front": pd.NA,
                    "print_back": pd.NA,
                    "sample_path": pd.NA,
                    "sample_bbox": pd.NA,
                }
            )
            log.append(
                {
                    "action": "missing_add",
                    "image_id": image_id,
                    "annotation_id": next_ann,
                    "from_n": None,
                    "to_n": n_number,
                    "nn_sim": float(best["sim"]),
                    "cluster_size": None,
                }
            )
            next_ann += 1

    if rows_to_add:
        review = pd.concat([review, pd.DataFrame(rows_to_add)], ignore_index=True)

    log_df = pd.DataFrame(log)
    missing_df = pd.DataFrame(missing_candidates)
    summary = {
        "input_review": str(args.review),
        "dry_run": bool(args.dry_run),
        "class_fix_rows": int((log_df["action"].eq("class_fix")).sum()) if not log_df.empty else 0,
        "duplicate_keep_rows": int((log_df["action"].eq("duplicate_keep")).sum()) if not log_df.empty else 0,
        "duplicate_drop_rows": int((log_df["action"].eq("duplicate_drop")).sum()) if not log_df.empty else 0,
        "missing_candidates": int(len(missing_df)),
        "missing_added": int(len(rows_to_add)),
        "review_rows_after": int(len(review)),
        "keep_rows_after": int(review["keep"].astype(int).sum()),
    }

    log_path = args.out_dir / "visual_nn_correction_log.csv"
    missing_path = args.out_dir / "visual_nn_missing_component_candidates.csv"
    summary_path = args.out_dir / "visual_nn_correction_summary.json"
    log_df.to_csv(log_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.dry_run:
        review.to_csv(args.review, index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("log:", log_path)
    print("missing candidates:", missing_path)


if __name__ == "__main__":
    main()
