#!/usr/bin/env python3
"""Build pseudo ground-truth CSV/COCO files from the review CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/test_pseudo_ground_truth_review.csv"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "working/reports/pill_class_number_map.csv"
DEFAULT_TEST_IMAGES = PROJECT_ROOT / "sprint_ai_project1_data/test_images"
DEFAULT_OUT_DIR = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def clean_n(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("N"):
        return f"N{int(text[1:]):02d}"
    return f"N{int(float(text)):02d}"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    review = pd.read_csv(args.review)
    class_map = pd.read_csv(args.class_map)
    n_to_meta = {
        f"N{int(row.class_no):02d}": {
            "category_id": int(row.category_id),
            "drug_name": str(row.name),
            "name_en": "" if pd.isna(row.name_en) else str(row.name_en),
            "company": "" if pd.isna(row.company) else str(row.company),
            "shape": "" if pd.isna(row.shape) else str(row.shape),
            "color": "" if pd.isna(row.color) else str(row.color),
            "print_front": "" if pd.isna(row.print_front) else str(row.print_front),
            "print_back": "" if pd.isna(row.print_back) else str(row.print_back),
        }
        for row in class_map.itertuples(index=False)
    }

    kept = review[review["keep"].astype(int).eq(1)].copy()
    kept["predicted_n_number"] = kept["predicted_n_number"].map(clean_n)
    kept["correct_n_number"] = kept["correct_n_number"].map(clean_n)
    kept["resolved_n_number"] = [
        correct if pd.notna(correct) and str(correct).strip() else predicted
        for predicted, correct in zip(kept["predicted_n_number"], kept["correct_n_number"], strict=True)
    ]
    missing_n = sorted({value for value in kept["resolved_n_number"] if value not in n_to_meta})
    if missing_n:
        raise ValueError(f"Unknown resolved_n_number values: {missing_n}")

    rows = []
    for new_id, row in enumerate(kept.sort_values(["image_id", "annotation_id"]).itertuples(index=False), 1):
        meta = n_to_meta[row.resolved_n_number]
        rows.append(
            {
                "annotation_id": new_id,
                "source_annotation_id": int(row.annotation_id),
                "image_id": int(row.image_id),
                "predicted_n_number": row.predicted_n_number,
                "correct_n_number": row.correct_n_number,
                "resolved_n_number": row.resolved_n_number,
                "predicted_category_id": int(row.category_id) if not pd.isna(row.category_id) else pd.NA,
                "category_id": int(meta["category_id"]),
                "drug_name": meta["drug_name"],
                "bbox_x": round(float(row.bbox_x), 2),
                "bbox_y": round(float(row.bbox_y), 2),
                "bbox_w": round(float(row.bbox_w), 2),
                "bbox_h": round(float(row.bbox_h), 2),
                "score": round(float(row.score), 6) if not pd.isna(row.score) else pd.NA,
                "source": row.source,
                "review_status": row.review_status,
                "review_note": row.review_note,
            }
        )
    pseudo = pd.DataFrame(rows)
    pseudo_csv = args.out_dir / "test_pseudo_ground_truth.csv"
    pseudo.to_csv(pseudo_csv, index=False)

    image_records = []
    for path in sorted(args.test_images.glob("*.png"), key=lambda p: int(p.stem)):
        with Image.open(path) as image:
            width, height = image.size
        image_records.append({"id": int(path.stem), "file_name": path.name, "width": width, "height": height})

    categories = []
    for row in class_map.sort_values("category_id").itertuples(index=False):
        n_number = f"N{int(row.class_no):02d}"
        categories.append(
            {
                "id": int(row.category_id),
                "name": str(row.name),
                "supercategory": "pill",
                "n_number": n_number,
                **n_to_meta[n_number],
            }
        )

    annotations = []
    for row in pseudo.itertuples(index=False):
        area = float(row.bbox_w) * float(row.bbox_h)
        annotations.append(
            {
                "id": int(row.annotation_id),
                "image_id": int(row.image_id),
                "category_id": int(row.category_id),
                "bbox": [float(row.bbox_x), float(row.bbox_y), float(row.bbox_w), float(row.bbox_h)],
                "area": round(area, 2),
                "iscrowd": 0,
                "n_number": row.resolved_n_number,
                "drug_name": row.drug_name,
                "source": row.source,
            }
        )

    coco = {
        "info": {
            "description": "Pseudo/manual ground truth built from test_pseudo_ground_truth_review.csv",
            "version": "visual_nn_review",
        },
        "images": image_records,
        "annotations": annotations,
        "categories": categories,
    }
    coco_json = args.out_dir / "test_pseudo_ground_truth_coco.json"
    coco_json.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "pseudo_csv": str(pseudo_csv),
        "coco_json": str(coco_json),
        "images": len(image_records),
        "annotations": int(len(pseudo)),
        "review_rows": int(len(review)),
        "kept_review_rows": int(len(kept)),
        "manual_added_rows": int(kept["source"].fillna("").str.startswith("manual_added").sum()),
        "visual_nn_added_rows": int(kept["source"].fillna("").eq("manual_added_visual_nn").sum()),
        "classes_used": int(pseudo["category_id"].nunique()),
        "classes_total": int(class_map["category_id"].nunique()),
    }
    (args.out_dir / "test_pseudo_ground_truth_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
