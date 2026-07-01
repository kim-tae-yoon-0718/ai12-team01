#!/usr/bin/env python3
"""Export pseudo ground-truth test annotations into practical review/use files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PSEUDO_GT = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/test_pseudo_ground_truth.csv"
DEFAULT_REVIEW = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/test_pseudo_ground_truth_review.csv"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "working/reports/pill_class_number_map.csv"
DEFAULT_TEST_IMAGES = PROJECT_ROOT / "sprint_ai_project1_data/test_images"
DEFAULT_PRED_BOXED = PROJECT_ROOT / "working/reports/test_submission_boxed_classification_numbered"
DEFAULT_PSEUDO_BOXED = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/pseudo_gt_boxed_images"
DEFAULT_REFERENCE_GRID = PROJECT_ROOT / "working/reports/pill_class_reference_grid_numbered.png"
DEFAULT_UNKNOWN_IGNORE = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/manual_unknown_ignore_boxes.csv"
DEFAULT_OUT = PROJECT_ROOT / "working/test_annotations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo-gt", type=Path, default=DEFAULT_PSEUDO_GT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--prediction-boxed", type=Path, default=DEFAULT_PRED_BOXED)
    parser.add_argument("--pseudo-boxed", type=Path, default=DEFAULT_PSEUDO_BOXED)
    parser.add_argument("--reference-grid", type=Path, default=DEFAULT_REFERENCE_GRID)
    parser.add_argument("--unknown-ignore", type=Path, default=DEFAULT_UNKNOWN_IGNORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def n_number(class_no: object) -> str:
    return f"N{int(class_no):02d}"


def clean_n_number(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("N"):
        return f"N{int(text[1:]):02d}"
    return n_number(text)


def image_dimensions(test_images: Path) -> dict[int, tuple[int, int, str, str]]:
    dims: dict[int, tuple[int, int, str, str]] = {}
    for path in sorted(test_images.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image_id = int(path.stem)
        with Image.open(path) as image:
            width, height = image.size
        dims[image_id] = (width, height, path.name, str(path))
    return dims


def add_image_and_box_columns(df: pd.DataFrame, dims: dict[int, tuple[int, int, str, str]]) -> pd.DataFrame:
    out = df.copy()
    widths: list[int] = []
    heights: list[int] = []
    file_names: list[str] = []
    image_paths: list[str] = []
    for image_id in out["image_id"].astype(int):
        width, height, file_name, image_path = dims[image_id]
        widths.append(width)
        heights.append(height)
        file_names.append(file_name)
        image_paths.append(image_path)
    out["file_name"] = file_names
    out["image_path"] = image_paths
    out["width"] = widths
    out["height"] = heights
    out["x_min"] = out["bbox_x"].astype(float)
    out["y_min"] = out["bbox_y"].astype(float)
    out["x_max"] = out["bbox_x"].astype(float) + out["bbox_w"].astype(float)
    out["y_max"] = out["bbox_y"].astype(float) + out["bbox_h"].astype(float)
    out["bbox_area"] = out["bbox_w"].astype(float) * out["bbox_h"].astype(float)
    out["bbox_in_image"] = (
        (out["x_min"] >= 0)
        & (out["y_min"] >= 0)
        & (out["x_max"] <= out["width"])
        & (out["y_max"] <= out["height"])
        & (out["bbox_w"] > 0)
        & (out["bbox_h"] > 0)
    )
    return out


def build_clean_annotations(pseudo_gt: pd.DataFrame, dims: dict[int, tuple[int, int, str, str]]) -> pd.DataFrame:
    clean = pseudo_gt.copy()
    clean["n_number"] = clean["resolved_n_number"].map(clean_n_number)
    clean["predicted_n_number"] = clean["predicted_n_number"].map(clean_n_number)
    clean["correct_n_number"] = clean["correct_n_number"].map(clean_n_number)
    clean["category_id"] = clean["category_id"].astype(int)
    clean = clean.sort_values(["image_id", "annotation_id"], kind="stable").reset_index(drop=True)
    clean.insert(0, "test_annotation_id", range(1, len(clean) + 1))
    clean = add_image_and_box_columns(clean, dims)
    columns = [
        "test_annotation_id",
        "image_id",
        "file_name",
        "image_path",
        "width",
        "height",
        "n_number",
        "category_id",
        "drug_name",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "bbox_area",
        "bbox_in_image",
        "score",
        "source",
        "predicted_n_number",
        "correct_n_number",
        "review_status",
        "review_note",
    ]
    return clean[columns]


def build_reviewable_annotations(
    review: pd.DataFrame,
    class_map: pd.DataFrame,
    dims: dict[int, tuple[int, int, str, str]],
    prediction_boxed: Path,
    pseudo_boxed: Path,
    reference_grid: Path,
) -> pd.DataFrame:
    n_to_meta = {
        n_number(row.class_no): {
            "resolved_category_id": int(row.category_id),
            "resolved_drug_name": row.name,
        }
        for row in class_map.itertuples(index=False)
    }

    out = review.copy()
    out["predicted_n_number"] = out["predicted_n_number"].map(clean_n_number)
    out["correct_n_number"] = out["correct_n_number"].map(clean_n_number)
    out["resolved_n_number"] = [
        correct if pd.notna(correct) and str(correct).strip() else predicted
        for predicted, correct in zip(out["predicted_n_number"], out["correct_n_number"], strict=True)
    ]
    out["resolved_category_id"] = [
        n_to_meta.get(value, {}).get("resolved_category_id") if value else pd.NA
        for value in out["resolved_n_number"]
    ]
    out["resolved_drug_name"] = [
        n_to_meta.get(value, {}).get("resolved_drug_name") if value else pd.NA
        for value in out["resolved_n_number"]
    ]
    out = add_image_and_box_columns(out, dims)
    out["prediction_boxed_image_path"] = [
        str(prediction_boxed / f"{int(image_id)}.jpg") for image_id in out["image_id"]
    ]
    out["pseudo_gt_boxed_image_path"] = [
        str(pseudo_boxed / f"{int(image_id)}.jpg") for image_id in out["image_id"]
    ]
    out["reference_grid_path"] = str(reference_grid)
    out["needs_review"] = out["review_status"].fillna("").ne("reviewed")
    out["action_hint"] = "accept"
    out.loc[out["keep"].astype(int).eq(0), "action_hint"] = "drop"
    out.loc[out["source"].fillna("").str.startswith("manual_added"), "action_hint"] = "manual_add"
    out.loc[
        out["correct_n_number"].notna()
        & out["predicted_n_number"].notna()
        & out["correct_n_number"].ne(out["predicted_n_number"]),
        "action_hint",
    ] = "class_fix"
    out["checked"] = ""
    out["review_keep_override"] = ""
    out["review_correct_n_number_override"] = ""
    out["review_comment"] = ""
    columns = [
        "keep",
        "needs_review",
        "action_hint",
        "checked",
        "review_keep_override",
        "review_correct_n_number_override",
        "review_comment",
        "review_status",
        "review_note",
        "source",
        "annotation_id",
        "image_id",
        "candidate_rank",
        "file_name",
        "image_path",
        "prediction_boxed_image_path",
        "pseudo_gt_boxed_image_path",
        "reference_grid_path",
        "width",
        "height",
        "predicted_n_number",
        "correct_n_number",
        "resolved_n_number",
        "category_id",
        "resolved_category_id",
        "predicted_drug_name",
        "resolved_drug_name",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "bbox_area",
        "bbox_in_image",
        "score",
        "name",
        "name_en",
        "company",
        "shape",
        "color",
        "print_front",
        "print_back",
        "sample_path",
        "sample_bbox",
    ]
    return out[columns].sort_values(["image_id", "candidate_rank", "annotation_id"], kind="stable")


def build_coco(
    clean: pd.DataFrame,
    class_map: pd.DataFrame,
    dims: dict[int, tuple[int, int, str, str]],
) -> dict[str, object]:
    images = [
        {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        }
        for image_id, (width, height, file_name, _path) in sorted(dims.items())
    ]
    categories = []
    for row in class_map.sort_values("category_id").itertuples(index=False):
        categories.append(
            {
                "id": int(row.category_id),
                "name": str(row.name),
                "supercategory": "pill",
                "n_number": n_number(row.class_no),
                "name_en": "" if pd.isna(row.name_en) else str(row.name_en),
                "company": "" if pd.isna(row.company) else str(row.company),
                "shape": "" if pd.isna(row.shape) else str(row.shape),
                "color": "" if pd.isna(row.color) else str(row.color),
                "print_front": "" if pd.isna(row.print_front) else str(row.print_front),
                "print_back": "" if pd.isna(row.print_back) else str(row.print_back),
            }
        )
    annotations = []
    for row in clean.itertuples(index=False):
        annotations.append(
            {
                "id": int(row.test_annotation_id),
                "image_id": int(row.image_id),
                "category_id": int(row.category_id),
                "bbox": [
                    round(float(row.bbox_x), 2),
                    round(float(row.bbox_y), 2),
                    round(float(row.bbox_w), 2),
                    round(float(row.bbox_h), 2),
                ],
                "area": round(float(row.bbox_area), 2),
                "iscrowd": 0,
                "score": round(float(row.score), 6) if not pd.isna(row.score) else None,
                "n_number": row.n_number,
                "drug_name": row.drug_name,
                "source": row.source,
            }
        )
    return {
        "info": {
            "description": "Pseudo ground-truth annotations for test images, exported from submission and manual fixes.",
            "version": "1.0",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def build_label_studio(clean: pd.DataFrame) -> list[dict[str, object]]:
    tasks = []
    for image_id, group in clean.groupby("image_id", sort=True):
        first = group.iloc[0]
        results = []
        for row in group.itertuples(index=False):
            results.append(
                {
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": float(row.bbox_x) / float(row.width) * 100.0,
                        "y": float(row.bbox_y) / float(row.height) * 100.0,
                        "width": float(row.bbox_w) / float(row.width) * 100.0,
                        "height": float(row.bbox_h) / float(row.height) * 100.0,
                        "rectanglelabels": [f"{row.n_number} {row.drug_name}"],
                    },
                    "score": float(row.score) if not pd.isna(row.score) else None,
                }
            )
        tasks.append(
            {
                "id": int(image_id),
                "data": {"image": "file://" + str(first.image_path)},
                "predictions": [
                    {
                        "model_version": "pseudo_gt_from_submission_review",
                        "result": results,
                    }
                ],
            }
        )
    return tasks


def write_readme(out_dir: Path, summary: dict[str, object]) -> None:
    text = f"""# Test Annotation Export

이 폴더는 테스트 이미지 pseudo ground-truth를 바로 쓰기 좋게 뽑은 결과입니다.

## 핵심 파일

- `test_annotations.csv`: keep=1인 최종 박스만 담은 학습/평가용 CSV
- `test_annotations_reviewable.csv`: 원본 후보, drop/manual/class_fix 상태, 이미지 경로를 포함한 검수용 CSV
- `test_annotations_coco.json`: COCO 형식 어노테이션
- `label_studio_preannotations.json`: Label Studio에 pre-annotation으로 넣기 좋은 JSON
- `summary.json`: export 검증 요약

## 현재 요약

- images: {summary["images"]}
- annotations: {summary["annotations"]}
- classes_used: {summary["classes_used"]}
- classes_total: {summary["classes_total"]}
- manual_added: {summary["manual_added"]}
- dropped_candidates: {summary["dropped_candidates"]}
- bbox_out_of_image: {summary["bbox_out_of_image"]}

## 수정 규칙

1. 바로 학습/평가에 쓸 때는 `test_annotations.csv` 또는 `test_annotations_coco.json`을 사용합니다.
2. 사람이 다시 볼 때는 `test_annotations_reviewable.csv`에서 `prediction_boxed_image_path`, `pseudo_gt_boxed_image_path`, `reference_grid_path`를 같이 봅니다.
3. 틀린 후보는 `keep=0`, 클래스만 틀린 후보는 `correct_n_number`에 `N17` 같은 번호를 넣는 방식이 원본 규칙입니다.
4. 현재 반영된 수정은 visual NN 보조 검수 결과입니다. 중복 박스 drop, 강한 class correction, confidence 높은 누락 component 추가가 포함됩니다.
5. 세부 변경 내역은 `../reports/test_pseudo_gt_eval/visual_nn_correction_log.csv`와 `../reports/test_pseudo_gt_eval/visual_nn_missing_component_candidates.csv`를 봅니다.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pseudo_gt = pd.read_csv(args.pseudo_gt)
    review = pd.read_csv(args.review)
    class_map = pd.read_csv(args.class_map)
    dims = image_dimensions(args.test_images)

    missing_images = sorted(set(pseudo_gt["image_id"].astype(int)) - set(dims))
    if missing_images:
        raise FileNotFoundError(f"Missing test images for image_id values: {missing_images[:10]}")

    clean = build_clean_annotations(pseudo_gt, dims)
    reviewable = build_reviewable_annotations(
        review,
        class_map,
        dims,
        args.prediction_boxed,
        args.pseudo_boxed,
        args.reference_grid,
    )
    coco = build_coco(clean, class_map, dims)
    label_studio = build_label_studio(clean)

    clean.to_csv(args.out / "test_annotations.csv", index=False)
    reviewable.to_csv(args.out / "test_annotations_reviewable.csv", index=False)
    unknown_ignore_rows = 0
    if args.unknown_ignore.exists():
        unknown_ignore = pd.read_csv(args.unknown_ignore)
        unknown_ignore.to_csv(args.out / "test_unknown_ignore_boxes.csv", index=False)
        unknown_ignore_rows = int(len(unknown_ignore))
    (args.out / "test_annotations_coco.json").write_text(
        json.dumps(coco, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out / "label_studio_preannotations.json").write_text(
        json.dumps(label_studio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "out_dir": str(args.out),
        "images": int(clean["image_id"].nunique()),
        "test_images_total": len(dims),
        "annotations": int(len(clean)),
        "reviewable_rows": int(len(reviewable)),
        "classes_used": int(clean["category_id"].nunique()),
        "classes_total": int(class_map["category_id"].nunique()),
        "manual_added": int(clean["source"].fillna("").str.startswith("manual_added").sum()),
        "dropped_candidates": int(reviewable["keep"].astype(int).eq(0).sum()),
        "bbox_out_of_image": int((~clean["bbox_in_image"]).sum()),
        "unknown_ignore_boxes": unknown_ignore_rows,
        "files": {
            "clean_csv": str(args.out / "test_annotations.csv"),
            "reviewable_csv": str(args.out / "test_annotations_reviewable.csv"),
            "coco_json": str(args.out / "test_annotations_coco.json"),
            "label_studio_json": str(args.out / "label_studio_preannotations.json"),
            "unknown_ignore_boxes": str(args.out / "test_unknown_ignore_boxes.csv"),
            "readme": str(args.out / "README.md"),
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
