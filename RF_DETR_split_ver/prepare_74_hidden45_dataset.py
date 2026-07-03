#!/usr/bin/env python3
"""Prepare a 74-class RF-DETR dataset from 56-class 45-fill + hidden N18.

This keeps the project class contract visible in the COCO ids used on disk:

- COCO category_id is the numeric K-code, e.g. K-001900 -> 1900
- known train classes: N01..N56 keep the basic trainset K-code ids
- hidden classes: N57..N74 use their AIHub/K-code ids

The hidden import is expected to be the crop export made from AIHub combo
images. It intentionally has one annotation per crop so unlabeled neighboring
pills from the original combo image do not leak into training.

RF-DETR remaps sparse COCO category IDs to contiguous internal labels during
custom dataset loading, so these sparse K-code ids are safe for training while
remaining compatible with the project submission/evaluation ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


DEFAULT_BASE56 = Path(
    "/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/aihub_prepared/train_56_45_merged_coco"
)
DEFAULT_HIDDEN18 = Path(
    "/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/aihub_prepared/hidden_train_import"
)
DEFAULT_OUT = Path(
    "/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/rfdetr_dataset_74_hidden45"
)


N_RE = re.compile(r"^N(\d{2})$")
K_RE = re.compile(r"K-(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base56-dir", type=Path, default=DEFAULT_BASE56)
    parser.add_argument("--hidden18-dir", type=Path, default=DEFAULT_HIDDEN18)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument(
        "--test-from",
        choices=["valid", "empty"],
        default="valid",
        help="Use the combined valid split as RF-DETR test by default.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        dst.symlink_to(src)


def parse_n_number(value: str) -> int:
    match = N_RE.match(value)
    if not match:
        raise ValueError(f"Invalid N-number: {value!r}")
    return int(match.group(1))


def n_number_id(category: dict[str, Any]) -> int:
    class_no = category.get("class_no")
    if class_no:
        return parse_n_number(str(class_no))
    hidden_n = category.get("hidden_n")
    if hidden_n:
        return parse_n_number(str(hidden_n))
    raise ValueError(f"Category is missing class_no/hidden_n: {category}")


def k_code_id(category: dict[str, Any]) -> int:
    for key in ("real_category_id", "source_category_id"):
        value = category.get(key)
        if value not in (None, ""):
            return int(value)
    for key in ("mapping_code", "dl_mapping_code"):
        value = category.get(key)
        if value:
            match = K_RE.search(str(value))
            if match:
                return int(match.group(1))
    raw_id = int(category["id"])
    if raw_id >= 100:
        return raw_id
    return raw_id + 1


def hidden_category_id(category: dict[str, Any]) -> int:
    hidden_n = category.get("hidden_n")
    if not hidden_n:
        raise ValueError(f"Hidden category is missing hidden_n: {category}")
    return parse_n_number(str(hidden_n))


def convert_categories(base_categories: list[dict[str, Any]], hidden_categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    used_k_codes: set[int] = set()
    used_n_numbers: set[int] = set()

    for category in sorted(base_categories, key=n_number_id):
        n_id = n_number_id(category)
        category_id = k_code_id(category)
        if not 1 <= n_id <= 56:
            raise ValueError(f"Base N-number should be N01..N56, got N{n_id:02d}: {category}")
        if category_id in used_k_codes:
            raise ValueError(f"Duplicate K-code category id {category_id}")
        new_category = dict(category)
        new_category["id"] = category_id
        new_category["name"] = str(category_id)
        new_category["n_number"] = f"N{n_id:02d}"
        new_category["k_code_category_id"] = category_id
        new_category["source_category_id"] = int(category["id"])
        new_category["source_dataset"] = "base56_45fill"
        converted.append(new_category)
        used_k_codes.add(category_id)
        used_n_numbers.add(n_id)

    for category in sorted(hidden_categories, key=hidden_category_id):
        n_id = hidden_category_id(category)
        category_id = k_code_id(category)
        if not 57 <= n_id <= 74:
            raise ValueError(f"Hidden N-number should be N57..N74, got N{n_id:02d}: {category}")
        if category_id in used_k_codes:
            raise ValueError(f"Duplicate K-code category id {category_id}")
        new_category = dict(category)
        if "drug_name" not in new_category:
            new_category["drug_name"] = str(new_category.get("name", ""))
        new_category["id"] = category_id
        new_category["name"] = str(category_id)
        new_category["n_number"] = f"N{n_id:02d}"
        new_category["k_code_category_id"] = category_id
        new_category["source_category_id"] = int(category["id"])
        new_category["source_dataset"] = "hidden18_45fill"
        converted.append(new_category)
        used_k_codes.add(category_id)
        used_n_numbers.add(n_id)

    expected = set(range(1, 75))
    missing = sorted(expected - used_n_numbers)
    if missing:
        raise ValueError(f"Missing N-number classes: {[f'N{i:02d}' for i in missing]}")
    return converted


def validate_bbox(image: dict[str, Any], ann: dict[str, Any]) -> None:
    x, y, w, h = [float(v) for v in ann["bbox"]]
    if w <= 0 or h <= 0:
        raise ValueError(f"Non-positive bbox: {ann}")
    width = float(image["width"])
    height = float(image["height"])
    if x < 0 or y < 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
        raise ValueError(f"Out-of-bounds bbox for image {image['file_name']}: {ann['bbox']} vs {width}x{height}")


def source_image_path(source_dir: Path, source_dataset: str, split: str, file_name: str) -> Path:
    if source_dataset == "base56_45fill":
        return source_dir / "images" / split / Path(file_name).name
    return source_dir / "images" / Path(file_name).name


def add_source_split(
    *,
    source_dir: Path,
    source_dataset: str,
    source_split: str,
    target_dir: Path,
    target_split: str,
    link_mode: str,
    category_map: dict[int, int],
    next_image_id: int,
    next_annotation_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    src_ann = read_json(source_dir / "annotations" / f"{source_split}.json")
    target_split_dir = target_dir / target_split
    image_id_map: dict[int, int] = {}
    image_by_old_id: dict[int, dict[str, Any]] = {}
    converted_images: list[dict[str, Any]] = []
    converted_annotations: list[dict[str, Any]] = []

    for image in src_ann.get("images", []):
        old_image_id = int(image["id"])
        original_name = Path(image["file_name"]).name
        source_path = source_image_path(source_dir, source_dataset, source_split, original_name)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        new_name = f"{source_dataset}__{original_name}"
        place_file(source_path, target_split_dir / new_name, link_mode)

        converted_image = dict(image)
        converted_image["id"] = next_image_id
        converted_image["file_name"] = new_name
        converted_image["source_dataset"] = source_dataset
        converted_image["source_split"] = source_split
        converted_image["source_file_name"] = original_name
        image_id_map[old_image_id] = next_image_id
        image_by_old_id[old_image_id] = converted_image
        converted_images.append(converted_image)
        next_image_id += 1

    for ann in src_ann.get("annotations", []):
        old_image_id = int(ann["image_id"])
        old_category_id = int(ann["category_id"])
        if old_category_id not in category_map:
            raise KeyError(f"{source_dataset} category id not mapped: {old_category_id}")
        image = image_by_old_id[old_image_id]
        validate_bbox(image, ann)

        converted_ann = dict(ann)
        converted_ann["id"] = next_annotation_id
        converted_ann["image_id"] = image_id_map[old_image_id]
        converted_ann["category_id"] = category_map[old_category_id]
        converted_ann["source_dataset"] = source_dataset
        converted_ann["source_split"] = source_split
        converted_ann["source_annotation_id"] = int(ann["id"])
        converted_ann["source_category_id"] = old_category_id
        converted_annotations.append(converted_ann)
        next_annotation_id += 1

    return converted_images, converted_annotations, next_image_id, next_annotation_id


def build_category_maps(base_categories: list[dict[str, Any]], hidden_categories: list[dict[str, Any]]) -> tuple[dict[int, int], dict[int, int]]:
    base_map = {int(category["id"]): k_code_id(category) for category in base_categories}
    hidden_map = {int(category["id"]): k_code_id(category) for category in hidden_categories}
    return base_map, hidden_map


def convert_split(
    *,
    base56_dir: Path,
    hidden18_dir: Path,
    out_dir: Path,
    source_split: str,
    target_split: str,
    link_mode: str,
    categories: list[dict[str, Any]],
    base_category_map: dict[int, int],
    hidden_category_map: dict[int, int],
) -> dict[str, Any]:
    next_image_id = 1
    next_annotation_id = 1
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for source_dir, source_dataset, category_map in [
        (base56_dir, "base56_45fill", base_category_map),
        (hidden18_dir, "hidden18_45fill", hidden_category_map),
    ]:
        new_images, new_annotations, next_image_id, next_annotation_id = add_source_split(
            source_dir=source_dir,
            source_dataset=source_dataset,
            source_split=source_split,
            target_dir=out_dir,
            target_split=target_split,
            link_mode=link_mode,
            category_map=category_map,
            next_image_id=next_image_id,
            next_annotation_id=next_annotation_id,
        )
        images.extend(new_images)
        annotations.extend(new_annotations)

    payload = {
        "info": {
            "description": f"RF-DETR {target_split} split from 56-class 45-fill + hidden N18 45-fill",
            "base56_dir": str(base56_dir),
            "hidden18_dir": str(hidden18_dir),
            "source_split": source_split,
            "category_id_semantics": "COCO category_id is the numeric K-code. RF-DETR remaps sparse ids internally.",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    write_json(out_dir / target_split / "_annotations.coco.json", payload)
    return {
        "split": target_split,
        "source_split": source_split,
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "base56_images": sum(1 for image in images if image["source_dataset"] == "base56_45fill"),
        "hidden18_images": sum(1 for image in images if image["source_dataset"] == "hidden18_45fill"),
    }


def write_empty_test(out_dir: Path, categories: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "info": {"description": "Empty RF-DETR test split placeholder"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    write_json(out_dir / "test" / "_annotations.coco.json", payload)
    return {"split": "test", "source_split": "empty", "images": 0, "annotations": 0, "categories": len(categories)}


def write_category_table(out_dir: Path, categories: list[dict[str, Any]]) -> None:
    rows = []
    for internal_label, category in enumerate(sorted(categories, key=lambda item: int(item["id"]))):
        rows.append(
            {
                "category_id": category["id"],
                "rfdetr_internal_label": internal_label,
                "n_number": category.get("n_number") or category.get("class_no") or category.get("hidden_n"),
                "source_dataset": category.get("source_dataset", ""),
                "source_category_id": category.get("source_category_id", ""),
                "name": category.get("drug_name") or category.get("name", ""),
                "mapping_code": category.get("mapping_code") or category.get("dl_mapping_code", ""),
                "print_front": category.get("print_front", ""),
                "print_back": category.get("print_back", ""),
                "candidate_status": category.get("candidate_status", ""),
                "decision_note": category.get("decision_note", ""),
            }
        )
    path = out_dir / "category_mapping.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.base56_dir.exists():
        raise FileNotFoundError(args.base56_dir)
    if not args.hidden18_dir.exists():
        raise FileNotFoundError(args.hidden18_dir)

    base_train = read_json(args.base56_dir / "annotations" / "train.json")
    hidden_train = read_json(args.hidden18_dir / "annotations" / "train.json")
    categories = convert_categories(base_train.get("categories", []), hidden_train.get("categories", []))
    base_category_map, hidden_category_map = build_category_maps(base_train.get("categories", []), hidden_train.get("categories", []))

    reset_dir(args.out_dir)
    summary = {
        "base56_dir": str(args.base56_dir),
        "hidden18_dir": str(args.hidden18_dir),
        "out_dir": str(args.out_dir),
        "link_mode": args.link_mode,
        "class_count": len(categories),
        "category_id_semantics": "COCO category_id is the numeric K-code. RF-DETR remaps sparse ids internally.",
        "splits": [
            convert_split(
                base56_dir=args.base56_dir,
                hidden18_dir=args.hidden18_dir,
                out_dir=args.out_dir,
                source_split="train",
                target_split="train",
                link_mode=args.link_mode,
                categories=categories,
                base_category_map=base_category_map,
                hidden_category_map=hidden_category_map,
            ),
            convert_split(
                base56_dir=args.base56_dir,
                hidden18_dir=args.hidden18_dir,
                out_dir=args.out_dir,
                source_split="val",
                target_split="valid",
                link_mode=args.link_mode,
                categories=categories,
                base_category_map=base_category_map,
                hidden_category_map=hidden_category_map,
            ),
        ],
    }
    if args.test_from == "valid":
        summary["splits"].append(
            convert_split(
                base56_dir=args.base56_dir,
                hidden18_dir=args.hidden18_dir,
                out_dir=args.out_dir,
                source_split="val",
                target_split="test",
                link_mode=args.link_mode,
                categories=categories,
                base_category_map=base_category_map,
                hidden_category_map=hidden_category_map,
            )
        )
    else:
        summary["splits"].append(write_empty_test(args.out_dir, categories))

    write_category_table(args.out_dir, categories)
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
