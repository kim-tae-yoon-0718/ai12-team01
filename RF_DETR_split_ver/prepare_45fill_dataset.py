#!/usr/bin/env python3
"""Prepare the 56-class 45-fill COCO dataset for RF-DETR training.

RF-DETR expects:

dataset/
  train/_annotations.coco.json
  valid/_annotations.coco.json
  test/_annotations.coco.json

The merged dataset produced by the detectionproject workflow stores images as
images/train, images/val and annotations/train.json, annotations/val.json. This
script creates the RF-DETR directory layout and shifts category ids from the
project's 0-based contiguous ids to RF-DETR's 1-based class ids with a dummy
background category at id=0.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/aihub_prepared/train_56_45_merged_coco"
)
DEFAULT_OUT = Path(
    "/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/rfdetr_dataset_45fill"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument(
        "--test-from",
        choices=["valid", "empty"],
        default="valid",
        help="RF-DETR docs show a test split. Use valid images as test by default.",
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


def convert_categories(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = [{"id": 0, "name": "pill", "supercategory": "none"}]
    for category in sorted(categories, key=lambda c: int(c["id"])):
        new_category = dict(category)
        old_id = int(new_category["id"])
        new_category["id"] = old_id + 1
        new_category["project_coco_category_id"] = old_id
        converted.append(new_category)
    return converted


def convert_split(source_dir: Path, out_dir: Path, source_split: str, target_split: str, link_mode: str) -> dict[str, Any]:
    src_ann = read_json(source_dir / "annotations" / f"{source_split}.json")
    target_dir = out_dir / target_split
    target_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for image in src_ann.get("images", []):
        file_name = Path(image["file_name"]).name
        src_image = source_dir / "images" / source_split / file_name
        if not src_image.exists():
            raise FileNotFoundError(src_image)
        place_file(src_image, target_dir / file_name, link_mode)
        converted_image = dict(image)
        converted_image["file_name"] = file_name
        converted_image.pop("source_dataset", None)
        images.append(converted_image)

    annotations = []
    for ann in src_ann.get("annotations", []):
        converted_ann = dict(ann)
        converted_ann["category_id"] = int(ann["category_id"]) + 1
        converted_ann["project_coco_category_id"] = int(ann["category_id"])
        annotations.append(converted_ann)

    payload = {
        "info": {
            "description": f"RF-DETR {target_split} split converted from 56-class 45-fill merged COCO",
            "source_dir": str(source_dir),
            "source_split": source_split,
            "category_id_semantics": "0 is dummy background; class ids are original contiguous ids + 1.",
        },
        "licenses": src_ann.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": convert_categories(src_ann.get("categories", [])),
    }
    write_json(target_dir / "_annotations.coco.json", payload)
    return {
        "split": target_split,
        "source_split": source_split,
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(payload["categories"]),
    }


def write_empty_test(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    src_ann = read_json(source_dir / "annotations" / "val.json")
    payload = {
        "info": {
            "description": "Empty RF-DETR test split placeholder",
            "source_dir": str(source_dir),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": convert_categories(src_ann.get("categories", [])),
    }
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    write_json(test_dir / "_annotations.coco.json", payload)
    return {"split": "test", "source_split": "empty", "images": 0, "annotations": 0, "categories": len(payload["categories"])}


def main() -> None:
    args = parse_args()
    if not args.source_dir.exists():
        raise FileNotFoundError(args.source_dir)
    reset_dir(args.out_dir)

    summary = {
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "link_mode": args.link_mode,
        "splits": [
            convert_split(args.source_dir, args.out_dir, "train", "train", args.link_mode),
            convert_split(args.source_dir, args.out_dir, "val", "valid", args.link_mode),
        ],
    }
    if args.test_from == "valid":
        summary["splits"].append(convert_split(args.source_dir, args.out_dir, "val", "test", args.link_mode))
    else:
        summary["splits"].append(write_empty_test(args.source_dir, args.out_dir))

    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
