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
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_BASE56 = Path(
    "/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/aihub_prepared/train_56_45_merged_coco"
)
DEFAULT_HIDDEN18 = Path(
    "/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/aihub_prepared/hidden_train_import"
)
DEFAULT_OUT = Path(
    "/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/working/rfdetr_dataset_74_hidden45"
)
DEFAULT_TEST_IMAGES = Path(
    "/Users/pio/Documents/AIENGINEERCOURSE/detectionproject/sprint_ai_project1_data/test_images"
)


N_RE = re.compile(r"^N(\d{2})$")
K_RE = re.compile(r"K-(\d+)")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base56-dir", type=Path, default=DEFAULT_BASE56)
    parser.add_argument("--hidden18-dir", type=Path, default=DEFAULT_HIDDEN18)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--test-images-dir", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument(
        "--target-count-per-class",
        type=int,
        default=45,
        help="Cap hidden N57-N74 train+valid annotations per class. Base/original N01-N56 annotations are kept as-is.",
    )
    parser.add_argument(
        "--test-from",
        choices=["original", "valid", "empty"],
        default="original",
        help="Use original downloaded test images by default. No test annotations are created.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Target validation annotation ratio for the group-aware train/valid split.",
    )
    parser.add_argument(
        "--valid-min-per-class",
        type=int,
        default=5,
        help="Minimum validation annotations to target per class while keeping split groups intact.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for group-aware train/valid split assignment.",
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


def validate_split_files(out_dir: Path, split: str) -> dict[str, Any]:
    ann_path = out_dir / split / "_annotations.coco.json"
    payload = read_json(ann_path)
    missing: list[str] = []
    for image in payload.get("images", []):
        image_path = out_dir / split / str(image["file_name"])
        if not image_path.exists():
            missing.append(str(image_path))

    if missing:
        sample = "\n".join(f"  - {path}" for path in missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"{split}: {len(missing)} image file(s) referenced by COCO JSON are missing\n"
            f"{sample}{more}"
        )

    return {
        "split": split,
        "images_checked": len(payload.get("images", [])),
        "missing_images": 0,
    }


def natural_image_key(path: Path) -> tuple[int, int | str]:
    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, path.name)


def image_id_from_name(path: Path, fallback: int) -> int:
    if path.stem.isdigit():
        return int(path.stem)
    return fallback


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


AnnotationKey = tuple[str, str, int]


def build_annotation_selection(
    *,
    base56_dir: Path,
    hidden18_dir: Path,
    base_category_map: dict[int, int],
    hidden_category_map: dict[int, int],
    target_count_per_class: int,
) -> tuple[set[AnnotationKey] | None, list[dict[str, Any]]]:
    if target_count_per_class <= 0:
        return None, []

    records_by_category: dict[int, dict[str, list[AnnotationKey]]] = defaultdict(lambda: defaultdict(list))
    for source_dir, source_dataset, category_map in [
        (hidden18_dir, "hidden18_45fill", hidden_category_map),
    ]:
        for source_split in ("train", "val"):
            src_ann = read_json(source_dir / "annotations" / f"{source_split}.json")
            for ann in src_ann.get("annotations", []):
                old_category_id = int(ann["category_id"])
                final_category_id = category_map[old_category_id]
                records_by_category[final_category_id][source_split].append(
                    (source_dataset, source_split, int(ann["id"]))
                )

    selected: set[AnnotationKey] = set()
    selection_rows: list[dict[str, Any]] = []
    for category_id in sorted(records_by_category):
        by_split = records_by_category[category_id]
        train_records = by_split.get("train", [])
        val_records = by_split.get("val", [])
        total_available = len(train_records) + len(val_records)
        if total_available <= target_count_per_class:
            chosen = train_records + val_records
        elif train_records and val_records:
            val_target = min(len(val_records), max(1, math.ceil(target_count_per_class * 0.1)))
            train_target = min(len(train_records), target_count_per_class - val_target)
            chosen = train_records[:train_target] + val_records[:val_target]
            if len(chosen) < target_count_per_class:
                already = set(chosen)
                remainder = [record for record in train_records + val_records if record not in already]
                chosen.extend(remainder[: target_count_per_class - len(chosen)])
        else:
            chosen = (train_records + val_records)[:target_count_per_class]

        selected.update(chosen)
        selection_rows.append(
            {
                "category_id": category_id,
                "available_total": total_available,
                "available_train": len(train_records),
                "available_val": len(val_records),
                "selected_total": len(chosen),
                "selected_train": sum(1 for record in chosen if record[1] == "train"),
                "selected_val": sum(1 for record in chosen if record[1] == "val"),
                "target_count_per_class": target_count_per_class,
            }
        )

    return selected, selection_rows


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
    selected_annotation_keys: set[AnnotationKey] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    src_ann = read_json(source_dir / "annotations" / f"{source_split}.json")
    target_split_dir = target_dir / target_split
    image_id_map: dict[int, int] = {}
    image_by_old_id: dict[int, dict[str, Any]] = {}
    converted_images: list[dict[str, Any]] = []
    converted_annotations: list[dict[str, Any]] = []
    raw_annotations = src_ann.get("annotations", [])
    if selected_annotation_keys is not None and source_dataset == "hidden18_45fill":
        raw_annotations = [
            ann
            for ann in raw_annotations
            if (source_dataset, source_split, int(ann["id"])) in selected_annotation_keys
        ]
    needed_image_ids = {int(ann["image_id"]) for ann in raw_annotations}

    for image in src_ann.get("images", []):
        old_image_id = int(image["id"])
        if old_image_id not in needed_image_ids:
            continue
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

    for ann in raw_annotations:
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


ANGLE_TOKENS = {"70", "75", "90"}


def raw_group_stem(image: dict[str, Any]) -> str:
    raw_name = (
        image.get("source_member")
        or image.get("source_file_name")
        or image.get("file_name")
        or ""
    )
    stem = Path(str(raw_name)).stem
    if stem.endswith("_crop"):
        stem = stem[:-5]
    generated_crop_match = re.match(r"^N\d{2}_K-\d{6}_\d{3}_(.+)$", stem)
    if generated_crop_match:
        stem = generated_crop_match.group(1)
    return stem


def angle_set_group_key(image: dict[str, Any]) -> str:
    parts = raw_group_stem(image).split("_")
    if len(parts) >= 3 and parts[-3] in ANGLE_TOKENS and parts[-2].isdigit() and parts[-1].isdigit():
        parts[-3] = "ANGLE"
    return "_".join(parts)


def collect_source_image_records(
    *,
    source_dir: Path,
    source_dataset: str,
    category_map: dict[int, int],
    selected_annotation_keys: set[AnnotationKey] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source_split in ("train", "val"):
        src_ann = read_json(source_dir / "annotations" / f"{source_split}.json")
        image_by_id = {int(image["id"]): image for image in src_ann.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in src_ann.get("annotations", []):
            ann_id = int(ann["id"])
            if (
                selected_annotation_keys is not None
                and source_dataset == "hidden18_45fill"
                and (source_dataset, source_split, ann_id) not in selected_annotation_keys
            ):
                continue
            annotations_by_image[int(ann["image_id"])].append(ann)

        for old_image_id in sorted(annotations_by_image):
            image = image_by_id[old_image_id]
            final_category_counts: Counter[int] = Counter()
            for ann in annotations_by_image[old_image_id]:
                old_category_id = int(ann["category_id"])
                final_category_counts[category_map[old_category_id]] += 1
            records.append(
                {
                    "source_dir": source_dir,
                    "source_dataset": source_dataset,
                    "source_split": source_split,
                    "old_image_id": old_image_id,
                    "image": image,
                    "annotations": annotations_by_image[old_image_id],
                    "category_map": category_map,
                    "split_group_key": angle_set_group_key(image),
                    "final_category_counts": final_category_counts,
                }
            )
    return records


def build_group_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["split_group_key"]
        group = groups.setdefault(
            key,
            {
                "split_group_key": key,
                "records": [],
                "category_counts": Counter(),
                "source_datasets": Counter(),
                "source_splits": Counter(),
                "sample_files": [],
            },
        )
        group["records"].append(record)
        group["category_counts"].update(record["final_category_counts"])
        group["source_datasets"][record["source_dataset"]] += 1
        group["source_splits"][record["source_split"]] += 1
        if len(group["sample_files"]) < 3:
            group["sample_files"].append(Path(record["image"]["file_name"]).name)
    return groups


def choose_validation_groups(
    groups: dict[str, dict[str, Any]],
    *,
    val_ratio: float,
    valid_min_per_class: int,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    if not groups:
        return {}, {"split_group_count": 0}

    rng = random.Random(seed)
    group_order = sorted(groups)
    rng.shuffle(group_order)

    total_by_category: Counter[int] = Counter()
    for group in groups.values():
        total_by_category.update(group["category_counts"])

    target_by_category = {
        category_id: min(
            count - 1,
            max(valid_min_per_class, math.ceil(count * val_ratio)),
        )
        if count > 1
        else 0
        for category_id, count in total_by_category.items()
    }
    target_total = max(1, round(sum(total_by_category.values()) * val_ratio))
    val_groups: set[str] = set()
    val_by_category: Counter[int] = Counter()
    val_total = 0

    def can_select(group: dict[str, Any]) -> bool:
        for category_id, count in group["category_counts"].items():
            if val_by_category[category_id] + count >= total_by_category[category_id]:
                return False
        return True

    def select(group_key: str) -> None:
        nonlocal val_total
        val_groups.add(group_key)
        group = groups[group_key]
        val_by_category.update(group["category_counts"])
        val_total += sum(group["category_counts"].values())

    while True:
        deficits = {
            category_id: target - val_by_category[category_id]
            for category_id, target in target_by_category.items()
            if val_by_category[category_id] < target
        }
        if not deficits:
            break

        best_key: str | None = None
        best_score: tuple[int, int, int, str] | None = None
        for group_key in group_order:
            if group_key in val_groups:
                continue
            group = groups[group_key]
            if not can_select(group):
                continue
            gain = sum(min(deficits.get(category_id, 0), count) for category_id, count in group["category_counts"].items())
            if gain <= 0:
                continue
            overshoot = sum(
                max(0, val_by_category[category_id] + count - target_by_category.get(category_id, 0))
                for category_id, count in group["category_counts"].items()
            )
            ann_count = sum(group["category_counts"].values())
            score = (-gain, overshoot, ann_count, group_key)
            if best_score is None or score < best_score:
                best_key = group_key
                best_score = score

        if best_key is None:
            break
        select(best_key)

    for group_key in group_order:
        if val_total >= target_total:
            break
        if group_key in val_groups:
            continue
        group = groups[group_key]
        if not can_select(group):
            continue
        select(group_key)

    assignment = {group_key: ("valid" if group_key in val_groups else "train") for group_key in groups}
    train_by_category = {
        category_id: total_by_category[category_id] - val_by_category[category_id]
        for category_id in total_by_category
    }
    summary = {
        "split_group_count": len(groups),
        "train_group_count": sum(1 for split in assignment.values() if split == "train"),
        "valid_group_count": sum(1 for split in assignment.values() if split == "valid"),
        "total_annotations": sum(total_by_category.values()),
        "train_annotations": sum(train_by_category.values()),
        "valid_annotations": sum(val_by_category.values()),
        "target_valid_ratio": val_ratio,
        "actual_valid_ratio": round(sum(val_by_category.values()) / max(1, sum(total_by_category.values())), 4),
        "valid_min_per_class_target": valid_min_per_class,
        "valid_min_per_class": min((val_by_category[category_id] for category_id in total_by_category), default=0),
        "valid_max_per_class": max((val_by_category[category_id] for category_id in total_by_category), default=0),
        "classes_below_requested_min_valid": sum(
            1
            for category_id in total_by_category
            if total_by_category[category_id] > valid_min_per_class
            and val_by_category[category_id] < valid_min_per_class
        ),
        "classes_with_zero_valid": sum(1 for category_id in total_by_category if val_by_category[category_id] == 0),
        "classes_with_zero_train": sum(1 for category_id in total_by_category if train_by_category[category_id] == 0),
    }
    return assignment, summary


def write_split_group_manifest(
    *,
    out_dir: Path,
    groups: dict[str, dict[str, Any]],
    assignment: dict[str, str],
    categories: list[dict[str, Any]],
) -> None:
    category_to_n = {
        int(category["id"]): category.get("n_number") or category.get("class_no") or category.get("hidden_n") or ""
        for category in categories
    }
    rows = []
    for group_key in sorted(groups):
        group = groups[group_key]
        category_ids = sorted(group["category_counts"])
        rows.append(
            {
                "split_group_key": group_key,
                "target_split": assignment[group_key],
                "images": len(group["records"]),
                "annotations": sum(group["category_counts"].values()),
                "category_ids": "|".join(str(category_id) for category_id in category_ids),
                "n_numbers": "|".join(category_to_n.get(category_id, "") for category_id in category_ids),
                "source_datasets": "|".join(f"{key}:{value}" for key, value in sorted(group["source_datasets"].items())),
                "source_splits": "|".join(f"{key}:{value}" for key, value in sorted(group["source_splits"].items())),
                "sample_files": "|".join(group["sample_files"]),
            }
        )
    with (out_dir / "split_group_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_group_aware_train_valid(
    *,
    base56_dir: Path,
    hidden18_dir: Path,
    out_dir: Path,
    link_mode: str,
    categories: list[dict[str, Any]],
    base_category_map: dict[int, int],
    hidden_category_map: dict[int, int],
    selected_annotation_keys: set[AnnotationKey] | None,
    val_ratio: float,
    valid_min_per_class: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(
        collect_source_image_records(
            source_dir=base56_dir,
            source_dataset="base56_45fill",
            category_map=base_category_map,
            selected_annotation_keys=None,
        )
    )
    records.extend(
        collect_source_image_records(
            source_dir=hidden18_dir,
            source_dataset="hidden18_45fill",
            category_map=hidden_category_map,
            selected_annotation_keys=selected_annotation_keys,
        )
    )
    groups = build_group_stats(records)
    assignment, split_summary = choose_validation_groups(
        groups,
        val_ratio=val_ratio,
        valid_min_per_class=valid_min_per_class,
        seed=seed,
    )
    write_split_group_manifest(out_dir=out_dir, groups=groups, assignment=assignment, categories=categories)

    split_records = {
        "train": [record for record in records if assignment[record["split_group_key"]] == "train"],
        "valid": [record for record in records if assignment[record["split_group_key"]] == "valid"],
    }
    split_summaries: list[dict[str, Any]] = []
    for target_split, target_records in split_records.items():
        target_split_dir = out_dir / target_split
        target_split_dir.mkdir(parents=True, exist_ok=True)
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        image_id_map: dict[tuple[str, str, int], int] = {}
        used_names: set[str] = set()
        next_image_id = 1
        next_annotation_id = 1

        for record in sorted(
            target_records,
            key=lambda item: (
                item["source_dataset"],
                item["split_group_key"],
                item["source_split"],
                item["old_image_id"],
            ),
        ):
            image = record["image"]
            original_name = Path(image["file_name"]).name
            source_path = source_image_path(
                record["source_dir"],
                record["source_dataset"],
                record["source_split"],
                original_name,
            )
            if not source_path.exists():
                raise FileNotFoundError(source_path)

            new_name = f"{record['source_dataset']}__{original_name}"
            if new_name in used_names:
                new_name = f"{record['source_dataset']}__{record['source_split']}__{record['old_image_id']}__{original_name}"
            used_names.add(new_name)
            place_file(source_path, target_split_dir / new_name, link_mode)

            converted_image = dict(image)
            converted_image["id"] = next_image_id
            converted_image["file_name"] = new_name
            converted_image["source_dataset"] = record["source_dataset"]
            converted_image["source_split"] = record["source_split"]
            converted_image["source_file_name"] = original_name
            converted_image["split_group_key"] = record["split_group_key"]
            converted_image["target_split"] = target_split
            image_key = (record["source_dataset"], record["source_split"], record["old_image_id"])
            image_id_map[image_key] = next_image_id
            images.append(converted_image)

            for ann in record["annotations"]:
                old_category_id = int(ann["category_id"])
                converted_ann = dict(ann)
                converted_ann["id"] = next_annotation_id
                converted_ann["image_id"] = next_image_id
                converted_ann["category_id"] = record["category_map"][old_category_id]
                converted_ann["source_dataset"] = record["source_dataset"]
                converted_ann["source_split"] = record["source_split"]
                converted_ann["source_annotation_id"] = int(ann["id"])
                converted_ann["source_category_id"] = old_category_id
                converted_ann["split_group_key"] = record["split_group_key"]
                validate_bbox(converted_image, converted_ann)
                annotations.append(converted_ann)
                next_annotation_id += 1
            next_image_id += 1

        payload = {
            "info": {
                "description": f"RF-DETR {target_split} split with angle-set group-aware validation holdout",
                "base56_dir": str(base56_dir),
                "hidden18_dir": str(hidden18_dir),
                "target_split": target_split,
                "group_key_rule": "source stem with 70/75/90 camera angle token replaced by ANGLE",
                "category_id_semantics": "COCO category_id is the numeric K-code. RF-DETR remaps sparse ids internally.",
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        write_json(out_dir / target_split / "_annotations.coco.json", payload)
        split_summaries.append(
            {
                "split": target_split,
                "source_split": "group_aware_all",
                "images": len(images),
                "annotations": len(annotations),
                "categories": len(categories),
                "base56_images": sum(1 for image in images if image["source_dataset"] == "base56_45fill"),
                "hidden18_images": sum(1 for image in images if image["source_dataset"] == "hidden18_45fill"),
            }
        )

    split_summary["split_group_manifest"] = str(out_dir / "split_group_manifest.csv")
    return split_summaries, split_summary


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
    selected_annotation_keys: set[AnnotationKey] | None,
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
            selected_annotation_keys=selected_annotation_keys,
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


def write_original_test(test_images_dir: Path, out_dir: Path, categories: list[dict[str, Any]], link_mode: str) -> dict[str, Any]:
    if not test_images_dir.exists():
        raise FileNotFoundError(test_images_dir)
    test_dir = out_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [path for path in test_images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_image_key,
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_images_dir}")

    images: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for index, image_path in enumerate(image_paths, start=1):
        image_id = image_id_from_name(image_path, index)
        if image_id in used_ids:
            raise ValueError(f"Duplicate test image id {image_id} from {image_path}")
        used_ids.add(image_id)

        with Image.open(image_path) as image:
            width, height = image.size
        place_file(image_path, test_dir / image_path.name, link_mode)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "source_dataset": "original_test",
                "source_file_name": image_path.name,
            }
        )

    payload = {
        "info": {
            "description": "Original downloaded competition test images for RF-DETR inference",
            "test_images_dir": str(test_images_dir),
            "category_id_semantics": "COCO category_id is the numeric K-code. Test annotations are intentionally empty.",
        },
        "licenses": [],
        "images": images,
        "annotations": [],
        "categories": categories,
    }
    write_json(test_dir / "_annotations.coco.json", payload)
    return {
        "split": "test",
        "source_split": "original_test",
        "images": len(images),
        "annotations": 0,
        "categories": len(categories),
        "test_images_dir": str(test_images_dir),
    }


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
    selected_annotation_keys, class_selection = build_annotation_selection(
        base56_dir=args.base56_dir,
        hidden18_dir=args.hidden18_dir,
        base_category_map=base_category_map,
        hidden_category_map=hidden_category_map,
        target_count_per_class=args.target_count_per_class,
    )

    reset_dir(args.out_dir)
    train_valid_splits, group_split_summary = write_group_aware_train_valid(
        base56_dir=args.base56_dir,
        hidden18_dir=args.hidden18_dir,
        out_dir=args.out_dir,
        link_mode=args.link_mode,
        categories=categories,
        base_category_map=base_category_map,
        hidden_category_map=hidden_category_map,
        selected_annotation_keys=selected_annotation_keys,
        val_ratio=args.val_ratio,
        valid_min_per_class=args.valid_min_per_class,
        seed=args.seed,
    )
    summary = {
        "base56_dir": str(args.base56_dir),
        "hidden18_dir": str(args.hidden18_dir),
        "test_images_dir": str(args.test_images_dir),
        "out_dir": str(args.out_dir),
        "link_mode": args.link_mode,
        "target_count_per_class": args.target_count_per_class,
        "val_ratio": args.val_ratio,
        "valid_min_per_class": args.valid_min_per_class,
        "seed": args.seed,
        "class_count": len(categories),
        "category_id_semantics": "COCO category_id is the numeric K-code. RF-DETR remaps sparse ids internally.",
        "split_strategy": "angle_set_group_aware",
        "group_split_summary": group_split_summary,
        "class_selection": class_selection,
        "splits": train_valid_splits,
    }
    if args.test_from == "original":
        summary["splits"].append(write_original_test(args.test_images_dir, args.out_dir, categories, args.link_mode))
    elif args.test_from == "valid":
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
                selected_annotation_keys=selected_annotation_keys,
            )
        )
    else:
        summary["splits"].append(write_empty_test(args.out_dir, categories))

    write_category_table(args.out_dir, categories)
    summary["file_validation"] = [validate_split_files(args.out_dir, split["split"]) for split in summary["splits"]]
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
