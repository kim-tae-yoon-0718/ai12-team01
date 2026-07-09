#!/usr/bin/env python3
"""Convert the local 5-fold COCO pill dataset to YOLO format.

This keeps real pill classes only. COCO category id 0, if present, is treated
as a background placeholder and is not written as a YOLO class. Extra
background-only images are added as empty label files so YOLO can learn
no-object/background regions without fake category-0 boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps


YOLO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = YOLO_ROOT.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", os.environ.get("PROJECT_ROOT", REPO_ROOT))).resolve()
DEFAULT_SOURCE = DATA_ROOT / "working/rfdetr_dataset_74_hidden45_canvas_balanced_5fold_cls0_mps"
DEFAULT_BACKGROUNDS = DATA_ROOT / "working/backgrounds/drive_1cbHdfMYasujFtEhs5OGbPr17rtgjaOfE"
DEFAULT_OUTPUT = DATA_ROOT / "working/yolo_74_5fold_bg_mps"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BACKGROUNDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--background-variants-per-source", type=int, default=10)
    parser.add_argument("--background-valid-variants-per-source", type=int, default=0)
    parser.add_argument("--background-width", type=int, default=976)
    parser.add_argument("--background-height", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(os.path.relpath(src, dst.parent))
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_category_rows(source: Path) -> dict[int, dict[str, str]]:
    mapping_path = source / "category_mapping.csv"
    if not mapping_path.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with mapping_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("is_placeholder") == "True":
                continue
            try:
                internal = int(row["rfdetr_internal_label"])
            except Exception:
                continue
            rows[internal] = row
    return rows


def make_class_map(coco: dict[str, Any], mapping_rows: dict[int, dict[str, str]]) -> tuple[dict[int, int], list[dict[str, Any]]]:
    real_categories = []
    for cat in coco.get("categories", []):
        cid = int(cat["id"])
        if cid == 0 or cat.get("is_placeholder"):
            continue
        real_categories.append(cat)
    real_categories.sort(key=lambda c: int(c["id"]))

    cat_to_yolo: dict[int, int] = {}
    out_rows: list[dict[str, Any]] = []
    for yolo_id, cat in enumerate(real_categories):
        cid = int(cat["id"])
        row = mapping_rows.get(cid, {})
        submission_category_id = int(row.get("category_id") or cat.get("original_category_id") or cat.get("name"))
        n_number = row.get("n_number") or f"N{cid:02d}"
        drug_name = row.get("name") or ""
        class_name = f"{n_number}_{submission_category_id}"
        cat_to_yolo[cid] = yolo_id
        out_rows.append(
            {
                "yolo_class": yolo_id,
                "coco_category_id": cid,
                "submission_category_id": submission_category_id,
                "n_number": n_number,
                "class_name": class_name,
                "drug_name": drug_name,
            }
        )
    return cat_to_yolo, out_rows


def yolo_line(bbox: list[float], image_w: int, image_h: int, yolo_cls: int) -> str | None:
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0 or image_w <= 0 or image_h <= 0:
        return None
    cx = (x + w / 2.0) / image_w
    cy = (y + h / 2.0) / image_h
    nw = w / image_w
    nh = h / image_h
    vals = [max(0.0, min(1.0, v)) for v in [cx, cy, nw, nh]]
    return f"{yolo_cls} " + " ".join(f"{v:.8f}" for v in vals)


def write_yaml(path: Path, train_dir: Path, val_dir: Path, class_rows: list[dict[str, Any]]) -> None:
    names = [row["class_name"] for row in class_rows]
    lines = [
        f"path: {path.parent.as_posix()}",
        f"train: {train_dir.as_posix()}",
        f"val: {val_dir.as_posix()}",
        f"nc: {len(names)}",
        "names:",
    ]
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {json.dumps(name, ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def list_backgrounds(background_dir: Path) -> list[Path]:
    if not background_dir.exists():
        return []
    return sorted(p for p in background_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def make_background_variant(src: Path, out: Path, size: tuple[int, int], rng: random.Random, quality: int) -> None:
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    scale *= rng.uniform(1.0, 1.18)
    new_size = (max(target_w, round(image.width * scale)), max(target_h, round(image.height * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    max_x = max(0, image.width - target_w)
    max_y = max(0, image.height - target_h)
    left = rng.randint(0, max_x) if max_x else 0
    top = rng.randint(0, max_y) if max_y else 0
    image = image.crop((left, top, left + target_w, top + target_h))
    if rng.random() < 0.75:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
    if rng.random() < 0.75:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.12))
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, quality=quality)


def convert_split(
    *,
    fold_dir: Path,
    split: str,
    out_fold: Path,
    source: Path,
    cat_to_yolo: dict[int, int],
    link_mode: str,
) -> dict[str, int]:
    coco_path = fold_dir / split / "_annotations.coco.json"
    coco = read_json(coco_path)
    images_by_id = {int(img["id"]): img for img in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images_by_id}
    dropped_placeholder = 0
    dropped_unknown = 0
    for ann in coco["annotations"]:
        cid = int(ann["category_id"])
        if cid == 0:
            dropped_placeholder += 1
            continue
        if cid not in cat_to_yolo:
            dropped_unknown += 1
            continue
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    image_out_dir = out_fold / "images" / ("val" if split == "valid" else split)
    label_out_dir = out_fold / "labels" / ("val" if split == "valid" else split)
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    ann_count = 0
    missing_images = 0
    for image_id, image in images_by_id.items():
        src = fold_dir / split / image["file_name"]
        if not src.exists():
            missing_images += 1
            continue
        stem = Path(image["file_name"]).stem
        dst = image_out_dir / image["file_name"]
        link_or_copy(src, dst, link_mode)
        lines: list[str] = []
        for ann in anns_by_image.get(image_id, []):
            cid = int(ann["category_id"])
            line = yolo_line(ann["bbox"], int(image["width"]), int(image["height"]), cat_to_yolo[cid])
            if line:
                lines.append(line)
                ann_count += 1
        (label_out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return {
        "images": len(images_by_id) - missing_images,
        "annotations": ann_count,
        "missing_images": missing_images,
        "dropped_placeholder": dropped_placeholder,
        "dropped_unknown": dropped_unknown,
    }


def add_backgrounds(
    *,
    out_fold: Path,
    split: str,
    backgrounds: list[Path],
    variants_per_source: int,
    size: tuple[int, int],
    seed: int,
    quality: int,
) -> list[dict[str, Any]]:
    if variants_per_source <= 0 or not backgrounds:
        return []
    image_dir = out_fold / "images" / split
    label_dir = out_fold / "labels" / split
    rows: list[dict[str, Any]] = []
    for bg_idx, bg_path in enumerate(backgrounds):
        for variant in range(variants_per_source):
            rng = random.Random(seed + bg_idx * 1009 + variant * 9176)
            out_name = f"background__{bg_idx:03d}_{variant:03d}.jpg"
            out_image = image_dir / out_name
            make_background_variant(bg_path, out_image, size, rng, quality)
            (label_dir / f"{Path(out_name).stem}.txt").write_text("", encoding="utf-8")
            rows.append(
                {
                    "split": split,
                    "file_name": out_name,
                    "source_background": str(bg_path),
                    "label_file": str(label_dir / f"{Path(out_name).stem}.txt"),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    out = args.out.resolve()
    reset_dir(out)

    fold0_train = read_json(source / "fold0/train/_annotations.coco.json")
    mapping_rows = load_category_rows(source)
    cat_to_yolo, class_rows = make_class_map(fold0_train, mapping_rows)
    backgrounds = list_backgrounds(args.background_dir.resolve())

    with (out / "label_map_yolo74.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_rows)

    summary: dict[str, Any] = {
        "source": str(source),
        "out": str(out),
        "background_dir": str(args.background_dir.resolve()),
        "background_sources": len(backgrounds),
        "background_variants_per_source": args.background_variants_per_source,
        "background_valid_variants_per_source": args.background_valid_variants_per_source,
        "classes": len(class_rows),
        "folds": [],
    }
    bg_manifest: list[dict[str, Any]] = []
    for fold in range(args.folds):
        fold_dir = source / f"fold{fold}"
        out_fold = out / f"fold{fold}"
        train_stats = convert_split(
            fold_dir=fold_dir,
            split="train",
            out_fold=out_fold,
            source=source,
            cat_to_yolo=cat_to_yolo,
            link_mode=args.link_mode,
        )
        valid_stats = convert_split(
            fold_dir=fold_dir,
            split="valid",
            out_fold=out_fold,
            source=source,
            cat_to_yolo=cat_to_yolo,
            link_mode=args.link_mode,
        )
        fold_bg_train = add_backgrounds(
            out_fold=out_fold,
            split="train",
            backgrounds=backgrounds,
            variants_per_source=args.background_variants_per_source,
            size=(args.background_width, args.background_height),
            seed=args.seed + fold * 100000,
            quality=args.jpeg_quality,
        )
        fold_bg_valid = add_backgrounds(
            out_fold=out_fold,
            split="val",
            backgrounds=backgrounds,
            variants_per_source=args.background_valid_variants_per_source,
            size=(args.background_width, args.background_height),
            seed=args.seed + fold * 100000 + 50000,
            quality=args.jpeg_quality,
        )
        bg_manifest.extend({"fold": fold, **row} for row in fold_bg_train + fold_bg_valid)
        write_yaml(
            out_fold / "data.yaml",
            train_dir=out_fold / "images/train",
            val_dir=out_fold / "images/val",
            class_rows=class_rows,
        )
        summary["folds"].append(
            {
                "fold": fold,
                "train": train_stats,
                "valid": valid_stats,
                "background_train_images": len(fold_bg_train),
                "background_valid_images": len(fold_bg_valid),
                "data_yaml": str(out_fold / "data.yaml"),
            }
        )

    with (out / "background_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["fold", "split", "file_name", "source_background", "label_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bg_manifest)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        f"""# YOLO 74-class 5-fold dataset with background negatives

Source COCO folds: `{source}`

Background source folder: `{args.background_dir.resolve()}`

Real classes are mapped to YOLO class ids `0..73`; the COCO/RF-DETR background
placeholder category `0` is intentionally omitted. Background files are added
as images with empty label files, not as category-0 boxes.

Run example:

```bash
python YOLO/tools/convert_coco5fold_to_yolo_with_backgrounds.py
```

Per-fold YAML files:

```text
{out}/fold0/data.yaml
...
{out}/fold4/data.yaml
```

Use `label_map_yolo74.csv` to convert YOLO class ids back to competition
`category_id` values for submission.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
