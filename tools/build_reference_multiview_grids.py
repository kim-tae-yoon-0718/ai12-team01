#!/usr/bin/env python3
"""Build multi-view pill reference grids for manual test review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = PROJECT_ROOT / "working/train.csv"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "working/reports/pill_class_number_map.csv"
DEFAULT_TRAIN_IMAGES = PROJECT_ROOT / "sprint_ai_project1_data/train_images"
DEFAULT_OUT_DIR = PROJECT_ROOT / "working/reports/pill_reference_multiview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples-per-class", type=int, default=4)
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = load_font(22, True)
FONT_META = load_font(15)
FONT_SMALL = load_font(12)
FONT_SHEET_TITLE = load_font(28, True)


def angle_tag(file_name: str) -> str:
    parts = Path(str(file_name)).stem.split("_")
    if len(parts) >= 4 and parts[-3].isdigit():
        return parts[-3]
    return "na"


def angle_group_key(file_name: str) -> str:
    parts = Path(str(file_name)).stem.split("_")
    if len(parts) >= 4 and parts[-3].isdigit():
        return "_".join([*parts[:-3], "ANGLE", *parts[-2:]])
    return Path(str(file_name)).stem


def parse_bbox(row: pd.Series) -> tuple[int, int, int, int]:
    return (
        int(round(float(row["bbox_x"]))),
        int(round(float(row["bbox_y"]))),
        int(round(float(row["bbox_w"]))),
        int(round(float(row["bbox_h"]))),
    )


def crop_pill(image_path: Path, bbox: tuple[int, int, int, int], pad_ratio: float = 0.18) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    x, y, w, h = bbox
    pad = int(max(w, h) * pad_ratio)
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(image.width, x + w + pad)
    bottom = min(image.height, y + h + pad)
    return image.crop((left, top, right, bottom))


def paste_contained(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int] = (255, 255, 255),
) -> None:
    x, y, w, h = box
    bg = Image.new("RGB", (w, h), fill)
    thumb = image.copy()
    thumb.thumbnail((w - 8, h - 8), Image.Resampling.LANCZOS)
    bg.paste(thumb, ((w - thumb.width) // 2, (h - thumb.height) // 2))
    canvas.paste(bg, (x, y))


def text_fit(text: object, max_chars: int) -> str:
    value = "" if pd.isna(text) else str(text)
    return value if len(value) <= max_chars else value[: max_chars - 1] + "…"


def choose_samples(group: pd.DataFrame, n: int) -> pd.DataFrame:
    group = group.drop_duplicates(subset=["file_name", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]).copy()
    group["angle_tag"] = group["file_name"].map(angle_tag)
    group["angle_group_key"] = group["file_name"].map(angle_group_key)
    angle_rank = {"75": 0, "70": 1, "90": 2, "na": 3}
    group["angle_rank"] = group["angle_tag"].map(lambda value: angle_rank.get(str(value), 9))
    group = group.sort_values(["angle_group_key", "angle_rank", "annotation_id"], kind="stable").reset_index(drop=True)

    # Keep one crop per underlying scene first. The train set often contains the
    # same physical view at 70/75/90 degrees; taking all three hides front/back variants.
    reps = group.groupby("angle_group_key", sort=False).head(1).reset_index(drop=True)
    if len(reps) >= n:
        if n == 1:
            return reps.head(1).copy()
        positions = [round(i * (len(reps) - 1) / max(1, n - 1)) for i in range(n)]
        return reps.iloc[positions].drop_duplicates().head(n).copy()

    chosen = reps.copy()
    rep_keys = set(zip(reps["file_name"], reps["bbox_x"], reps["bbox_y"], reps["bbox_w"], reps["bbox_h"], strict=False))
    remaining = group[
        ~group[["file_name", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]]
        .apply(lambda row: tuple(row.tolist()) in rep_keys, axis=1)
    ].copy()
    if len(chosen) < n and not remaining.empty:
        chosen = pd.concat([chosen, remaining.head(n - len(chosen))], ignore_index=True)
    return chosen.head(n).copy()


def build_multiview_grid(selected: pd.DataFrame, class_map: pd.DataFrame, out_path: Path) -> None:
    card_w, card_h = 660, 300
    cols = 2
    rows = math.ceil(len(class_map) / cols)
    pad = 18
    header_h = 74
    canvas_w = cols * card_w + (cols + 1) * pad
    canvas_h = header_h + rows * card_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), "Pill Class Reference Multi-view", font=FONT_SHEET_TITLE, fill=(20, 24, 32))
    draw.text(
        (pad, 50),
        "Actual train crops only. Use this next to the compact numbered grid when imprints are rotated or faint.",
        font=FONT_META,
        fill=(84, 92, 106),
    )

    for idx, meta in enumerate(class_map.sort_values("class_no").itertuples(index=False)):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (card_w + pad)
        y0 = header_h + pad + row * (card_h + pad)
        draw.rounded_rectangle([x0, y0, x0 + card_w, y0 + card_h], radius=8, fill=(255, 255, 255), outline=(218, 224, 235), width=1)
        draw.rectangle([x0, y0, x0 + card_w, y0 + 38], fill=(232, 239, 252))
        header = f"N{int(meta.class_no):02d}  ID {int(meta.category_id):05d}  {text_fit(meta.name, 26)}"
        draw.text((x0 + 12, y0 + 8), header, font=FONT_TITLE, fill=(19, 27, 45))
        info = f"front={text_fit(meta.print_front, 14)}   back={text_fit(meta.print_back, 10)}   shape={text_fit(meta.shape, 8)}   color={text_fit(meta.color, 8)}   n={int(meta.count)}"
        draw.text((x0 + 12, y0 + 46), info, font=FONT_META, fill=(62, 72, 90))

        samples = selected[selected["category_id"].eq(int(meta.category_id))].sort_values("slot")
        thumb_y = y0 + 76
        thumb_w, thumb_h = 150, 165
        gap = 9
        for sample_idx, sample in enumerate(samples.itertuples(index=False)):
            sx = x0 + 12 + sample_idx * (thumb_w + gap)
            image_path = DEFAULT_TRAIN_IMAGES / str(sample.file_name)
            if not image_path.exists():
                continue
            crop = crop_pill(image_path, (int(sample.bbox_x), int(sample.bbox_y), int(sample.bbox_w), int(sample.bbox_h)))
            paste_contained(canvas, crop, (sx, thumb_y, thumb_w, thumb_h))
            draw.rectangle([sx, thumb_y, sx + thumb_w, thumb_y + thumb_h], outline=(198, 207, 220), width=1)
            label = f"{sample.angle_tag}deg · img {int(sample.image_id)}"
            draw.text((sx + 4, thumb_y + thumb_h + 6), label, font=FONT_SMALL, fill=(51, 63, 82))
            draw.text((sx + 4, thumb_y + thumb_h + 23), f"bbox {int(sample.bbox_w)}x{int(sample.bbox_h)}", font=FONT_SMALL, fill=(95, 105, 122))

    canvas.save(out_path, quality=95)


def build_rotation_grid(selected: pd.DataFrame, class_map: pd.DataFrame, out_path: Path) -> None:
    card_w, card_h = 520, 260
    cols = 3
    rows = math.ceil(len(class_map) / cols)
    pad = 16
    header_h = 76
    canvas_w = cols * card_w + (cols + 1) * pad
    canvas_h = header_h + rows * card_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), "Pill Class Reference Rotation Aid", font=FONT_SHEET_TITLE, fill=(20, 24, 32))
    draw.text(
        (pad, 52),
        "Augmented view for manual review only: each first crop is rotated 0/90/180/270 degrees.",
        font=FONT_META,
        fill=(84, 92, 106),
    )

    for idx, meta in enumerate(class_map.sort_values("class_no").itertuples(index=False)):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (card_w + pad)
        y0 = header_h + pad + row * (card_h + pad)
        draw.rounded_rectangle([x0, y0, x0 + card_w, y0 + card_h], radius=8, fill=(255, 255, 255), outline=(218, 224, 235), width=1)
        draw.rectangle([x0, y0, x0 + card_w, y0 + 34], fill=(236, 245, 240))
        header = f"N{int(meta.class_no):02d}  {text_fit(meta.name, 23)}"
        draw.text((x0 + 10, y0 + 7), header, font=FONT_TITLE, fill=(19, 27, 45))
        draw.text((x0 + 10, y0 + 40), f"front={text_fit(meta.print_front, 12)}  back={text_fit(meta.print_back, 8)}", font=FONT_META, fill=(62, 72, 90))

        samples = selected[selected["category_id"].eq(int(meta.category_id))].sort_values("slot")
        if samples.empty:
            continue
        sample = samples.iloc[0]
        image_path = DEFAULT_TRAIN_IMAGES / str(sample["file_name"])
        if not image_path.exists():
            continue
        crop = crop_pill(
            image_path,
            (int(sample["bbox_x"]), int(sample["bbox_y"]), int(sample["bbox_w"]), int(sample["bbox_h"])),
        )
        thumb_w, thumb_h = 112, 136
        y = y0 + 75
        for rot_idx, deg in enumerate([0, 90, 180, 270]):
            x = x0 + 10 + rot_idx * (thumb_w + 12)
            rotated = crop.rotate(deg, expand=True, fillcolor=(255, 255, 255))
            paste_contained(canvas, rotated, (x, y, thumb_w, thumb_h))
            draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(198, 207, 220), width=1)
            draw.text((x + 6, y + thumb_h + 6), f"aug {deg}deg", font=FONT_SMALL, fill=(51, 63, 82))

    canvas.save(out_path, quality=95)


def build_per_class_sheets(selected: pd.DataFrame, class_map: pd.DataFrame, out_dir: Path) -> None:
    sheet_dir = out_dir / "per_class_selected_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    for meta in class_map.sort_values("class_no").itertuples(index=False):
        samples = selected[selected["category_id"].eq(int(meta.category_id))].sort_values("slot")
        if samples.empty:
            continue
        cell_w, cell_h = 210, 240
        canvas = Image.new("RGB", (len(samples) * cell_w, cell_h), (247, 248, 250))
        draw = ImageDraw.Draw(canvas)
        for idx, sample in enumerate(samples.itertuples(index=False)):
            x0 = idx * cell_w
            image_path = DEFAULT_TRAIN_IMAGES / str(sample.file_name)
            if not image_path.exists():
                continue
            crop = crop_pill(image_path, (int(sample.bbox_x), int(sample.bbox_y), int(sample.bbox_w), int(sample.bbox_h)))
            paste_contained(canvas, crop, (x0 + 10, 10, cell_w - 20, 165))
            draw.rectangle([x0 + 10, 10, x0 + cell_w - 10, 175], outline=(198, 207, 220), width=1)
            draw.text((x0 + 10, 184), f"N{int(meta.class_no):02d} {sample.angle_tag}deg img {int(sample.image_id)}", font=FONT_SMALL, fill=(25, 32, 45))
            draw.text((x0 + 10, 204), text_fit(sample.file_name, 27), font=FONT_SMALL, fill=(80, 88, 104))
        canvas.save(sheet_dir / f"N{int(meta.class_no):02d}_{int(meta.category_id):05d}.jpg", quality=95)


def meaningful_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() == "nan" else text


def two_sided_classes(class_map: pd.DataFrame) -> pd.DataFrame:
    out = class_map.copy()
    out["print_front_clean"] = out["print_front"].map(meaningful_text)
    out["print_back_clean"] = out["print_back"].map(meaningful_text)
    out = out[
        out["print_front_clean"].ne("")
        & out["print_back_clean"].ne("")
        & out["print_front_clean"].ne(out["print_back_clean"])
    ].copy()
    return out.sort_values("class_no").reset_index(drop=True)


def build_two_sided_grid(
    train: pd.DataFrame,
    class_map: pd.DataFrame,
    train_images: Path,
    out_path: Path,
    samples_per_class: int = 8,
) -> pd.DataFrame:
    two_sided = two_sided_classes(class_map)
    selected_rows: list[dict[str, object]] = []
    for meta in two_sided.itertuples(index=False):
        group = train[train["category_id"].astype(int).eq(int(meta.category_id))].copy()
        samples = choose_samples(group, samples_per_class)
        for slot, (_, sample) in enumerate(samples.iterrows(), 1):
            selected_rows.append(
                {
                    "class_no": int(meta.class_no),
                    "category_id": int(meta.category_id),
                    "name": str(meta.name),
                    "front_imprint": meaningful_text(meta.print_front),
                    "back_imprint": meaningful_text(meta.print_back),
                    "slot": slot,
                    "annotation_id": int(sample.annotation_id),
                    "image_id": int(sample.image_id),
                    "file_name": str(sample.file_name),
                    "angle_tag": angle_tag(str(sample.file_name)),
                    "bbox_x": int(round(float(sample.bbox_x))),
                    "bbox_y": int(round(float(sample.bbox_y))),
                    "bbox_w": int(round(float(sample.bbox_w))),
                    "bbox_h": int(round(float(sample.bbox_h))),
                    "image_path": str(train_images / str(sample.file_name)),
                }
            )
    selected = pd.DataFrame(selected_rows)

    card_w, card_h = 760, 540
    cols = 2
    rows = math.ceil(len(two_sided) / cols)
    pad = 18
    header_h = 82
    canvas_w = cols * card_w + (cols + 1) * pad
    canvas_h = header_h + rows * card_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), "Two-sided Imprint Reference", font=FONT_SHEET_TITLE, fill=(20, 24, 32))
    draw.text(
        (pad, 52),
        "Actual train crops for classes whose front/back imprints differ. This is the grid to use for flipped-over pills.",
        font=FONT_META,
        fill=(84, 92, 106),
    )

    for idx, meta in enumerate(two_sided.itertuples(index=False)):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (card_w + pad)
        y0 = header_h + pad + row * (card_h + pad)
        draw.rounded_rectangle([x0, y0, x0 + card_w, y0 + card_h], radius=8, fill=(255, 255, 255), outline=(218, 224, 235), width=1)
        draw.rectangle([x0, y0, x0 + card_w, y0 + 42], fill=(255, 244, 229))
        header = f"N{int(meta.class_no):02d}  ID {int(meta.category_id):05d}  {text_fit(meta.name, 28)}"
        draw.text((x0 + 12, y0 + 9), header, font=FONT_TITLE, fill=(19, 27, 45))

        front = meaningful_text(meta.print_front)
        back = meaningful_text(meta.print_back)
        draw.text((x0 + 12, y0 + 52), f"FRONT: {front}", font=FONT_TITLE, fill=(16, 94, 50))
        draw.text((x0 + 310, y0 + 52), f"BACK: {back}", font=FONT_TITLE, fill=(126, 58, 10))
        draw.text(
            (x0 + 12, y0 + 82),
            f"shape={text_fit(meta.shape, 8)}  color={text_fit(meta.color, 10)}  train crops={int(meta.count)}",
            font=FONT_META,
            fill=(62, 72, 90),
        )

        samples = selected[selected["category_id"].eq(int(meta.category_id))].sort_values("slot")
        thumb_w, thumb_h = 172, 170
        gap = 10
        start_y = y0 + 112
        for sample_idx, sample in enumerate(samples.itertuples(index=False)):
            sx = x0 + 12 + (sample_idx % 4) * (thumb_w + gap)
            sy = start_y + (sample_idx // 4) * (thumb_h + 34)
            image_path = train_images / str(sample.file_name)
            if not image_path.exists():
                continue
            crop = crop_pill(image_path, (int(sample.bbox_x), int(sample.bbox_y), int(sample.bbox_w), int(sample.bbox_h)))
            paste_contained(canvas, crop, (sx, sy, thumb_w, thumb_h))
            draw.rectangle([sx, sy, sx + thumb_w, sy + thumb_h], outline=(198, 207, 220), width=1)
            label = f"{sample.angle_tag}deg · img {int(sample.image_id)}"
            draw.text((sx + 4, sy + thumb_h + 5), label, font=FONT_SMALL, fill=(51, 63, 82))

    canvas.save(out_path, quality=95)
    return selected


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_csv)
    class_map = pd.read_csv(args.class_map)

    selected_rows: list[dict[str, object]] = []
    for meta in class_map.sort_values("class_no").itertuples(index=False):
        group = train[train["category_id"].astype(int).eq(int(meta.category_id))].copy()
        samples = choose_samples(group, args.samples_per_class)
        for slot, (_, sample) in enumerate(samples.iterrows(), 1):
            selected_rows.append(
                {
                    "class_no": int(meta.class_no),
                    "category_id": int(meta.category_id),
                    "name": str(meta.name),
                    "slot": slot,
                    "annotation_id": int(sample.annotation_id),
                    "image_id": int(sample.image_id),
                    "file_name": str(sample.file_name),
                    "angle_tag": angle_tag(str(sample.file_name)),
                    "bbox_x": int(round(float(sample.bbox_x))),
                    "bbox_y": int(round(float(sample.bbox_y))),
                    "bbox_w": int(round(float(sample.bbox_w))),
                    "bbox_h": int(round(float(sample.bbox_h))),
                    "image_path": str(args.train_images / str(sample.file_name)),
                }
            )
    selected = pd.DataFrame(selected_rows)
    selected_csv = args.out_dir / "pill_class_reference_multiview_samples.csv"
    selected.to_csv(selected_csv, index=False)

    multiview_grid = args.out_dir / "pill_class_reference_grid_multiview.png"
    rotation_grid = args.out_dir / "pill_class_reference_grid_rotation_augmented.png"
    two_sided_grid = args.out_dir / "pill_class_reference_grid_two_sided_imprints.png"
    build_multiview_grid(selected, class_map, multiview_grid)
    build_rotation_grid(selected, class_map, rotation_grid)
    build_per_class_sheets(selected, class_map, args.out_dir)
    two_sided_selected = build_two_sided_grid(train, class_map, args.train_images, two_sided_grid)
    two_sided_selected_csv = args.out_dir / "pill_class_reference_two_sided_samples.csv"
    two_sided_selected.to_csv(two_sided_selected_csv, index=False)
    two_sided_csv = args.out_dir / "pill_class_reference_two_sided_classes.csv"
    two_sided_classes(class_map).to_csv(two_sided_csv, index=False)

    summary = {
        "out_dir": str(args.out_dir),
        "multiview_grid": str(multiview_grid),
        "rotation_augmented_grid": str(rotation_grid),
        "two_sided_grid": str(two_sided_grid),
        "selected_samples_csv": str(selected_csv),
        "two_sided_classes_csv": str(two_sided_csv),
        "two_sided_samples_csv": str(two_sided_selected_csv),
        "classes": int(class_map["category_id"].nunique()),
        "two_sided_classes": int(len(two_sided_classes(class_map))),
        "selected_samples": int(len(selected)),
        "samples_per_class": int(args.samples_per_class),
        "note": "Rotation grid is augmented for human review only; do not treat rotations as new annotations.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
