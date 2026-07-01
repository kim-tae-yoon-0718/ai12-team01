#!/usr/bin/env python3
"""Draw crop contact sheets grouped by final test pseudo-GT N-number."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PSEUDO_GT = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/test_pseudo_ground_truth.csv"
DEFAULT_CLASS_MAP = PROJECT_ROOT / "working/reports/pill_class_number_map.csv"
DEFAULT_TEST_IMAGES = PROJECT_ROOT / "sprint_ai_project1_data/test_images"
DEFAULT_UNKNOWN_IGNORE = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/manual_unknown_ignore_boxes.csv"
DEFAULT_OUT = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/test_pseudo_gt_by_n_number_sheets"

FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT_PATH = next((Path(path) for path in FONT_CANDIDATES if Path(path).exists()), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo-gt", type=Path, default=DEFAULT_PSEUDO_GT)
    parser.add_argument("--class-map", type=Path, default=DEFAULT_CLASS_MAP)
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--unknown-ignore", type=Path, default=DEFAULT_UNKNOWN_IGNORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=156)
    parser.add_argument("--cell-width", type=int, default=170)
    parser.add_argument("--cell-height", type=int, default=232)
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    if FONT_PATH is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(FONT_PATH), size=size, index=0)


FONT_TITLE = font(24)
FONT_SUBTITLE = font(15)
FONT_LABEL = font(13)
FONT_TINY = font(11)


def n_sort_key(value: str) -> int:
    text = str(value).strip().upper()
    if text.startswith("N"):
        return int(text[1:])
    return int(float(text))


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def crop_with_padding(image: Image.Image, row: object, pad_ratio: float = 0.12) -> Image.Image:
    x = float(getattr(row, "bbox_x"))
    y = float(getattr(row, "bbox_y"))
    w = float(getattr(row, "bbox_w"))
    h = float(getattr(row, "bbox_h"))
    pad = max(w, h) * pad_ratio
    box = (
        max(0, int(math.floor(x - pad))),
        max(0, int(math.floor(y - pad))),
        min(image.width, int(math.ceil(x + w + pad))),
        min(image.height, int(math.ceil(y + h + pad))),
    )
    return image.crop(box).convert("RGB")


def fit_on_canvas(crop: Image.Image, size: int) -> Image.Image:
    thumb = ImageOps.contain(crop, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(thumb, ((size - thumb.width) // 2, (size - thumb.height) // 2))
    return canvas


def draw_cell(
    row: object,
    image: Image.Image,
    crop_size: int,
    cell_width: int,
    cell_height: int,
) -> Image.Image:
    pred = clean_text(getattr(row, "predicted_n_number"))
    correct = clean_text(getattr(row, "correct_n_number"))
    final = clean_text(getattr(row, "resolved_n_number"))
    corrected = bool(correct) and correct != pred
    source = clean_text(getattr(row, "source"))
    manual = source.startswith("manual_added")
    border = "#f97316" if corrected else ("#16a34a" if manual else "#64748b")
    bg = "#fff7ed" if corrected else ("#ecfdf5" if manual else "#f8fafc")

    cell = Image.new("RGB", (cell_width, cell_height), bg)
    draw = ImageDraw.Draw(cell)
    draw.rectangle((0, 0, cell_width - 1, cell_height - 1), outline=border, width=3)
    label = f"img {int(getattr(row, 'image_id'))} ann {int(getattr(row, 'source_annotation_id'))}"
    score = clean_text(getattr(row, "score"))
    score_text = f"s {float(score):.2f}" if score else "manual"
    transition = f"{pred}->{final}" if corrected else final
    draw.text((7, 6), label, font=FONT_LABEL, fill="#111827")
    draw.text((7, 23), f"{transition} {score_text}", font=FONT_LABEL, fill="#7f1d1d" if corrected else "#334155")
    if manual:
        draw.text((cell_width - 52, 23), "manual", font=FONT_TINY, fill="#166534")
    crop = fit_on_canvas(crop_with_padding(image, row), crop_size)
    cell.paste(crop, ((cell_width - crop_size) // 2, 48))
    return cell


def draw_sheet(
    rows: pd.DataFrame,
    n_number: str,
    drug_name: str,
    out_path: Path,
    test_images: Path,
    columns: int,
    crop_size: int,
    cell_width: int,
    cell_height: int,
) -> None:
    rows = rows.sort_values(["image_id", "source_annotation_id"], kind="stable")
    header_height = 78
    count = len(rows)
    sheet_width = columns * cell_width
    sheet_height = header_height + max(1, math.ceil(count / columns)) * cell_height
    sheet = Image.new("RGB", (sheet_width, sheet_height), "#e5e7eb")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet_width, header_height), fill="#111827")
    draw.text((18, 12), f"{n_number}  count={count}", font=FONT_TITLE, fill="white")
    draw.text((18, 43), drug_name[:90], font=FONT_SUBTITLE, fill="#d1d5db")

    image_cache: dict[int, Image.Image] = {}
    manifest_rows = []
    for idx, row in enumerate(rows.itertuples(index=False)):
        image_id = int(row.image_id)
        if image_id not in image_cache:
            image_cache[image_id] = Image.open(test_images / f"{image_id}.png").convert("RGB")
        cell = draw_cell(row, image_cache[image_id], crop_size, cell_width, cell_height)
        x = (idx % columns) * cell_width
        y = header_height + (idx // columns) * cell_height
        sheet.paste(cell, (x, y))
        manifest_rows.append(
            {
                "n_number": n_number,
                "sheet": str(out_path),
                "cell_index": idx + 1,
                "image_id": image_id,
                "source_annotation_id": int(row.source_annotation_id),
                "predicted_n_number": row.predicted_n_number,
                "correct_n_number": clean_text(row.correct_n_number),
                "resolved_n_number": row.resolved_n_number,
                "score": clean_text(row.score),
                "bbox_x": row.bbox_x,
                "bbox_y": row.bbox_y,
                "bbox_w": row.bbox_w,
                "bbox_h": row.bbox_h,
                "review_status": clean_text(row.review_status),
                "review_note": clean_text(row.review_note),
            }
        )
    sheet.save(out_path, quality=92)
    return manifest_rows


def draw_unknown_sheet(unknown: pd.DataFrame, out_path: Path, test_images: Path) -> list[dict[str, object]]:
    if unknown.empty:
        return []
    columns = 6
    cell_width = 220
    cell_height = 252
    crop_size = 170
    header_height = 78
    unknown = unknown.sort_values(["image_id", "previous_annotation_id"], kind="stable")
    sheet = Image.new("RGB", (columns * cell_width, header_height + math.ceil(len(unknown) / columns) * cell_height), "#fee2e2")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, sheet.width, header_height), fill="#7f1d1d")
    draw.text((18, 12), f"UNKNOWN_IGNORE  count={len(unknown)}", font=FONT_TITLE, fill="white")
    draw.text((18, 43), "Dropped pseudo-GT boxes kept as unknown/ignore evidence", font=FONT_SUBTITLE, fill="#fecaca")

    image_cache: dict[int, Image.Image] = {}
    manifest_rows = []
    for idx, row in enumerate(unknown.itertuples(index=False)):
        image_id = int(row.image_id)
        if image_id not in image_cache:
            image_cache[image_id] = Image.open(test_images / f"{image_id}.png").convert("RGB")
        cell = Image.new("RGB", (cell_width, cell_height), "#fff1f2")
        d = ImageDraw.Draw(cell)
        d.rectangle((0, 0, cell_width - 1, cell_height - 1), outline="#be123c", width=3)
        d.text((7, 6), f"img {image_id} ann {clean_text(row.previous_annotation_id)}", font=FONT_LABEL, fill="#111827")
        d.text((7, 24), f"prev {clean_text(row.previous_n_number)}", font=FONT_LABEL, fill="#7f1d1d")
        d.text((7, 42), clean_text(row.observed_imprint)[:28], font=FONT_TINY, fill="#334155")
        crop = fit_on_canvas(crop_with_padding(image_cache[image_id], row), crop_size)
        cell.paste(crop, ((cell_width - crop_size) // 2, 68))
        x = (idx % columns) * cell_width
        y = header_height + (idx // columns) * cell_height
        sheet.paste(cell, (x, y))
        manifest_rows.append(
            {
                "image_id": image_id,
                "previous_annotation_id": clean_text(row.previous_annotation_id),
                "previous_n_number": clean_text(row.previous_n_number),
                "observed_imprint": clean_text(row.observed_imprint),
                "sheet": str(out_path),
                "cell_index": idx + 1,
            }
        )
    sheet.save(out_path, quality=92)
    return manifest_rows


def draw_index(summary_rows: list[dict[str, object]], out_path: Path) -> None:
    columns = 2
    row_height = 34
    header_height = 66
    rows_per_col = math.ceil(len(summary_rows) / columns)
    width = 1400
    height = header_height + rows_per_col * row_height + 22
    sheet = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, width, header_height), fill="#0f172a")
    draw.text((18, 12), "Test pseudo-GT crop sheets by final N-number", font=FONT_TITLE, fill="white")
    draw.text((18, 43), "Use each sheet to audit within-class consistency after manual corrections", font=FONT_SUBTITLE, fill="#cbd5e1")
    col_width = width // columns
    for idx, row in enumerate(summary_rows):
        col = idx // rows_per_col
        local = idx % rows_per_col
        x = 18 + col * col_width
        y = header_height + local * row_height + 8
        fill = "#e2e8f0" if local % 2 == 0 else "#f8fafc"
        draw.rectangle((col * col_width, y - 4, (col + 1) * col_width - 10, y + row_height - 5), fill=fill)
        text = f"{row['n_number']}  count={row['count']:>3}  {row['drug_name']}"
        draw.text((x, y), text[:80], font=FONT_LABEL, fill="#111827")
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*"):
        if old.is_file() and old.suffix.lower() in {".jpg", ".csv", ".json"}:
            old.unlink()

    pseudo = pd.read_csv(args.pseudo_gt)
    class_map = pd.read_csv(args.class_map)
    meta = {
        f"N{int(row.class_no):02d}": {
            "drug_name": str(row.name),
            "category_id": int(row.category_id),
        }
        for row in class_map.itertuples(index=False)
    }

    manifest_rows = []
    summary_rows = []
    for n_number in sorted(pseudo["resolved_n_number"].dropna().unique(), key=n_sort_key):
        rows = pseudo[pseudo["resolved_n_number"].eq(n_number)].copy()
        drug_name = meta.get(n_number, {}).get("drug_name", str(rows["drug_name"].iloc[0]))
        out_path = args.out / f"{n_number}_count{len(rows):03d}.jpg"
        manifest_rows.extend(
            draw_sheet(
                rows,
                n_number,
                drug_name,
                out_path,
                args.test_images,
                args.columns,
                args.crop_size,
                args.cell_width,
                args.cell_height,
            )
        )
        summary_rows.append(
            {
                "n_number": n_number,
                "category_id": meta.get(n_number, {}).get("category_id", int(rows["category_id"].iloc[0])),
                "drug_name": drug_name,
                "count": int(len(rows)),
                "sheet": str(out_path),
            }
        )

    unknown_manifest = []
    if args.unknown_ignore.exists():
        unknown = pd.read_csv(args.unknown_ignore)
        unknown_manifest = draw_unknown_sheet(unknown, args.out / f"UNKNOWN_IGNORE_count{len(unknown):03d}.jpg", args.test_images)

    summary = pd.DataFrame(summary_rows).sort_values("n_number", key=lambda s: s.map(n_sort_key))
    summary_path = args.out / "manifest.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(manifest_rows).to_csv(args.out / "crop_manifest.csv", index=False)
    pd.DataFrame(unknown_manifest).to_csv(args.out / "unknown_ignore_crop_manifest.csv", index=False)
    draw_index(summary_rows, args.out / "00_index.jpg")

    payload = {
        "out_dir": str(args.out),
        "sheets": int(len(summary_rows)),
        "annotations": int(len(pseudo)),
        "unknown_ignore_boxes": int(len(unknown_manifest)),
        "manifest": str(summary_path),
        "crop_manifest": str(args.out / "crop_manifest.csv"),
        "index": str(args.out / "00_index.jpg"),
    }
    (args.out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
