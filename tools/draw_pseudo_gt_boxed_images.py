#!/usr/bin/env python3
"""Draw boxed pseudo-GT images from test_pseudo_ground_truth.csv."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PSEUDO_GT = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/test_pseudo_ground_truth.csv"
DEFAULT_TEST_IMAGES = PROJECT_ROOT / "sprint_ai_project1_data/test_images"
DEFAULT_OUT = PROJECT_ROOT / "working/reports/test_pseudo_gt_eval/pseudo_gt_boxed_images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo-gt", type=Path, default=DEFAULT_PSEUDO_GT)
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT_PATH = next((Path(path) for path in FONT_CANDIDATES if Path(path).exists()), None)


def font(size: int) -> ImageFont.ImageFont:
    if FONT_PATH is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(FONT_PATH), size=size, index=0)


FONT_LABEL = font(22)
FONT_SMALL = font(14)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=FONT_LABEL)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    top = max(0, y - th - 8)
    draw.rounded_rectangle((x, top, min(x + tw + 12, 975), top + th + 8), radius=4, fill=color)
    draw.text((x + 6, top + 4), text, font=FONT_LABEL, fill="white")


def make_contact_sheet(image_paths: list[Path], out_path: Path) -> None:
    thumbs = []
    for path in image_paths[:24]:
        img = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(img, (240, 315), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (240, 315), "white")
        canvas.paste(thumb, ((240 - thumb.width) // 2, (315 - thumb.height) // 2))
        d = ImageDraw.Draw(canvas)
        d.text((8, 8), path.stem, font=FONT_SMALL, fill="#111827")
        thumbs.append(canvas)
    cols = 6
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows * 315), "#f8fafc")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 240, (idx // cols) * 315))
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    if args.out.exists():
        for old in args.out.glob("*.jpg"):
            old.unlink()
    args.out.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(args.pseudo_gt)
    manifest = []
    colors = ["#2563eb", "#dc2626", "#16a34a", "#f97316", "#9333ea", "#0891b2", "#be123c", "#4d7c0f"]
    output_paths: list[Path] = []

    for image_path in sorted(args.test_images.glob("*.png"), key=lambda p: int(p.stem)):
        image_id = int(image_path.stem)
        rows = gt[gt["image_id"].astype(int).eq(image_id)].copy()
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for idx, row in enumerate(rows.itertuples(index=False)):
            color = colors[idx % len(colors)]
            x, y, w, h = [float(getattr(row, col)) for col in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]]
            draw.rectangle((x, y, x + w, y + h), outline=color, width=5)
            source = str(row.source)
            marker = " [manual]" if source.startswith("manual_added") else ""
            text = f"{row.resolved_n_number} {int(row.category_id):05d} {str(row.drug_name)[:15]}{marker}"
            draw_label(draw, (x, y), text, color)
            manifest.append(
                {
                    "image_id": image_id,
                    "output_file": str(args.out / f"{image_id}.jpg"),
                    "resolved_n_number": row.resolved_n_number,
                    "category_id": int(row.category_id),
                    "drug_name": row.drug_name,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "source": source,
                }
            )
        out_path = args.out / f"{image_id}.jpg"
        img.save(out_path, quality=92)
        output_paths.append(out_path)

    manifest_path = args.out / "pseudo_gt_boxed_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    preview_path = args.out / "pseudo_gt_preview_contact_sheet.jpg"
    make_contact_sheet(output_paths, preview_path)
    summary = {
        "output_dir": str(args.out),
        "images": len(output_paths),
        "boxes": len(manifest),
        "manual_added_boxes": int(pd.Series([m["source"] for m in manifest]).str.startswith("manual_added").sum())
        if manifest
        else 0,
        "manifest": str(manifest_path),
        "preview_contact_sheet": str(preview_path),
    }
    (args.out / "pseudo_gt_boxed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
