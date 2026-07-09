#!/usr/bin/env python3
"""Audit YOLO-vs-RF-DETR training parameters before running YOLO."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RFDETR_REPO = Path(os.environ.get("RFDETR_REPO", PROJECT_ROOT.parent / "ai12-team01-rfdetr"))
DEFAULT_RFDETR_CONFIG = (
    DEFAULT_RFDETR_REPO
    / "RF_DETR_split_ver"
    / "config_74_hidden45_mps_rfdetr_large_v18_5fold_p00.yaml"
)
DEFAULT_RFDETR_FOLDS = PROJECT_ROOT / "working/rfdetr_dataset_74_hidden45_canvas_balanced_5fold_cls0_mps"
DEFAULT_YOLO_DATASET = PROJECT_ROOT / "working/yolo_74_5fold_bg_mps"
DEFAULT_OUTPUT = PROJECT_ROOT / "working/reports/yolo_rfdetr_param_audit_20260709"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rfdetr-config", type=Path, default=DEFAULT_RFDETR_CONFIG)
    parser.add_argument("--rfdetr-folds", type=Path, default=DEFAULT_RFDETR_FOLDS)
    parser.add_argument("--yolo-dataset", type=Path, default=DEFAULT_YOLO_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--yolo-model", default="yolo11m.pt")
    parser.add_argument("--yolo-epochs", type=int, default=10)
    parser.add_argument("--yolo-imgsz", type=int, default=960)
    parser.add_argument("--yolo-batch", type=int, default=4)
    parser.add_argument("--yolo-workers", type=int, default=2)
    parser.add_argument("--yolo-device", default="mps")
    parser.add_argument("--yolo-scale-min", type=float, default=0.9)
    parser.add_argument("--yolo-scale-max", type=float, default=1.0)
    parser.add_argument("--yolo-best-metric", default="map75_95")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare_class_maps(rfdetr_folds: Path, yolo_dataset: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rf_rows: dict[int, dict[str, str]] = {}
    with (rfdetr_folds / "category_mapping.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("is_placeholder") == "True":
                continue
            rf_rows[int(row["category_id"])] = row

    yolo_rows: dict[int, dict[str, str]] = {}
    with (yolo_dataset / "label_map_yolo74.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            yolo_rows[int(row["submission_category_id"])] = row

    all_ids = sorted(set(rf_rows) | set(yolo_rows))
    rows: list[dict[str, Any]] = []
    mismatches = 0
    for cat_id in all_ids:
        rf = rf_rows.get(cat_id, {})
        yo = yolo_rows.get(cat_id, {})
        ok = bool(rf) and bool(yo) and rf.get("n_number") == yo.get("n_number")
        if not ok:
            mismatches += 1
        rows.append(
            {
                "submission_category_id": cat_id,
                "rfdetr_internal_label": rf.get("rfdetr_internal_label", ""),
                "yolo_class": yo.get("yolo_class", ""),
                "rfdetr_n_number": rf.get("n_number", ""),
                "yolo_n_number": yo.get("n_number", ""),
                "rfdetr_name": rf.get("name", ""),
                "yolo_name": yo.get("drug_name", ""),
                "ok": ok,
            }
        )
    return rows, {"class_count": len(all_ids), "mismatches": mismatches}


def count_coco_annotations(fold_dir: Path, fold: int, split: str) -> int:
    coco = read_json(fold_dir / f"fold{fold}" / split / "_annotations.coco.json")
    return len(coco.get("annotations", []))


def fold_rows(rfdetr_ready: dict[str, Any], yolo_summary: dict[str, Any], rfdetr_folds: Path) -> list[dict[str, Any]]:
    yolo_by_fold = {int(row["fold"]): row for row in yolo_summary["folds"]}
    rows = []
    for rf in rfdetr_ready["folds"]:
        fold = int(rf["fold"])
        yo = yolo_by_fold[fold]
        rf_train_annotations = count_coco_annotations(rfdetr_folds, fold, "train")
        rf_valid_annotations = count_coco_annotations(rfdetr_folds, fold, "valid")
        rows.append(
            {
                "fold": fold,
                "rfdetr_train_images": rf["train_images"],
                "yolo_train_images": yo["train"]["images"] + yo["background_train_images"],
                "background_train_images": yo["background_train_images"],
                "rfdetr_valid_images": rf["valid_images"],
                "yolo_valid_images": yo["valid"]["images"] + yo["background_valid_images"],
                "rfdetr_train_annotations": rf_train_annotations,
                "yolo_train_annotations": yo["train"]["annotations"],
                "rfdetr_valid_annotations": rf_valid_annotations,
                "yolo_valid_annotations": yo["valid"]["annotations"],
                "train_annotation_count_match": rf_train_annotations == yo["train"]["annotations"],
                "valid_annotation_count_match": rf_valid_annotations == yo["valid"]["annotations"],
                "group_leakage": rf["group_leakage"],
                "rfdetr_valid_canvas_images": rf["valid_canvas_images"],
                "yolo_background_valid_images": yo["background_valid_images"],
                "valid_classes": rf["valid_classes"],
                "valid_min_class_annotations": rf["valid_min_class_annotations"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rf_cfg = read_yaml(args.rfdetr_config)
    rf_ready = read_json(args.rfdetr_folds / "_5fold_ready.json")
    yo_summary = read_json(args.yolo_dataset / "summary.json")

    rf_train = rf_cfg["train"]
    rf_model = rf_cfg["model"]
    param_rows = [
        {
            "area": "model",
            "setting": "base model",
            "rfdetr": rf_model.get("variant"),
            "yolo": args.yolo_model,
            "audit": "intentional_change",
            "note": "YOLO11m is the proposed faster local detector; RF-DETR used large.",
        },
        {
            "area": "training",
            "setting": "epochs",
            "rfdetr": rf_train.get("epochs"),
            "yolo": args.yolo_epochs,
            "audit": "intentional_change",
            "note": "YOLO pilot run is capped at 10 epochs; RF-DETR config was long-run/early-stop oriented.",
        },
        {
            "area": "training",
            "setting": "effective batch",
            "rfdetr": f"{rf_train.get('batch_size')} x accum {rf_train.get('grad_accum_steps')}",
            "yolo": args.yolo_batch,
            "audit": "watch",
            "note": "YOLO MPS batch 4 may need reduction to 2 if memory errors occur.",
        },
        {
            "area": "image_size",
            "setting": "input resolution",
            "rfdetr": rf_train.get("resolution"),
            "yolo": args.yolo_imgsz,
            "audit": "intentional_change",
            "note": "YOLO uses larger input to preserve imprint detail.",
        },
        {
            "area": "device",
            "setting": "device/workers",
            "rfdetr": f"{rf_train.get('device')}/workers={rf_train.get('num_workers')}",
            "yolo": f"{args.yolo_device}/workers={args.yolo_workers}",
            "audit": "aligned",
            "note": "Both local MPS with workers=2.",
        },
        {
            "area": "augmentation",
            "setting": "rotation",
            "rfdetr": rf_train.get("aug_config", {}).get("Affine", {}).get("rotate"),
            "yolo": "degrees=25",
            "audit": "watch",
            "note": "RF-DETR allowed ±90; YOLO starts lower to avoid destabilizing 10-epoch pilot.",
        },
        {
            "area": "augmentation",
            "setting": "scale/translate",
            "rfdetr": f"scale={rf_train.get('aug_config', {}).get('Affine', {}).get('scale')}, translate={rf_train.get('aug_config', {}).get('Affine', {}).get('translate_percent')}",
            "yolo": f"scale=({args.yolo_scale_min}, {args.yolo_scale_max}), translate=0.04",
            "audit": "roughly_aligned",
            "note": "Ultralytics RandomPerspective accepts tuple scale as absolute min/max factors.",
        },
        {
            "area": "checkpoint",
            "setting": "best checkpoint metric",
            "rfdetr": "map75 wrapper selection",
            "yolo": args.yolo_best_metric,
            "audit": "aligned",
            "note": "YOLO runner validates saved epoch checkpoints and copies best_map75_95.pt.",
        },
        {
            "area": "background",
            "setting": "background handling",
            "rfdetr": "class0 placeholder only; no class0 boxes",
            "yolo": f"{yo_summary['background_sources']} bg sources x {yo_summary['background_variants_per_source']} per fold as empty labels",
            "audit": "intentional_change",
            "note": "YOLO trains real background negatives without fake category-0 boxes.",
        },
        {
            "area": "loss",
            "setting": "OHEM",
            "rfdetr": "not enabled in RF-DETR config",
            "yolo": "OHEMv8DetectionLoss patch enabled by default",
            "audit": "intentional_change",
            "note": "Foreground anchors kept; only hard background anchors contribute to cls loss.",
        },
        {
            "area": "validation",
            "setting": "validation synthetic/bg",
            "rfdetr": f"valid_canvas={rf_ready.get('valid_include_canvas')}",
            "yolo": "background_valid_images=0",
            "audit": "aligned",
            "note": "Validation stays clean; train gets background negatives only.",
        },
    ]
    write_csv(out / "parameter_crosscheck.csv", param_rows)

    class_rows, class_summary = compare_class_maps(args.rfdetr_folds, args.yolo_dataset)
    write_csv(out / "class_mapping_crosscheck.csv", class_rows)

    folds = fold_rows(rf_ready, yo_summary, args.rfdetr_folds)
    write_csv(out / "fold_data_crosscheck.csv", folds)

    summary = {
        "rfdetr_config": str(args.rfdetr_config),
        "rfdetr_folds": str(args.rfdetr_folds),
        "yolo_dataset": str(args.yolo_dataset),
        "output": str(out),
        "class_summary": class_summary,
        "fold_count": len(folds),
        "warnings": [
            row for row in param_rows if row["audit"] in {"watch"}
        ],
        "blocking_issues": [
            "class_mapping_mismatch" if class_summary["mismatches"] else "",
            "fold_annotation_count_mismatch"
            if any(not row["train_annotation_count_match"] or not row["valid_annotation_count_match"] for row in folds)
            else "",
        ],
    }
    summary["blocking_issues"] = [x for x in summary["blocking_issues"] if x]
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# YOLO vs RF-DETR Parameter Crosscheck",
        "",
        f"- RF-DETR config: `{args.rfdetr_config}`",
        f"- RF-DETR folds: `{args.rfdetr_folds}`",
        f"- YOLO dataset: `{args.yolo_dataset}`",
        f"- Class mapping mismatches: `{class_summary['mismatches']}` / `{class_summary['class_count']}`",
        "",
        "## Key Decisions",
        "",
        "- Use `yolo11m.pt` for the 10-epoch local MPS pilot.",
        "- Add background negatives as empty-label images, not category-0 boxes.",
        "- Enable OHEM for classification loss so easy background anchors are downweighted/ignored.",
        "- Keep validation clean: no background negatives in valid.",
        "- Use YOLO `imgsz=960` to preserve imprint detail, while RF-DETR used `resolution=384`.",
        "",
        "## Watch Items",
        "",
    ]
    for row in summary["warnings"]:
        md.append(f"- `{row['setting']}`: {row['note']}")
    if summary["blocking_issues"]:
        md.extend(["", "## Blocking Issues", ""])
        for issue in summary["blocking_issues"]:
            md.append(f"- {issue}")
    else:
        md.extend(["", "No blocking issues found in class mapping or fold structure."])
    (out / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
