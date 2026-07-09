#!/usr/bin/env python3
"""Train Ultralytics YOLO on the prepared 74-class 5-fold MPS dataset.

Default model is YOLO11m for the main local MPS experiment. Use YOLO11n/s for
smoke runs or YOLO11l if local resources permit.

The prepared dataset already includes background-only images as empty YOLO
label files. By default this runner also enables a local Ultralytics loss patch
that keeps all foreground anchors and only hard background anchors for
classification loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


YOLO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = YOLO_ROOT.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", os.environ.get("PROJECT_ROOT", REPO_ROOT))).resolve()
DEFAULT_DATASET = DATA_ROOT / "working/yolo_74_5fold_bg_mps"
DEFAULT_OUTPUT = DATA_ROOT / "working/yolo_outputs/yolo11m_74_5fold_bg_mps_10ep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="yolo11m.pt", help="Examples: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt")
    parser.add_argument("--folds", default="0,1,2,3,4", help="Comma-separated fold ids.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--select-best-map75-95", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-ohem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-ohem", action="store_true", help="Abort unless the loss-level OHEM patch is enabled.")
    parser.add_argument("--ohem-negative-ratio", type=float, default=0.25)
    parser.add_argument("--ohem-min-neg", type=int, default=16)
    parser.add_argument("--ohem-max-neg", type=int, default=2048)
    parser.add_argument("--ohem-bg-neg", type=int, default=32)
    parser.add_argument("--ohem-neg-weight", type=float, default=0.25)
    parser.add_argument("--ohem-note", default="foreground anchors kept; hard background anchors only")
    return parser.parse_args()


def enable_ohem_patch(args: argparse.Namespace) -> dict:
    if not args.enable_ohem:
        return {"enabled": False, "reason": "--no-enable-ohem"}
    try:
        from yolo_ohem_patch import enable_yolo_detection_ohem

        return enable_yolo_detection_ohem(
            negative_ratio=args.ohem_negative_ratio,
            min_neg_per_image=args.ohem_min_neg,
            max_neg_per_image=args.ohem_max_neg,
            background_only_neg_per_image=args.ohem_bg_neg,
            negative_loss_weight=args.ohem_neg_weight,
        )
    except Exception as exc:
        if args.require_ohem:
            raise
        return {"enabled": False, "reason": f"{type(exc).__name__}: {exc}"}


def import_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(
            "ultralytics is not installed in this Python environment.\n"
            "Install once with:\n"
            "  python -m pip install -U ultralytics\n"
            f"Original error: {exc}"
        ) from exc
    return YOLO


def metric_map75_95(metrics) -> float:
    import numpy as np

    all_ap = getattr(getattr(metrics, "box", None), "all_ap", None)
    if all_ap is None or len(all_ap) == 0:
        return 0.0
    return float(np.asarray(all_ap)[:, 5:].mean())


def select_best_map75_95(*, yolo_cls, fold_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict:
    """Validate saved checkpoints and copy the best IoU 0.75:0.95 checkpoint."""

    weights_dir = fold_dir / "weights"
    candidates = sorted(weights_dir.glob("epoch*.pt"))
    for extra in [weights_dir / "best.pt", weights_dir / "last.pt"]:
        if extra.exists() and extra not in candidates:
            candidates.append(extra)
    rows = []
    best = {"score": -1.0, "path": None}
    for ckpt in candidates:
        print(f"validating for map75_95: {ckpt}", flush=True)
        model = yolo_cls(str(ckpt))
        metrics = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            plots=False,
            verbose=False,
        )
        score = metric_map75_95(metrics)
        row = {
            "checkpoint": str(ckpt),
            "map75_95": score,
            "map75": float(getattr(metrics.box, "map75", 0.0)),
            "map50_95": float(getattr(metrics.box, "map", 0.0)),
            "map50": float(getattr(metrics.box, "map50", 0.0)),
        }
        rows.append(row)
        if score > best["score"]:
            best = {"score": score, "path": ckpt}

    import csv
    import shutil

    out_csv = fold_dir / "best_map75_95_selection.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if best["path"] is not None:
        out_pt = weights_dir / "best_map75_95.pt"
        shutil.copy2(best["path"], out_pt)
        summary = {
            "best_checkpoint": str(best["path"]),
            "best_map75_95": best["score"],
            "copied_to": str(out_pt),
            "selection_csv": str(out_csv),
            "note": "Selected by mean AP over IoU thresholds 0.75,0.80,0.85,0.90,0.95.",
        }
    else:
        summary = {
            "best_checkpoint": None,
            "best_map75_95": None,
            "copied_to": None,
            "selection_csv": str(out_csv),
            "note": "No checkpoint candidates found.",
        }
    (fold_dir / "best_map75_95_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    ohem_patch = enable_ohem_patch(args)
    if args.require_ohem and not ohem_patch.get("enabled"):
        raise SystemExit(f"Loss-level OHEM patch is not enabled: {ohem_patch}")

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "output": str(output),
        "model": args.model,
        "folds": folds,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "ohem_patch": ohem_patch,
        "ohem_negative_ratio": args.ohem_negative_ratio,
        "ohem_min_neg": args.ohem_min_neg,
        "ohem_max_neg": args.ohem_max_neg,
        "ohem_bg_neg": args.ohem_bg_neg,
        "ohem_neg_weight": args.ohem_neg_weight,
        "ohem_note": args.ohem_note,
        "best_checkpoint_metric": "map75_95" if args.select_best_map75_95 else "ultralytics_default_best",
        "augmentation": {
            "scale": [args.scale_min, args.scale_max],
            "translate": 0.04,
            "degrees": 25.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.25,
            "hsv_v": 0.20,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_config, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("Dry run only. No training launched.")
        return

    YOLO = import_yolo()
    for fold in folds:
        data_yaml = dataset / f"fold{fold}" / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(data_yaml)
        print(f"\\n=== Training fold {fold}: {data_yaml} ===", flush=True)
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            project=str(output),
            name=f"fold{fold}",
            exist_ok=True,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed + fold,
            pretrained=True,
            optimizer="auto",
            cos_lr=True,
            amp=False,
            cache=False,
            plots=True,
            save=True,
            save_period=1,
            val=True,
            verbose=True,
            resume=args.resume,
            degrees=25.0,
            translate=0.04,
            scale=(args.scale_min, args.scale_max),
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            hsv_h=0.015,
            hsv_s=0.25,
            hsv_v=0.20,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            close_mosaic=0,
        )
        if args.select_best_map75_95:
            select_best_map75_95(
                yolo_cls=YOLO,
                fold_dir=output / f"fold{fold}",
                data_yaml=data_yaml,
                args=args,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
