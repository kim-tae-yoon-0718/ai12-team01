#!/usr/bin/env python3
"""Train RF-DETR on prepared 45-fill datasets without k-fold wrapping."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from model import get_rfdetr_model


SPECIAL_TRAIN_KWARGS = {"device", "resolution"}
EXPLICIT_TRAIN_KWARGS = {"dataset_dir", "output_dir"}
FALLBACK_TRAIN_KWARGS = {
    "epochs",
    "batch_size",
    "grad_accum_steps",
    "auto_batch_target_effective",
    "auto_batch_max_targets_per_image",
    "auto_batch_ema_headroom",
    "lr",
    "lr_encoder",
    "weight_decay",
    "lr_scheduler",
    "warmup_epochs",
    "lr_min_factor",
    "checkpoint_interval",
    "eval_interval",
    "early_stopping",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "early_stopping_use_ema",
    "use_ema",
    "tensorboard",
    "progress_bar",
    "num_workers",
    "persistent_workers",
    "prefetch_factor",
    "pin_memory",
    "amp_dtype",
    "seed",
    "multi_scale",
    "expanded_scales",
    "aug_config",
    "augmentation_backend",
    "log_per_class_metrics",
    "run_test",
    "compute_val_loss",
    "compute_test_loss",
    "eval_max_dets",
    "class_names",
    "resume",
}


CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.ckpt$")
DATASET_PREFLIGHT_SPLITS = ("train", "valid")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def allowed_train_kwargs() -> set[str]:
    try:
        from rfdetr.config import TrainConfig
    except Exception:
        return FALLBACK_TRAIN_KWARGS | SPECIAL_TRAIN_KWARGS

    return (set(TrainConfig.model_fields) - EXPLICIT_TRAIN_KWARGS) | SPECIAL_TRAIN_KWARGS


def compact_train_kwargs(train_cfg: dict[str, Any]) -> dict[str, Any]:
    allowed = allowed_train_kwargs()
    ignored = sorted(key for key, value in train_cfg.items() if key not in allowed and value is not None)
    if ignored:
        print(f"ignored train config keys: {ignored}")
    return {key: value for key, value in train_cfg.items() if key in allowed and value is not None}


def latest_epoch_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, float, Path]] = []
    for path in output_dir.glob("checkpoint_*.ckpt"):
        match = CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        candidates.append((int(match.group(1)), path.stat().st_mtime, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def missing_split_images(dataset_dir: Path, split: str) -> list[str]:
    ann_path = dataset_dir / split / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"RF-DETR {split} annotations are missing: {ann_path}")

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    split_dir = dataset_dir / split
    missing: list[str] = []
    for image in payload.get("images", []):
        file_name = image.get("file_name")
        if not file_name:
            continue
        image_path = split_dir / str(file_name)
        if not image_path.exists():
            missing.append(str(image_path))
    return missing


def validate_dataset_files(dataset_dir: Path, splits: tuple[str, ...] = DATASET_PREFLIGHT_SPLITS) -> dict[str, int]:
    summary: dict[str, int] = {}
    messages: list[str] = []
    for split in splits:
        missing = missing_split_images(dataset_dir, split)
        summary[f"{split}_missing_images"] = len(missing)
        if missing:
            sample = "\n".join(f"  - {path}" for path in missing[:20])
            more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
            messages.append(f"{split}: {len(missing)} missing image file(s)\n{sample}{more}")

    if messages:
        details = "\n\n".join(messages)
        raise FileNotFoundError(
            "RF-DETR dataset image preflight failed. Regenerate the dataset with "
            "`prepare_74_hidden45_dataset.py` or fix the image links before training.\n"
            f"{details}"
        )
    return summary


def apply_auto_resume(train_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    auto_resume = bool(train_cfg.pop("auto_resume", False))
    train_cfg.pop("resume_glob", None)
    explicit_resume = train_cfg.get("resume")
    if explicit_resume:
        print(f"resume checkpoint explicitly configured: {explicit_resume}")
        return {
            "auto_resume": auto_resume,
            "resume_checkpoint": str(explicit_resume),
            "resume_source": "explicit",
        }

    if not auto_resume:
        return {
            "auto_resume": False,
            "resume_checkpoint": "",
            "resume_source": "disabled",
        }

    checkpoint = latest_epoch_checkpoint(output_dir)
    if checkpoint is None:
        return {
            "auto_resume": True,
            "resume_checkpoint": "",
            "resume_source": "none_found",
        }

    train_cfg["resume"] = str(checkpoint)
    print(f"auto-resume checkpoint selected: {checkpoint}")
    return {
        "auto_resume": True,
        "resume_checkpoint": str(checkpoint),
        "resume_source": "latest_epoch_checkpoint",
    }


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def best_map75_epoch(metrics_csv: Path) -> dict[str, Any] | None:
    if not metrics_csv.exists():
        return None

    best: dict[str, Any] | None = None
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            score = safe_float(row.get("val/mAP_75"))
            epoch_value = safe_float(row.get("epoch"))
            if score is None or epoch_value is None:
                continue
            epoch = int(epoch_value)
            step = safe_float(row.get("step"))
            if best is None or score > best["map75"]:
                best = {
                    "epoch": epoch,
                    "step": int(step) if step is not None else None,
                    "map75": score,
                    "map50_95": safe_float(row.get("val/mAP_50_95")),
                    "ema_map50_95": safe_float(row.get("val/ema_mAP_50_95")),
                }
    return best


def save_map75_checkpoint(output_dir: Path, backup_dir: Path, tag: str) -> dict[str, Any]:
    metrics_csv = output_dir / "metrics.csv"
    best = best_map75_epoch(metrics_csv)
    if best is None:
        print("mAP75 checkpoint not saved: val/mAP_75 was not found in metrics.csv")
        return {"map75_checkpoint": "", "map75_summary": None}

    src = output_dir / f"checkpoint_{best['epoch']}.ckpt"
    summary = {
        **best,
        "source_checkpoint": str(src),
        "output_checkpoint": "",
        "backup_checkpoint": "",
    }
    if not src.exists():
        print(f"mAP75 checkpoint not saved: missing {src}")
        (output_dir / "map75_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {"map75_checkpoint": "", "map75_summary": summary}

    output_dst = output_dir / "checkpoint_best_map75.ckpt"
    backup_dst = backup_dir / f"{tag}_best_map75.ckpt"
    shutil.copy2(src, output_dst)
    shutil.copy2(src, backup_dst)
    summary["output_checkpoint"] = str(output_dst)
    summary["backup_checkpoint"] = str(backup_dst)
    (output_dir / "map75_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"best mAP75 checkpoint copied: {backup_dst} (epoch={best['epoch']}, mAP75={best['map75']:.6f})")
    return {"map75_checkpoint": str(backup_dst), "map75_summary": summary}


def copy_best_total_checkpoint(output_dir: Path, backup_dir: Path, tag: str) -> str:
    best_src = output_dir / "checkpoint_best_total.pth"
    if not best_src.exists():
        print("checkpoint_best_total.pth not found")
        return ""

    best_dst = backup_dir / f"{tag}_best.pth"
    shutil.copy2(best_src, best_dst)
    print(f"best checkpoint copied: {best_dst}")
    return str(best_dst)


def finalize_outputs(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    output_cfg = config["output"]
    tag = model_cfg.get("tag", model_cfg.get("variant", "rfdetr"))
    output_dir = Path(output_cfg["local_output_dir"]) / tag
    backup_dir = Path(output_cfg["backup_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "run_summary.json"
    if summary_path.exists():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        run_summary = {
            "output_dir": str(output_dir),
            "backup_dir": str(backup_dir),
            "model_variant": model_cfg["variant"],
            "model_tag": tag,
        }

    run_summary["best_checkpoint"] = copy_best_total_checkpoint(output_dir, backup_dir, tag)
    run_summary.update(save_map75_checkpoint(output_dir, backup_dir, tag))
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_summary


def train_once(config: dict[str, Any], epochs_override: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = dict(config["train"])
    output_cfg = config["output"]
    if epochs_override is not None:
        train_cfg["epochs"] = epochs_override

    dataset_dir = Path(data_cfg["dataset_dir"])
    if not (dataset_dir / "train" / "_annotations.coco.json").exists():
        raise FileNotFoundError(f"RF-DETR dataset is not prepared: {dataset_dir}")
    if not (dataset_dir / "valid" / "_annotations.coco.json").exists():
        raise FileNotFoundError(f"RF-DETR valid annotations are missing: {dataset_dir}")
    dataset_file_summary = validate_dataset_files(dataset_dir)

    tag = model_cfg.get("tag", model_cfg.get("variant", "rfdetr"))
    output_dir = Path(output_cfg["local_output_dir"]) / tag
    backup_dir = Path(output_cfg["backup_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    fold_label = dataset_dir.name if dataset_dir.name.startswith("fold") else "single"

    resume_summary = apply_auto_resume(train_cfg, output_dir)
    train_kwargs = compact_train_kwargs(train_cfg)
    run_summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "backup_dir": str(backup_dir),
        "model_variant": model_cfg["variant"],
        "model_tag": tag,
        "train_kwargs": train_kwargs,
        "dataset_file_summary": dataset_file_summary,
        "resume": resume_summary,
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        },
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "\n"
        f"===== AI12 RF-DETR TRAIN START | {fold_label} | tag={tag} =====\n"
        f"dataset_dir: {dataset_dir}\n"
        f"output_dir: {output_dir}\n"
        f"device: {train_kwargs.get('device')} | epochs: {train_kwargs.get('epochs')}"
        "\n"
        "============================================================",
        flush=True,
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))

    if dry_run:
        print("dry-run: dataset and config validated; training skipped.")
        return run_summary

    model = get_rfdetr_model(model_cfg["variant"])
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        **train_kwargs,
    )

    run_summary["best_checkpoint"] = copy_best_total_checkpoint(output_dir, backup_dir, tag)
    run_summary.update(save_map75_checkpoint(output_dir, backup_dir, tag))

    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"===== AI12 RF-DETR TRAIN DONE | {fold_label} | tag={tag} =====", flush=True)
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_45fill.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Do not train; copy checkpoint_best_total.pth and best val/mAP_75 epoch checkpoint to backup_dir.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.finalize_only:
        finalize_outputs(config)
    else:
        train_once(config, epochs_override=args.epochs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
