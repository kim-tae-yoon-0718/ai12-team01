#!/usr/bin/env python3
"""Train RF-DETR on prepared 45-fill datasets without k-fold wrapping."""

from __future__ import annotations

import argparse
import json
import os
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
    "augmentation_backend",
    "log_per_class_metrics",
    "run_test",
    "eval_max_dets",
    "class_names",
    "resume",
}


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

    tag = model_cfg.get("tag", model_cfg.get("variant", "rfdetr"))
    output_dir = Path(output_cfg["local_output_dir"]) / tag
    backup_dir = Path(output_cfg["backup_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = compact_train_kwargs(train_cfg)
    run_summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "backup_dir": str(backup_dir),
        "model_variant": model_cfg["variant"],
        "model_tag": tag,
        "train_kwargs": train_kwargs,
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        },
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
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

    best_src = output_dir / "checkpoint_best_total.pth"
    if best_src.exists():
        best_dst = backup_dir / f"{tag}_best.pth"
        shutil.copy2(best_src, best_dst)
        run_summary["best_checkpoint"] = str(best_dst)
        print(f"best checkpoint copied: {best_dst}")
    else:
        run_summary["best_checkpoint"] = ""
        print("checkpoint_best_total.pth not found")

    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_45fill.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_once(load_config(args.config), epochs_override=args.epochs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
