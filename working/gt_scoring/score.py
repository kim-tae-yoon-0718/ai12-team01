#!/usr/bin/env python3
"""Score a submission against the canonical class-reviewed test GT."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
GT_DIR = HERE / "ground_truth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a submission against the canonical 74-class test GT."
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(HERE / "score_submission_against_gt.py"),
        "--submission",
        str(args.submission.resolve()),
        "--ground-truth",
        str(GT_DIR / "test_ground_truth.csv"),
        "--class-map",
        str(GT_DIR / "pill_class_number_map_74.csv"),
        "--unknown-ignore-boxes",
        str(PROJECT_ROOT / "working" / "test_annotations" / "test_unknown_ignore_boxes.csv"),
        "--out-dir",
        str(args.out_dir.resolve()),
        "--image-filter",
        "all",
        "--iou-threshold",
        str(args.iou_threshold),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
