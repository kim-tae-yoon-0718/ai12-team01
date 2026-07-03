#!/usr/bin/env python3
"""Train RF-DETR on the 74-class 45-fill dataset without k-fold wrapping."""

from __future__ import annotations

import argparse
import os

from train_45fill import load_config, train_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config_74_hidden45.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_once(load_config(args.config), epochs_override=args.epochs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
