"""Train ResNet50 binary defect classifier (Phase 2a).

Usage:
    uv run python scripts/train_resnet.py
    uv run python scripts/train_resnet.py --config configs/resnet50.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path when run as a script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.defect_dataset import (
    DefectDataset,
    build_train_transform,
    build_val_transform,
    load_split,
)
from src.models.resnet_baseline import run_training


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet50.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))
    input_size: int = cfg.get("input_size", 512)
    batch_size: int = cfg.get("batch_size", 32)
    num_workers: int = cfg.get("num_workers", 4)

    # ── Training data ─────────────────────────────────────────────────────────
    train_df = load_split(cfg["train_manifests"], split="train")
    val_df = load_split(cfg["val_manifests"], split="val")

    n_pos = int(train_df["has_defect"].sum())
    n_neg = int((~train_df["has_defect"]).sum())
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Train: {len(train_df):,} images  |  +{n_pos} defect / -{n_neg} normal  |  pos_weight={pos_weight:.3f}")
    print(f"Val:   {len(val_df):,} images")

    train_tfm = build_train_transform(input_size)
    val_tfm = build_val_transform(input_size)

    train_loader = DataLoader(
        DefectDataset(train_df, train_tfm),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        DefectDataset(val_df, val_tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # ── Evaluation sets ───────────────────────────────────────────────────────
    eval_loaders: dict[str, DataLoader] = {}
    for name, ecfg in cfg.get("eval_sets", {}).items():
        sup_path = ecfg.get("supplement_normals_from")
        sup_split = ecfg.get("supplement_split")
        df = load_split(
            ecfg["manifests"],
            split=ecfg["split"],
            supplement_normals_path=sup_path,
            supplement_split=sup_split,
        )
        print(f"Eval '{name}': {len(df):,} images  (+{int(df['has_defect'].sum())} / -{int((~df['has_defect']).sum())})")
        eval_loaders[name] = DataLoader(
            DefectDataset(df, val_tfm),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    run_training(
        train_loader,
        val_loader,
        eval_loaders,
        epochs=cfg.get("epochs", 20),
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
        pos_weight=pos_weight,
        threshold=cfg.get("threshold", 0.5),
        checkpoint_path=Path(cfg["checkpoint"]),
        results_dir=Path(cfg["results_dir"]),
    )


if __name__ == "__main__":
    main()
