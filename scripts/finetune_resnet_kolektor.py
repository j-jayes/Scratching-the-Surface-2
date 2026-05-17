"""Fine-tune ResNet50 on kolektor domain data (Phase 2a+).

Loads the best Phase-2a checkpoint (models/resnet50_best.pt) and fine-tunes on
kolektor training images alongside the original Severstal / NEU-DET data to
prevent catastrophic forgetting.

Usage:
    uv run python scripts/finetune_resnet_kolektor.py
    uv run python scripts/finetune_resnet_kolektor.py --config configs/resnet50_kolektor_ft.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.defect_dataset import (
    DefectDataset,
    build_train_transform,
    build_train_transform_rrc,
    build_val_transform,
    build_tta_transforms,
    load_split,
)
from src.models.resnet_baseline import build_model, evaluate, get_device, train_epoch


def evaluate_tta(
    model: nn.Module,
    df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold: float,
    input_size: int = 224,
) -> dict:
    """Multi-crop TTA: top / center / bottom crops, averaged then scored."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    )

    tta_tfms = build_tta_transforms(input_size)
    all_probs: list[np.ndarray] = []
    gt_labels: np.ndarray | None = None

    model.eval()
    with torch.no_grad():
        for tfm in tta_tfms:
            loader = DataLoader(
                DefectDataset(df, tfm),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
            )
            batch_probs, batch_labels = [], []
            for imgs, labels in loader:
                imgs = imgs.to(device)
                probs = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
                batch_probs.append(probs)
                batch_labels.append(labels.numpy())
            all_probs.append(np.concatenate(batch_probs))
            if gt_labels is None:
                gt_labels = np.concatenate(batch_labels).astype(int)

    probs = np.mean(all_probs, axis=0)
    preds = (probs >= threshold).astype(int)
    labels = gt_labels  # type: ignore[assignment]

    metrics: dict = {
        "n": int(len(labels)),
        "n_pos": int(labels.sum()),
        "n_neg": int((1 - labels).sum()),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    return metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FocalLossWithLogits(nn.Module):
    """Focal loss for binary classification from raw logits.

    FL(pt) = -α_t * (1 - pt)^γ * log(pt)

    Down-weights easy examples (high pt) to focus training on hard ones.
    γ=0 recovers standard BCE; γ=2 is the standard setting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as _F
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        bce_loss = _F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.pos_weight is not None:
            bce_loss = torch.where(targets == 1, bce_loss * self.pos_weight, bce_loss)
        loss = focal_weight * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet50_kolektor_ft.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))
    input_size: int = cfg.get("input_size", 224)
    batch_size: int = cfg.get("batch_size", 32)
    num_workers: int = cfg.get("num_workers", 0)
    epochs: int = cfg.get("epochs", 15)
    lr: float = cfg.get("lr", 5e-5)
    weight_decay: float = cfg.get("weight_decay", 1e-4)
    threshold: float = cfg.get("threshold", 0.5)
    checkpoint_path = Path(cfg["checkpoint"])
    results_dir = Path(cfg["results_dir"])
    pretrained_weights = Path(cfg["pretrained_weights"])

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df = load_split(cfg["train_manifests"], split="train")
    val_df = load_split(cfg["val_manifests"], split="val")

    # Optional: hold out a fraction of kolektor train for kolektor-val checkpointing.
    # If kolektor_holdout_frac > 0, we split kolektor train images and use a composite
    # metric (severstal_val_F1 + kolektor_holdout_F1) / 2 for checkpoint selection.
    kolektor_holdout_frac: float = cfg.get("kolektor_holdout_frac", 0.0)
    kolektor_holdout_df = None
    if kolektor_holdout_frac > 0:
        kol_mask = train_df["path"].str.contains("kolektor", case=False)
        kol_df = train_df[kol_mask].copy()
        # Stratified split by defect label to preserve class ratio in holdout
        holdout_parts = []
        train_parts = [train_df[~kol_mask]]
        for label_val in [True, False]:
            subset = kol_df[kol_df["has_defect"] == label_val].sample(
                frac=1, random_state=cfg.get("seed", 42)
            )
            n_hold = max(1, int(len(subset) * kolektor_holdout_frac))
            holdout_parts.append(subset.iloc[:n_hold])
            train_parts.append(subset.iloc[n_hold:])
        kolektor_holdout_df = pd.concat(holdout_parts, ignore_index=True)
        train_df = pd.concat(train_parts, ignore_index=True)
        print(f"Kolektor holdout val: {len(kolektor_holdout_df)} imgs ({kolektor_holdout_df['has_defect'].sum()} defect)")

    n_pos = int(train_df["has_defect"].sum())
    n_neg = int((~train_df["has_defect"]).sum())
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Train: {len(train_df):,} images  |  +{n_pos} defect / -{n_neg} normal  |  pos_weight={pos_weight:.3f}")
    print(f"Val:   {len(val_df):,} images")

    augmentation = cfg.get("augmentation", "standard")
    if augmentation == "rrc":
        train_tfm = build_train_transform_rrc(input_size)
        print(f"Augmentation: RandomResizedCrop (rrc) — better for tall/narrow images")
    else:
        train_tfm = build_train_transform(input_size)
        print(f"Augmentation: standard (Resize+CenterCrop)")
    val_tfm = build_val_transform(input_size)

    # Optional kolektor domain oversampling via WeightedRandomSampler
    kolektor_oversample = cfg.get("kolektor_oversample_factor", 1.0)
    train_sampler = None
    train_shuffle = True
    if kolektor_oversample > 1.0:
        import torch as _torch
        from torch.utils.data import WeightedRandomSampler
        kol_mask_train = train_df["path"].str.contains("kolektor", case=False).values
        weights = _torch.ones(len(train_df))
        weights[kol_mask_train] = kolektor_oversample
        train_sampler = WeightedRandomSampler(
            weights, num_samples=len(train_df), replacement=True
        )
        train_shuffle = False  # mutually exclusive with sampler
        print(f"Kolektor oversampling: {kolektor_oversample}× ({kol_mask_train.sum()} kolektor images)")

    train_loader = DataLoader(
        DefectDataset(train_df, train_tfm),
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
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

    # Optional kolektor holdout loader for composite checkpoint selection
    kolektor_holdout_loader = None
    if kolektor_holdout_df is not None:
        kolektor_holdout_loader = DataLoader(
            DefectDataset(kolektor_holdout_df, val_tfm),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        print(f"Kolektor holdout loader: {len(kolektor_holdout_df)} images")

    # ── Eval sets ──────────────────────────────────────────────────────────────
    eval_loaders: dict[str, DataLoader] = {}
    eval_dfs: dict[str, pd.DataFrame] = {}
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
        eval_dfs[name] = df

    # ── Model — load from pretrained checkpoint ────────────────────────────────
    device = get_device()
    print(f"\nDevice: {device}")
    model = build_model().to(device)

    if pretrained_weights.exists():
        model.load_state_dict(torch.load(pretrained_weights, map_location=device, weights_only=True))
        print(f"Loaded pretrained weights from {pretrained_weights}")
    else:
        print(f"WARNING: pretrained weights not found at {pretrained_weights}, training from ImageNet init")

    pw = torch.tensor([pos_weight], device=device)
    loss_fn: str = cfg.get("loss_fn", "bce")
    if loss_fn == "focal":
        focal_gamma: float = float(cfg.get("focal_gamma", 2.0))
        criterion: nn.Module = FocalLossWithLogits(gamma=focal_gamma, pos_weight=pw)
        print(f"Loss: FocalLossWithLogits (gamma={focal_gamma}, pos_weight={pos_weight:.3f})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"Loss: BCEWithLogitsLoss (pos_weight={pos_weight:.3f})")

    # Optional differential learning rates: backbone gets backbone_lr_factor × lr
    backbone_lr_factor: float = cfg.get("backbone_lr_factor", 1.0)
    if backbone_lr_factor != 1.0:
        # ResNet50: model.fc is the head; everything else is backbone
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        head_params     = [p for n, p in model.named_parameters() if n.startswith("fc.")]
        param_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_factor},
            {"params": head_params,     "lr": lr},
        ]
        print(f"Differential LR: backbone={lr * backbone_lr_factor:.2e}, head={lr:.2e}")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch)
        val_metrics = evaluate(model, val_loader, device, threshold)
        scheduler.step()

        # Composite checkpoint metric: severstal_val_F1 (always) + kolektor_holdout_F1 (optional)
        checkpoint_score = val_metrics["f1"]
        kol_holdout_f1 = None
        if kolektor_holdout_loader is not None:
            kol_holdout_metrics = evaluate(model, kolektor_holdout_loader, device, threshold)
            kol_holdout_f1 = kol_holdout_metrics["f1"]
            checkpoint_score = (val_metrics["f1"] + kol_holdout_f1) / 2.0

        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "elapsed_s": round(elapsed, 1),
        }
        if kol_holdout_f1 is not None:
            row["kol_holdout_f1"] = round(kol_holdout_f1, 4)
            row["composite_score"] = round(checkpoint_score, 4)
        history.append(row)

        kol_str = f"  kol_holdout_f1={kol_holdout_f1:.4f}" if kol_holdout_f1 is not None else ""
        print(
            f"Ep {epoch:02d}/{epochs}  loss={train_loss:.4f}"
            f"  val_f1={val_metrics['f1']:.4f}"
            f"  val_auc={val_metrics.get('roc_auc', 0):.4f}"
            f"{kol_str}"
            f"  ({elapsed:.0f}s)"
        )

        if checkpoint_score > best_val_f1:
            best_val_f1 = checkpoint_score
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ saved checkpoint (score={best_val_f1:.4f})")

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\nLoading best checkpoint for final eval...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    tta_eval: bool = cfg.get("tta_eval", False)
    if tta_eval:
        print("TTA eval enabled (3-crop: top / center / bottom)")

    results_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, dict] = {}
    for name, loader in eval_loaders.items():
        if tta_eval:
            m = evaluate_tta(model, eval_dfs[name], batch_size, num_workers, device, threshold, input_size)
            tag = " [TTA]"
        else:
            m = evaluate(model, loader, device, threshold)
            tag = ""
        all_metrics[name] = m
        print(
            f"  {name:20s}  F1={m['f1']:.4f}  AUC={m.get('roc_auc', float('nan')):.4f}"
            f"  n={m['n']} (+{m['n_pos']} / -{m['n_neg']}){tag}"
        )

    output = {
        "model": "resnet50_kolektor_ft",
        "pretrained_from": str(pretrained_weights),
        "fine_tune_lr": lr,
        "best_val_f1": best_val_f1,
        "threshold": threshold,
        "history": history,
        "eval": all_metrics,
    }
    out_path = results_dir / "metrics.json"
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFine-tune metrics saved → {out_path}")


if __name__ == "__main__":
    main()
