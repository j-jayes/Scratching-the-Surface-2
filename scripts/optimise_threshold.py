"""Threshold optimisation and per-class error analysis for ResNet50 (Phase 2a).

Run after train_resnet.py completes:
    uv run python scripts/optimise_threshold.py

Run on fine-tuned model:
    uv run python scripts/optimise_threshold.py \
        --checkpoint models/resnet50_kolektor_ft_best.pt \
        --results-dir results/resnet50_kolektor_ft \
        --figures-dir figures/resnet50_kolektor_ft

Re-runs inference on val + all test sets, sweeps thresholds,
picks the best by val F1, saves:
  <results-dir>/threshold_sweep.json
  <figures-dir>/threshold_sweep.png
  <figures-dir>/roc_curves.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from src.data.defect_dataset import (
    DefectDataset,
    build_val_transform,
    build_tta_transforms,
    load_split,
)
from src.models.resnet_baseline import build_model, get_device

CHECKPOINT   = Path("models/resnet50_best.pt")
RESULTS_DIR  = Path("results/resnet50")
FIGURES_DIR  = Path("figures/resnet50")
CONFIG_PATH  = Path("configs/resnet50.yaml")


@torch.no_grad()
def get_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        logits = model(imgs.to(device)).squeeze(1).cpu()
        all_probs.append(torch.sigmoid(logits).numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels).astype(int)


@torch.no_grad()
def get_probs_tta(model, df, device, batch_size: int, input_size: int) -> tuple[np.ndarray, np.ndarray]:
    """3-crop TTA inference: average top / center / bottom crop predictions."""
    from torch.utils.data import DataLoader
    tta_tfms = build_tta_transforms(input_size)
    all_crop_probs = []
    gt_labels = None
    model.eval()
    for tfm in tta_tfms:
        loader = DataLoader(DefectDataset(df, tfm), batch_size=batch_size, shuffle=False, num_workers=0)
        batch_probs, batch_labels = [], []
        for imgs, labels in loader:
            logits = model(imgs.to(device)).squeeze(1).cpu()
            batch_probs.append(torch.sigmoid(logits).numpy())
            batch_labels.append(labels.numpy())
        all_crop_probs.append(np.concatenate(batch_probs))
        if gt_labels is None:
            gt_labels = np.concatenate(batch_labels).astype(int)
    return np.mean(all_crop_probs, axis=0), gt_labels


def best_threshold_by_f1(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(labels, (probs >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def plot_threshold_sweep(probs_val, labels_val, out_path: Path) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(labels_val, (probs_val >= t).astype(int), zero_division=0) for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, f1s, "b-", lw=1.5)
    ax.axvline(best_t, color="r", ls="--", label=f"best τ={best_t:.2f}")
    ax.axvline(0.5, color="gray", ls=":", label="τ=0.50 (default)")
    ax.set(title="Threshold sweep — Val F1", xlabel="Threshold τ", ylabel="F1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(best_t)


def plot_roc_curves(results: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    for i, (name, (probs, labels)) in enumerate(results.items()):
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        label = name.replace("_", " ")
        ax.plot(fpr, tpr, color=colors[i % 10], lw=1.5, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title="ROC curves — ResNet50", xlabel="FPR", ylabel="TPR", xlim=[0, 1], ylim=[0, 1])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default=str(CHECKPOINT),  help="Path to .pt checkpoint")
    parser.add_argument("--results-dir",  default=str(RESULTS_DIR), help="Where to write threshold_sweep.json")
    parser.add_argument("--figures-dir",  default=str(FIGURES_DIR), help="Where to write figures")
    parser.add_argument("--tta", action="store_true", help="Use 3-crop TTA (top/center/bottom) for tall images")
    args = parser.parse_args()

    checkpoint   = Path(args.checkpoint)
    results_dir  = Path(args.results_dir)
    figures_dir  = Path(args.figures_dir)

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint} — run training script first.")
        return

    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    model  = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    input_size: int = cfg.get("input_size", 224)
    batch_size: int = cfg.get("batch_size", 32)
    tfm = build_val_transform(input_size)

    from torch.utils.data import DataLoader

    def make_loader(manifests, split, *, sup_path=None, sup_split=None):
        df = load_split(manifests, split, supplement_normals_path=sup_path, supplement_split=sup_split)
        return DataLoader(DefectDataset(df, tfm), batch_size=batch_size, shuffle=False, num_workers=0)

    def make_df(manifests, split, *, sup_path=None, sup_split=None):
        return load_split(manifests, split, supplement_normals_path=sup_path, supplement_split=sup_split)

    # Build all sets
    if args.tta:
        print("TTA enabled — using 3-crop (top/center/bottom) inference")
        sets_df: dict[str, object] = {
            "val":            make_df(cfg["val_manifests"], "val"),
            "severstal_test": make_df(["data/processed/severstal_manifest.parquet"], "test"),
            "kolektor_test":  make_df(["data/processed/kolektor_manifest.parquet"], "test"),
            "gc10_test":      make_df(
                ["data/processed/gc10_manifest.parquet"], "test",
                sup_path="data/processed/severstal_manifest.parquet", sup_split="test"
            ),
        }
    else:
        sets_df = {}

    sets: dict[str, object] = {
        "val":            make_loader(cfg["val_manifests"], "val"),
        "severstal_test": make_loader(["data/processed/severstal_manifest.parquet"], "test"),
        "kolektor_test":  make_loader(["data/processed/kolektor_manifest.parquet"], "test"),
        "gc10_test":      make_loader(
            ["data/processed/gc10_manifest.parquet"], "test",
            sup_path="data/processed/severstal_manifest.parquet", sup_split="test"
        ),
    }

    print("Running inference on all sets...")
    prob_label: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in sets:
        if args.tta:
            p, l = get_probs_tta(model, sets_df[name], device, batch_size, input_size)
        else:
            p, l = get_probs(model, sets[name], device)
        prob_label[name] = (p, l)
        print(f"  {name}: {len(p)} images  {l.sum()} pos")

    # Find best threshold on val
    val_p, val_l = prob_label["val"]
    best_t = plot_threshold_sweep(val_p, val_l, figures_dir / "threshold_sweep.png")
    print(f"\nBest threshold (val F1): τ={best_t:.3f}")

    # Score all sets at both default (0.5) and optimised threshold
    sweep_results: dict = {}
    print(f"\n{'Dataset':<22} {'F1@0.50':>8} {'F1@τ_opt':>8} {'AUC':>7}")
    print("-" * 55)
    for name, (probs, labels) in prob_label.items():
        f1_default = f1_score(labels, (probs >= 0.5).astype(int), zero_division=0)
        f1_opt     = f1_score(labels, (probs >= best_t).astype(int), zero_division=0)
        auc_str    = f"{roc_auc_score(labels, probs):.4f}" if len(np.unique(labels)) > 1 else "  n/a"
        print(f"{name:<22} {f1_default:>8.4f} {f1_opt:>8.4f} {auc_str:>7}")
        sweep_results[name] = {
            "f1_at_0.5": round(f1_default, 4),
            "f1_at_opt_threshold": round(f1_opt, 4),
            "opt_threshold": best_t,
        }
        if len(np.unique(labels)) > 1:
            sweep_results[name]["roc_auc"] = round(float(roc_auc_score(labels, probs)), 4)

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "threshold_sweep.json"
    with out.open("w") as f:
        json.dump({"best_threshold": best_t, "sets": sweep_results}, f, indent=2)
    print(f"\nSweep results → {out}")

    plot_roc_curves(
        {k: v for k, v in prob_label.items() if k != "neu_det_val"},
        figures_dir / "roc_curves.png",
    )


if __name__ == "__main__":
    main()
