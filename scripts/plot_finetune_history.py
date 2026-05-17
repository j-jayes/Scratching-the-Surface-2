"""Plot ResNet50 fine-tune training history (val_F1 over epochs).

Usage:
    uv run python scripts/plot_finetune_history.py
    uv run python scripts/plot_finetune_history.py \
        --metrics results/resnet50_kolektor_ft/metrics.json \
        --out figures/resnet50_kolektor_ft/training_curve.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="results/resnet50_kolektor_ft/metrics.json",
        help="Path to fine-tune metrics.json",
    )
    parser.add_argument(
        "--baseline",
        default="results/resnet50/metrics.json",
        help="Path to base model metrics.json (for comparison horizontal line)",
    )
    parser.add_argument(
        "--out",
        default="figures/resnet50_kolektor_ft/training_curve.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        print("Run finetune_resnet_kolektor.py first.")
        return

    with metrics_path.open() as f:
        m = json.load(f)

    history = m.get("history", [])
    if not history:
        print("No history found in metrics.json")
        return

    epochs = [r["epoch"] for r in history]
    val_f1 = [r["val_f1"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    elapsed = [r.get("elapsed_s", 0) for r in history]
    kol_holdout_f1 = [r.get("kol_holdout_f1") for r in history]  # may be None per epoch

    # Load baseline for comparison line
    baseline_severstal_f1 = None
    if Path(args.baseline).exists():
        with open(args.baseline) as f:
            base_m = json.load(f)
        baseline_severstal_f1 = base_m.get("eval", {}).get("severstal_test", {}).get("f1")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Val F1
    ax1.plot(epochs, val_f1, "b-o", ms=5, label="Val F1 (severstal)")
    if any(v is not None for v in kol_holdout_f1):
        kol_ep = [e for e, v in zip(epochs, kol_holdout_f1) if v is not None]
        kol_vals = [v for v in kol_holdout_f1 if v is not None]
        ax1.plot(kol_ep, kol_vals, "g--s", ms=5, label="Kolektor holdout F1")
    if baseline_severstal_f1 is not None:
        ax1.axhline(baseline_severstal_f1, color="gray", ls="--", lw=1.2,
                    label=f"Baseline val F1 ({baseline_severstal_f1:.4f})")
    ax1.set(ylabel="Val F1", title="ResNet50+FT Training — Severstal Val F1 & Kolektor Holdout F1")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    _all_f1 = [v for v in val_f1 + kol_vals if v is not None] if any(v is not None for v in kol_holdout_f1) else val_f1
    ax1.set_ylim(max(0, min(_all_f1) - 0.05), min(1.0, max(_all_f1) + 0.05))

    # Train loss
    ax2.plot(epochs, train_loss, "r-o", ms=5, label="Train loss (BCE)")
    ax2.set(xlabel="Epoch", ylabel="Train loss")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Epoch time annotation
    avg_elapsed = sum(elapsed) / len(elapsed) if elapsed else 0
    fig.text(0.99, 0.01, f"Avg {avg_elapsed:.0f}s/epoch", ha="right", va="bottom",
             fontsize=8, color="gray")

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curve → {out_path}")

    # Summary stats
    best_ep = epochs[val_f1.index(max(val_f1))]
    print(f"\nBest val F1: {max(val_f1):.4f} at epoch {best_ep}")
    print(f"Final val F1: {val_f1[-1]:.4f}")
    print(f"Total training time: {sum(elapsed)/60:.1f} min ({len(epochs)} epochs)")


if __name__ == "__main__":
    main()
