"""Post-training analysis for Phase 2a ResNet50.

Run after train_resnet.py completes:
    uv run python scripts/analyse_resnet.py

Prints a full metrics table, plots training curves and per-set ROC curves,
saves figures to figures/resnet50/, and prints a GO/NO-GO recommendation
for starting Phase 2b YOLO training.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("results/resnet50/metrics.json")
FIGURES_DIR  = Path("figures/resnet50")
GATE_VAL_F1  = 0.75
GATE_KOLEKTOR_AUC = 0.65  # cross-domain, genuinely hard


def print_table(eval_dict: dict) -> None:
    header = f"{'Dataset':<22} {'N':>6} {'Pos':>6} {'Neg':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>7}"
    print(header)
    print("-" * len(header))
    for name, m in eval_dict.items():
        auc = f"{m.get('roc_auc', float('nan')):.4f}" if "roc_auc" in m else "  n/a "
        print(
            f"{name:<22} {m['n']:>6} {m['n_pos']:>6} {m['n_neg']:>6}"
            f"  {m['f1']:>6.4f}  {m['precision']:>6.4f}  {m['recall']:>6.4f}  {auc:>7}"
        )


def plot_training_curve(history: list[dict], out_path: Path) -> None:
    epochs = [r["epoch"] for r in history]
    losses = [r["train_loss"] for r in history]
    f1s    = [r["val_f1"] for r in history]
    aucs   = [r.get("val_roc_auc", None) for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, losses, "b-o", ms=4)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="BCE loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, f1s, "g-o", ms=4, label="Val F1")
    if any(a is not None for a in aucs):
        axes[1].plot(epochs, [a or 0 for a in aucs], "r--s", ms=4, label="Val AUC")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"  → {out_path}")
    plt.close(fig)


def go_no_go(m: dict, best_val_f1: float) -> None:
    eval_dict = m["eval"]
    kol_auc = eval_dict.get("kolektor_test", {}).get("roc_auc", 0)
    gc10_f1 = eval_dict.get("gc10_test", {}).get("f1", 0)

    print("\n" + "=" * 55)
    print("  PHASE 2a → 2b GO/NO-GO GATE")
    print("=" * 55)
    checks = [
        ("Best val F1",          best_val_f1, GATE_VAL_F1,        "≥"),
        ("Kolektor AUC",         kol_auc,     GATE_KOLEKTOR_AUC,  "≥"),
    ]
    passed = []
    for label, value, threshold, op in checks:
        ok = (value >= threshold) if op == "≥" else (value <= threshold)
        status = "✅" if ok else "❌"
        passed.append(ok)
        print(f"  {status}  {label:<22} {value:.4f}  (threshold {op} {threshold})")

    print()
    if all(passed):
        print("  ✅  ALL GATES PASSED — proceed to Phase 2b (train_yolo.py)")
    else:
        print("  ❌  GATE FAILED — review suggestions below before starting 2b")
        if best_val_f1 < GATE_VAL_F1:
            print("      • val F1 low: try more epochs (30), label smoothing, or focal loss")
        if kol_auc < GATE_KOLEKTOR_AUC:
            print("      • Kolektor AUC low: acceptable if GC10 AUC > 0.70 (different domain)")
            gc10_auc = eval_dict.get("gc10_test", {}).get("roc_auc", 0)
            print(f"        GC10 AUC = {gc10_auc:.4f}")
    print("=" * 55)


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"Metrics not found at {RESULTS_PATH} — training may still be running.")
        return

    with RESULTS_PATH.open() as f:
        m = json.load(f)

    print(f"\nResNet50 training summary")
    print(f"  Epochs:       {len(m['history'])}")
    print(f"  Best val F1:  {m['best_val_f1']:.4f}")
    print(f"  Threshold:    {m['threshold']}")

    print("\n--- Per-set evaluation ---")
    print_table(m["eval"])

    plot_training_curve(m["history"], FIGURES_DIR / "training_curve.png")

    go_no_go(m, m["best_val_f1"])


if __name__ == "__main__":
    main()
