"""Cost trade-off figure for slides section K.

Story: if you can pay a VLM $X per image to label data, when does it make
economic sense to spend N×$X labelling images and train a free classifier
instead of paying the per-call VLM cost forever?

Curve:
    Y = total $ to process N images at inference time
    X = N (log scale, 100 → 1,000,000 images)

Lines:
  - VLM-only            : N × $vlm_per_call
  - Hybrid YOLO→VLM     : N × $yolo + 0.30·N × $vlm_per_call
  - Classical-trained   : C_label × N_labels + N × $resnet_inference
                          where C_label ∈ {$0.05, $0.20, $0.50}

Numerical constants are pulled from results/vlm/flagship_summary*.json
and the existing cost_ledger.csv where possible, falling back to
hand-set defaults so the script always runs.

Outputs figures/bakeoff/labels_vs_vlm_cost.png and an annotated
break-even table (CSV) for the slides speaker notes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("figures/bakeoff")
RESULTS = Path("results/vlm")

# Defaults — overridden from flagship_summary if present
DEFAULT_VLM_COST = 0.007           # gpt-5.4 per image (from smoke test)
YOLO_INFER_COST  = 0.000_05        # ≈ 50 µs of GPU time, ~$5 / 100k imgs
RESNET_INFER_COST = 0.000_03
HYBRID_VLM_TRIGGER = 0.30          # 30% of images get escalated to VLM
N_TRAIN_LABELS = 2_332             # full Kolektor train + GC10 train pool

LABEL_BANDS = {
    "$0.05 / label (in-house GPU annotators)":  ("#1b9e77", 0.05),
    "$0.20 / label (offshore crowdworkers)":    ("#d95f02", 0.20),
    "$0.50 / label (expert metallurgist)":      ("#7570b3", 0.50),
}


def load_vlm_cost() -> float:
    summaries = sorted(RESULTS.glob("flagship_summary_*.json"))
    if not summaries:
        return DEFAULT_VLM_COST
    data = json.loads(summaries[-1].read_text())
    costs: list[float] = []
    for prov in data.get("results", {}).values():
        for ds in prov.values():
            n_valid = ds.get("n_valid") or 0
            cost = ds.get("total_cost_usd") or 0
            if n_valid:
                costs.append(cost / n_valid)
    return float(np.mean(costs)) if costs else DEFAULT_VLM_COST


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    vlm_cost = load_vlm_cost()
    print(f"Using mean VLM cost = ${vlm_cost:.4f} / image")

    N = np.logspace(2, 6, 200)          # 100 → 1,000,000

    vlm_curve    = N * vlm_cost
    hybrid_curve = N * YOLO_INFER_COST + HYBRID_VLM_TRIGGER * N * vlm_cost

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.loglog(N, vlm_curve,    color="#e7298a", lw=3,   label=f"VLM only (gpt-5.4, ${vlm_cost:.3f}/img)")
    ax.loglog(N, hybrid_curve, color="#66a61e", lw=2.5, label="Hybrid YOLO→VLM (30% escalation)")

    breakevens: list[dict] = []
    for label, (colour, cl) in LABEL_BANDS.items():
        classical = cl * N_TRAIN_LABELS + N * RESNET_INFER_COST
        ax.loglog(N, classical, color=colour, lw=2.5, linestyle="--", label=label)
        # break-even vs VLM-only
        diff = vlm_curve - classical
        cross_idx = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
        if len(cross_idx):
            n_break = int(N[cross_idx[0]])
            ax.axvline(n_break, color=colour, alpha=0.18, linestyle=":")
            breakevens.append({"label_cost": cl, "break_even_N_vs_VLM": n_break})

    # Annotate one-time labelling cost as horizontal floors
    for label, (colour, cl) in LABEL_BANDS.items():
        floor = cl * N_TRAIN_LABELS
        ax.axhline(floor, color=colour, alpha=0.10, linestyle="-")

    ax.set(
        xlabel="Number of images processed (production lifetime)",
        ylabel="Total cost (USD)",
        title="Per-image VLMs vs one-off labelling + classical inference",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    note = (f"Labelling pool: N_train = {N_TRAIN_LABELS:,} images\n"
            f"Hybrid escalation rate: {HYBRID_VLM_TRIGGER:.0%} of images → VLM\n"
            f"ResNet inference: ${RESNET_INFER_COST*1000:.2f} / 1k imgs")
    ax.text(0.97, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc",
                      edgecolor="#999"))

    out = FIG_DIR / "labels_vs_vlm_cost.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")

    if breakevens:
        df = pd.DataFrame(breakevens)
        csv = FIG_DIR / "labels_vs_vlm_breakeven.csv"
        df.to_csv(csv, index=False)
        print(df.to_string(index=False))
        print(f"  → {csv}")


if __name__ == "__main__":
    main()
