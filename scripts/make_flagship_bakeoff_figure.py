"""Render the flagship VLM bake-off bar chart from the latest summary JSON.

Reads results/vlm/flagship_summary_<latest>.json and produces
figures/bakeoff/flagship_bakeoff.png — a 2-panel bar chart
(F1 + cost per call) across providers × datasets.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DARK2 = {"openai": "#1b9e77", "gemini": "#d95f02", "azure": "#7570b3"}
RESULTS = Path("results/vlm")
FIG = Path("figures/bakeoff/flagship_bakeoff.png")


def latest_summary() -> Path:
    cands = sorted(RESULTS.glob("flagship_summary_*.json"))
    if not cands:
        raise SystemExit("No flagship_summary_*.json found.")
    return cands[-1]


def main() -> None:
    data = json.loads(latest_summary().read_text())
    providers = list(data["results"].keys())
    all_datasets = list(next(iter(data["results"].values())).keys())
    # Kolektor-only deck — drop other domains if present.
    datasets = [d for d in all_datasets if "kolektor" in d.lower()] or all_datasets

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(datasets))
    w = 0.8 / len(providers)

    # F1
    for i, prov in enumerate(providers):
        f1s = [data["results"][prov][d].get("f1", 0) for d in datasets]
        bars = axes[0].bar(x + i * w, f1s, w, label=f"{prov} ({data['results'][prov][datasets[0]].get('jsonl','').split('_')[-3] if False else prov})",
                           color=DARK2.get(prov, "#666"))
        for b, v in zip(bars, f1s):
            axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    axes[0].set(title=f"Flagship VLM bake-off — F1 (n={data['n_per_dataset']}/domain)",
                xlabel="Dataset", ylabel="F1",
                xticks=x + w * (len(providers) - 1) / 2,
                xticklabels=[d.replace("_test", "") for d in datasets],
                ylim=[0, 1.05])
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    # Cost per call
    for i, prov in enumerate(providers):
        costs = []
        for d in datasets:
            n_valid = data["results"][prov][d].get("n_valid", 1) or 1
            total = data["results"][prov][d].get("total_cost_usd", 0)
            costs.append(total / max(n_valid, 1))
        bars = axes[1].bar(x + i * w, costs, w, color=DARK2.get(prov, "#666"), label=prov)
        for b, v in zip(bars, costs):
            axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0002,
                         f"${v:.4f}", ha="center", va="bottom", fontsize=8.5)
    axes[1].set(title="Cost per inference call (USD)",
                xlabel="Dataset", ylabel="USD / image",
                xticks=x + w * (len(providers) - 1) / 2,
                xticklabels=[d.replace("_test", "") for d in datasets])
    axes[1].grid(axis="y", alpha=0.3)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {FIG}")


if __name__ == "__main__":
    main()
