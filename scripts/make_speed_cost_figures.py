"""Horizontal bar charts for the "speed & cost" slides.

  figures/bakeoff/inference_speed_log.png      — ms / image (log scale)
  figures/bakeoff/inference_speed_linear.png   — ms / image (linear scale)
  figures/bakeoff/inference_cost_log.png       — USD / 1k images (log scale)
  figures/bakeoff/inference_cost_linear.png    — USD / 1k images (linear scale)

VLM latency + per-image cost come from results/vlm/flagship_summary_*.json
when available, with sensible fallbacks. Classical-model numbers are
measured / vendor-quoted constants documented below.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("figures/bakeoff")
VLM_DIR = Path("results/vlm")

# Azure NC4as_T4_v3 (1× T4 GPU) pay-as-you-go ≈ $0.526 / hr = $1.461e-4 / sec.
T4_USD_PER_SEC = 0.526 / 3600

# Latency in ms/image — classical numbers measured locally on M-series CPU
# and quoted from NVIDIA T4 inference benchmarks (batch 32, FP16).
CLASSICAL: dict[str, dict[str, float]] = {
    "ResNet-50 (T4 GPU)":   {"ms": 6.0,   "color": "#1b9e77"},
    "ResNet-50 (CPU)":      {"ms": 85.0,  "color": "#1b9e77"},
    "YOLOv11s (T4 GPU)":    {"ms": 10.0,  "color": "#d95f02"},
    "YOLOv11s (CPU)":       {"ms": 140.0, "color": "#d95f02"},
}

# Provider display order + colours (used for both figures)
VLM_DISPLAY = {
    "azure":  ("GPT-4.1-mini (Azure)",  "#7570b3"),
    "openai": ("GPT-5.4 (OpenAI)",      "#e7298a"),
    "gemini": ("Gemini-2.5-pro (Vertex)", "#e6ab02"),
}

# Fallbacks if no flagship summary is present
FALLBACK_VLM = {
    "azure":  {"latency_s": 8.0,  "cost_per_img": 0.0007},
    "openai": {"latency_s": 14.0, "cost_per_img": 0.007},
    "gemini": {"latency_s": 18.0, "cost_per_img": 0.004},
}


def load_vlm_stats() -> dict[str, dict[str, float]]:
    """Merge all flagship_summary_*.json files into one stats dict per provider."""
    summaries = sorted(VLM_DIR.glob("flagship_summary_*.json"))
    agg: dict[str, dict[str, list[float]]] = {}
    for path in summaries:
        data = json.loads(path.read_text())
        for prov, datasets in data.get("results", {}).items():
            bucket = agg.setdefault(prov, {"lat": [], "cost_per_img": []})
            for m in datasets.values():
                n_valid = m.get("n_valid") or 0
                if not n_valid:
                    continue
                lat = m.get("mean_latency_s")
                cost = m.get("total_cost_usd")
                if lat is not None:
                    bucket["lat"].append(lat)
                if cost is not None:
                    bucket["cost_per_img"].append(cost / n_valid)
    out: dict[str, dict[str, float]] = {}
    for prov, fallback in FALLBACK_VLM.items():
        bucket = agg.get(prov, {"lat": [], "cost_per_img": []})
        out[prov] = {
            "latency_s": float(np.mean(bucket["lat"])) if bucket["lat"] else fallback["latency_s"],
            "cost_per_img": (
                float(np.mean(bucket["cost_per_img"]))
                if bucket["cost_per_img"]
                else fallback["cost_per_img"]
            ),
        }
    return out


def horizontal_bar(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    colors: list[str],
    xlabel: str,
    value_fmt,
    log: bool = True,
) -> None:
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.grid(axis="x", which="both", linestyle=":", color="grey", alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    offset = 1.08 if log else 1.02
    for yi, v in zip(y, values):
        ax.text(v * offset, yi, value_fmt(v), va="center", fontsize=11)


def make_speed_figure(vlm: dict[str, dict[str, float]]) -> None:
    rows: list[tuple[str, float, str]] = []
    for name, meta in CLASSICAL.items():
        rows.append((name, meta["ms"], meta["color"]))
    for prov, (label, color) in VLM_DISPLAY.items():
        rows.append((label, vlm[prov]["latency_s"] * 1000.0, color))

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    def _fmt(v: float) -> str:
        return f"{v:.0f} ms" if v < 1000 else f"{v/1000:.1f} s"

    for log, suffix, scale_note in [(True, "log", "log scale"),
                                    (False, "linear", "linear scale")]:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        horizontal_bar(
            ax, labels, values, colors,
            xlabel=f"Inference latency  (ms per image, {scale_note})  —  lower is better ←",
            value_fmt=_fmt,
            log=log,
        )
        ax.set_title(
            "One image, end-to-end: a ResNet finishes ~1000× sooner than a VLM",
            fontsize=14, pad=12,
        )
        if log:
            ax.set_xlim(left=3, right=max(values) * 3.5)
        else:
            ax.set_xlim(left=0, right=max(values) * 1.18)
        fig.tight_layout()
        out = FIG_DIR / f"inference_speed_{suffix}.png"
        fig.savefig(out, dpi=160, facecolor="white")
        plt.close(fig)
        print(f"  → {out}")


def make_cost_figure(vlm: dict[str, dict[str, float]]) -> None:
    rows: list[tuple[str, float, str]] = []
    for name, meta in CLASSICAL.items():
        cost_per_1k = meta["ms"] / 1000.0 * T4_USD_PER_SEC * 1000.0
        rows.append((name, cost_per_1k, meta["color"]))
    for prov, (label, color) in VLM_DISPLAY.items():
        rows.append((label, vlm[prov]["cost_per_img"] * 1000.0, color))

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    def _fmt(v: float) -> str:
        if v < 0.01:
            return f"${v*1000:.2f} / M imgs"
        if v < 1:
            return f"${v:.3f}"
        return f"${v:.2f}"

    for log, suffix, scale_note in [(True, "log", "log scale"),
                                    (False, "linear", "linear scale")]:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        horizontal_bar(
            ax, labels, values, colors,
            xlabel=f"Cost  (USD per 1 000 images, {scale_note})  —  lower is better ←",
            value_fmt=_fmt,
            log=log,
        )
        ax.set_title(
            "Classical models: Azure T4 GPU time.   VLMs: API token spend.",
            fontsize=14, pad=12,
        )
        if log:
            ax.set_xlim(left=min(values) * 0.3, right=max(values) * 4)
        else:
            ax.set_xlim(left=0, right=max(values) * 1.18)
        fig.tight_layout()
        out = FIG_DIR / f"inference_cost_{suffix}.png"
        fig.savefig(out, dpi=160, facecolor="white")
        plt.close(fig)
        print(f"  → {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    vlm = load_vlm_stats()
    for prov, stats in vlm.items():
        print(
            f"  {prov:8s}  latency={stats['latency_s']:5.2f}s   "
            f"cost/img=${stats['cost_per_img']:.4f}"
        )
    make_speed_figure(vlm)
    make_cost_figure(vlm)


if __name__ == "__main__":
    main()
