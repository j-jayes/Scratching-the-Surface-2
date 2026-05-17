"""Phase 5 — bake-off comparison figure generator.

Reads results JSONs from all four approaches and generates the headline
'cross-domain generalisation gap' figure for the slide deck.

Requires:
  results/resnet50/metrics.json         (Phase 2a)
  results/yolo/eval_metrics.json        (Phase 2b)
  results/vlm/summary_phase3a_*.json    (Phase 3a — uses most recent)
  results/hybrid/summary_*.json         (Phase 3b — uses most recent)

Output:
  figures/bakeoff/f1_comparison.png
  figures/bakeoff/auc_comparison.png
  figures/bakeoff/cost_comparison.png

Usage:
    uv run python scripts/make_comparison_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("figures/bakeoff")
RESULTS_DIR = Path("results")

DATASETS_ORDER = ["kolektor_test", "gc10_test"]
DATASET_LABELS = {"kolektor_test": "KolektorSDD2\n(held-out)", "gc10_test": "GC10-DET\n(held-out)"}

APPROACH_COLORS = {
    "ResNet50":          "#1f77b4",
    "ResNet50+FT":       "#aec7e8",
    "YOLO11s":           "#ff7f0e",
    "VLM zero-shot":     "#2ca02c",
    "YOLO→VLM":          "#d62728",
    "YOLO-bootstrap":    "#9467bd",
}


def load_resnet() -> dict[str, dict]:
    p = RESULTS_DIR / "resnet50" / "metrics.json"
    if not p.exists():
        return {}
    with p.open() as f:
        m = json.load(f)

    # Prefer threshold-optimised F1 if available
    sweep_path = RESULTS_DIR / "resnet50" / "threshold_sweep.json"
    sweep: dict = {}
    if sweep_path.exists():
        with sweep_path.open() as f:
            sweep = json.load(f).get("sets", {})

    out = {}
    for ds in DATASETS_ORDER:
        if ds not in m.get("eval", {}):
            continue
        base = m["eval"][ds]
        f1_val = sweep.get(ds, {}).get("f1_at_opt_threshold") or base["f1"]
        out[ds] = {"f1": f1_val, "roc_auc": base.get("roc_auc")}
    return out


def load_resnet_ft() -> dict[str, dict]:
    """Load fine-tuned ResNet50 results (kolektor domain adaptation).

    Prefers metrics.json eval F1 (which may use TTA) over threshold_sweep
    opt-threshold F1, for backward compatibility with v1 results.
    """
    p = RESULTS_DIR / "resnet50_kolektor_ft" / "metrics.json"
    if not p.exists():
        return {}
    with p.open() as f:
        m = json.load(f)

    # Fall back to threshold_sweep only if metrics.json eval is missing
    sweep_path = RESULTS_DIR / "resnet50_kolektor_ft" / "threshold_sweep.json"
    sweep: dict = {}
    if sweep_path.exists():
        with sweep_path.open() as f:
            sweep = json.load(f).get("sets", {})

    out = {}
    for ds in DATASETS_ORDER:
        base = m.get("eval", {}).get(ds, {})
        if base.get("f1") is not None:
            # Prefer metrics.json eval F1 (TTA-aware)
            out[ds] = {"f1": base["f1"], "roc_auc": base.get("roc_auc")}
        elif ds in sweep:
            # Fall back to threshold_sweep
            sw = sweep[ds]
            f1_val = sw.get("f1_at_opt_threshold") or sw.get("f1_at_0.5")
            out[ds] = {"f1": f1_val, "roc_auc": sw.get("roc_auc")}
    return out


def load_yolo_main() -> dict[str, dict]:
    p = RESULTS_DIR / "yolo" / "eval_metrics.json"
    if not p.exists():
        return {}
    with p.open() as f:
        m = json.load(f)
    return {ds: {"f1": m[ds]["best_f1"], "roc_auc": m[ds].get("roc_auc")}
            for ds in DATASETS_ORDER if ds in m}


def load_vlm_zeroshot() -> dict[str, dict]:
    """Load the most recent Phase 3a summary, picking the best provider by average F1."""
    summaries = sorted(RESULTS_DIR.glob("vlm/summary_phase3a_*.json"))
    if not summaries:
        return {}
    with summaries[-1].open() as f:
        m = json.load(f)
    providers = m.get("providers", {})
    if not providers:
        return {}
    # Pick provider with highest average F1 across datasets
    best_prov, best_prov_results = None, None
    best_avg_f1 = -1.0
    for prov_name, prov_results in providers.items():
        f1s = [prov_results.get(ds, {}).get("f1", 0) for ds in DATASETS_ORDER if ds in prov_results]
        avg_f1 = sum(f1s) / len(f1s) if f1s else 0
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_prov = prov_name
            best_prov_results = prov_results
    if best_prov_results is None:
        return {}
    return {ds: {"f1": best_prov_results.get(ds, {}).get("f1", 0),
                 "roc_auc": best_prov_results.get(ds, {}).get("roc_auc")}
            for ds in DATASETS_ORDER if ds in best_prov_results}


def load_hybrid() -> dict[str, dict]:
    summaries = sorted(RESULTS_DIR.glob("hybrid/summary_*.json"))
    if not summaries:
        return {}
    with summaries[-1].open() as f:
        m = json.load(f)
    return {ds: {"f1": m["results"][ds]["f1"], "roc_auc": m["results"][ds].get("roc_auc")}
            for ds in DATASETS_ORDER if ds in m.get("results", {})}


def load_bootstrap_yolo() -> dict[str, dict]:
    """Eval results for YOLO trained on VLM pseudo-labels."""
    p = RESULTS_DIR / "yolo_bootstrap" / "eval_metrics.json"
    if not p.exists():
        return {}
    with p.open() as f:
        m = json.load(f)
    return {ds: {"f1": m[ds]["best_f1"], "roc_auc": m[ds].get("roc_auc")}
            for ds in DATASETS_ORDER if ds in m}


def bar_chart(
    data: dict[str, dict[str, dict]],  # approach → dataset → metrics
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple = (0, 1.05),
    datasets: list[str] | None = None,
) -> None:
    approaches = list(data.keys())
    datasets   = datasets if datasets is not None else DATASETS_ORDER
    x = np.arange(len(datasets))
    width = 0.8 / max(len(approaches), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, approach in enumerate(approaches):
        values = [data[approach].get(ds, {}).get(metric, 0) or 0 for ds in datasets]
        bars = ax.bar(
            x + i * width, values, width,
            label=approach,
            color=APPROACH_COLORS.get(approach, f"C{i}"),
        )
        for bar, v in zip(bars, values):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set(
        title=title,
        xlabel="Evaluation dataset",
        ylabel=ylabel,
        xticks=x + width * (len(approaches) - 1) / 2,
        xticklabels=[DATASET_LABELS.get(d, d) for d in datasets],
        ylim=ylim,
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> None:
    # Load all available results
    all_data: dict[str, dict[str, dict]] = {}

    resnet = load_resnet()
    if resnet:
        all_data["ResNet50"] = resnet
        print("✓ ResNet50 results loaded")
    else:
        print("⚠ ResNet50 results not found — run train_resnet.py + analyse_resnet.py")

    resnet_ft = load_resnet_ft()
    if resnet_ft:
        all_data["ResNet50+FT"] = resnet_ft
        print("✓ ResNet50+FT results loaded")
    else:
        print("ℹ ResNet50+FT results not yet available")

    yolo = load_yolo_main()
    if yolo:
        all_data["YOLO11s"] = yolo
        print("✓ YOLO11s results loaded")
    else:
        print("⚠ YOLO11s results not found — run eval_yolo.py")

    vlm = load_vlm_zeroshot()
    if vlm:
        all_data["VLM zero-shot"] = vlm
        print("✓ VLM zero-shot results loaded")
    else:
        print("⚠ VLM zero-shot results not found — run eval_vlm_zeroshot.py")

    hybrid = load_hybrid()
    if hybrid:
        all_data["YOLO→VLM"] = hybrid
        print("✓ Hybrid results loaded")
    else:
        print("⚠ Hybrid results not found — run eval_hybrid.py")

    bootstrap = load_bootstrap_yolo()
    if bootstrap:
        all_data["YOLO-bootstrap"] = bootstrap
        print("✓ YOLO-bootstrap results loaded")
    else:
        print("ℹ YOLO-bootstrap results not yet available")

    if not all_data:
        print("\nNo results available yet. Run experiments first.")
        return

    print(f"\nGenerating figures with {len(all_data)} approaches...")

    bar_chart(all_data, "f1",      "Cross-Domain F1 — Bake-Off",      "F1 Score",  FIGURES_DIR / "f1_comparison.png")
    bar_chart(all_data, "roc_auc", "Cross-Domain ROC-AUC — Bake-Off", "ROC-AUC",   FIGURES_DIR / "auc_comparison.png")

    # Classical-only Kolektor-only variant: used on the slide that motivates
    # the pivot to VLMs, before the VLM has been introduced.
    classical_only = {k: v for k, v in all_data.items() if k not in {"VLM zero-shot", "YOLO→VLM"}}
    if classical_only:
        bar_chart(classical_only, "f1", "Classical recipes — held-out F1 on KolektorSDD2", "F1 Score",
                  FIGURES_DIR / "f1_comparison_classical.png",
                  datasets=["kolektor_test"])

    # VLM-vs-classical Kolektor F1 chart — pull best VLM F1 from latest flagship summary.
    flagship = sorted(RESULTS_DIR.glob("vlm/flagship_summary_*.json"))
    if flagship and classical_only:
        with flagship[-1].open() as f:
            fs = json.load(f)
        best_prov, best_f1 = None, -1.0
        prov_label_map = {"openai": "GPT-5.4", "gemini": "Gemini-2.5-pro", "azure": "GPT-4.1-mini"}
        for prov, ds in fs.get("results", {}).items():
            f1v = ds.get("kolektor_test", {}).get("f1")
            if f1v is not None and f1v > best_f1:
                best_f1 = f1v
                best_prov = prov
        if best_prov is not None:
            vlm_label = f"VLM ({prov_label_map.get(best_prov, best_prov)}, zero-shot)"
            combined = dict(classical_only)
            combined[vlm_label] = {"kolektor_test": {"f1": best_f1}}
            APPROACH_COLORS[vlm_label] = "#2ca02c"
            bar_chart(combined, "f1",
                      "Best VLM vs every classical recipe — KolektorSDD2 F1",
                      "F1 Score",
                      FIGURES_DIR / "f1_kolektor_vlm_vs_classical.png",
                      datasets=["kolektor_test"])

    # Print summary table
    print(f"\n{'Approach':<20} {'kolektor F1':>12} {'gc10 F1':>10} {'kolektor AUC':>14} {'gc10 AUC':>10}")
    print("-" * 70)
    for approach, ds_metrics in all_data.items():
        kol_f1  = ds_metrics.get("kolektor_test", {}).get("f1", None)
        gc10_f1 = ds_metrics.get("gc10_test",     {}).get("f1", None)
        kol_auc = ds_metrics.get("kolektor_test", {}).get("roc_auc", None)
        gc10_auc= ds_metrics.get("gc10_test",     {}).get("roc_auc", None)
        fmt = lambda v: f"{v:.4f}" if v is not None else "   n/a"
        print(f"{approach:<20} {fmt(kol_f1):>12} {fmt(gc10_f1):>10} {fmt(kol_auc):>14} {fmt(gc10_auc):>10}")

    # Bootstrap-focused comparison (for bootstrap results slide)
    bootstrap_story_approaches = ["ResNet50", "ResNet50+FT", "VLM zero-shot", "YOLO-bootstrap"]
    bootstrap_data = {k: v for k, v in all_data.items() if k in bootstrap_story_approaches}
    if len(bootstrap_data) >= 2:
        bar_chart(
            bootstrap_data,
            "f1",
            "Bootstrap story: can pseudo-labels replace human annotation?",
            "F1 Score",
            FIGURES_DIR / "bootstrap_comparison.png",
        )


if __name__ == "__main__":
    main()
