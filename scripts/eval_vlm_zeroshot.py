"""Phase 3a — Zero-shot VLM defect detection batch evaluation.

Runs the structured CoT prompt over a stratified sample of both
held-out test domains (KolektorSDD2 + GC10-DET) for one or more providers.

Usage:
    # Dry-run cost estimate (no API calls)
    uv run python scripts/eval_vlm_zeroshot.py --dry-run

    # Default: gpt-4o-mini on 100 GC10 + 100 Kolektor images
    uv run python scripts/eval_vlm_zeroshot.py

    # All providers
    uv run python scripts/eval_vlm_zeroshot.py --providers openai azure gemini

    # Bigger sample, flagship model
    uv run python scripts/eval_vlm_zeroshot.py --model gpt-4o --n 200

Results written to:
    results/vlm/<provider>_<model>_<set>_<timestamp>.jsonl  (per-image)
    results/vlm/summary_<timestamp>.json                     (aggregate)

Figures written to:
    figures/vlm/roc_curves.png
    figures/vlm/provider_comparison.png
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import f1_score, roc_auc_score

from src.config import OPENAI_MODELS, COST_LEDGER_PATH
from src.prompts.defect_analysis import JSON_SCHEMA, SYSTEM, USER_T3_COT
from src.vlm_clients import call_openai, call_azure, call_gemini
from PIL import Image

RESULTS_DIR = Path("results/vlm")
FIGURES_DIR = Path("figures/vlm")
PROCESSED_DIR = Path("data/processed")

DATASETS = {
    "kolektor_test": {
        "manifest":   PROCESSED_DIR / "kolektor_manifest.parquet",
        "split":      "test",
        "has_normals": True,   # KolektorSDD2 has both defect and normal images
    },
    "gc10_test": {
        "manifest":   PROCESSED_DIR / "gc10_manifest.parquet",
        "split":      "test",
        "has_normals": False,  # all-defect; normals come from Severstal held-out
    },
}

SEVERSTAL_NORMALS_MANIFEST = PROCESSED_DIR / "severstal_manifest.parquet"

PROVIDER_DEFAULTS = {
    "openai": OPENAI_MODELS["mini"],   # gpt-4o-mini
    "azure":  None,                    # uses AOAI_DEPLOYMENT from .env
    "gemini": "gemini-2.0-flash",
}


# ── Sampling ─────────────────────────────────────────────────────────────────

def sample_dataset(name: str, cfg: dict, n: int, seed: int = 42) -> pd.DataFrame:
    """Return a balanced sample of `n` images (n//2 pos + n//2 neg where possible)."""
    df = pd.read_parquet(cfg["manifest"])
    df = df[df["split"] == cfg["split"]].copy()

    if not cfg["has_normals"]:
        # Supplement with Severstal test normals
        sev = pd.read_parquet(SEVERSTAL_NORMALS_MANIFEST)
        normals = sev[(sev["split"] == "test") & (~sev["has_defect"])].copy()
        rng = np.random.default_rng(seed)
        normals = normals.sample(min(len(df), len(normals)), random_state=seed)
        df = pd.concat([df, normals], ignore_index=True)

    positives = df[df["has_defect"]].sample(min(n // 2, df["has_defect"].sum()), random_state=seed)
    negatives = df[~df["has_defect"]].sample(min(n // 2, (~df["has_defect"]).sum()), random_state=seed)
    sampled = pd.concat([positives, negatives], ignore_index=True).sample(frac=1, random_state=seed)
    return sampled.reset_index(drop=True)


# ── Cost estimate ─────────────────────────────────────────────────────────────

def estimate_cost(n_images: int, model: str) -> float:
    """Rough estimate: 1,800 input tokens + 400 output tokens per image."""
    from src.config import PRICING
    prices = PRICING.get(model, {"in": 2.50, "out": 15.00})
    cost = n_images * (1_800 * prices["in"] + 400 * prices["out"]) / 1_000_000
    return cost


# ── Single-image inference ────────────────────────────────────────────────────

def run_one(
    provider: str,
    model: str | None,
    image_path: str,
    label: int,
    dataset: str,
    img_id: str,
) -> dict:
    img = Image.open(image_path).convert("RGB")
    t0 = time.perf_counter()

    if provider == "openai":
        resp = call_openai(
            model=model or OPENAI_MODELS["mini"],
            system=SYSTEM,
            user=USER_T3_COT,
            image=img,
            schema=JSON_SCHEMA,
            phase="phase3a",
            note=f"{dataset}/{img_id}",
        )
    elif provider == "azure":
        resp = call_azure(
            system=SYSTEM,
            user=USER_T3_COT,
            image=img,
            schema=JSON_SCHEMA,
            phase="phase3a",
            note=f"{dataset}/{img_id}",
        )
    elif provider == "gemini":
        from src.config import GEMINI_MODELS
        resp = call_gemini(
            model=model or GEMINI_MODELS["mid"],
            system=SYSTEM,
            user=USER_T3_COT,
            image=img,
            schema=JSON_SCHEMA,
            phase="phase3a",
            note=f"{dataset}/{img_id}",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    pred_defect: bool | None = None
    confidence: float = 0.5
    if resp.parsed and isinstance(resp.parsed, dict):
        pred_defect = bool(resp.parsed.get("has_defect"))
        confidence = float(resp.parsed.get("confidence", 0.5))

    return {
        "img_id":       img_id,
        "dataset":      dataset,
        "label":        label,
        "pred_defect":  pred_defect,
        "confidence":   confidence,
        "provider":     provider,
        "model":        resp.model,
        "in_tok":       resp.in_tok,
        "out_tok":      resp.out_tok,
        "cost_usd":     resp.cost_usd,
        "latency_s":    resp.latency_s,
        "error":        resp.error,
        "parsed":       resp.parsed,
        "raw":          resp.raw,
    }


# ── Metrics aggregation ───────────────────────────────────────────────────────

def summarise(records: list[dict]) -> dict:
    valid = [r for r in records if r["pred_defect"] is not None and r["error"] is None]
    if not valid:
        return {"n": len(records), "n_valid": 0}

    labels  = np.array([r["label"]                    for r in valid])
    preds   = np.array([int(r["pred_defect"])          for r in valid])
    # Score = probability of defect: conf if predicted positive, 1-conf if predicted negative
    confs   = np.array([r["confidence"] if r["pred_defect"] else 1.0 - r["confidence"]
                        for r in valid])

    metrics: dict = {
        "n":            len(records),
        "n_valid":      len(valid),
        "n_errors":     len(records) - len(valid),
        "f1":           round(float(f1_score(labels, preds, zero_division=0)), 4),
        "total_cost_usd": round(sum(r["cost_usd"] for r in records), 4),
        "mean_latency_s": round(float(np.mean([r["latency_s"] for r in valid])), 2),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(labels, confs)), 4)
    return metrics


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_provider_comparison(summary_by_provider: dict[str, dict[str, dict]], out_path: Path) -> None:
    """Bar chart: F1 per provider × dataset."""
    providers = list(summary_by_provider.keys())
    datasets = list(next(iter(summary_by_provider.values())).keys())
    x = np.arange(len(datasets))
    width = 0.8 / max(len(providers), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors
    for i, prov in enumerate(providers):
        f1s = [summary_by_provider[prov].get(ds, {}).get("f1", 0) for ds in datasets]
        bars = ax.bar(x + i * width, f1s, width, label=prov, color=colors[i % 10])
        for bar, f1 in zip(bars, f1s):
            if f1 > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{f1:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set(
        title="VLM Zero-Shot F1 by Provider & Dataset",
        xlabel="Dataset",
        ylabel="F1 Score",
        xticks=x + width * (len(providers) - 1) / 2,
        xticklabels=[d.replace("_", " ") for d in datasets],
        ylim=[0, 1.05],
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", default=["openai"],
                        choices=["openai", "azure", "gemini"],
                        help="Which VLM providers to evaluate.")
    parser.add_argument("--model", default=None,
                        help="Model name override (applies to openai and gemini).")
    parser.add_argument("--n", type=int, default=100,
                        help="Images per dataset (split 50/50 pos/neg where possible).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost without making API calls.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Cost estimate
    total_images = args.n * len(DATASETS)
    for provider in args.providers:
        model = args.model or PROVIDER_DEFAULTS.get(provider, "gpt-4o-mini")
        est = estimate_cost(total_images, model or "gpt-4o-mini")
        print(f"Estimated cost for {provider}/{model}: ${est:.2f}  ({total_images} images)")

    if args.dry_run:
        print("\nDry-run complete — no API calls made.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_records: list[dict] = []
    summary_by_provider: dict[str, dict[str, dict]] = {}

    for provider in args.providers:
        model = args.model or PROVIDER_DEFAULTS.get(provider)
        print(f"\n{'='*60}")
        print(f"Provider: {provider}  model: {model}")
        print("="*60)
        summary_by_provider[provider] = {}

        for ds_name, ds_cfg in DATASETS.items():
            sample = sample_dataset(ds_name, ds_cfg, args.n, seed=args.seed)
            print(f"\nDataset: {ds_name}  ({len(sample)} images: "
                  f"{sample['has_defect'].sum()} pos + {(~sample['has_defect']).sum()} neg)")

            jsonl_path = RESULTS_DIR / f"{provider}_{(model or 'default').replace('/', '-')}_{ds_name}_{ts}.jsonl"
            records: list[dict] = []

            with jsonl_path.open("w") as f:
                for i, row in sample.iterrows():
                    result = run_one(
                        provider=provider,
                        model=model,
                        image_path=str(row["path"]),
                        label=int(row["has_defect"]),
                        dataset=ds_name,
                        img_id=Path(row["path"]).stem,
                    )
                    records.append(result)
                    all_records.append(result)
                    f.write(json.dumps(result) + "\n")
                    f.flush()

                    status = "✓" if result["error"] is None else "✗"
                    print(
                        f"  [{i+1:3d}/{len(sample)}] {status} "
                        f"label={result['label']} pred={result['pred_defect']} "
                        f"conf={result['confidence']:.2f}  ${result['cost_usd']:.4f}  "
                        f"{result['latency_s']:.1f}s"
                        + (f"  ERR: {result['error'][:60]}" if result["error"] else ""),
                        flush=True,
                    )

            metrics = summarise(records)
            summary_by_provider[provider][ds_name] = metrics
            print(f"\n  Summary: F1={metrics.get('f1', 'n/a')}  "
                  f"AUC={metrics.get('roc_auc', 'n/a')}  "
                  f"Cost=${metrics['total_cost_usd']:.3f}  "
                  f"Errors={metrics.get('n_errors', 0)}")
            print(f"  Results → {jsonl_path}")

    # Save aggregate summary
    summary_path = RESULTS_DIR / f"summary_phase3a_{ts}.json"
    with summary_path.open("w") as f:
        json.dump({"timestamp": ts, "providers": summary_by_provider}, f, indent=2)
    print(f"\nAggregate summary → {summary_path}")

    # Plot comparison figure
    if len(args.providers) > 1:
        plot_provider_comparison(summary_by_provider, FIGURES_DIR / f"provider_comparison_{ts}.png")

    # Print final comparison table
    print(f"\n{'Provider':<15} {'Dataset':<22} {'F1':>6} {'AUC':>7} {'Cost $':>8}")
    print("-" * 62)
    for prov, ds_results in summary_by_provider.items():
        for ds, m in ds_results.items():
            auc_s = f"{m.get('roc_auc', float('nan')):.4f}" if "roc_auc" in m else "  n/a "
            print(f"{prov:<15} {ds:<22} {m.get('f1', 0):>6.4f} {auc_s:>7} {m.get('total_cost_usd', 0):>8.3f}")


if __name__ == "__main__":
    main()
