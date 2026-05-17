"""Overnight learning-curve: how many VLM-labelled images do we need?

Story this script answers
-------------------------
Pretend we have a fresh, unlabelled production stream (proxy: Kolektor train,
~2 300 images, ~10 % defective). We use the **best VLM we measured** (GPT-5.4
zero-shot) to auto-label every image we draw. That VLM is not perfect — its
per-class accuracy on Kolektor test is measurable from the existing JSONL
results — so the labels we get are noisy.

We then train a small ResNet on those noisy labels and ask: how does
held-out F1 on the real Kolektor test set grow with the number of
VLM-labelled images N?

Key methodological points (per the user's brief)
------------------------------------------------
1. **Draws are random.** No class balancing, no "give me more defects"
   stratification — you wouldn't know what's there. Pool reflects the natural
   ~10 / 90 imbalance.
2. **Labels are noisy.** Each ground-truth label is flipped according to the
   measured per-class error rate of the VLM (1 − TPR for positives,
   1 − TNR for negatives), simulating what GPT-5.4 would actually produce.
3. **Multiple seeds** at each N so we get a mean and a spread.
4. **Cost is tracked.** N × mean VLM call cost — gives the dollar-axis you'd
   actually justify the labelling spend against.

Outputs
-------
- results/labelling_curve/runs_<ts>.csv      (one row per (N, seed))
- results/labelling_curve/summary_<ts>.json  (config, per-N aggregates, cost)
- figures/bakeoff/labelling_curve.png        (F1 vs N with seed band + cost)

Usage (overnight, all defaults):
    nohup uv run python scripts/labelling_curve.py \
        > results/labelling_curve/run.log 2>&1 &

Or fast sanity check:
    uv run python scripts/labelling_curve.py --sizes 50 200 --seeds 0 --epochs 2
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.defect_dataset import (
    DefectDataset,
    build_train_transform,
    build_val_transform,
)
from src.models.resnet_baseline import (
    build_model,
    evaluate,
    get_device,
    train_epoch,
)

POOL_MANIFEST = Path("data/processed/kolektor_manifest.parquet")
VLM_JSONL_GLOB = "results/vlm/flagship_openai_kolektor_test_*.jsonl"
RESULTS_DIR = Path("results/labelling_curve")
FIG_PATH = Path("figures/bakeoff/labelling_curve.png")
DEFAULT_SIZES = [25, 50, 100, 200, 400, 800, 1600]
DEFAULT_SEEDS = [0, 1, 2]


@dataclass
class VLMNoise:
    """Per-class accuracy of the labelling VLM, plus its cost per call."""
    tpr: float
    tnr: float
    cost_per_call: float
    source: str
    n_pos: int = 0
    n_neg: int = 0


def derive_vlm_noise() -> VLMNoise:
    files = sorted(glob.glob(VLM_JSONL_GLOB))
    if not files:
        raise FileNotFoundError(f"No flagship Kolektor JSONL matching {VLM_JSONL_GLOB}")
    path = files[-1]
    tp = fn = tn = fp = 0
    costs: list[float] = []
    for line in open(path):
        r = json.loads(line)
        y = r.get("label")
        p = r.get("pred_defect")
        if y is None or p is None:
            continue
        if y == 1 and p:
            tp += 1
        elif y == 1 and not p:
            fn += 1
        elif y == 0 and not p:
            tn += 1
        elif y == 0 and p:
            fp += 1
        if r.get("cost_usd") is not None:
            costs.append(float(r["cost_usd"]))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    cost = float(np.mean(costs)) if costs else 0.007
    return VLMNoise(
        tpr=tpr, tnr=tnr, cost_per_call=cost,
        source=path, n_pos=tp + fn, n_neg=tn + fp,
    )


def noisy_labels(true: np.ndarray, noise: VLMNoise, rng: np.random.Generator) -> np.ndarray:
    """Flip each label per the VLM's empirical per-class error rate."""
    out = true.copy().astype(int)
    pos = out == 1
    neg = out == 0
    flip_pos = rng.random(pos.sum()) > noise.tpr      # 1 - TPR  ⇒ false negative
    flip_neg = rng.random(neg.sum()) > noise.tnr      # 1 - TNR  ⇒ false positive
    out[np.where(pos)[0][flip_pos]] = 0
    out[np.where(neg)[0][flip_neg]] = 1
    return out


def one_run(
    pool: pd.DataFrame,
    test: pd.DataFrame,
    n: int,
    seed: int,
    noise: VLMNoise,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    input_size: int,
    num_workers: int,
    device: torch.device,
) -> dict:
    rng = np.random.default_rng(seed)
    if n >= len(pool):
        sample = pool.sample(n=len(pool), random_state=seed).reset_index(drop=True)
    else:
        sample = pool.sample(n=n, random_state=seed).reset_index(drop=True)

    true = sample["has_defect"].to_numpy().astype(int)
    noisy = noisy_labels(true, noise, rng)
    sample = sample.assign(has_defect=noisy)

    label_acc = float((true == noisy).mean())
    n_pos = int(noisy.sum())
    n_neg = int(n - n_pos)

    train_tfm = build_train_transform(input_size)
    val_tfm = build_val_transform(input_size)
    train_loader = DataLoader(
        DefectDataset(sample, train_tfm),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        DefectDataset(test, val_tfm),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    torch.manual_seed(seed)
    model = build_model().to(device)
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(epochs):
        train_epoch(model, train_loader, optimiser, criterion, device, epoch=ep)
    metrics = evaluate(model, test_loader, device, threshold=0.5)
    elapsed = time.time() - t0

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "n_train": n,
        "seed": seed,
        "n_pos_noisy": n_pos,
        "n_neg_noisy": n_neg,
        "label_accuracy": label_acc,
        "test_f1": metrics["f1"],
        "test_auc": metrics.get("roc_auc"),
        "test_acc": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "label_cost_usd": n * noise.cost_per_call,
        "train_time_s": elapsed,
    }


def render_figure(df: pd.DataFrame, noise: VLMNoise, baseline_f1: float | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = df.groupby("n_train").agg(
        f1_mean=("test_f1", "mean"),
        f1_std=("test_f1", "std"),
        cost=("label_cost_usd", "mean"),
    ).reset_index()
    agg["f1_std"] = agg["f1_std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(agg["n_train"], agg["f1_mean"], "-o", color="#1b9e77",
            label=f"ResNet50 on VLM-labelled pool  (mean over seeds)")
    ax.fill_between(
        agg["n_train"],
        agg["f1_mean"] - agg["f1_std"],
        agg["f1_mean"] + agg["f1_std"],
        color="#1b9e77", alpha=0.18, label="±1 σ (seeds)",
    )
    if baseline_f1 is not None:
        ax.axhline(baseline_f1, color="#7570b3", ls="--",
                   label=f"GPT-5.4 zero-shot F1 = {baseline_f1:.2f}")

    ax.set_xscale("log")
    ax.set_xlabel("N images sent to the VLM for labelling (log)")
    ax.set_ylabel("Held-out F1 on Kolektor test")
    ax.set_ylim(0, 1)
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_title(
        "How many VLM-labelled images before the student catches the teacher?\n"
        f"Noise model: TPR={noise.tpr:.2f}, TNR={noise.tnr:.2f}, "
        f"cost ≈ ${noise.cost_per_call:.4f}/img"
    )

    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ticks = [t for t in agg["n_train"]]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"${n*noise.cost_per_call:.1f}" for n in ticks], fontsize=9)
    ax2.set_xlabel("Cumulative labelling cost (USD)")

    ax.legend(loc="lower right")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {FIG_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--baseline-f1", type=float, default=0.79,
                    help="VLM zero-shot F1 to draw as horizontal reference.")
    ap.add_argument("--tpr", type=float, default=None,
                    help="Override measured TPR.")
    ap.add_argument("--tnr", type=float, default=None,
                    help="Override measured TNR.")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_csv = RESULTS_DIR / f"runs_{ts}.csv"
    summary_path = RESULTS_DIR / f"summary_{ts}.json"

    noise = derive_vlm_noise()
    if args.tpr is not None: noise.tpr = args.tpr
    if args.tnr is not None: noise.tnr = args.tnr
    print(
        f"VLM noise: TPR={noise.tpr:.3f}  TNR={noise.tnr:.3f}  "
        f"cost=${noise.cost_per_call:.4f}/call  (from {Path(noise.source).name})"
    )

    pool_all = pd.read_parquet(POOL_MANIFEST)
    pool = pool_all[pool_all["split"] == "train"].reset_index(drop=True)
    test = pool_all[pool_all["split"] == "test"].reset_index(drop=True)
    print(
        f"Pool (kolektor train):  N={len(pool)}  "
        f"defects={int(pool['has_defect'].sum())} "
        f"({pool['has_defect'].mean():.1%})"
    )
    print(
        f"Test (kolektor test):   N={len(test)}  "
        f"defects={int(test['has_defect'].sum())}"
    )

    device = get_device()
    print(f"Device: {device}")
    print(f"Sweep: N={args.sizes}  seeds={args.seeds}  epochs={args.epochs}")

    rows: list[dict] = []
    total = len(args.sizes) * len(args.seeds)
    k = 0
    for n in args.sizes:
        for seed in args.seeds:
            k += 1
            print(f"\n[{k}/{total}] N={n} seed={seed} ─────────────────────────")
            try:
                row = one_run(
                    pool, test, n, seed, noise,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    input_size=args.input_size,
                    num_workers=args.num_workers,
                    device=device,
                )
            except Exception as e:
                print(f"  ! FAILED: {e}")
                continue
            rows.append(row)
            print(
                f"  → F1={row['test_f1']:.3f}  AUC={row['test_auc']:.3f}  "
                f"label_acc={row['label_accuracy']:.3f}  "
                f"cost=${row['label_cost_usd']:.2f}  "
                f"({row['train_time_s']:.0f}s)"
            )
            pd.DataFrame(rows).to_csv(runs_csv, index=False)

    df = pd.DataFrame(rows)
    summary = {
        "timestamp": ts,
        "noise": {
            "tpr": noise.tpr, "tnr": noise.tnr,
            "cost_per_call": noise.cost_per_call,
            "source": noise.source,
            "n_pos_observed": noise.n_pos, "n_neg_observed": noise.n_neg,
        },
        "config": vars(args),
        "per_n": (
            df.groupby("n_train").agg(
                f1_mean=("test_f1", "mean"),
                f1_std=("test_f1", "std"),
                auc_mean=("test_auc", "mean"),
                cost=("label_cost_usd", "mean"),
            ).reset_index().to_dict(orient="records")
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n  → {runs_csv}\n  → {summary_path}")
    render_figure(df, noise, args.baseline_f1)


if __name__ == "__main__":
    main()
