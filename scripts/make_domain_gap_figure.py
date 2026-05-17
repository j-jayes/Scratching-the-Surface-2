"""Generate the train-vs-test domain-gap figure used in slides section F.

Two figures:
  1. domain_gap_train_vs_test.png — 2 rows × 4 cols, plain
     - row 1: 4 train-domain defect images (Severstal + NEU-DET)
     - row 2: 4 test-domain defect images (KolektorSDD2 + GC10-DET)
  2. domain_gap_with_baseline_preds.png — same images, overlaid with
     ResNet50 baseline probability + ground-truth label, to make the
     generalisation gap visceral.

Usage:
    uv run python scripts/make_domain_gap_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.defect_dataset import build_val_transform
from src.models.resnet_baseline import build_model, get_device

PROCESSED = Path("data/processed")
OUT_DIR = Path("figures/bakeoff")
CHECKPOINT = Path("models/resnet50_best.pt")

TRAIN_PICKS = [
    ("severstal_manifest.parquet", "Severstal", "train"),
    ("severstal_manifest.parquet", "Severstal", "train"),
    ("neu_det_manifest.parquet",   "NEU-DET",   "train"),
    ("neu_det_manifest.parquet",   "NEU-DET",   "train"),
]
TEST_PICKS = [
    ("kolektor_manifest.parquet", "KolektorSDD2", "test"),
    ("kolektor_manifest.parquet", "KolektorSDD2", "test"),
    ("kolektor_manifest.parquet", "KolektorSDD2", "test"),
    ("kolektor_manifest.parquet", "KolektorSDD2", "test"),
]


def pick_defects(picks: list[tuple[str, str, str]], seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    seen: dict[str, pd.DataFrame] = {}
    for manifest, label, split in picks:
        if manifest not in seen:
            seen[manifest] = pd.read_parquet(PROCESSED / manifest)
        df = seen[manifest]
        df = df[(df["split"] == split) & (df["has_defect"])].copy()
        idx = int(rng.integers(0, len(df)))
        r = df.iloc[idx]
        rows.append({"path": r["path"], "dataset": label, "image_id": r["image_id"]})
    return rows


@torch.no_grad()
def score(model, paths: list[str], tfm, device) -> np.ndarray:
    batch = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths]).to(device)
    return torch.sigmoid(model(batch).squeeze(1)).cpu().numpy()


def make_grid(rows: list[dict], titles: list[str], colours: list[str], out: Path,
              suptitle: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for ax, r, t, c in zip(axes.flat, rows, titles, colours):
        img = Image.open(r["path"]).convert("RGB")
        ax.imshow(img)
        ax.set_title(t, fontsize=11, color=c, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(c); spine.set_linewidth(3); spine.set_visible(True)
    # Row labels
    axes[0, 0].set_ylabel("TRAIN domain\n(Severstal + NEU-DET)",
                          fontsize=13, fontweight="bold", color="#1b9e77")
    axes[1, 0].set_ylabel("TEST domain\n(KolektorSDD2 — held-out)",
                          fontsize=13, fontweight="bold", color="#d95f02")
    fig.suptitle(suptitle, fontsize=15, y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = pick_defects(TRAIN_PICKS, seed=7)
    test_rows  = pick_defects(TEST_PICKS,  seed=11)
    rows = train_rows + test_rows

    # Plain grid (no preds)
    titles_plain = [f"{r['dataset']}\n{r['image_id']}" for r in rows]
    colours_plain = ["#1b9e77"] * 4 + ["#d95f02"] * 4
    make_grid(rows, titles_plain, colours_plain,
              OUT_DIR / "domain_gap_train_vs_test.png",
              "Train ≠ Test — every defect below is real, ground-truth labelled")

    # Annotated grid with baseline ResNet50 probability
    device = get_device()
    model = build_model().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    tfm = build_val_transform()
    probs = score(model, [r["path"] for r in rows], tfm, device)

    titles_pred: list[str] = []
    colours_pred: list[str] = []
    THRESH = 0.475   # canonical baseline threshold (resnet50)
    for r, p in zip(rows, probs):
        pred = "defect" if p >= THRESH else "normal"
        correct = pred == "defect"   # all picks are defects
        c = "#2ca02c" if correct else "#d62728"
        titles_pred.append(f"{r['dataset']}\nGT=defect | ResNet={pred} (p={p:.2f})")
        colours_pred.append(c)
    make_grid(rows, titles_pred, colours_pred,
              OUT_DIR / "domain_gap_with_baseline_preds.png",
              "Same images — ResNet50 baseline trained on top row, predicting on both rows")

    # Persist picks so the Grad-CAM script can reuse the exact same images
    pd.DataFrame(rows).assign(prob=probs).to_csv(
        OUT_DIR / "domain_gap_picks.csv", index=False
    )
    print(f"  → {OUT_DIR / 'domain_gap_picks.csv'}")


if __name__ == "__main__":
    main()
