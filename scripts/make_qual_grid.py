"""Generate a 4×N qualitative prediction grid for the slide deck.

Loads the best ResNet50 checkpoint and runs on a sample of test images,
sorting by error type: TP, FP, FN, TN. Saves a composite figure.

Usage:
    uv run python scripts/make_qual_grid.py
    uv run python scripts/make_qual_grid.py --dataset gc10_test --n 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.defect_dataset import build_val_transform, load_split
from src.models.resnet_baseline import build_model, get_device

CHECKPOINT = Path("models/resnet50_best.pt")
FIGURES_DIR = Path("figures/qual")
PROCESSED_DIR = Path("data/processed")

DATASETS = {
    "severstal_test": {
        "manifests": [PROCESSED_DIR / "severstal_manifest.parquet"],
        "split": "test",
    },
    "kolektor_test": {
        "manifests": [PROCESSED_DIR / "kolektor_manifest.parquet"],
        "split": "test",
    },
    "gc10_test": {
        "manifests": [PROCESSED_DIR / "gc10_manifest.parquet"],
        "split": "test",
        "supplement": PROCESSED_DIR / "severstal_manifest.parquet",
        "supplement_split": "test",
    },
}

CATEGORY_COLORS = {"TP": "#2ca02c", "FP": "#ff7f0e", "FN": "#d62728", "TN": "#1f77b4"}
CATEGORY_ORDER  = ["TP", "FP", "FN", "TN"]


@torch.no_grad()
def score_images(
    model: torch.nn.Module,
    paths: list[str],
    transform,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    all_probs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch = torch.stack([
            transform(Image.open(p).convert("RGB"))
            for p in batch_paths
        ]).to(device)
        logits = model(batch).squeeze(1).cpu()
        all_probs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(all_probs)


def make_grid(
    images: list[Image.Image],
    titles: list[str],
    colors: list[str],
    out_path: Path,
    cols: int = 4,
) -> None:
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes_flat = axes.flatten() if rows > 1 else list(axes)

    for ax, img, title, color in zip(axes_flat, images, titles, colors):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, color=color, fontweight="bold", pad=3)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    for ax in axes_flat[len(images):]:
        ax.axis("off")

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=k) for k, c in CATEGORY_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=10,
               title="Prediction category", title_fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(out_path.stem.replace("_", " ").title(), fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gc10_test", choices=list(DATASETS.keys()))
    parser.add_argument("--n",       type=int, default=12, help="Images per error category")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT} — run train_resnet.py first.")
        return

    device = get_device()
    model  = build_model().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    transform = build_val_transform(224)

    ds_cfg = DATASETS[args.dataset]
    df = load_split(
        [str(m) for m in ds_cfg["manifests"]],
        split=ds_cfg["split"],
        supplement_normals_path=str(ds_cfg["supplement"]) if "supplement" in ds_cfg else None,
        supplement_split=ds_cfg.get("supplement_split"),
    )
    print(f"Scoring {len(df)} images for '{args.dataset}'...")
    probs = score_images(model, df["path"].tolist(), transform, device)
    preds = (probs >= args.threshold).astype(int)
    labels = df["has_defect"].astype(int).values

    # Categorise
    categories: dict[str, list[int]] = {"TP": [], "FP": [], "FN": [], "TN": []}
    for i, (label, pred) in enumerate(zip(labels, preds)):
        if label == 1 and pred == 1:  categories["TP"].append(i)
        elif label == 0 and pred == 1: categories["FP"].append(i)
        elif label == 1 and pred == 0: categories["FN"].append(i)
        else:                           categories["TN"].append(i)

    print(f"  TP={len(categories['TP'])}  FP={len(categories['FP'])}"
          f"  FN={len(categories['FN'])}  TN={len(categories['TN'])}")

    rng = np.random.default_rng(args.seed)
    selected: list[tuple[int, str]] = []
    for cat in CATEGORY_ORDER:
        idxs = categories[cat]
        sample_idxs = rng.choice(idxs, min(args.n, len(idxs)), replace=False).tolist()
        selected.extend((i, cat) for i in sample_idxs)

    # Sort by category order for display
    cat_order_map = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    selected.sort(key=lambda t: cat_order_map[t[1]])

    images_pil, titles, colors = [], [], []
    for idx, cat in selected:
        row = df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img.thumbnail((224, 224), Image.LANCZOS)
        score = probs[idx]
        images_pil.append(img)
        titles.append(f"{cat}  p={score:.2f}")
        colors.append(CATEGORY_COLORS[cat])

    out_path = FIGURES_DIR / f"resnet_qual_{args.dataset}.png"
    make_grid(images_pil, titles, colors, out_path, cols=args.n)
    print(f"Grid saved → {out_path}")


if __name__ == "__main__":
    main()
