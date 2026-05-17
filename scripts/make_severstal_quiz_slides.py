"""Render close-up Severstal sample slides showing defect masks.

Produces four PNGs in figures/datasets/severstal_quiz/ — two "guess" frames
showing raw close-ups, two "reveal" frames overlaying the ground-truth
pixel mask + class label. Each frame holds 2 images side-by-side.

Class IDs follow the Severstal Kaggle convention (1..4); names are the
community-standard labels used in the few-shot grid.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SEV = ROOT / "data/raw/severstal"
TRAIN_CSV = SEV / "train.csv"
IMG_DIR = SEV / "train_images"
OUT = ROOT / "figures/datasets/severstal_quiz"
OUT.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 1600, 256

CLASS_NAMES = {1: "pitting", 2: "inclusion", 3: "scratch", 4: "patch"}
CLASS_COLORS = {
    1: "#e7298a",   # pink — pitting
    2: "#1b9e77",   # green — inclusion
    3: "#d95f02",   # orange — scratch
    4: "#7570b3",   # purple — patch
}


def rle_to_mask(rle: str, w: int = IMG_W, h: int = IMG_H) -> np.ndarray:
    """Decode Severstal column-major RLE → binary HxW mask."""
    mask = np.zeros(w * h, dtype=np.uint8)
    nums = list(map(int, rle.split()))
    for s, l in zip(nums[0::2], nums[1::2]):
        mask[s - 1 : s - 1 + l] = 1
    return mask.reshape((w, h)).T  # column-major → (H, W)


def crop_around_mask(img: np.ndarray, mask: np.ndarray, crop_w: int = 700) -> tuple[np.ndarray, np.ndarray]:
    """Crop a `crop_w`-wide window centered on the mask's defect centroid.

    For 1600×256 strips this gives a meaningful close-up while keeping the
    full strip height.
    """
    h, w = img.shape[:2]
    if mask.sum() == 0:
        cx = w // 2
    else:
        cols = np.where(mask.any(axis=0))[0]
        cx = int((cols.min() + cols.max()) / 2)
    half = crop_w // 2
    x1 = max(0, min(cx - half, w - crop_w))
    x2 = x1 + crop_w
    return img[:, x1:x2], mask[:, x1:x2]


def pick_image(df: pd.DataFrame, class_id: int, min_pixels: int = 600,
               max_pixels: int = 5000, offset: int = 0) -> tuple[str, str]:
    """Pick a `class_id` defect whose mask area is in [min_pixels, max_pixels]."""
    sub = df[df["ClassId"] == class_id].copy()
    sub["n_pix"] = sub["EncodedPixels"].apply(
        lambda r: sum(int(x) for x in r.split()[1::2])
    )
    sub = sub[(sub["n_pix"] >= min_pixels) & (sub["n_pix"] <= max_pixels)]
    sub = sub.sort_values("n_pix").reset_index(drop=True)
    row = sub.iloc[(len(sub) // 2 + offset) % len(sub)]
    return row["ImageId"], row["EncodedPixels"]


def pick_normal(df: pd.DataFrame, all_ids: list[str], offset: int = 0) -> str:
    """Pick a clean normal: not in defect CSV, with mid-range brightness.

    Filters out the all-black edge frames and over-exposed frames so the
    audience sees a representative grey strip.
    """
    defect_ids = set(df["ImageId"])
    candidates = [i for i in all_ids if i not in defect_ids]
    clean = []
    for img_id in candidates[offset : offset + 800]:
        arr = np.array(Image.open(IMG_DIR / img_id).convert("L"))
        # Require: bright, low contrast, no big dark patches (which read as defects)
        if (
            arr.mean() > 110
            and arr.std() < 28
            and (arr < 60).mean() < 0.01
        ):
            clean.append(img_id)
        if len(clean) >= 5:
            break
    return clean[0] if clean else candidates[offset]


def render_pair(
    panel_specs: list[dict],
    out_path: Path,
    reveal: bool,
    suptitle: str,
) -> None:
    """Render one slide-figure with two close-up panels."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5))
    for ax, spec in zip(axes, panel_specs):
        img = np.array(Image.open(spec["path"]).convert("RGB"))
        mask = spec.get("mask")
        crop_img, crop_mask = crop_around_mask(img, mask if mask is not None else np.zeros(img.shape[:2], dtype=np.uint8))

        ax.imshow(crop_img)

        if reveal and mask is not None and crop_mask.sum() > 0:
            colour = np.array(matplotlib.colors.to_rgb(spec["colour"]))
            overlay = np.zeros((*crop_mask.shape, 4))
            overlay[crop_mask == 1] = [*colour, 0.55]
            ax.imshow(overlay)
            ax.set_title(
                f"DEFECT — {spec['label']}",
                fontsize=14, fontweight="bold",
                color=spec["colour"], loc="left", pad=4,
            )
        elif reveal:
            ax.set_title("NORMAL", fontsize=14, fontweight="bold",
                         color="#666", loc="left", pad=4)
        else:
            ax.set_title("?", fontsize=14, fontweight="bold",
                         color="#333", loc="left", pad=4)

        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out_path.relative_to(ROOT)}")


def main() -> None:
    df = pd.read_csv(TRAIN_CSV).dropna(subset=["EncodedPixels"])
    all_ids = sorted(p.name for p in IMG_DIR.glob("*.jpg"))

    # Pair A: normal + pitting (subtle, audience hard)
    pit_id, pit_rle = pick_image(df, 1, min_pixels=300, max_pixels=1500, offset=5)
    norm_a = pick_normal(df, all_ids, offset=200)

    # Pair B: scratch + patch (more obvious — visceral reveal)
    scr_id, scr_rle = pick_image(df, 3, min_pixels=4000, max_pixels=12000, offset=20)
    pat_id, pat_rle = pick_image(df, 4, min_pixels=10000, max_pixels=40000, offset=10)

    pair_a = [
        {"path": IMG_DIR / norm_a, "mask": None, "label": None, "colour": None},
        {"path": IMG_DIR / pit_id,
         "mask": rle_to_mask(pit_rle),
         "label": CLASS_NAMES[1], "colour": CLASS_COLORS[1]},
    ]
    pair_b = [
        {"path": IMG_DIR / scr_id,
         "mask": rle_to_mask(scr_rle),
         "label": CLASS_NAMES[3], "colour": CLASS_COLORS[3]},
        {"path": IMG_DIR / pat_id,
         "mask": rle_to_mask(pat_rle),
         "label": CLASS_NAMES[4], "colour": CLASS_COLORS[4]},
    ]

    render_pair(pair_a, OUT / "quiz_a_guess.png", reveal=False,
                suptitle="Severstal hot-rolled strip — which one is defective?")
    render_pair(pair_a, OUT / "quiz_a_reveal.png", reveal=True,
                suptitle="…and here is the pixel-level ground truth")
    render_pair(pair_b, OUT / "quiz_b_guess.png", reveal=False,
                suptitle="Another pair — both have defects. Where? What kind?")
    render_pair(pair_b, OUT / "quiz_b_reveal.png", reveal=True,
                suptitle="…annotated by a metallurgist, pixel by pixel")

    print(f"\nPicked: pit={pit_id} scr={scr_id} pat={pat_id} norm={norm_a}")


if __name__ == "__main__":
    main()
