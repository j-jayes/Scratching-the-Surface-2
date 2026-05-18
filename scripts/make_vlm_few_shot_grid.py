"""Render the few-shot grid with coloured row backgrounds.

Green (#1b9e77) for the no-defect row, orange (#d95f02) for all defect rows
— both taken from the ColorBrewer Dark2 palette used throughout the
presentation.

Images
  • no_defect  — 3 normal strips from the Severstal manifest
  • pitting    — data/raw/neu/pitted_surface
  • inclusion  — data/raw/neu/inclusion
  • scratch    — data/raw/neu/scratches
  • patch      — data/raw/neu/patches

Output: website/assets/explainers/vlm_few_shot_grid.png
"""
from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "website/assets/explainers/vlm_few_shot_grid.png"

# ColorBrewer Dark2
GREEN  = "#1b9e77"
ORANGE = "#d95f02"

SHOTS = 3

CLASSES: list[dict] = [
    {
        "label": "No Defect",
        "color": GREEN,
        "images": None,   # populated at runtime from the Severstal manifest
        "description": (
            "Uniform grey surface. May show normal mill texture, faint roller lines, "
            "mild brightness gradient, or thin horizontal banding from the rolling "
            "process. NO dark spots, NO embedded particles, NO long linear marks, "
            "NO discoloured patches. Mill texture is NOT a defect."
        ),
    },
    {
        "label": "Pitting",
        "color": ORANGE,
        "images": ROOT / "data/raw/neu/pitted_surface",
        "description": (
            "One or more small, dark, roughly circular spots or shallow holes in the "
            "surface. Spots are typically 2–15 px wide, often clustered, and sit "
            "BELOW the surface plane (darker than the surrounding metal)."
        ),
    },
    {
        "label": "Inclusion",
        "color": ORANGE,
        "images": ROOT / "data/raw/neu/inclusion",
        "description": (
            "Foreign material embedded IN the steel: irregular dark blobs or streaks "
            "with high local contrast against a clean grey background. Edges are "
            "jagged, not linear. Often elongated along the rolling direction but "
            "not pencil-thin like a scratch."
        ),
    },
    {
        "label": "Scratch",
        "color": ORANGE,
        "images": ROOT / "data/raw/neu/scratches",
        "description": (
            "Long, thin, straight or gently curved LINEAR mark, much longer than it "
            "is wide. Aspect ratio > 10:1. Usually a single bright or dark line; can "
            "appear in groups of parallel lines from a tool drag."
        ),
    },
    {
        "label": "Patch",
        "color": ORANGE,
        "images": ROOT / "data/raw/neu/patches",
        "description": (
            "A LARGE region of the strip whose brightness or texture differs from "
            "the surrounding metal. Covers > 10% of the visible area. Edges are "
            "diffuse, not sharp. Often appears as a lighter or darker zone of "
            "rolled-in scale or oxidation."
        ),
    },
]


def main() -> None:
    # ── populate no-defect images — hand-picked clean Severstal strips ─────
    _sev = ROOT / "data/raw/severstal/train_images"
    CLASSES[0]["images"] = [
        _sev / "00031f466.jpg",   # uniform mid-grey, faint roller lines
        _sev / "000418bfc.jpg",   # clean dark strip, minimal texture
        _sev / "007f28bba.jpg",   # clean bright strip, normal longitudinal texture
    ]

    n_rows = len(CLASSES)
    n_cols = SHOTS + 1   # image columns + description column

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.4, n_rows * 2.4),
        gridspec_kw={"width_ratios": [1] * SHOTS + [3.0]},
    )
    fig.patch.set_facecolor("white")

    for ri, cls in enumerate(CLASSES):
        row_color = cls["color"]
        bg_rgba   = mcolors.to_rgba(row_color, alpha=0.13)

        # ── resolve image list ─────────────────────────────────────────────
        if isinstance(cls["images"], list):
            imgs = cls["images"]
        else:
            imgs = sorted(cls["images"].glob("*.jpg"))[:SHOTS]

        # ── image columns ──────────────────────────────────────────────────
        for ci in range(SHOTS):
            ax = axes[ri, ci]
            ax.set_facecolor(bg_rgba)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(row_color)
                spine.set_linewidth(2)

            if ci < len(imgs):
                img = Image.open(imgs[ci]).convert("L")
                ax.imshow(img, cmap="gray", aspect="auto")

            # Row label on the left edge of col-0
            if ci == 0:
                ax.set_ylabel(
                    cls["label"],
                    fontsize=18,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=14,
                    fontweight="bold",
                    color=row_color,
                )

        # ── description column ─────────────────────────────────────────────
        ax_desc = axes[ri, -1]
        ax_desc.set_facecolor(bg_rgba)
        ax_desc.axis("off")
        for spine in ax_desc.spines.values():
            spine.set_edgecolor(row_color)
            spine.set_linewidth(2)

        wrapped = "\n".join(wrap(cls["description"], width=46))
        ax_desc.text(
            0.05, 0.5,
            wrapped,
            fontsize=13,
            va="center",
            ha="left",
            linespacing=1.45,
            color="#1a1a2e",
            transform=ax_desc.transAxes,
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor=mcolors.to_rgba(row_color, alpha=0.10),
                edgecolor=row_color,
                linewidth=1.8,
            ),
        )

    fig.suptitle(
        "Few-shot prompt: 3 reference images × 5 classes + visual description",
        fontsize=17,
        fontweight="bold",
        color="#1a1a2e",
        y=1.01,
    )
    fig.tight_layout(pad=0.4, h_pad=0.9, w_pad=0.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
