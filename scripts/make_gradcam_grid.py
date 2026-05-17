"""Grad-CAM visualisation for ResNet50 baseline vs ResNet50+FT v2.

Two figures driving slides section G + J:

  1. gradcam_train_vs_test.png — 2 rows × 4 cols
     Same 8 images as `make_domain_gap_figure.py` (re-uses domain_gap_picks.csv).
     Shows the baseline ResNet50 layer4[-1] CAM overlaid on each image.
     Train-domain CAMs should peak on the defect; test-domain CAMs scatter.

  2. gradcam_vs_vlm_rationale.png — N rows × 3 cols  (image | CAM | VLM rationale text)
     For section J — contrasts the heatmap with the VLM's natural-language reasoning.
     VLM rationale is read from results/vlm/flagship_rationale_*.jsonl
     (produced by scripts/eval_vlm_flagship.py).

Usage:
    uv run python scripts/make_gradcam_grid.py
    uv run python scripts/make_gradcam_grid.py --rationale-file results/vlm/flagship_rationale_<ts>.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.defect_dataset import build_val_transform, IMAGENET_MEAN, IMAGENET_STD
from src.models.resnet_baseline import build_model, get_device

PICKS_CSV   = Path("figures/bakeoff/domain_gap_picks.csv")
OUT_DIR     = Path("figures/explainers")
CHECKPOINT  = Path("models/resnet50_best.pt")
INPUT_SIZE  = 224


def load_picks() -> pd.DataFrame:
    if not PICKS_CSV.exists():
        raise SystemExit(
            f"Missing {PICKS_CSV}. Run scripts/make_domain_gap_figure.py first."
        )
    return pd.read_csv(PICKS_CSV)


def prep_input(path: str) -> tuple[Image.Image, torch.Tensor, np.ndarray]:
    """Return PIL square, normalized tensor, and float32 [0,1] rgb array for CAM overlay."""
    img = Image.open(path).convert("RGB")
    tfm = build_val_transform(INPUT_SIZE)
    tensor = tfm(img).unsqueeze(0)
    # rebuild the un-normalised square for overlay
    square = img
    w, h = square.size
    scale = INPUT_SIZE / min(w, h)
    sq = square.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    left = (sq.size[0] - INPUT_SIZE) // 2
    top  = (sq.size[1] - INPUT_SIZE) // 2
    sq = sq.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    rgb = np.asarray(sq, dtype=np.float32) / 255.0
    return sq, tensor, rgb


def compute_cam(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    target_layer = model.layer4[-1]
    cam_engine = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam_engine(input_tensor=tensor.to(device), targets=None)[0]
    return grayscale_cam   # H×W in [0,1]


def grid_train_vs_test(picks: pd.DataFrame, model, device, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for ax, (_, row) in zip(axes.flat, picks.iterrows()):
        sq, tensor, rgb = prep_input(row["path"])
        cam = compute_cam(model, tensor, device)
        overlay = show_cam_on_image(rgb, cam, use_rgb=True)
        ax.imshow(overlay)
        ax.set_title(f"{row['dataset']}\np(defect)={row['prob']:.2f}",
                     fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("TRAIN domain", fontsize=13, fontweight="bold", color="#1b9e77")
    axes[1, 0].set_ylabel("TEST domain",  fontsize=13, fontweight="bold", color="#d95f02")
    fig.suptitle("What is the ResNet looking at? — Grad-CAM (layer4[-1])",
                 fontsize=15, y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def grid_cam_vs_rationale(picks: pd.DataFrame, model, device,
                          rationales: dict[str, str], out: Path,
                          max_rows: int = 4) -> None:
    rows = picks.head(max_rows)
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3.6 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1, 1.6]})
    if len(rows) == 1:
        axes = axes.reshape(1, -1)
    for i, (_, row) in enumerate(rows.iterrows()):
        sq, tensor, rgb = prep_input(row["path"])
        cam = compute_cam(model, tensor, device)
        overlay = show_cam_on_image(rgb, cam, use_rgb=True)

        axes[i, 0].imshow(sq); axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].set_title(f"{row['dataset']} — input", fontsize=10, fontweight="bold")

        axes[i, 1].imshow(overlay); axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        axes[i, 1].set_title(f"ResNet Grad-CAM (p={row['prob']:.2f})",
                             fontsize=10, fontweight="bold", color="#7570b3")

        axes[i, 2].axis("off")
        key = row["image_id"]
        text = rationales.get(key, "(no VLM rationale captured for this image)")
        wrapped = "\n".join(textwrap.wrap(text, width=48))
        axes[i, 2].text(0.02, 0.5, wrapped, fontsize=10.5, va="center",
                        family="serif",
                        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff8dc",
                                  edgecolor="#d95f02", linewidth=1.5))
        axes[i, 2].set_title("VLM (gpt-5.4) reasoning",
                             fontsize=10, fontweight="bold", color="#d95f02")
    fig.suptitle("Two languages of explanation — pixel heatmap vs natural-language rationale",
                 fontsize=14, y=0.995)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def load_rationales(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            img_id = r.get("image_id") or r.get("img_id")
            reasoning = (r.get("parsed") or {}).get("reasoning", "")
            if img_id and reasoning:
                out[img_id] = reasoning
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationale-file", type=Path, default=None,
                        help="JSONL produced by eval_vlm_flagship.py with reasoning text.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picks = load_picks()

    device = get_device()
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))

    grid_train_vs_test(picks, model, device, OUT_DIR / "gradcam_train_vs_test.png")

    rationales = load_rationales(args.rationale_file)
    grid_cam_vs_rationale(picks, model, device, rationales,
                          OUT_DIR / "gradcam_vs_vlm_rationale.png", max_rows=4)


if __name__ == "__main__":
    main()
