"""Side-by-side qualitative: ResNet Grad-CAM heatmap vs YOLO bounding boxes.

For each of four hand-picked images we render:
  input | ResNet Grad-CAM (+ p̂) | YOLO11s detections (+ box conf)

The point of the slide: the two classical models produce *different shaped
outputs* — a scalar with a heatmap vs. one-or-more typed bounding boxes — so
the operational consequences are different even when the scoreboard F1s are
close.

Output: figures/bakeoff/resnet_vs_yolo_qual.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.defect_dataset import build_val_transform
from src.models.resnet_baseline import build_model, get_device

RESNET_CKPT = Path("models/resnet50_best.pt")
YOLO_CKPT = Path("results/yolo/yolo11s_defect/weights/best.pt")
OUT_FIG = Path("figures/bakeoff/resnet_vs_yolo_qual.png")
INPUT_SIZE = 224
YOLO_IMGSZ = 1024
YOLO_CONF = 0.10

PICKS: list[dict] = [
    {
        "tag": "IN-DOMAIN",
        "dataset": "Severstal",
        "image_id": "0014fce06.jpg",
        "path": "data/raw/severstal/train_images/0014fce06.jpg",
    },
    {
        "tag": "IN-DOMAIN",
        "dataset": "Severstal",
        "image_id": "008621629.jpg",
        "path": "data/raw/severstal/train_images/008621629.jpg",
    },
    {
        "tag": "OUT-OF-DOMAIN",
        "dataset": "KolektorSDD2",
        "image_id": "20042.png",
        "path": "data/raw/kolektor/test/20042.png",
    },
    {
        "tag": "OUT-OF-DOMAIN",
        "dataset": "GC10-DET",
        "image_id": "img_05_425505000_00051.jpg",
        "path": "data/raw/gc10/GC10-DET/images/img_05_425505000_00051.jpg",
    },
]


def prep_resnet(path: str):
    img = Image.open(path).convert("RGB")
    tfm = build_val_transform(INPUT_SIZE)
    tensor = tfm(img).unsqueeze(0)
    rgb_full = np.asarray(img, dtype=np.float32) / 255.0
    return img, tensor, rgb_full


def cam_for(model, tensor, device, target_hw: tuple[int, int]) -> np.ndarray:
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])(
        input_tensor=tensor.to(device), targets=None
    )[0]
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (target_hw[1], target_hw[0]), Image.BILINEAR
    )
    return np.asarray(cam_img, dtype=np.float32) / 255.0


def render() -> None:
    device = get_device()
    resnet = build_model().to(device).eval()
    resnet.load_state_dict(torch.load(RESNET_CKPT, map_location=device))
    yolo = YOLO(str(YOLO_CKPT))

    n = len(PICKS)
    fig, axes = plt.subplots(
        n, 3, figsize=(15, 3.6 * n),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, pick in enumerate(PICKS):
        sq, tensor, rgb_full = prep_resnet(pick["path"])
        cam = cam_for(resnet, tensor, device, rgb_full.shape[:2])
        overlay = show_cam_on_image(rgb_full, cam, use_rgb=True)
        with torch.no_grad():
            p = torch.sigmoid(resnet(tensor.to(device))).item()

        full = sq
        yres = yolo.predict(
            source=str(pick["path"]), imgsz=YOLO_IMGSZ,
            verbose=False, conf=YOLO_CONF,
        )[0]
        boxes = yres.boxes
        confs = (
            boxes.conf.cpu().numpy().tolist() if boxes is not None and len(boxes) else []
        )
        xyxys = (
            boxes.xyxy.cpu().numpy().tolist() if boxes is not None and len(boxes) else []
        )

        tag_color = "#1b9e77" if "IN" in pick["tag"] and "OUT" not in pick["tag"] else "#d95f02"

        axes[i, 0].imshow(full)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].set_title(
            f"{pick['tag']} — {pick['dataset']}",
            fontsize=11, fontweight="bold", color=tag_color,
        )

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        axes[i, 1].set_title(
            f"ResNet50 Grad-CAM   p̂={p:.2f}",
            fontsize=11, fontweight="bold", color="#7570b3",
        )

        axes[i, 2].imshow(full)
        axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])
        for (x1, y1, x2, y2), c in zip(xyxys, confs):
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="#00ffff", facecolor="none",
            )
            axes[i, 2].add_patch(rect)
            axes[i, 2].text(
                x1, max(0, y1 - 4), f"defect {c:.2f}",
                fontsize=9, color="white",
                bbox=dict(facecolor="#005577", edgecolor="none", pad=1.5),
            )
        n_boxes = len(xyxys)
        max_conf = max(confs) if confs else 0.0
        title = (
            f"YOLO11s — {n_boxes} box{'es' if n_boxes != 1 else ''}"
            + (f"   max conf={max_conf:.2f}" if n_boxes else "   (no detection)")
        )
        axes[i, 2].set_title(title, fontsize=11, fontweight="bold", color="#1f77b4")

    fig.suptitle(
        "Two output modalities — scalar+heatmap (ResNet) vs typed bounding boxes (YOLO)",
        fontsize=14, y=0.995,
    )
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {OUT_FIG}")


if __name__ == "__main__":
    render()
