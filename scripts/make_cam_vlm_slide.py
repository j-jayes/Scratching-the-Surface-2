"""Build the section-J slide: 1 normal + 3 defective close-ups, each shown as
input | Grad-CAM | live GPT-5.4 rationale.

The 4 picks are hand-curated (one per dataset family) so the speaker can talk
about variety. The script calls gpt-5.4 once per image — total cost ≈ $0.02 —
and writes the rationales to a timestamped JSONL alongside the figure.

Outputs
  figures/explainers/gradcam_vs_vlm_rationale.png
  results/vlm/section_j_rationale_<ts>.jsonl
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime
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

from src.config import OPENAI_MODELS
from src.data.defect_dataset import build_val_transform
from src.models.resnet_baseline import build_model, get_device
from src.prompts.defect_analysis import JSON_SCHEMA, SYSTEM, USER_T3_COT
from src.vlm_clients import call_openai

CHECKPOINT = Path("models/resnet50_best.pt")
INPUT_SIZE = 224
OUT_FIG = Path("figures/explainers/gradcam_vs_vlm_rationale.png")
RESULTS = Path("results/vlm")

PICKS: list[dict] = [
    {
        "tag": "NORMAL",
        "dataset": "Severstal",
        "image_id": "87b82c9cb.jpg",
        "path": "data/raw/severstal/train_images/87b82c9cb.jpg",
        "label": 0,
    },
    {
        "tag": "DEFECT",
        "dataset": "KolektorSDD2",
        "image_id": "20042.png",
        "path": "data/raw/kolektor/test/20042.png",
        "label": 1,
    },
    {
        "tag": "DEFECT",
        "dataset": "Severstal",
        "image_id": "c6d3371a8.jpg",
        "path": "data/raw/severstal/train_images/c6d3371a8.jpg",
        "label": 1,
    },
    {
        "tag": "DEFECT",
        "dataset": "GC10-DET",
        "image_id": "img_05_425505000_00051.jpg",
        "path": "data/raw/gc10/GC10-DET/images/img_05_425505000_00051.jpg",
        "label": 1,
    },
]


def prep_input(path: str) -> tuple[Image.Image, torch.Tensor, np.ndarray]:
    img = Image.open(path).convert("RGB")
    tfm = build_val_transform(INPUT_SIZE)
    tensor = tfm(img).unsqueeze(0)
    w, h = img.size
    scale = INPUT_SIZE / min(w, h)
    sq = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    left = (sq.size[0] - INPUT_SIZE) // 2
    top = (sq.size[1] - INPUT_SIZE) // 2
    sq = sq.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    rgb = np.asarray(sq, dtype=np.float32) / 255.0
    return sq, tensor, rgb


def compute_cam(model, tensor, device) -> np.ndarray:
    target_layer = model.layer4[-1]
    engine = GradCAM(model=model, target_layers=[target_layer])
    return engine(input_tensor=tensor.to(device), targets=None)[0]


def call_gpt5(image_path: str, image_id: str, dataset: str) -> dict:
    img = Image.open(image_path).convert("RGB")
    resp = call_openai(
        model=OPENAI_MODELS["flagship"],
        system=SYSTEM,
        user=USER_T3_COT,
        image=img,
        schema=JSON_SCHEMA,
        phase="phaseJ",
        note=f"section_j/{dataset}/{image_id}",
    )
    parsed = resp.parsed if isinstance(resp.parsed, dict) else {}
    return {
        "image_id": image_id,
        "dataset": dataset,
        "model": resp.model,
        "in_tok": resp.in_tok,
        "out_tok": resp.out_tok,
        "cost_usd": resp.cost_usd,
        "latency_s": resp.latency_s,
        "error": resp.error,
        "parsed": resp.parsed,
        "raw": resp.raw,
        "has_defect": parsed.get("has_defect"),
        "confidence": parsed.get("confidence"),
        "reasoning": parsed.get("reasoning", "") or "",
    }


def render(picks: list[dict], rationales: dict[str, dict], model, device) -> None:
    n = len(picks)
    fig, axes = plt.subplots(
        n, 3, figsize=(15, 3.5 * n),
        gridspec_kw={"width_ratios": [1, 1, 1.8]},
    )
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, pick in enumerate(picks):
        sq, tensor, rgb = prep_input(pick["path"])
        cam = compute_cam(model, tensor, device)
        overlay = show_cam_on_image(rgb, cam, use_rgb=True)

        with torch.no_grad():
            prob = torch.sigmoid(model(tensor.to(device))).item()

        tag_color = "#1b9e77" if pick["tag"] == "NORMAL" else "#d95f02"

        axes[i, 0].imshow(sq)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 0].set_title(
            f"{pick['tag']} — {pick['dataset']}",
            fontsize=11, fontweight="bold", color=tag_color,
        )

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        axes[i, 1].set_title(
            f"ResNet Grad-CAM   (p̂={prob:.2f})",
            fontsize=11, fontweight="bold", color="#7570b3",
        )

        axes[i, 2].axis("off")
        r = rationales.get(pick["image_id"], {})
        pred = r.get("has_defect")
        conf = r.get("confidence")
        verdict = (
            "DEFECT" if pred is True else "NORMAL" if pred is False else "—"
        )
        header = f"VLM verdict: {verdict}"
        if conf is not None:
            header += f"   (conf {float(conf):.2f})"
        reasoning = r.get("reasoning") or "(no rationale captured)"
        wrapped = "\n".join(textwrap.wrap(reasoning, width=46))
        body = f"{header}\n\n{wrapped}"
        axes[i, 2].text(
            0.02, 0.5, body, fontsize=10.5, va="center", family="serif",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#fff8dc",
                edgecolor="#d95f02",
                linewidth=1.5,
            ),
        )
        axes[i, 2].set_title(
            "GPT-5.4 reasoning",
            fontsize=11, fontweight="bold", color="#d95f02",
        )

    fig.suptitle(
        "Two languages of explanation — pixel heatmap vs natural-language rationale",
        fontsize=14, y=0.995,
    )
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {OUT_FIG}")


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_out = RESULTS / f"section_j_rationale_{ts}.jsonl"
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"Calling GPT-5.4 on {len(PICKS)} images …")
    rationales: dict[str, dict] = {}
    total_cost = 0.0
    with jsonl_out.open("w") as f:
        for pick in PICKS:
            print(f"  [{pick['tag']:6s}] {pick['dataset']:14s} {pick['image_id']}")
            rec = call_gpt5(pick["path"], pick["image_id"], pick["dataset"])
            rec["label"] = pick["label"]
            f.write(json.dumps(rec) + "\n")
            rationales[pick["image_id"]] = rec
            total_cost += rec["cost_usd"] or 0
            print(
                f"          → has_defect={rec['has_defect']}  "
                f"conf={rec['confidence']}  ${rec['cost_usd']:.4f}  "
                f"{rec['latency_s']:.1f}s"
            )
    print(f"  → {jsonl_out}   total ≈ ${total_cost:.4f}")

    print("Rendering figure …")
    device = get_device()
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    render(PICKS, rationales, model, device)


if __name__ == "__main__":
    main()
