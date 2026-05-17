"""Gradio live demo — AQ Group rolled-metal defect detection bake-off.

Shows all four approaches side-by-side for any uploaded image:
  1. ResNet50 binary score
  2. YOLO11s detection bbox overlay
  3. VLM zero-shot JSON analysis
  4. Hybrid YOLO→VLM verdict

Launch:
    uv run python app/app.py
    uv run gradio app/app.py   # with auto-reload

Deploy to Hugging Face Spaces:
    gradio deploy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Model paths ────────────────────────────────────────────────────────────────
RESNET_CHECKPOINT = Path("models/resnet50_best.pt")
YOLO_WEIGHTS      = Path("results/yolo/yolo11s_defect/weights/best.pt")

# ── Gallery: sample images from GC10 test set ─────────────────────────────────
GALLERY_MANIFEST  = Path("data/processed/gc10_manifest.parquet")
N_GALLERY         = 12


# ── Lazy model loading (load once on first call) ──────────────────────────────
_resnet_model = None
_yolo_model   = None


def _load_resnet():
    global _resnet_model
    if _resnet_model is None and RESNET_CHECKPOINT.exists():
        import torch
        from src.models.resnet_baseline import build_model, get_device
        from src.data.defect_dataset import build_val_transform

        device = get_device()
        m = build_model().to(device)
        m.load_state_dict(torch.load(RESNET_CHECKPOINT, map_location=device, weights_only=True))
        m.eval()
        _resnet_model = (m, device, build_val_transform(224))
    return _resnet_model


def _load_yolo():
    global _yolo_model
    if _yolo_model is None and YOLO_WEIGHTS.exists():
        from ultralytics import YOLO
        _yolo_model = YOLO(str(YOLO_WEIGHTS))
    return _yolo_model


# ── Inference helpers ─────────────────────────────────────────────────────────

def resnet_predict(img: Image.Image) -> tuple[float | None, str]:
    state = _load_resnet()
    if state is None:
        return None, "(ResNet50 checkpoint not found — train Phase 2a first)"
    import torch
    m, device, tfm = state
    tensor = tfm(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = m(tensor).squeeze().item()
    prob = float(torch.sigmoid(torch.tensor(logit)).item())
    verdict = "DEFECT" if prob >= 0.5 else "NORMAL"
    return prob, f"ResNet50: **{verdict}** (p={prob:.3f})"


def yolo_predict(img: Image.Image, conf: float = 0.25) -> tuple[Image.Image, str]:
    model = _load_yolo()
    img_rgb = img.convert("RGB")
    if model is None:
        return img_rgb, "(YOLO weights not found — train Phase 2b first)"
    result = model.predict(source=img_rgb, imgsz=1024, verbose=False, conf=conf)
    boxes = result[0].boxes

    # Draw boxes on image
    annotated = img_rgb.copy()
    draw = ImageDraw.Draw(annotated)
    n_detections = 0
    if boxes is not None and len(boxes):
        n_detections = len(boxes)
        W, H = annotated.size
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            c = float(box.conf[0].item())
            draw.rectangle([x1, y1, x2, y2], outline="#ff0000", width=3)
            draw.text((x1, max(0, y1 - 16)), f"defect {c:.2f}", fill="#ff0000")

    verdict = f"YOLO11s: **{n_detections} detection(s)** @ conf≥{conf:.2f}"
    return annotated, verdict


def vlm_predict(img: Image.Image, provider: str = "openai", model: str = "gpt-4o-mini") -> tuple[str, str]:
    try:
        from src.prompts.defect_analysis import JSON_SCHEMA, SYSTEM, USER_T3_COT
        from src.vlm_clients import call_openai, call_azure
        from src.config import OPENAI_MODELS

        if provider == "openai":
            resp = call_openai(
                model=model or OPENAI_MODELS["mini"],
                system=SYSTEM,
                user=USER_T3_COT,
                image=img,
                schema=JSON_SCHEMA,
                phase="demo",
                note="gradio_demo",
            )
        elif provider == "azure":
            resp = call_azure(
                system=SYSTEM,
                user=USER_T3_COT,
                image=img,
                schema=JSON_SCHEMA,
                phase="demo",
                note="gradio_demo",
            )
        else:
            return "{}", f"Unknown provider: {provider}"

        if resp.error:
            return resp.raw or "{}", f"VLM error: {resp.error}"

        parsed = resp.parsed or {}
        verdict = "DEFECT" if parsed.get("has_defect") else "NORMAL"
        conf = parsed.get("confidence", 0)
        dtype = parsed.get("defect_type", "unknown")
        reasoning = parsed.get("reasoning", "")

        summary = (
            f"**{provider.upper()} {model}**: **{verdict}** "
            f"(conf={conf:.2f})  type=`{dtype}`\n\n"
            f"_{reasoning}_\n\n"
            f"Cost: ${resp.cost_usd:.4f}  |  {resp.latency_s:.1f}s"
        )
        return json.dumps(parsed, indent=2), summary

    except Exception as e:
        return "{}", f"VLM error: {e}"


def hybrid_verdict(img: Image.Image) -> str:
    """YOLO (high recall) → VLM adjudication → final verdict."""
    yolo = _load_yolo()
    if yolo is None:
        return "(YOLO not available)"

    result = yolo.predict(source=img.convert("RGB"), imgsz=1024, verbose=False, conf=0.15)
    boxes = result[0].boxes
    if boxes is None or len(boxes) == 0:
        return "Hybrid: **NORMAL** (YOLO found no candidates)"

    # Use best box crop for VLM
    best = int(boxes.conf.argmax().item())
    xywhn = boxes.xywhn[best].tolist()[:4]
    W, H = img.size
    cx, cy, bw, bh = xywhn
    x1 = max(0, int((cx - bw / 2 - 0.05) * W))
    y1 = max(0, int((cy - bh / 2 - 0.05) * H))
    x2 = min(W, int((cx + bw / 2 + 0.05) * W))
    y2 = min(H, int((cy + bh / 2 + 0.05) * H))
    crop = img.crop((x1, y1, x2, y2)) if (x2 - x1 > 8 and y2 - y1 > 8) else img

    _, vlm_summary = vlm_predict(crop)
    return f"Hybrid (YOLO→VLM):\n{vlm_summary}"


# ── Gallery loading ───────────────────────────────────────────────────────────

def load_gallery() -> list[str]:
    if not GALLERY_MANIFEST.exists():
        return []
    import pandas as pd
    df = pd.read_parquet(GALLERY_MANIFEST)
    # Mix defect classes for visual variety
    sample = df.sample(min(N_GALLERY, len(df)), random_state=0)
    return [str(p) for p in sample["path"] if Path(p).exists()]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_app():
    import gradio as gr

    gallery_paths = load_gallery()

    def analyse(
        image,
        yolo_conf: float,
        vlm_provider: str,
        vlm_model: str,
        run_vlm: bool,
    ):
        if image is None:
            return None, "", "", "", "", ""

        img = Image.fromarray(image).convert("RGB") if not isinstance(image, Image.Image) else image

        # ResNet
        r_prob, r_summary = resnet_predict(img)
        r_bar = r_prob if r_prob is not None else 0.0

        # YOLO
        annotated_np, y_summary = yolo_predict(img, conf=yolo_conf)

        # VLM (optional — costs money)
        if run_vlm:
            vlm_json, vlm_summary = vlm_predict(img, vlm_provider, vlm_model)
            hybrid = hybrid_verdict(img)
        else:
            vlm_json = '{"note": "VLM skipped — enable with checkbox"}'
            vlm_summary = "_VLM disabled_"
            hybrid = "_Enable VLM for hybrid verdict_"

        return annotated_np, r_summary, float(r_bar), y_summary, vlm_summary, vlm_json, hybrid

    with gr.Blocks(title="AQ Group Defect Detection Bake-Off") as demo:
        gr.Markdown(
            "## AQ Group — Rolled-Metal Defect Detection Bake-Off\n"
            "Upload a metal surface image (or select from the gallery) to compare "
            "**ResNet50**, **YOLO11s**, **VLM zero-shot**, and **Hybrid YOLO→VLM** approaches."
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Input image", type="numpy")
                if gallery_paths:
                    gr.Examples(examples=gallery_paths, inputs=inp, label="GC10-DET samples")

                with gr.Accordion("Settings", open=False):
                    yolo_conf_sl = gr.Slider(0.05, 0.9, value=0.25, step=0.05,
                                             label="YOLO confidence threshold")
                    run_vlm_cb   = gr.Checkbox(value=False,
                                               label="Run VLM (costs ~$0.004/image — uncheck to skip)")
                    vlm_prov_dd  = gr.Dropdown(["openai", "azure"], value="openai",
                                               label="VLM provider")
                    vlm_model_dd = gr.Dropdown(
                        ["gpt-4o-mini", "gpt-4o"], value="gpt-4o-mini",
                        label="OpenAI model"
                    )

                run_btn = gr.Button("Analyse", variant="primary")

            with gr.Column(scale=2):
                with gr.Tab("ResNet50"):
                    r_summary_md = gr.Markdown()
                    r_bar_num    = gr.Number(label="Defect probability", precision=3)

                with gr.Tab("YOLO11s"):
                    y_img    = gr.Image(label="YOLO detections")
                    y_summary_md = gr.Markdown()

                with gr.Tab("VLM zero-shot"):
                    vlm_summary_md = gr.Markdown()
                    vlm_json_box   = gr.Code(language="json", label="VLM JSON response")

                with gr.Tab("Hybrid YOLO→VLM"):
                    hybrid_md = gr.Markdown()

        run_btn.click(
            fn=analyse,
            inputs=[inp, yolo_conf_sl, vlm_prov_dd, vlm_model_dd, run_vlm_cb],
            outputs=[y_img, r_summary_md, r_bar_num, y_summary_md, vlm_summary_md, vlm_json_box, hybrid_md],
        )
        inp.change(
            fn=analyse,
            inputs=[inp, yolo_conf_sl, vlm_prov_dd, vlm_model_dd, run_vlm_cb],
            outputs=[y_img, r_summary_md, r_bar_num, y_summary_md, vlm_summary_md, vlm_json_box, hybrid_md],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
