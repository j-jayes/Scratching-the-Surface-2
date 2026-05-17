"""Phase 3b — Hybrid YOLO→VLM pipeline evaluation.

YOLO runs at high recall (low confidence threshold) to flag candidate crops.
Each crop is sent to the best VLM (from Phase 3a) for adjudication.
Image is marked defective iff ≥1 YOLO proposal is confirmed by the VLM.

Usage:
    uv run python scripts/eval_hybrid.py
    uv run python scripts/eval_hybrid.py --provider openai --model gpt-4o-mini
    uv run python scripts/eval_hybrid.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from src.config import OPENAI_MODELS
from src.prompts.defect_analysis import JSON_SCHEMA, SYSTEM, USER_T3_COT
from src.vlm_clients import call_openai, call_azure, call_gemini

RESULTS_DIR = Path("results/hybrid")
FIGURES_DIR = Path("figures/hybrid")
PROCESSED_DIR = Path("data/processed")

ROOT = Path(__file__).resolve().parents[1]
YOLO_WEIGHTS = ROOT / "results/yolo/yolo11s_defect/weights/best.pt"
YOLO_LOW_CONF = 0.15   # recall-optimised YOLO threshold (many false positives OK)
CONTEXT_MARGIN = 0.05  # fraction of image size to pad around each YOLO crop

EVAL_DATASETS = {
    "gc10_test": {
        "manifest":   PROCESSED_DIR / "gc10_manifest.parquet",
        "split":      "test",
        "has_normals": False,
        "supplement": PROCESSED_DIR / "severstal_manifest.parquet",
    },
    "kolektor_test": {
        "manifest":   PROCESSED_DIR / "kolektor_manifest.parquet",
        "split":      "test",
        "has_normals": True,
    },
}

PROVIDER_DEFAULTS = {
    "openai": OPENAI_MODELS["mini"],
    "azure":  None,
    "gemini": "gemini-2.0-flash",
}


def load_set(name: str, cfg: dict, n: int | None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_parquet(cfg["manifest"])
    df = df[df["split"] == cfg["split"]].copy()
    if not cfg["has_normals"] and cfg.get("supplement"):
        sev = pd.read_parquet(cfg["supplement"])
        normals = sev[(sev["split"] == "test") & (~sev["has_defect"])].copy()
        normals = normals.sample(min(len(df), len(normals)), random_state=seed)
        df = pd.concat([df, normals], ignore_index=True)
    if n:
        pos = df[df["has_defect"]].sample(min(n // 2, df["has_defect"].sum()), random_state=seed)
        neg = df[~df["has_defect"]].sample(min(n // 2, (~df["has_defect"]).sum()), random_state=seed)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=seed)
    return df.reset_index(drop=True)


def crop_with_margin(img: Image.Image, box_xywhn: list[float]) -> Image.Image:
    """Crop a YOLO normalised box [cx, cy, w, h] with context margin."""
    W, H = img.size
    cx, cy, bw, bh = box_xywhn
    x1 = (cx - bw / 2 - CONTEXT_MARGIN) * W
    y1 = (cy - bh / 2 - CONTEXT_MARGIN) * H
    x2 = (cx + bw / 2 + CONTEXT_MARGIN) * W
    y2 = (cy + bh / 2 + CONTEXT_MARGIN) * H
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(W, int(x2)); y2 = min(H, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return img   # too small — use full image as fallback
    return img.crop((x1, y1, x2, y2))


def vlm_adjudicate(
    provider: str,
    model: str | None,
    crop: Image.Image,
    img_id: str,
    dataset: str,
) -> dict:
    note = f"hybrid/{dataset}/{img_id}"
    if provider == "openai":
        resp = call_openai(
            model=model or OPENAI_MODELS["mini"],
            system=SYSTEM,
            user=USER_T3_COT,
            image=crop,
            schema=JSON_SCHEMA,
            phase="phase3b",
            note=note,
        )
    elif provider == "azure":
        resp = call_azure(
            system=SYSTEM,
            user=USER_T3_COT,
            image=crop,
            schema=JSON_SCHEMA,
            phase="phase3b",
            note=note,
        )
    elif provider == "gemini":
        from src.config import GEMINI_MODELS
        resp = call_gemini(
            model=model or GEMINI_MODELS["mid"],
            system=SYSTEM,
            user=USER_T3_COT,
            image=crop,
            schema=JSON_SCHEMA,
            phase="phase3b",
            note=note,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    pred = None
    conf = 0.5
    if resp.parsed and isinstance(resp.parsed, dict):
        pred = bool(resp.parsed.get("has_defect"))
        conf = float(resp.parsed.get("confidence", 0.5))
    return {"pred": pred, "confidence": conf, "in_tok": resp.in_tok, "out_tok": resp.out_tok,
            "cost_usd": resp.cost_usd, "latency_s": resp.latency_s, "error": resp.error}


def process_image(
    yolo_model,
    provider: str,
    model: str | None,
    img_path: str,
    label: int,
    dataset: str,
    imgsz: int = 640,
) -> dict:
    img_id = Path(img_path).stem
    img = Image.open(img_path).convert("RGB")

    # 1. YOLO proposals at low threshold
    result = yolo_model.predict(source=img_path, imgsz=imgsz, verbose=False, conf=YOLO_LOW_CONF)
    boxes = result[0].boxes
    n_proposals = 0 if boxes is None else len(boxes)
    yolo_score = 0.0
    if boxes is not None and len(boxes):
        yolo_score = float(boxes.conf.max().item())

    if n_proposals == 0:
        # YOLO found nothing → no VLM calls needed → predict normal
        return {
            "img_id": img_id, "dataset": dataset, "label": label,
            "yolo_n_proposals": 0, "yolo_max_conf": 0.0,
            "vlm_calls": 0, "hybrid_pred": 0, "hybrid_conf": 0.0,
            "cost_usd": 0.0, "latency_s": 0.0, "error": None,
        }

    # 2. VLM adjudication on the highest-confidence YOLO crop only (cost control)
    #    Take the single most confident box to adjudicate
    best_box_idx = int(boxes.conf.argmax().item())
    xywhn = boxes.xywhn[best_box_idx].tolist()[:4]
    crop = crop_with_margin(img, xywhn)

    adj = vlm_adjudicate(provider, model, crop, img_id, dataset)

    hybrid_pred = 1 if (adj["pred"] is True) else 0
    hybrid_conf = adj["confidence"] if adj["pred"] is True else (1 - adj["confidence"])

    return {
        "img_id": img_id, "dataset": dataset, "label": label,
        "yolo_n_proposals": n_proposals, "yolo_max_conf": yolo_score,
        "vlm_calls": 1, "hybrid_pred": hybrid_pred, "hybrid_conf": hybrid_conf,
        "cost_usd": adj["cost_usd"], "latency_s": adj["latency_s"], "error": adj["error"],
        "vlm_result": adj,
    }


def summarise(records: list[dict]) -> dict:
    labels  = np.array([r["label"]       for r in records])
    preds   = np.array([r["hybrid_pred"] for r in records])
    confs   = np.array([r["hybrid_conf"] for r in records])
    return {
        "n":              len(records),
        "f1":             round(float(f1_score(labels, preds, zero_division=0)), 4),
        "precision":      round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall":         round(float(recall_score(labels, preds, zero_division=0)), 4),
        "roc_auc":        round(float(roc_auc_score(labels, confs)), 4) if len(np.unique(labels)) > 1 else None,
        "vlm_calls":      int(sum(r["vlm_calls"] for r in records)),
        "total_cost_usd": round(float(sum(r["cost_usd"] for r in records)), 4),
        "mean_latency_s": round(float(np.mean([r["latency_s"] for r in records if r["vlm_calls"] > 0] or [0])), 2),
        "yolo_passthrough_pct": round(100 * sum(r["yolo_n_proposals"] > 0 for r in records) / len(records), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="openai", choices=["openai", "azure", "gemini"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--n", type=int, default=100, help="Images per dataset (balanced)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not YOLO_WEIGHTS.exists():
        print(f"YOLO weights not found: {YOLO_WEIGHTS}")
        print("Run train_yolo.py first (Phase 2b).")
        raise SystemExit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = args.model or PROVIDER_DEFAULTS.get(args.provider)
    n_total = args.n * len(EVAL_DATASETS)
    from src.config import PRICING
    prices = PRICING.get(model or "gpt-4o-mini", {"in": 0.15, "out": 0.60})
    # Estimate: assume ~50% of images pass YOLO (each gets ~1k in + 400 out tokens)
    est_cost = 0.5 * n_total * (1_000 * prices["in"] + 400 * prices["out"]) / 1_000_000
    print(f"Estimated cost (50% YOLO pass-through): ${est_cost:.2f} for {n_total} images")

    if args.dry_run:
        print("Dry-run complete — no API calls made.")
        return

    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_WEIGHTS))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summary: dict = {}

    for ds_name, ds_cfg in EVAL_DATASETS.items():
        sample = load_set(ds_name, ds_cfg, args.n, seed=args.seed)
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}  {len(sample)} images  provider={args.provider}  model={model}")
        print("="*60)

        jsonl_path = RESULTS_DIR / f"{args.provider}_{ds_name}_{ts}.jsonl"
        records: list[dict] = []

        with jsonl_path.open("w") as f:
            for _, row in sample.iterrows():
                rec = process_image(
                    yolo_model=yolo,
                    provider=args.provider,
                    model=model,
                    img_path=str(row["path"]),
                    label=int(row["has_defect"]),
                    dataset=ds_name,
                )
                records.append(rec)
                f.write(json.dumps({k: v for k, v in rec.items() if k != "vlm_result"}) + "\n")
                f.flush()

                print(
                    f"  [{len(records):3d}/{len(sample)}] "
                    f"yolo={rec['yolo_n_proposals']} vlm={rec['vlm_calls']} "
                    f"pred={rec['hybrid_pred']} label={rec['label']} "
                    f"${rec['cost_usd']:.4f}",
                    flush=True,
                )

        m = summarise(records)
        all_summary[ds_name] = m
        print(f"\n  F1={m['f1']}  AUC={m['roc_auc']}  Prec={m['precision']}  Rec={m['recall']}")
        print(f"  YOLO pass-through={m['yolo_passthrough_pct']}%  VLM calls={m['vlm_calls']}")
        print(f"  Total cost=${m['total_cost_usd']:.3f}  → {jsonl_path}")

    summary_path = RESULTS_DIR / f"summary_{ts}.json"
    with summary_path.open("w") as f:
        json.dump({"timestamp": ts, "provider": args.provider, "model": model,
                   "yolo_low_conf": YOLO_LOW_CONF, "results": all_summary}, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
