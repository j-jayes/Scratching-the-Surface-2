"""Phase 4 — VLM auto-label bootstrap: cold-start YOLO on the unseen domain.

Workflow:
  1. Take N GC10-DET images (labels hidden).
  2. Run best VLM (gpt-4o default) with CoT prompt → pseudo-labels.
  3. Convert VLM bbox estimates to YOLO format.
  4. Build `data/yolo_bootstrap/` dataset tree.
  5. Train a fresh YOLO11s on pseudo-labels only.
  6. Evaluate on held-out GC10 test set vs ground truth.

This script implements steps 1–4 (pseudo-labelling + dataset build).
Run scripts/train_yolo_bootstrap.py for step 5 (training).
Run scripts/eval_yolo.py --weights results/yolo_bootstrap/... for step 6.

Usage:
    uv run python scripts/bootstrap_labels.py
    uv run python scripts/bootstrap_labels.py --n 1500 --model gpt-4o
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import OPENAI_MODELS
from src.prompts.defect_analysis import JSON_SCHEMA, SYSTEM, USER_T3_COT
from src.vlm_clients import call_openai

PROCESSED_DIR  = Path("data/processed")
RESULTS_DIR    = Path("results/bootstrap")
BOOTSTRAP_YOLO = Path("data/yolo_bootstrap")
GC10_MANIFEST  = PROCESSED_DIR / "gc10_manifest.parquet"

# Keep test split as evaluation; bootstrap only from "train" rows — but GC10 manifest
# marks everything as split="test". We therefore treat a random subset as the "bootstrap
# training pool" and keep the rest as a held-out evaluation partition.
BOOTSTRAP_FRACTION = 0.70   # 70% of GC10 for pseudo-label training


def estimate_cost(n: int, model: str) -> float:
    from src.config import PRICING
    prices = PRICING.get(model, {"in": 2.50, "out": 15.00})
    return n * (1_800 * prices["in"] + 400 * prices["out"]) / 1_000_000


def vlm_pseudolabel(img_path: str, model: str, img_id: str) -> dict:
    img = Image.open(img_path).convert("RGB")
    resp = call_openai(
        model=model,
        system=SYSTEM,
        user=USER_T3_COT,
        image=img,
        schema=JSON_SCHEMA,
        phase="phase4",
        note=f"bootstrap/{img_id}",
    )
    result: dict = {
        "img_id":    img_id,
        "img_path":  img_path,
        "in_tok":    resp.in_tok,
        "out_tok":   resp.out_tok,
        "cost_usd":  resp.cost_usd,
        "latency_s": resp.latency_s,
        "error":     resp.error,
    }
    if resp.parsed and isinstance(resp.parsed, dict):
        result["has_defect"]  = bool(resp.parsed.get("has_defect", False))
        result["confidence"]  = float(resp.parsed.get("confidence", 0.5))
        result["defect_type"] = resp.parsed.get("defect_type", "unknown")
        result["bbox"]        = resp.parsed.get("bbox")   # [cx, cy, w, h] in [0,1] or None
    else:
        result["has_defect"]  = None
        result["confidence"]  = 0.0
        result["defect_type"] = None
        result["bbox"]        = None
    return result


def write_yolo_label(txt_path: Path, bbox: list[float] | None) -> None:
    """Write a YOLO label file. If bbox is [cx,cy,w,h], use it; else full image."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if bbox and len(bbox) == 4 and all(0 <= v <= 1 for v in bbox):
        cx, cy, w, h = [round(v, 6) for v in bbox]
    else:
        # Fallback: full-image box
        cx, cy, w, h = 0.5, 0.5, 1.0, 1.0
    txt_path.write_text(f"0 {cx} {cy} {w} {h}\n")


def build_bootstrap_yolo_dataset(
    records: list[dict],
    seed: int = 42,
    train_frac: float = 0.85,
) -> Path:
    """Build data/yolo_bootstrap/ from pseudo-label records."""
    defect_records = [r for r in records if r.get("has_defect") and r.get("error") is None]
    rng = np.random.default_rng(seed)
    rng.shuffle(defect_records)
    n_train = int(len(defect_records) * train_frac)
    splits = {"train": defect_records[:n_train], "val": defect_records[n_train:]}

    for split, recs in splits.items():
        img_dir = BOOTSTRAP_YOLO / "images" / split
        lbl_dir = BOOTSTRAP_YOLO / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for r in recs:
            src = Path(r["img_path"])
            # Symlink images
            link = img_dir / src.name
            if not link.exists():
                link.symlink_to(src.resolve())
            # Write label
            write_yolo_label(lbl_dir / (src.stem + ".txt"), r.get("bbox"))

    # Write defect.yaml
    yaml_content = f"""\
# YOLO bootstrap dataset — pseudo-labelled by VLM (Phase 4)
path: {BOOTSTRAP_YOLO.resolve()}
train: images/train
val: images/val
nc: 1
names: ['defect']
"""
    yaml_path = BOOTSTRAP_YOLO / "defect_bootstrap.yaml"
    yaml_path.write_text(yaml_content)

    n_train = len(splits["train"])
    n_val   = len(splits["val"])
    print(f"Bootstrap YOLO dataset: {n_train} train / {n_val} val")
    print(f"  → {yaml_path}")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=1500, help="Images to pseudo-label")
    parser.add_argument("--model", default=OPENAI_MODELS["mid"],
                        help="OpenAI model for pseudo-labelling (gpt-4o default for quality)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(GC10_MANIFEST)
    # All GC10 are defect-only with split="test". Take a bootstrap training pool.
    df = df.sample(min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"Bootstrap pool: {len(df)} GC10 images (all defect)")

    est = estimate_cost(len(df), args.model)
    print(f"Estimated cost ({args.model}): ${est:.2f}")

    if args.dry_run:
        print("Dry-run complete — no API calls made.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = RESULTS_DIR / f"pseudolabels_{ts}.jsonl"
    records: list[dict] = []

    with jsonl_path.open("w") as f:
        for i, row in df.iterrows():
            rec = vlm_pseudolabel(
                img_path=str(row["path"]),
                model=args.model,
                img_id=Path(row["path"]).stem,
            )
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            status = "✓" if rec["error"] is None else "✗"
            print(
                f"  [{i+1:4d}/{len(df)}] {status} "
                f"defect={rec['has_defect']} conf={rec['confidence']:.2f} "
                f"${rec['cost_usd']:.4f}  {rec['latency_s']:.1f}s"
                + (f"  ERR: {rec['error'][:50]}" if rec["error"] else ""),
                flush=True,
            )

    # Quality metrics vs ground truth (all GC10 are defect=True)
    valid = [r for r in records if r["has_defect"] is not None and not r["error"]]
    n_agree = sum(1 for r in valid if r["has_defect"] is True)
    agree_pct = 100 * n_agree / len(valid) if valid else 0
    total_cost = sum(r["cost_usd"] for r in records)
    print(f"\nVLM-as-annotator agreement with ground truth: {n_agree}/{len(valid)} = {agree_pct:.1f}%")
    print(f"Total cost: ${total_cost:.3f}")

    # Build YOLO dataset from pseudo-labels
    yaml_path = build_bootstrap_yolo_dataset(records, seed=args.seed)

    # Summary
    summary = {
        "timestamp": ts,
        "model": args.model,
        "n_images": len(df),
        "n_valid": len(valid),
        "n_errors": len(records) - len(valid),
        "vlm_agreement_pct": round(agree_pct, 1),
        "total_cost_usd": round(total_cost, 4),
        "yolo_dataset_yaml": str(yaml_path),
    }
    summary_path = RESULTS_DIR / f"summary_{ts}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")
    print(f"\nNext: uv run python scripts/train_yolo_bootstrap.py --data {yaml_path}")


if __name__ == "__main__":
    main()
