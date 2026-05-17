"""Phase F — Flagship VLM bake-off (gpt-5.4 + gemini-3.1-pro-preview + gpt-4.1-mini reference).

Runs the canonical defect_analysis prompt on N images per domain
(default: 50 per dataset), separate from the earlier mini eval.
Persists per-image JSONL including the model's reasoning text so the
downstream "Grad-CAM vs VLM rationale" figure can read it.

Usage:
    uv run python scripts/eval_vlm_flagship.py --dry-run
    uv run python scripts/eval_vlm_flagship.py --n 50
    uv run python scripts/eval_vlm_flagship.py --n 50 --providers openai gemini azure
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GEMINI_MODELS, OPENAI_MODELS, PRICING
from src.prompts.defect_analysis import JSON_SCHEMA, SYSTEM, USER_T3_COT
from src.vlm_clients import call_azure, call_gemini, call_openai
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

# Re-use sampling helper from the existing script
from scripts.eval_vlm_zeroshot import DATASETS, sample_dataset

RESULTS_DIR = Path("results/vlm")


PROVIDER_MODEL = {
    "openai": OPENAI_MODELS["flagship"],          # gpt-5.4
    "gemini": GEMINI_MODELS["flagship"],          # gemini-2.5-pro
    "azure":  None,                               # uses AOAI_DEPLOYMENT (gpt-4.1-mini)
}


def call_provider(provider: str, model: str | None, img: Image.Image,
                  system: str, user: str, note: str):
    if provider == "openai":
        return call_openai(model=model or OPENAI_MODELS["flagship"],
                           system=system, user=user, image=img,
                           schema=JSON_SCHEMA, phase="phaseF", note=note)
    if provider == "gemini":
        return call_gemini(model=model or GEMINI_MODELS["flagship"],
                           system=system, user=user, image=img,
                           schema=JSON_SCHEMA, phase="phaseF", note=note)
    if provider == "azure":
        return call_azure(system=system, user=user, image=img,
                          schema=JSON_SCHEMA, phase="phaseF", note=note)
    raise ValueError(provider)


def run_one(provider: str, model: str | None, image_path: str,
            label: int, dataset: str, img_id: str,
            system: str = SYSTEM, user: str = USER_T3_COT) -> dict:
    img = Image.open(image_path).convert("RGB")
    resp = call_provider(provider, model, img, system, user, note=f"{dataset}/{img_id}")
    parsed = resp.parsed if isinstance(resp.parsed, dict) else {}
    pred = parsed.get("has_defect")
    conf = float(parsed.get("confidence", 0.5)) if parsed else 0.5
    return {
        "image_id":   img_id,
        "img_id":     img_id,            # keep both keys
        "dataset":    dataset,
        "label":      label,
        "pred_defect": bool(pred) if pred is not None else None,
        "confidence": conf,
        "provider":   provider,
        "model":      resp.model,
        "in_tok":     resp.in_tok,
        "out_tok":    resp.out_tok,
        "cost_usd":   resp.cost_usd,
        "latency_s":  resp.latency_s,
        "error":      resp.error,
        "parsed":     resp.parsed,
        "raw":        resp.raw,
    }


def summarise(records: list[dict]) -> dict:
    total_cost = round(sum(r["cost_usd"] for r in records), 4)
    valid = [r for r in records if r["pred_defect"] is not None and r["error"] is None]
    if not valid:
        return {"n": len(records), "n_valid": 0, "n_errors": len(records),
                "total_cost_usd": total_cost}
    y = np.array([r["label"] for r in valid])
    p = np.array([int(r["pred_defect"]) for r in valid])
    s = np.array([r["confidence"] if r["pred_defect"] else 1 - r["confidence"]
                  for r in valid])
    out: dict = {
        "n": len(records),
        "n_valid": len(valid),
        "n_errors": len(records) - len(valid),
        "f1": round(float(f1_score(y, p, zero_division=0)), 4),
        "precision": round(float((p[y == 1] == 1).mean()) if (y == 1).any() else 0, 4),
        "recall":    round(float((p[y == 1] == 1).mean()) if (y == 1).any() else 0, 4),
        "total_cost_usd": total_cost,
        "mean_latency_s": round(float(np.mean([r["latency_s"] for r in valid])), 2),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y, s)), 4)
    return out


def estimate_cost(n_per_ds: int, model: str) -> float:
    pr = PRICING.get(model, {"in": 2.5, "out": 15.0})
    return (n_per_ds * 2) * (1800 * pr["in"] + 500 * pr["out"]) / 1_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+",
                    default=["openai", "gemini", "azure"],
                    choices=list(PROVIDER_MODEL.keys()))
    ap.add_argument("--n", type=int, default=50,
                    help="Images per dataset (50/50 pos/neg).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"\nFlagship VLM bake-off — {args.n} imgs × {len(DATASETS)} datasets × "
          f"{len(args.providers)} providers\n")

    total_est = 0.0
    for prov in args.providers:
        model = PROVIDER_MODEL[prov] or "gpt-4.1-mini"
        est = estimate_cost(args.n, model)
        total_est += est
        print(f"  {prov:7s} {model:30s}  ≈ ${est:.2f}")
    print(f"  total ≈ ${total_est:.2f}\n")
    if args.dry_run:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: dict = {"timestamp": ts, "n_per_dataset": args.n, "results": {}}

    for prov in args.providers:
        model = PROVIDER_MODEL[prov]
        summary["results"][prov] = {}
        for ds_name, ds_cfg in DATASETS.items():
            sample = sample_dataset(ds_name, ds_cfg, args.n, seed=args.seed)
            print(f"\n{prov}/{model or 'AOAI'}  •  {ds_name}  "
                  f"({len(sample)} imgs: {sample['has_defect'].sum()} pos)")
            jsonl_path = RESULTS_DIR / f"flagship_{prov}_{ds_name}_{ts}.jsonl"
            records: list[dict] = []
            t_start = time.time()
            with jsonl_path.open("w") as f:
                for i, row in sample.iterrows():
                    rec = run_one(prov, model, row["path"],
                                  int(row["has_defect"]), ds_name,
                                  Path(row["path"]).stem)
                    records.append(rec)
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    ok = "✓" if rec["error"] is None else "✗"
                    print(f"  [{i+1:3d}/{len(sample)}] {ok} "
                          f"y={rec['label']} ŷ={rec['pred_defect']} "
                          f"conf={rec['confidence']:.2f} ${rec['cost_usd']:.4f} "
                          f"{rec['latency_s']:.1f}s"
                          + (f"  ERR:{rec['error'][:50]}" if rec['error'] else ""),
                          flush=True)
            metrics = summarise(records)
            metrics["wall_time_s"] = round(time.time() - t_start, 1)
            metrics["jsonl"] = str(jsonl_path)
            summary["results"][prov][ds_name] = metrics
            print(f"  → F1={metrics.get('f1','n/a')}  AUC={metrics.get('roc_auc','n/a')}  "
                  f"cost=${metrics['total_cost_usd']:.2f}  errs={metrics['n_errors']}")

    summary_path = RESULTS_DIR / f"flagship_summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  → {summary_path}")


if __name__ == "__main__":
    main()
