"""Cross-domain evaluation for trained YOLO11s (Phase 2b).

Sweeps confidence thresholds to produce image-level binary metrics
on all four eval sets, matching the ResNet50 output format so both
can be plotted on the same comparison chart.

Usage:
    uv run python scripts/eval_yolo.py
    uv run python scripts/eval_yolo.py --weights results/yolo/yolo11s_defect/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import f1_score, roc_auc_score

from src.data.defect_dataset import load_split

RESULTS_DIR  = Path("results/yolo")
FIGURES_DIR  = Path("figures/yolo")
CONFIG_PATH  = Path("configs/yolo11s.yaml")


def yolo_image_conf(
    model, img_paths: list[str], imgsz: int = 1024, batch: int = 32
) -> np.ndarray:
    """Return max-confidence score per image (0 if no detection)."""
    scores: list[float] = []
    for i in range(0, len(img_paths), batch):
        chunk = img_paths[i : i + batch]
        results = model.predict(source=chunk, imgsz=imgsz, verbose=False, conf=0.01)
        for res in results:
            boxes = res.boxes
            if boxes is not None and len(boxes):
                scores.append(float(boxes.conf.max().item()))
            else:
                scores.append(0.0)
    return np.array(scores)


def evaluate_at_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> dict[str, list]:
    out: dict[str, list] = {"threshold": [], "f1": [], "precision": [], "recall": []}
    for t in thresholds:
        preds = (scores >= t).astype(int)
        out["threshold"].append(t)
        out["f1"].append(float(f1_score(labels, preds, zero_division=0)))
        from sklearn.metrics import precision_score, recall_score
        out["precision"].append(float(precision_score(labels, preds, zero_division=0)))
        out["recall"].append(float(recall_score(labels, preds, zero_division=0)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/yolo11s.yaml")
    parser.add_argument("--weights",    default=None, help="Override weight path")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for eval_metrics.json and figures (default: results/yolo)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ROOT = Path(__file__).resolve().parents[1]
    # Try absolute project path first (from fixed train_yolo.py), then relative
    _abs_weights = ROOT / cfg["project"] / cfg["name"] / "weights" / "best.pt"
    _rel_weights = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    weights = args.weights or (
        str(_abs_weights) if _abs_weights.exists() else str(_rel_weights)
    )
    if not Path(weights).exists():
        print(f"Weights not found: {weights} — run train_yolo.py first.")
        raise SystemExit(1)

    # Allow redirecting output for bootstrap eval
    results_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    figures_dir = Path(args.output_dir).parent / "figures" / Path(args.output_dir).name if args.output_dir else FIGURES_DIR

    from ultralytics import YOLO
    model = YOLO(weights)
    imgsz: int = cfg.get("imgsz", 640)
    thresholds: list[float] = cfg.get("conf_sweep", [0.1, 0.2, 0.3, 0.4, 0.5])

    # Build eval sets (same as ResNet for comparability)
    eval_sets: dict[str, tuple[list[str], np.ndarray]] = {}
    for name, ecfg in cfg.get("eval_sets", {}).items():
        sup_path  = ecfg.get("supplement_normals_from")
        sup_split = ecfg.get("supplement_split")
        df = load_split(
            [ecfg["manifest"]],
            split=ecfg["split"],
            supplement_normals_path=sup_path,
            supplement_split=sup_split,
        )
        eval_sets[name] = (df["path"].tolist(), df["has_defect"].astype(int).values)
        print(f"Eval '{name}': {len(df)} images  (+{df['has_defect'].sum()} / -{(~df['has_defect']).sum()})")

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    all_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, (paths, labels) in eval_sets.items():
        print(f"\nRunning YOLO inference on '{name}' ({len(paths)} images)...")
        scores = yolo_image_conf(model, paths, imgsz)
        all_scores[name] = (scores, labels)

        sweep = evaluate_at_thresholds(scores, labels, thresholds)
        best_f1_idx = int(np.argmax(sweep["f1"]))
        best_t  = sweep["threshold"][best_f1_idx]
        best_f1 = sweep["f1"][best_f1_idx]

        auc_val = None
        auc_str = "n/a"
        if len(np.unique(labels)) > 1:
            auc_val = float(roc_auc_score(labels, scores))
            auc_str = f"{auc_val:.4f}"

        print(f"  Best F1={best_f1:.4f} @ conf={best_t:.2f}  AUC={auc_str}")
        all_results[name] = {
            "best_conf_threshold": best_t,
            "best_f1": round(best_f1, 4),
            "roc_auc": round(auc_val, 4) if auc_val is not None else None,
            "sweep": sweep,
        }

    # Save
    out_path = results_dir / "eval_metrics.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nYOLO eval metrics → {out_path}")

    # PR curves figure
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for i, (name, res) in enumerate(all_results.items()):
        ax.plot(res["sweep"]["recall"], res["sweep"]["precision"],
                color=colors[i % 10], marker="o", ms=4, label=name.replace("_", " "))
    ax.set(title="YOLO11s Precision-Recall (conf sweep)", xlabel="Recall", ylabel="Precision")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pr_path = figures_dir / "pr_curves.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    print(f"PR curves → {pr_path}")


if __name__ == "__main__":
    main()
