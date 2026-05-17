"""Train YOLO11s binary defect detector (Phase 2b).

GATED: Do not run until Phase 2a ResNet50 results are reviewed.

Usage:
    uv run python scripts/train_yolo.py
    uv run python scripts/train_yolo.py --config configs/yolo11s.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_2a_gate(results_dir: str = "results/resnet50") -> bool:
    """Return True only if Phase 2a metrics exist and pass quality gates."""
    metrics_path = Path(results_dir) / "metrics.json"
    if not metrics_path.exists():
        print("⛔  Phase 2a metrics not found — run train_resnet.py first.")
        return False
    with metrics_path.open() as f:
        m = json.load(f)

    val_f1 = m.get("best_val_f1", 0)
    kolektor_f1 = m.get("eval", {}).get("kolektor_test", {}).get("f1", 0)
    print(f"Phase 2a gate check:")
    print(f"  best_val_f1      = {val_f1:.4f}  (threshold ≥ 0.70)")
    print(f"  kolektor_test_f1 = {kolektor_f1:.4f}  (informational)")

    if val_f1 < 0.70:
        print("⛔  val_f1 below threshold — review ResNet50 training before proceeding.")
        return False
    print("✅  Gate passed — starting YOLO training.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yolo11s.yaml")
    parser.add_argument("--force", action="store_true", help="Skip 2a gate check")
    args = parser.parse_args()

    if not args.force and not check_2a_gate():
        raise SystemExit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from ultralytics import YOLO

    model = YOLO(cfg["model"])

    # Use absolute project path to avoid Ultralytics global runs_dir prepending
    ROOT = Path(__file__).resolve().parents[1]
    project_abs = str(ROOT / cfg["project"])

    train_args = {
        "data":          str(ROOT / cfg["data"]),
        "imgsz":         cfg["imgsz"],
        "batch":         cfg["batch"],
        "epochs":        cfg["epochs"],
        "patience":      cfg["patience"],
        "lr0":           cfg["lr0"],
        "lrf":           cfg["lrf"],
        "weight_decay":  cfg["weight_decay"],
        "warmup_epochs": cfg["warmup_epochs"],
        "mosaic":        cfg["mosaic"],
        "mixup":         cfg["mixup"],
        "degrees":       cfg["degrees"],
        "flipud":        cfg["flipud"],
        "fliplr":        cfg["fliplr"],
        "hsv_v":         cfg["hsv_v"],
        "project":       project_abs,
        "name":          cfg["name"],
        "seed":          cfg["seed"],
        "device":        cfg["device"],
        "exist_ok":      True,
        "verbose":       True,
    }

    print(f"\nStarting YOLO11s training on {cfg['data']}")
    print(f"  imgsz={cfg['imgsz']}  batch={cfg['batch']}  epochs={cfg['epochs']}")

    results = model.train(**train_args)

    # Save a clean summary alongside the YOLO run dir
    best_path = ROOT / cfg["project"] / cfg["name"] / "weights" / "best.pt"
    summary = {
        "model": "yolo11s",
        "best_weights": str(best_path),
        "results_dir": str(ROOT / cfg["project"] / cfg["name"]),
    }
    out = ROOT / cfg["project"] / "yolo_summary.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nYOLO training complete. Summary → {out}")


if __name__ == "__main__":
    main()
