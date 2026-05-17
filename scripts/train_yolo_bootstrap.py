"""Train a fresh YOLO11s on VLM-bootstrapped pseudo-labels (Phase 4 step 5).

Usage:
    uv run python scripts/train_yolo_bootstrap.py
    uv run python scripts/train_yolo_bootstrap.py --data data/yolo_bootstrap/defect_bootstrap.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_DATA = "data/yolo_bootstrap/defect_bootstrap.yaml"
DEFAULT_PROJECT = "results/yolo_bootstrap"
DEFAULT_NAME    = "yolo11s_bootstrap"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default=DEFAULT_DATA)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name",    default=DEFAULT_NAME)
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=32)
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Bootstrap dataset not found: {args.data}")
        print("Run scripts/bootstrap_labels.py first.")
        raise SystemExit(1)

    ROOT = Path(__file__).resolve().parents[1]

    from ultralytics import YOLO
    model = YOLO("yolo11s.pt")

    print(f"Training YOLO-bootstrap on pseudo-labels: {args.data}")
    print(f"  epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")

    model.train(
        data=str(ROOT / args.data) if not Path(args.data).is_absolute() else args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=10,
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_v=0.2,
        device="mps",
        project=str(ROOT / args.project),
        name=args.name,
        seed=42,
        exist_ok=True,
    )

    weights = ROOT / args.project / args.name / "weights" / "best.pt"
    print(f"\nBootstrap YOLO trained. Best weights → {weights}")
    print(f"Next: uv run python scripts/eval_yolo.py --weights {weights} --output-dir results/yolo_bootstrap")


if __name__ == "__main__":
    main()
