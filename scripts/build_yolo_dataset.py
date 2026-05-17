"""Build YOLO-format dataset tree from manifests (Phase 2b prep).

Creates:
  data/yolo/
    images/train/*.{jpg,png}  (symlinks to originals)
    images/val/*.{jpg,png}
    labels/train/*.txt         (one file per image, YOLO bbox format)
    labels/val/*.txt
    defect.yaml                (YOLO dataset descriptor)

Usage:
    uv run python scripts/build_yolo_dataset.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR

YOLO_ROOT = Path("data/yolo")
BINARY_MODE = True  # collapse all classes → class 0 "defect"


def write_label(label_path: Path, bboxes_yolo: str, binary: bool) -> None:
    """Write YOLO .txt label file from the pipe-separated bboxes_yolo column."""
    lines = []
    if pd.notna(bboxes_yolo) and bboxes_yolo.strip():
        for entry in bboxes_yolo.split("|"):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split()
            if len(parts) != 5:
                continue
            cls_idx = 0 if binary else int(parts[0])
            cx, cy, bw, bh = parts[1], parts[2], parts[3], parts[4]
            lines.append(f"{cls_idx} {cx} {cy} {bw} {bh}")
    label_path.write_text("\n".join(lines))


def build_yolo_dataset(binary: bool = BINARY_MODE) -> None:
    manifests = {
        "severstal": PROCESSED_DIR / "severstal_manifest.parquet",
        "neu_det":   PROCESSED_DIR / "neu_det_manifest.parquet",
    }

    for split in ("train", "val"):
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_images = 0
    for source, mpath in manifests.items():
        df = pd.read_parquet(mpath)
        # Map neu_det "val" → yolo "val", severstal "val" → yolo "val"
        split_map = {"train": "train", "val": "val", "test": None}
        for _, row in df.iterrows():
            yolo_split = split_map.get(row["split"])
            if yolo_split is None:
                continue  # skip severstal test — held out
            if not row["has_defect"] and (not pd.notna(row.get("bboxes_yolo", "")) or not str(row.get("bboxes_yolo", "")).strip()):
                # Normal image — write empty label (background)
                pass

            img_src = Path(row["path"])
            img_dst = YOLO_ROOT / "images" / yolo_split / img_src.name
            lbl_dst = YOLO_ROOT / "labels" / yolo_split / (img_src.stem + ".txt")

            # Symlink image (avoid copying GBs of data)
            if not img_dst.exists():
                img_dst.symlink_to(img_src.resolve())

            write_label(lbl_dst, row.get("bboxes_yolo", ""), binary=binary)
            n_images += 1

    # Write YOLO dataset YAML
    nc = 1 if binary else 10
    names = ["defect"] if binary else [
        "defect_1", "defect_2", "defect_3", "defect_4",   # severstal
        "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",  # neu
    ]
    dataset_yaml = {
        "path": str(YOLO_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names,
    }
    yaml_path = YOLO_ROOT / "defect.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"YOLO dataset written to {YOLO_ROOT}")
    print(f"  {n_images:,} image entries (symlinked)")
    print(f"  Classes: {nc} ({'binary' if binary else 'multi-class'})")
    print(f"  Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    build_yolo_dataset()
