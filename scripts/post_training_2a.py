#!/usr/bin/env python3
"""Master post-training runner for Phase 2a completion.

Run this once after train_resnet.py completes:

    uv run python scripts/post_training_2a.py

Steps:
  1. analyse_resnet.py     — metrics table + training curves + GO/NO-GO
  2. optimise_threshold.py — threshold sweep + ROC curves
  3. make_qual_grid.py     — qualitative prediction grids for slide deck
  4. make_comparison_figures.py — bake-off chart (partial — ResNet only)
  5. If gate passes → launch YOLO training (train_yolo.py)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "resnet50" / "metrics.json"


def run(script: str, *args: str, check: bool = True) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print(f"\n{'='*60}")
    print(f"  Running: {script} {' '.join(args)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=ROOT)
    if check and result.returncode != 0:
        print(f"\n  ✗ {script} failed (exit {result.returncode})")
        raise SystemExit(result.returncode)
    return result.returncode


def check_gate() -> bool:
    import json
    if not METRICS.exists():
        return False
    with METRICS.open() as f:
        m = json.load(f)
    val_f1    = m.get("best_val_f1", 0)
    kol_auc   = m.get("eval", {}).get("kolektor_test", {}).get("roc_auc", 0)
    passed = val_f1 >= 0.75 and kol_auc >= 0.65
    print(f"\nGate check: val_F1={val_f1:.4f} (≥0.75: {'✓' if val_f1>=0.75 else '✗'})"
          f"  kolektor_AUC={kol_auc:.4f} (≥0.65: {'✓' if kol_auc>=0.65 else '✗'})")
    return passed


def main() -> None:
    if not METRICS.exists():
        print(f"ERROR: metrics not found at {METRICS}")
        print("Ensure train_resnet.py has finished before running this script.")
        raise SystemExit(1)

    # Step 1 — analysis
    run("analyse_resnet.py")

    # Step 2 — threshold optimisation
    run("optimise_threshold.py")

    # Step 3 — qualitative grids (don't fail the pipeline on grid errors)
    for dataset in ("severstal_test", "kolektor_test", "gc10_test"):
        run("make_qual_grid.py", "--dataset", dataset, "--n", "6", check=False)

    # Step 4 — bake-off figure (partial — ResNet only so far)
    run("make_comparison_figures.py", check=False)

    # Step 5 — gate check & YOLO launch
    print("\n" + "=" * 60)
    if check_gate():
        print("  ✅  Gate PASSED — launching Phase 2b YOLO training...")
        print("=" * 60)
        run("train_yolo.py")
    else:
        print("  ❌  Gate FAILED — review analyse_resnet.py output above.")
        print("      Run with --force to skip gate: uv run python scripts/train_yolo.py --force")
        print("=" * 60)


if __name__ == "__main__":
    main()
