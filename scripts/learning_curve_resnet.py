"""Learning curve for ResNet50 fine-tune (STUB — run overnight).

TODO (Jonathan, overnight job):
    Run this script with N in {50, 100, 250, 500, 1000, 2000, 2332}.
    For each N:
      - sample N training images stratified across (severstal_train,
        kolektor_train, gc10_train)
      - fine-tune the ResNet50 baseline for 5 epochs (lr=1e-4, RRC+TTA recipe)
      - evaluate on kolektor_test + gc10_test
      - record {N, kol_f1, gc10_f1, train_minutes, total_label_cost($0.20/img)}
    Persist to results/learning_curve/curve.json
    Then re-run make_cost_tradeoff_figure.py to overlay the empirical
    "F1 vs labels" curve onto the cost trade-off plot.

Usage:
    uv run python scripts/learning_curve_resnet.py --n 250 --epochs 5

Until this is wired up, the slides reference an analytical break-even from
make_cost_tradeoff_figure.py rather than an empirical curve.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

print(__doc__)
print("\n[STUB] No-op. Implement the sampling + training loop above to populate")
print("       results/learning_curve/curve.json before showing the slide.")
sys.exit(0)


def main() -> None:  # pragma: no cover  — placeholder
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.parse_args()


if __name__ == "__main__":
    main()
