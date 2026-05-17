#!/usr/bin/env bash
# Post-fine-tune pipeline: threshold sweep → comparison figures → bootstrap YOLO.
# Run this after finetune_resnet_kolektor.py completes.
#
# Usage:
#   bash scripts/run_post_finetune_pipeline.sh
#
# Requirements:
#   models/resnet50_kolektor_ft_best.pt  (created by finetune_resnet_kolektor.py)
#   results/resnet50_kolektor_ft/metrics.json  (created by finetune_resnet_kolektor.py)

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Post-fine-tune pipeline started at $(date) ==="

# ── Step 1: Threshold optimisation on fine-tuned model ─────────────────────
echo ""
echo "▶ Step 1: Threshold sweep on ResNet50+FT"
if [[ -f "models/resnet50_kolektor_ft_best.pt" ]]; then
    uv run python scripts/optimise_threshold.py \
        --checkpoint models/resnet50_kolektor_ft_best.pt \
        --results-dir results/resnet50_kolektor_ft \
        --figures-dir figures/resnet50_kolektor_ft && \
        echo "✓ Threshold sweep done → results/resnet50_kolektor_ft/threshold_sweep.json" || \
        echo "✗ Threshold sweep failed (continuing)"

# ── Step 1b: Training history plot ────────────────────────────────────────
echo ""
echo "▶ Step 1b: Training history plot"
uv run python scripts/plot_finetune_history.py && \
    echo "✓ Training curve → figures/resnet50_kolektor_ft/training_curve.png" || \
    echo "✗ Training history plot failed (continuing)"
else
    echo "✗ Checkpoint not found: models/resnet50_kolektor_ft_best.pt"
    echo "  Run finetune_resnet_kolektor.py first."
    exit 1
fi

# ── Step 2: Bake-off comparison figures ────────────────────────────────────
echo ""
echo "▶ Step 2: Bake-off comparison figures"
uv run python scripts/make_comparison_figures.py && \
    echo "✓ Comparison figures → figures/bakeoff/" || \
    echo "✗ make_comparison_figures failed (continuing)"

# ── Step 3: Print ResNet50+FT results summary ─────────────────────────────
echo ""
echo "▶ Step 3: Results summary"
uv run python - <<'PYEOF'
import json
from pathlib import Path

p = Path("results/resnet50_kolektor_ft/metrics.json")
if p.exists():
    m = json.loads(p.read_text())
    print("\n=== ResNet50+FT eval metrics ===")
    for ds, vals in m.get("eval", {}).items():
        print(f"  {ds:20s}  F1={vals.get('f1', 0):.4f}  AUC={vals.get('roc_auc', float('nan')):.4f}")
    print(f"\n  Best val F1: {m.get('best_val_f1', 0):.4f}")

sp = Path("results/resnet50_kolektor_ft/threshold_sweep.json")
if sp.exists():
    s = json.loads(sp.read_text())
    print(f"\n=== Threshold sweep (opt τ={s.get('best_threshold', 0.5):.3f}) ===")
    for ds, vals in s.get("sets", {}).items():
        if ds in ("kolektor_test", "gc10_test", "severstal_test"):
            print(f"  {ds:20s}  F1@τ={vals.get('f1_at_opt_threshold', 0):.4f}  "
                  f"F1@0.5={vals.get('f1_at_0.5', 0):.4f}")
PYEOF

# ── Step 4: Auto-update slides.qmd ────────────────────────────────────────
echo ""
echo "▶ Step 4: Auto-update slides with ResNet50+FT results"
uv run python scripts/update_slides_results.py && \
    echo "✓ Slides updated" || \
    echo "✗ Slides update failed (check update_slides_results.py)"

echo ""
echo "=== Pipeline complete at $(date) ==="
echo ""
echo "Next steps:"
echo "  1. Verify website/slides.qmd 'Supervised adaptation' table numbers look correct"
echo "  2. Run bootstrap YOLO:"
echo "     uv run python scripts/train_yolo_bootstrap.py --batch 16"
echo "  3. After bootstrap YOLO completes:"
echo "     uv run python scripts/eval_yolo.py --weights results/yolo_bootstrap/yolo11s_bootstrap/weights/best.pt --output-dir results/yolo_bootstrap"
echo "  4. Final comparison figures: uv run python scripts/make_comparison_figures.py"
