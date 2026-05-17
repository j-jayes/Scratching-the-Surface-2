#!/usr/bin/env bash
# Wait for fine-tune v1 to complete, then automatically run the post-pipeline.
# Polls every 60 seconds. Safe to kill and restart.
#
# Usage:
#   bash scripts/wait_and_run_pipeline.sh
#   bash scripts/wait_and_run_pipeline.sh --skip-wait  (if already done)

# Use -u (unset var error) but NOT -e (exit on error) — we want to continue
# through pipeline steps even if one step fails.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="models/resnet50_kolektor_ft_best.pt"
METRICS="results/resnet50_kolektor_ft/metrics.json"
FINETUNE_PID="${FINETUNE_PID:-}"

skip_wait=false
for arg in "$@"; do
  [[ "$arg" == "--skip-wait" ]] && skip_wait=true
done

if [[ "$skip_wait" == "false" ]]; then
  echo "=== Waiting for fine-tune v1 to complete... ==="
  echo "    Monitoring: $METRICS"
  echo "    (Poll interval: 60s — kill this script if needed)"
  echo ""

  while true; do
    if [[ -f "$METRICS" ]]; then
      echo ""
      echo "✓ Fine-tune complete! Metrics found at: $METRICS"
      break
    fi

    # Also check if the fine-tune PID is dead with no metrics (failed)
    if [[ -n "$FINETUNE_PID" ]] && ! kill -0 "$FINETUNE_PID" 2>/dev/null; then
      if [[ ! -f "$METRICS" ]]; then
        echo "✗ Fine-tune process $FINETUNE_PID died without producing metrics!"
        echo "  Check the terminal running finetune_resnet_kolektor.py for errors."
        exit 1
      fi
    fi

    now=$(date '+%H:%M:%S')
    # Check checkpoint age as progress indicator
    if [[ -f "$CHECKPOINT" ]]; then
      chk_age=$(( $(date +%s) - $(stat -f%m "$CHECKPOINT") ))
      chk_min=$(( chk_age / 60 ))
      echo -n "  [$now] Training in progress (checkpoint last updated ${chk_min}min ago)..."
    else
      echo -n "  [$now] Waiting for checkpoint to appear..."
    fi
    echo ""
    sleep 60
  done
fi

echo ""
echo "=== Running post-fine-tune pipeline ==="
bash scripts/run_post_finetune_pipeline.sh

# ── Decide whether to run v2 ──────────────────────────────────────────────────
echo ""
echo "=== Evaluating whether to run v2 training ==="
RUN_V2=false
KOL_F1=$(uv run python - <<'PYEOF'
import json, sys
from pathlib import Path

metrics_path = Path("results/resnet50_kolektor_ft/metrics.json")
if not metrics_path.exists():
    print("0.0")
    sys.exit(0)

with metrics_path.open() as f:
    m = json.load(f)

sweep_path = Path("results/resnet50_kolektor_ft/threshold_sweep.json")
sweep = {}
if sweep_path.exists():
    with sweep_path.open() as f:
        sweep = json.load(f).get("sets", {})

def get_f1(ds):
    sw = sweep.get(ds, {})
    if sw.get("f1_at_opt_threshold") is not None:
        return sw["f1_at_opt_threshold"]
    return m.get("eval", {}).get(ds, {}).get("f1", 0)

kol_f1 = get_f1("kolektor_test")
gc10_f1 = get_f1("gc10_test")
vlm_kol = 0.7925

print(f"\n[v1 Results]")
print(f"  kolektor_test F1 = {kol_f1:.4f}  (VLM baseline: {vlm_kol:.4f})")
print(f"  gc10_test     F1 = {gc10_f1:.4f}")
# Final line is the f1 value for bash
import sys
sys.stdout.flush()
print(f"__KOL_F1__={kol_f1:.4f}")
PYEOF
)
KOL_F1_NUM=$(echo "$KOL_F1" | grep "__KOL_F1__" | cut -d= -f2)
echo "$KOL_F1" | grep -v "__KOL_F1__"

if (( $(echo "$KOL_F1_NUM < 0.60" | bc -l) )); then
    echo ""
    echo "→ kolektor F1 ($KOL_F1_NUM) below 0.60 — running v2 with RRC augmentation"
    uv run python scripts/finetune_resnet_kolektor.py --config configs/resnet50_kolektor_ft_v2.yaml

    echo ""
    echo "▶ v2 threshold sweep"
    uv run python scripts/optimise_threshold.py \
        --checkpoint models/resnet50_kolektor_ft_v2_best.pt \
        --results-dir results/resnet50_kolektor_ft_v2 \
        --figures-dir figures/resnet50_kolektor_ft_v2

    echo ""
    echo "▶ v2 training history"
    uv run python scripts/plot_finetune_history.py \
        --metrics results/resnet50_kolektor_ft_v2/metrics.json \
        --out figures/resnet50_kolektor_ft_v2/training_curve.png

    # Promote v2 as the fine-tune to use in slides if it's better
    V2_KOL=$(uv run python -c "
import json
from pathlib import Path
p = Path('results/resnet50_kolektor_ft_v2/threshold_sweep.json')
if p.exists():
    import json
    s = json.loads(p.read_text()).get('sets', {})
    print(s.get('kolektor_test', {}).get('f1_at_opt_threshold', 0))
else:
    p2 = Path('results/resnet50_kolektor_ft_v2/metrics.json')
    if p2.exists():
        m = json.loads(p2.read_text())
        print(m.get('eval', {}).get('kolektor_test', {}).get('f1', 0))
    else:
        print(0)
" 2>/dev/null)
    if (( $(echo "$V2_KOL > $KOL_F1_NUM" | bc -l) )); then
        echo ""
        echo "✓ v2 kolektor F1 ($V2_KOL) > v1 ($KOL_F1_NUM) — promoting v2 as primary"
        # Symlink or copy v2 results as the canonical ft results for slides
        cp results/resnet50_kolektor_ft_v2/metrics.json results/resnet50_kolektor_ft/metrics.json
        [[ -f results/resnet50_kolektor_ft_v2/threshold_sweep.json ]] && \
            cp results/resnet50_kolektor_ft_v2/threshold_sweep.json results/resnet50_kolektor_ft/threshold_sweep.json
        echo "  (v2 results copied to results/resnet50_kolektor_ft/ for slides update)"
    else
        echo ""
        echo "  v2 kolektor F1 ($V2_KOL) not better than v1 ($KOL_F1_NUM) — keeping v1"
    fi
else
    echo ""
    echo "→ kolektor F1 ($KOL_F1_NUM) >= 0.60 — v2 not needed (v1 sufficient)"
fi

# ── Bootstrap YOLO (runs after fine-tune frees MPS memory) ───────────────────
echo ""
echo "=== Starting bootstrap YOLO training ==="
if [[ -f "data/yolo_bootstrap/defect_bootstrap.yaml" ]]; then
    uv run python scripts/train_yolo_bootstrap.py --batch 16
    echo "✓ Bootstrap YOLO trained"

    echo ""
    echo "▶ Bootstrap YOLO eval"
    YOLO_WEIGHTS="results/yolo_bootstrap/yolo11s_bootstrap/weights/best.pt"
    if [[ -f "$YOLO_WEIGHTS" ]]; then
        uv run python scripts/eval_yolo.py \
            --weights "$YOLO_WEIGHTS" \
            --output-dir results/yolo_bootstrap
        echo "✓ Bootstrap YOLO eval complete"

        echo ""
        echo "▶ Regenerating all comparison figures"
        uv run python scripts/make_comparison_figures.py
        echo "✓ Comparison figures updated"

        echo ""
        echo "▶ Updating slides with all results"
        uv run python scripts/update_slides_results.py
        echo "✓ Slides updated"
    else
        echo "✗ Bootstrap YOLO weights not found at: $YOLO_WEIGHTS"
    fi
else
    echo "✗ Bootstrap dataset not found: data/yolo_bootstrap/defect_bootstrap.yaml"
    echo "  Run: uv run python scripts/bootstrap_labels.py"
fi

echo ""
echo "=== Full pipeline complete at $(date) ==="
echo ""
echo "Final deliverables:"
echo "  - results/resnet50_kolektor_ft/metrics.json"
echo "  - results/yolo_bootstrap/eval_metrics.json"
echo "  - figures/bakeoff/f1_comparison.png"
echo "  - figures/bakeoff/bootstrap_comparison.png"
echo "  - website/slides.qmd (updated with actual numbers)"
echo "  - website/index.qmd  (updated with actual numbers)"
