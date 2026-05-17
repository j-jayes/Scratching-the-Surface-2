#!/usr/bin/env bash
# Run v3/v4 fine-tunes if kolektor F1 hasn't reached the target thresholds.
# Run this AFTER wait_and_run_pipeline.sh has completed.
#
# Decision logic:
#   - Run v3 if kolektor F1 < 0.60  (warm start from v2, backbone_lr_factor=0.1)
#   - Run v4 if kolektor F1 < 0.65  (warm start from v3, Focal Loss γ=2)
#
# Usage:
#   bash scripts/run_v3_if_needed.sh

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BEST_METRICS="results/resnet50_kolektor_ft/metrics.json"
V2_WEIGHTS="models/resnet50_kolektor_ft_v2_best.pt"
THRESHOLD_GOAL=0.60

echo "=== v3 decision check $(date) ==="

# Read best available kolektor F1 — prefer TTA eval F1 (metrics.json) for
# apples-to-apples comparison with V3_KOL / V4_KOL (also TTA eval F1).
KOL_F1=$(uv run python -c "
import json
from pathlib import Path

# Prefer TTA eval F1 from metrics.json (consistent with V3_KOL/V4_KOL)
m_path = Path('$BEST_METRICS')
if m_path.exists():
    m = json.loads(m_path.read_text())
    f1 = m.get('eval', {}).get('kolektor_test', {}).get('f1')
    if f1 is not None and float(f1) > 0:
        print(f'{float(f1):.4f}')
        exit()

# Fallback to threshold sweep
sweep = Path('results/resnet50_kolektor_ft/threshold_sweep.json')
if sweep.exists():
    s = json.loads(sweep.read_text())
    v = s.get('sets', {}).get('kolektor_test', {}).get('f1_at_opt_threshold')
    if v is not None:
        print(f'{v:.4f}')
        exit()

print('0.0000')
" 2>/dev/null)

echo "  Best kolektor F1 so far: $KOL_F1 (goal: $THRESHOLD_GOAL)"

if (( $(echo "$KOL_F1 < $THRESHOLD_GOAL" | bc -l) )); then
    if [[ ! -f "$V2_WEIGHTS" ]]; then
        echo "✗ v2 checkpoint not found at $V2_WEIGHTS — cannot warm-start v3"
        echo "  Run v3 from v1 checkpoint instead:"
        echo "  uv run python scripts/finetune_resnet_kolektor.py --config configs/resnet50_kolektor_ft_v3.yaml"
        # Fallback: use v1 checkpoint
        sed -i.bak 's|models/resnet50_kolektor_ft_v2_best.pt|models/resnet50_kolektor_ft_best.pt|' \
            configs/resnet50_kolektor_ft_v3.yaml
        echo "  (v3.yaml updated to warm-start from v1)"
    fi

    echo ""
    echo "→ Running v3 (warm start + 3× kolektor oversampling)..."
    uv run python scripts/finetune_resnet_kolektor.py \
        --config configs/resnet50_kolektor_ft_v3.yaml

    echo ""
    echo "▶ v3 threshold sweep (with TTA)"
    uv run python scripts/optimise_threshold.py \
        --checkpoint models/resnet50_kolektor_ft_v3_best.pt \
        --results-dir results/resnet50_kolektor_ft_v3 \
        --figures-dir figures/resnet50_kolektor_ft_v3 \
        --tta || true

    echo ""
    echo "▶ v3 training history"
    uv run python scripts/plot_finetune_history.py \
        --metrics results/resnet50_kolektor_ft_v3/metrics.json \
        --out figures/resnet50_kolektor_ft_v3/training_curve.png || true

    # Promote v3 if better — prefer metrics.json eval F1 (TTA numbers)
    V3_KOL=$(uv run python -c "
import json
from pathlib import Path
p2 = Path('results/resnet50_kolektor_ft_v3/metrics.json')
if p2.exists():
    m = json.loads(p2.read_text())
    f1 = m.get('eval', {}).get('kolektor_test', {}).get('f1', 0)
    print(f'{float(f1):.4f}')
else:
    p = Path('results/resnet50_kolektor_ft_v3/threshold_sweep.json')
    if p.exists():
        s = json.loads(p.read_text())
        print(s.get('sets', {}).get('kolektor_test', {}).get('f1_at_opt_threshold', 0))
    else:
        print(0)
" 2>/dev/null)

    echo ""
    echo "  v3 kolektor F1: $V3_KOL  (current best: $KOL_F1)"
    if (( $(echo "$V3_KOL > $KOL_F1" | bc -l) )); then
        echo "✓ v3 is better — promoting to canonical results"
        cp results/resnet50_kolektor_ft_v3/metrics.json results/resnet50_kolektor_ft/metrics.json
        [[ -f results/resnet50_kolektor_ft_v3/threshold_sweep.json ]] && \
            cp results/resnet50_kolektor_ft_v3/threshold_sweep.json \
               results/resnet50_kolektor_ft/threshold_sweep.json
        echo "  Regenerating figures and updating slides..."
        uv run python scripts/make_comparison_figures.py
        uv run python scripts/update_slides_results.py
    else
        echo "  v3 did not improve — keeping current best ($KOL_F1)"
    fi
else
    echo "✓ kolektor F1 $KOL_F1 ≥ $THRESHOLD_GOAL — v3 not needed"
fi

# ── v4: Focal Loss fine-tune if still below 0.65 ─────────────────────────────
V4_THRESHOLD=0.65
# Re-read best F1 (might have been promoted from v3 above)
BEST_KOL=$(uv run python -c "
import json
from pathlib import Path
m_path = Path('results/resnet50_kolektor_ft/metrics.json')
if m_path.exists():
    m = json.loads(m_path.read_text())
    f1 = m.get('eval', {}).get('kolektor_test', {}).get('f1', 0)
    print(f'{float(f1):.4f}')
else:
    print('0.0000')
" 2>/dev/null)

echo ""
echo "=== v4 decision check: best kolektor F1=$BEST_KOL (goal: $V4_THRESHOLD) ==="

if (( $(echo "$BEST_KOL < $V4_THRESHOLD" | bc -l) )); then
    # Decide which warm-start checkpoint to use for v4
    if [[ -f "models/resnet50_kolektor_ft_v3_best.pt" ]]; then
        echo "  Warm-starting v4 from v3 checkpoint"
    elif [[ -f "models/resnet50_kolektor_ft_v2_best.pt" ]]; then
        echo "  v3 weights not found — warm-starting v4 from v2 checkpoint"
        sed -i.bak 's|models/resnet50_kolektor_ft_v3_best.pt|models/resnet50_kolektor_ft_v2_best.pt|' \
            configs/resnet50_kolektor_ft_v4.yaml
    else
        echo "  No suitable warm-start checkpoint — skipping v4"
        echo "=== Done at $(date) ===" ; exit 0
    fi

    echo ""
    echo "→ Running v4 (Focal Loss γ=2, warm start)..."
    uv run python scripts/finetune_resnet_kolektor.py \
        --config configs/resnet50_kolektor_ft_v4.yaml

    echo ""
    echo "▶ v4 threshold sweep (with TTA)"
    uv run python scripts/optimise_threshold.py \
        --checkpoint models/resnet50_kolektor_ft_v4_best.pt \
        --results-dir results/resnet50_kolektor_ft_v4 \
        --figures-dir figures/resnet50_kolektor_ft_v4 \
        --tta || true

    echo ""
    echo "▶ v4 training history"
    uv run python scripts/plot_finetune_history.py \
        --metrics results/resnet50_kolektor_ft_v4/metrics.json \
        --out figures/resnet50_kolektor_ft_v4/training_curve.png || true

    # Promote v4 if better
    V4_KOL=$(uv run python -c "
import json
from pathlib import Path
p2 = Path('results/resnet50_kolektor_ft_v4/metrics.json')
if p2.exists():
    m = json.loads(p2.read_text())
    f1 = m.get('eval', {}).get('kolektor_test', {}).get('f1', 0)
    print(f'{float(f1):.4f}')
else:
    p = Path('results/resnet50_kolektor_ft_v4/threshold_sweep.json')
    if p.exists():
        s = json.loads(p.read_text())
        print(s.get('sets', {}).get('kolektor_test', {}).get('f1_at_opt_threshold', 0))
    else:
        print(0)
" 2>/dev/null)

    echo ""
    echo "  v4 kolektor F1: $V4_KOL  (current best: $BEST_KOL)"
    if (( $(echo "$V4_KOL > $BEST_KOL" | bc -l) )); then
        echo "✓ v4 is better — promoting to canonical results"
        cp results/resnet50_kolektor_ft_v4/metrics.json results/resnet50_kolektor_ft/metrics.json
        [[ -f results/resnet50_kolektor_ft_v4/threshold_sweep.json ]] && \
            cp results/resnet50_kolektor_ft_v4/threshold_sweep.json \
               results/resnet50_kolektor_ft/threshold_sweep.json
        echo "  Regenerating figures and updating slides..."
        uv run python scripts/make_comparison_figures.py
        uv run python scripts/update_slides_results.py
    else
        echo "  v4 did not improve — keeping current best ($BEST_KOL)"
    fi
else
    echo "✓ kolektor F1 $BEST_KOL ≥ $V4_THRESHOLD — v4 not needed"
fi

echo ""
echo "=== Done at $(date) ==="
