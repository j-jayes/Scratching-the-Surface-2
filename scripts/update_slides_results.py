"""Auto-update slides.qmd with actual ResNet50+FT and YOLO-bootstrap results.

Reads the results JSON files and replaces placeholder dashes with real values.

Usage:
    uv run python scripts/update_slides_results.py

Safe to run multiple times — idempotent on the numeric values.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


SLIDES_PATH = Path("website/slides.qmd")
INDEX_PATH  = Path("website/index.qmd")

RESNET_FT_METRICS   = Path("results/resnet50_kolektor_ft/metrics.json")
RESNET_FT_SWEEP     = Path("results/resnet50_kolektor_ft/threshold_sweep.json")
BOOTSTRAP_METRICS   = Path("results/yolo_bootstrap/eval_metrics.json")


def load_resnet_ft_f1() -> dict[str, float | None]:
    """Return dict with kolektor_f1, gc10_f1.

    Prefers the eval F1 from metrics.json (which uses TTA when enabled) and
    falls back to f1_at_opt_threshold from threshold_sweep.json for backward
    compatibility (v1 results which don't have TTA).
    """
    if not RESNET_FT_METRICS.exists():
        return {}

    with RESNET_FT_METRICS.open() as f:
        m = json.load(f)

    sweep: dict = {}
    if RESNET_FT_SWEEP.exists():
        with RESNET_FT_SWEEP.open() as f:
            sweep = json.load(f).get("sets", {})

    def get_f1(ds: str) -> float | None:
        # Prefer metrics.json eval section (may use TTA)
        base = m.get("eval", {}).get(ds, {})
        if base.get("f1") is not None:
            return float(base["f1"])
        # Fall back to threshold_sweep opt-threshold F1
        sw = sweep.get(ds, {})
        if sw.get("f1_at_opt_threshold") is not None:
            return float(sw["f1_at_opt_threshold"])
        return None

    return {
        "kolektor_f1": get_f1("kolektor_test"),
        "gc10_f1":     get_f1("gc10_test"),
    }


def load_bootstrap_f1() -> dict[str, float | None]:
    if not BOOTSTRAP_METRICS.exists():
        return {}
    with BOOTSTRAP_METRICS.open() as f:
        m = json.load(f)
    return {
        "kolektor_f1": m.get("kolektor_test", {}).get("best_f1"),
        "gc10_f1":     m.get("gc10_test",     {}).get("best_f1"),
    }


def fmt(v: float | None) -> str:
    """Format F1 value as 3dp string, or '—' if unavailable."""
    if v is None:
        return "—"
    return f"{v:.3f}"


def main() -> None:
    slides = SLIDES_PATH.read_text(encoding="utf-8")

    ft = load_resnet_ft_f1()
    bootstrap = load_bootstrap_f1()

    changes: list[str] = []

    # ── Update "Supervised adaptation vs. zero-shot" table ───────────────────
    # Pattern: | ResNet50+FT | — | — | ✓ kolektor train |
    # Also handles already-filled pattern (v2 promotion over v1)
    if ft.get("kolektor_f1") is not None and ft.get("gc10_f1") is not None:
        kol_f1 = fmt(ft["kolektor_f1"])
        gc10_f1 = fmt(ft["gc10_f1"])
        new = f"| ResNet50+FT | **{kol_f1}** | {gc10_f1} | ✓ kolektor train |"
        old_placeholder = "| ResNet50+FT | — | — | ✓ kolektor train |"
        if old_placeholder in slides:
            slides = slides.replace(old_placeholder, new)
            changes.append(f"  Supervised adaptation table: kolektor={kol_f1}, gc10={gc10_f1}")
        else:
            # Try to replace any previously-filled value (re-run idempotency for v2 promotion)
            import re as _re
            pattern = r"\| ResNet50\+FT \| \*?\*?[\d.]+\*?\*? \| [\d.]+ \| ✓ kolektor train \|"
            if _re.search(pattern, slides):
                slides = _re.sub(pattern, new, slides)
                changes.append(f"  Supervised adaptation table (update): kolektor={kol_f1}, gc10={gc10_f1}")

        # Also update inline "0.166 → X.XXX" finding text in the same slide
        import re as _re
        _baseline_str = "0.166"
        _gap_pat = rf"closes the gap significantly \({_baseline_str} → [\d.]+\)"
        _gap_new = f"closes the gap significantly ({_baseline_str} → {kol_f1})"
        if _re.search(_gap_pat, slides):
            slides = _re.sub(_gap_pat, _gap_new, slides)
            changes.append(f"  Supervised adaptation inline gap text: → {kol_f1}")

    # ── Update "Bootstrap results" table — ResNet50+FT row ───────────────────
    # Pattern: | ResNet50+FT | — | — | human labels + kolektor data |
    if ft.get("gc10_f1") is not None and ft.get("kolektor_f1") is not None:
        gc10_f1 = fmt(ft["gc10_f1"])
        kol_f1 = fmt(ft["kolektor_f1"])
        new = f"| ResNet50+FT | {gc10_f1} | **{kol_f1}** | human labels + kolektor data |"
        old_placeholder = "| ResNet50+FT | — | — | human labels + kolektor data |"
        if old_placeholder in slides:
            slides = slides.replace(old_placeholder, new)
            changes.append(f"  Bootstrap table ResNet50+FT: gc10={gc10_f1}, kolektor={kol_f1}")
        else:
            import re as _re
            pattern = r"\| ResNet50\+FT \| [\d.]+ \| \*?\*?[\d.]+\*?\*? \| human labels \+ kolektor data \|"
            if _re.search(pattern, slides):
                slides = _re.sub(pattern, new, slides)
                changes.append(f"  Bootstrap table ResNet50+FT (update): gc10={gc10_f1}, kolektor={kol_f1}")

    # ── Update "Bootstrap results" table — YOLO-bootstrap row ────────────────
    # Pattern: | **YOLO-bootstrap** | — | — | $2.33 API |
    if bootstrap.get("gc10_f1") is not None and bootstrap.get("kolektor_f1") is not None:
        gc10_b = fmt(bootstrap["gc10_f1"])
        kol_b = fmt(bootstrap["kolektor_f1"])
        old = "| **YOLO-bootstrap** | — | — | $2.33 API |"
        new = f"| **YOLO-bootstrap** | **{gc10_b}** | {kol_b} | $2.33 API |"
        if old in slides:
            slides = slides.replace(old, new)
            changes.append(f"  Bootstrap table YOLO-bootstrap: gc10={gc10_b}, kolektor={kol_b}")

    if changes:
        SLIDES_PATH.write_text(slides, encoding="utf-8")
        print(f"Updated {SLIDES_PATH}:")
        for c in changes:
            print(c)
    else:
        print("No slides.qmd updates. Results may not yet be available or values already filled.")

    # ── Also update index.qmd ─────────────────────────────────────────────────
    index = INDEX_PATH.read_text(encoding="utf-8")
    index_changes: list[str] = []

    # ResNet50+FT row in index.qmd:
    # | **ResNet50+FT** | — | — | — | fine-tuning on kolektor train in progress |
    if ft.get("gc10_f1") is not None and ft.get("kolektor_f1") is not None:
        gc10_f1 = fmt(ft["gc10_f1"])
        kol_f1 = fmt(ft["kolektor_f1"])
        new = f"| **ResNet50+FT** | — | {gc10_f1} | **{kol_f1}** | supervised domain adaptation |"
        old_placeholder = "| **ResNet50+FT** | — | — | — | fine-tuning on kolektor train in progress |"
        if old_placeholder in index:
            index = index.replace(old_placeholder, new)
            index_changes.append(f"  index.qmd ResNet50+FT: gc10={gc10_f1}, kolektor={kol_f1}")
        else:
            import re as _re
            pattern = r"\| \*\*ResNet50\+FT\*\* \| — \| [\d.]+ \| \*?\*?[\d.]+\*?\*? \| supervised domain adaptation \|"
            if _re.search(pattern, index):
                index = _re.sub(pattern, new, index)
                index_changes.append(f"  index.qmd ResNet50+FT (update): gc10={gc10_f1}, kolektor={kol_f1}")

    # Inline text mentions of ResNet50+FT kolektor F1 (progress note + central finding)
    if ft.get("kolektor_f1") is not None:
        import re as _re
        kol_f1_val = ft["kolektor_f1"]
        kol_f1_str = fmt(kol_f1_val)
        _baseline_kol = 0.1622  # ResNet50 baseline kolektor F1
        ratio = kol_f1_val / _baseline_kol
        ratio_str = f"{ratio:.1f}×"
        # "kolektor F1 = **0.425** (2.6× improvement over baseline)"
        _prog_pat = r"kolektor F1 = \*\*[\d.]+\*\* \([\d.]+× improvement over baseline\)"
        _prog_new = f"kolektor F1 = **{kol_f1_str}** ({ratio_str} improvement over baseline)"
        if _re.search(_prog_pat, index):
            index = _re.sub(_prog_pat, _prog_new, index)
            index_changes.append(f"  index.qmd inline kolektor F1: {kol_f1_str} ({ratio_str})")
        # "the VLM still leads: 0.79 vs. 0.43."  ([\d]+\.[\d]+ won't eat trailing period)
        _vlm_pat = r"the VLM still leads: \d+\.\d+ vs\. \d+\.\d+"
        _vlm_new = f"the VLM still leads: 0.79 vs. {kol_f1_str}"
        if _re.search(_vlm_pat, index):
            index = _re.sub(_vlm_pat, _vlm_new, index)
            index_changes.append(f"  index.qmd VLM vs FT comparison: 0.79 vs {kol_f1_str}")

    # YOLO-bootstrap row in index.qmd:
    # | **YOLO-bootstrap** | — | — | — | training on 484 pseudo-labels |
    if bootstrap.get("gc10_f1") is not None and bootstrap.get("kolektor_f1") is not None:
        gc10_b = fmt(bootstrap["gc10_f1"])
        kol_b = fmt(bootstrap["kolektor_f1"])
        old = "| **YOLO-bootstrap** | — | — | — | training on 484 pseudo-labels |"
        new = f"| **YOLO-bootstrap** | — | **{gc10_b}** | {kol_b} | trained on $2.33 VLM pseudo-labels |"
        if old in index:
            index = index.replace(old, new)
            index_changes.append(f"  index.qmd YOLO-bootstrap: gc10={gc10_b}, kolektor={kol_b}")

    if index_changes:
        INDEX_PATH.write_text(index, encoding="utf-8")
        print(f"Updated {INDEX_PATH}:")
        for c in index_changes:
            print(c)

    if not ft:
        print("  ⚠ ResNet50+FT results not found — run finetune_resnet_kolektor.py first")
    if not bootstrap:
        print("  ℹ YOLO-bootstrap results not yet available")


if __name__ == "__main__":
    main()
