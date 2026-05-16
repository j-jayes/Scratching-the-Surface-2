"""Unified configuration: paths, models, and constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
CONFIGS_DIR = REPO_ROOT / "configs"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── OpenAI native (May 2026 lineup) ──────────────────────────────────────────
OPENAI_API_KEY = os.getenv("NATIVE_OPENAI_API_KEY", "")
OPENAI_MODELS = {
    "flagship": "gpt-5.5",      # $5 / $30 per 1M — qualitative showcase + ablation only
    "mid": "gpt-5.4",            # $2.50 / $15 — bootstrap pseudo-labels
    "mini": "gpt-5.4-mini",      # $0.75 / $4.50 — bulk eval, hybrid filter
}

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT", "")
AOAI_API_KEY = os.getenv("AOAI_API_KEY", "")
AOAI_API_VERSION = os.getenv("AOAI_API_VERSION", "2024-08-01-preview")
AOAI_DEPLOYMENT = os.getenv("AOAI_DEPLOYMENT", "")
AOAI_MODEL = os.getenv("AOAI_MODEL", "")

# ── GCP / Vertex (Gemini 3.x lineup, May 2026) ───────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GEMINI_MODELS = {
    "flagship": "gemini-3.1-pro-preview",   # frontier reasoning
    "mid":      "gemini-3-flash-preview",   # frontier-class, cheaper
    "mini":     "gemini-3.1-flash-lite",    # stable, bulk/cheap
}

# ── Pricing table for cost tracking ($ per 1M tokens) ────────────────────────
PRICING = {
    "gpt-5.5":      {"in": 5.00,  "out": 30.00},
    "gpt-5.4":      {"in": 2.50,  "out": 15.00},
    "gpt-5.4-mini": {"in": 0.75,  "out":  4.50},
    "gpt-4.1":      {"in": 2.00,  "out":  8.00},   # Azure approx
    "gemini-3.1-pro-preview":  {"in": 2.00, "out": 12.00},  # approximate, verify on probe
    "gemini-3-flash-preview":  {"in": 0.30, "out":  2.50},
    "gemini-3.1-flash-lite":   {"in": 0.10, "out":  0.40},
}

# ── Budget cap ───────────────────────────────────────────────────────────────
VLM_BUDGET_USD = 50.0
COST_LEDGER_PATH = RESULTS_DIR / "cost_ledger.csv"

# ── Dataset class taxonomy ───────────────────────────────────────────────────
SEVERSTAL_CLASSES = ["defect_1", "defect_2", "defect_3", "defect_4"]
NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
# KolektorSDD2 replaces GC10 as the held-out cross-domain test set.
# Binary: defect / normal (from GT mask). Images are 229×645 grayscale RGB.
KOLEKTOR_CLASSES = ["defect"]  # binary
# GC10-DET: second held-out cross-domain test set (10 defect classes → binary eval)
GC10_CLASSES = [
    "punching_hole", "weld_line", "crescent_gap", "water_spot", "oil_spot",
    "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding",
]

# ── Image preprocessing targets ──────────────────────────────────────────────
RESNET_INPUT = 512        # square after centre-crop
YOLO_IMGSZ = 1024         # letterbox square
VLM_INPUT_MAX = 1024      # longest side then pad to square
