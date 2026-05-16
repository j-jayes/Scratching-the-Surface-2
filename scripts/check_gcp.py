"""Probe GCP / Vertex: check gcloud auth, list available Gemini models, sanity caption."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GCP_PROJECT_ID
from src.prompts.defect_analysis import BLIND_SCHEMA, SYSTEM, USER_BLIND
from src.vlm_clients import call_gemini


def _sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or p.stderr).strip()


# Gemini lineup as of May 2026 (https://ai.google.dev/gemini-api/docs/models).
# Gemini 3 Pro Preview was shut down 2026-03-09 → use 3.1 Pro Preview.
CANDIDATE_MODELS = [
    "gemini-3.1-pro-preview",       # flagship reasoning
    "gemini-3-flash-preview",       # frontier-class, cheaper
    "gemini-3.1-flash-lite",        # stable, bulk/cheap
    "gemini-2.5-pro",               # stable fallback
    "gemini-2.5-flash",             # stable fallback
]


def main() -> int:
    print("=== gcloud config ===")
    code, out = _sh(["gcloud", "config", "list", "--format=json"])
    if code != 0:
        print(f"  gcloud not available: {out}")
        return 1
    cfg = json.loads(out)
    gcloud_proj = cfg.get("core", {}).get("project")
    print(f"  account: {cfg.get('core', {}).get('account')}")
    print(f"  gcloud project: {gcloud_proj}")
    print(f"  .env GCP_PROJECT_ID: {GCP_PROJECT_ID}")
    if gcloud_proj and GCP_PROJECT_ID and gcloud_proj != GCP_PROJECT_ID:
        print("  WARNING: gcloud project and .env GCP_PROJECT_ID differ — Vertex calls will use .env value.")

    print("\n=== ADC status ===")
    adc_code, adc_out = _sh(["gcloud", "auth", "application-default", "print-access-token"])
    if adc_code != 0:
        print("  ADC token unavailable. Run:")
        print(f"    gcloud auth application-default login")
        print(f"    gcloud auth application-default set-quota-project {GCP_PROJECT_ID}")
        return 1
    print("  ADC token OK")

    img = Image.new("RGB", (256, 256), (200, 200, 200))
    ImageDraw.Draw(img).line([(20, 200), (240, 60)], fill=(40, 40, 40), width=3)

    print("\n=== Probe Gemini models ===")
    for m in CANDIDATE_MODELS:
        r = call_gemini(m, SYSTEM, USER_BLIND, img, schema=BLIND_SCHEMA, phase="phase0", note=f"probe-{m}")
        status = "ok" if r.error is None else "FAIL"
        print(f"  {m:<28} {status}  tokens={r.in_tok}/{r.out_tok}  parsed={r.parsed}  err={r.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
