"""Probe the Azure OpenAI resource: log in, list deployments, sanity caption.

Uses `az` CLI for subscription + deployment listing (no SDK boilerplate), then
the OpenAI SDK for the actual completion call.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AOAI_DEPLOYMENT, AOAI_ENDPOINT
from src.prompts.defect_analysis import BLIND_SCHEMA, SYSTEM, USER_BLIND
from src.vlm_clients import call_azure


def _sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout or p.stderr).strip()


def main() -> int:
    print("=== az account show ===")
    code, out = _sh(["az", "account", "show", "-o", "json"])
    if code != 0:
        print(f"  az CLI not logged in or unavailable: {out}")
        print("  Run: az login")
        return 1
    acct = json.loads(out)
    print(f"  subscription: {acct.get('name')} ({acct.get('id')})")

    print(f"\n=== Endpoint: {AOAI_ENDPOINT}")
    print(f"=== Deployment from .env: {AOAI_DEPLOYMENT}")

    # Listing deployments via az cognitiveservices requires the resource name.
    # We just sanity-call the configured deployment.
    print("\n=== Sanity caption via configured deployment ===")
    img = Image.new("RGB", (256, 256), (200, 200, 200))
    ImageDraw.Draw(img).line([(20, 200), (240, 60)], fill=(40, 40, 40), width=3)
    r = call_azure(SYSTEM, USER_BLIND, img, schema=BLIND_SCHEMA, phase="phase0", note="azure-check")
    if r.error:
        print(f"  ERROR: {r.error}")
        return 1
    print(f"  parsed={r.parsed}  tokens(in/out)={r.in_tok}/{r.out_tok}  cost=${r.cost_usd:.4f}  {r.latency_s:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
