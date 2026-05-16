"""Probe the native OpenAI key: list vision-capable models + try a tiny captioning call."""
from __future__ import annotations

import sys
from io import BytesIO

from PIL import Image, ImageDraw

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.config import OPENAI_API_KEY, OPENAI_MODELS
from src.prompts.defect_analysis import BLIND_SCHEMA, SYSTEM, USER_BLIND
from src.vlm_clients import call_openai


def _sample_image() -> Image.Image:
    img = Image.new("RGB", (256, 256), (200, 200, 200))
    d = ImageDraw.Draw(img)
    d.line([(20, 200), (240, 60)], fill=(40, 40, 40), width=3)
    return img


def main() -> int:
    if not OPENAI_API_KEY:
        print("NATIVE_OPENAI_API_KEY is not set in .env")
        return 1

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    print("=== Listing available OpenAI models ===")
    try:
        models = sorted(m.id for m in client.models.list().data)
        vision_like = [m for m in models if any(k in m for k in ("gpt-5", "gpt-4", "o4", "o3", "vision"))]
        print(f"Total: {len(models)} | vision-capable candidates: {len(vision_like)}")
        for m in vision_like[:40]:
            print(f"  {m}")
    except Exception as e:  # noqa: BLE001
        print(f"  list failed: {e}")

    print("\n=== Sanity captioning on planned model tiers ===")
    img = _sample_image()
    for tier, name in OPENAI_MODELS.items():
        print(f"-- {tier}: {name}")
        r = call_openai(name, SYSTEM, USER_BLIND, img, schema=BLIND_SCHEMA, phase="phase0", note=f"check-{tier}")
        if r.error:
            print(f"   ERROR: {r.error}")
        else:
            print(f"   parsed={r.parsed}  tokens(in/out)={r.in_tok}/{r.out_tok}  cost=${r.cost_usd:.4f}  {r.latency_s:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
