"""Unified VLM client wrapper across OpenAI native, Azure OpenAI, and Google Vertex Gemini.

Every provider returns the same shape::

    {
        "parsed": dict,       # JSON object matching the requested schema (best-effort)
        "raw": str,           # raw text the model produced
        "in_tok": int,
        "out_tok": int,
        "cost_usd": float,
        "model": str,
        "provider": str,
        "latency_s": float,
        "error": Optional[str],
    }

Each call is logged to the shared cost ledger and will raise ``BudgetExceeded``
once the global $50 cap is hit.
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

from PIL import Image

from .config import AOAI_API_KEY, AOAI_API_VERSION, AOAI_DEPLOYMENT, AOAI_ENDPOINT, AOAI_MODEL, GCP_PROJECT_ID, OPENAI_API_KEY
from .cost import log_call
from .data.transforms import png_bytes, vlm_transform


def _b64(img: Image.Image) -> str:
    return base64.b64encode(png_bytes(vlm_transform(img))).decode()


@dataclass
class VLMResponse:
    parsed: Optional[dict]
    raw: str
    in_tok: int
    out_tok: int
    cost_usd: float
    model: str
    provider: str
    latency_s: float
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed": self.parsed,
            "raw": self.raw,
            "in_tok": self.in_tok,
            "out_tok": self.out_tok,
            "cost_usd": self.cost_usd,
            "model": self.model,
            "provider": self.provider,
            "latency_s": self.latency_s,
            "error": self.error,
        }


# ── OpenAI native ────────────────────────────────────────────────────────────
def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def call_openai(
    model: str,
    system: str,
    user: str,
    image: Image.Image,
    schema: dict | None = None,
    phase: str = "phase3",
    note: str = "",
) -> VLMResponse:
    client = _openai_client()
    b64 = _b64(image)
    content = [
        {"type": "text", "text": user},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    }
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "defect_analysis", "strict": True, "schema": schema},
        }
    t0 = time.perf_counter()
    err: Optional[str] = None
    raw = ""
    parsed: Optional[dict] = None
    in_tok = out_tok = 0
    try:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    except Exception as e:  # noqa: BLE001 — surface every provider error to caller
        err = f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - t0
    cost = log_call("openai", model, in_tok, out_tok, phase=phase, note=note)
    return VLMResponse(parsed, raw, in_tok, out_tok, cost, model, "openai", latency, err)


# ── Azure OpenAI ─────────────────────────────────────────────────────────────
def _azure_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        api_key=AOAI_API_KEY,
        api_version=AOAI_API_VERSION,
        azure_endpoint=AOAI_ENDPOINT,
    )


def call_azure(
    system: str,
    user: str,
    image: Image.Image,
    schema: dict | None = None,
    deployment: str = "",
    phase: str = "phase3",
    note: str = "",
) -> VLMResponse:
    client = _azure_client()
    deployment = deployment or AOAI_DEPLOYMENT
    model_for_pricing = AOAI_MODEL or "gpt-4.1"
    b64 = _b64(image)
    content = [
        {"type": "text", "text": user},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    kwargs: dict[str, Any] = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    }
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "defect_analysis", "strict": True, "schema": schema},
        }
    t0 = time.perf_counter()
    err: Optional[str] = None
    raw = ""
    parsed: Optional[dict] = None
    in_tok = out_tok = 0
    try:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - t0
    cost = log_call("azure-openai", model_for_pricing, in_tok, out_tok, phase=phase, note=note)
    return VLMResponse(parsed, raw, in_tok, out_tok, cost, model_for_pricing, "azure-openai", latency, err)


# ── Gemini (Vertex via google-genai) ─────────────────────────────────────────
def call_gemini(
    model: str,
    system: str,
    user: str,
    image: Image.Image,
    schema: dict | None = None,
    phase: str = "phase3",
    note: str = "",
) -> VLMResponse:
    from google import genai
    from google.genai import types

    location = os.environ.get("GCP_LOCATION", "us-central1")
    client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=location)
    img_bytes = png_bytes(vlm_transform(image))
    parts = [
        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        types.Part.from_text(text=user),
    ]
    config: dict[str, Any] = {"system_instruction": system}
    if schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = schema
    t0 = time.perf_counter()
    err: Optional[str] = None
    raw = ""
    parsed: Optional[dict] = None
    in_tok = out_tok = 0
    try:
        resp = client.models.generate_content(model=model, contents=parts, config=config)
        raw = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            in_tok = getattr(usage, "prompt_token_count", 0) or 0
            out_tok = getattr(usage, "candidates_token_count", 0) or 0
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    latency = time.perf_counter() - t0
    cost = log_call("vertex-gemini", model, in_tok, out_tok, phase=phase, note=note)
    return VLMResponse(parsed, raw, in_tok, out_tok, cost, model, "vertex-gemini", latency, err)
