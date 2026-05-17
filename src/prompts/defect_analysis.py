"""Structured chain-of-thought (CoT) prompt for VLM defect analysis."""
from __future__ import annotations

SYSTEM = """You are an automated quality-assurance agent for a rolled-metal production line.
Your job is to inspect a single static image of a metal surface and decide whether it shows a
manufacturing defect. You must reason step-by-step and return your verdict as strict JSON
matching the schema given by the user."""

USER_T3_COT = """Analyse the image of a rolled-metal surface using the following procedure.

1. UNDERSTAND
   State, in one sentence, what kind of metal surface is shown (e.g. hot-rolled steel strip,
   flat sheet steel) and the apparent lighting/imaging condition.

2. ANALYZE — extract a structured T3 attribute matrix for any anomaly you observe:
   - shape:        {linear | curved | round | irregular | patch | none}
   - direction:    {horizontal | vertical | diagonal | none}
   - distribution: {isolated | scattered | clustered | continuous | none}
   - count:        integer (0 if normal)
   - position:     {top-left | top | top-right | left | center | right | bottom-left | bottom | bottom-right | edge | full | none}
   - scale:        approximate fraction of image covered (0.0–1.0)
   - polarity:     {darker_than_background | lighter_than_background | none}
   - saliency:     {low | medium | high | none}

3. REASON
   Briefly explain (≤2 sentences) what physical process most plausibly caused the anomaly
   (e.g. roller misalignment, thermal scale, mechanical scratch, surface oxidation, water
   spot, weld seam), or state that the surface is consistent with healthy metal.

4. SYNTHESIZE & CONCLUDE
   Decide: defect present (true) or not (false). Estimate a normalised bounding box around the
   defect IF and ONLY IF defect=true. Coordinates are in [0,1] of the padded square image,
   format [x_center, y_center, width, height]. Give a confidence in [0,1].

Return ONLY a single JSON object matching the schema. Do not include markdown fences."""


JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["has_defect", "defect_type", "attributes", "reasoning", "bbox", "confidence"],
    "properties": {
        "has_defect": {"type": "boolean"},
        "defect_type": {
            "type": "string",
            "description": "Short label such as 'scratch', 'crack', 'pitting', 'inclusion', 'scale', 'weld_seam', 'oil_spot', 'water_spot', 'crease', 'hole', 'none'.",
        },
        "attributes": {
            "type": "object",
            "additionalProperties": False,
            "required": ["shape", "direction", "distribution", "count", "position", "scale", "polarity", "saliency"],
            "properties": {
                "shape":        {"type": "string"},
                "direction":    {"type": "string"},
                "distribution": {"type": "string"},
                "count":        {"type": "integer", "minimum": 0},
                "position":     {"type": "string"},
                "scale":        {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "polarity":     {"type": "string"},
                "saliency":     {"type": "string"},
            },
        },
        "reasoning": {"type": "string"},
        "bbox": {
            "type": "array",
            "description": "[x_center, y_center, width, height] in [0,1]; [0,0,0,0] if has_defect=false.",
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "minItems": 4,
            "maxItems": 4,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


# Blind-prompt baseline used for the ablation slide
USER_BLIND = "Is this image of a metal surface defective? Reply with strict JSON: {\"has_defect\": bool, \"confidence\": number in [0,1]}."

BLIND_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["has_defect", "confidence"],
    "properties": {
        "has_defect": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}
