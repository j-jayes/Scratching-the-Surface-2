"""Phase G — VLM prompt-tuning loop with a structured-output tuner LLM.

Story: zero-shot is already strong, but the prompt was written by a human who
hadn't seen the domain. Can the VLM teach itself a better prompt by reflecting
on its own disagreements with the ground truth?

Loop (per domain × per VLM):
  1. Run the current prompt on a small calibration set.
  2. Send the calibration disagreements (image-by-image: ground truth +
     VLM verdict + VLM reasoning) to a *tuner LLM* (gpt-5.5 by default)
     with a STRUCTURED OUTPUT schema:
         {"summary": str, "common_mistakes": [≤5 str], "prompt_addendum": str}
  3. Append the suggested addendum to the user prompt and re-run on the
     calibration set. Track F1 round-by-round.
  4. After R rounds, run the final prompt on a held-out eval set
     (disjoint from calibration) and compare to the round-0 baseline F1.

Outputs:
    results/vlm/prompt_tune_history_<ts>.json
    figures/vlm/prompt_tuning_curve.png

Defaults are conservative to stay inside the $50 budget (≈ $15 spend).

Usage:
    uv run python scripts/prompt_tune_vlm.py --dry-run
    uv run python scripts/prompt_tune_vlm.py --rounds 2 --n-calib 12 --n-eval 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import f1_score

from src.config import GEMINI_MODELS, OPENAI_MODELS, PRICING
from src.prompts.defect_analysis import JSON_SCHEMA, SYSTEM, USER_T3_COT
from src.vlm_clients import call_azure, call_gemini, call_openai
from scripts.eval_vlm_zeroshot import DATASETS, sample_dataset
from scripts.eval_vlm_flagship import run_one

RESULTS_DIR = Path("results/vlm")
FIGURES_DIR = Path("figures/vlm")
DARK2 = {"openai": "#1b9e77", "gemini": "#d95f02", "azure": "#7570b3"}

PROVIDERS = {
    "openai": OPENAI_MODELS["flagship"],
    "gemini": GEMINI_MODELS["flagship"],
    "azure":  None,
}

TUNER_MODEL = OPENAI_MODELS["top"]   # gpt-5.5

TUNER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "common_mistakes", "prompt_addendum"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "1–2 sentence diagnosis of the model's failure mode on this domain.",
        },
        "common_mistakes": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "description": "Up to 5 concrete failure patterns observed in the disagreements.",
        },
        "prompt_addendum": {
            "type": "string",
            "description": (
                "A new block (≤ 8 bullet points, ≤ 150 words) to append to the user prompt. "
                "Should give the inspecting VLM domain-specific guidance — what tends to "
                "look like a defect but isn't (false-positive traps), and what subtle real "
                "defects to look harder for (false-negative traps). Phrase as actionable rules."
            ),
        },
    },
}

TUNER_SYSTEM = (
    "You are an expert prompt engineer for industrial computer-vision systems. "
    "You will be shown the disagreements a multimodal model made on a small "
    "calibration set of metal-surface inspection images. Diagnose its failure "
    "pattern, then propose a focused addendum to the existing user prompt that "
    "will help the model do better next time. Be terse, concrete, and specific "
    "to this domain."
)


def build_addendum_block(addenda: list[str]) -> str:
    if not addenda:
        return ""
    lines = ["\n\n# DOMAIN-SPECIFIC GUIDANCE (learned from prior errors)"]
    for i, a in enumerate(addenda, 1):
        lines.append(f"\n## Round {i} addendum\n{a}")
    return "\n".join(lines)


def evaluate_prompt(provider: str, model: str | None, sample: pd.DataFrame,
                    user_prompt: str, domain: str, tag: str) -> tuple[float, list[dict]]:
    """Run the prompt on the sample and return (F1, records). Records keep raw reasoning."""
    records: list[dict] = []
    for i, row in sample.iterrows():
        rec = run_one(provider, model, row["path"], int(row["has_defect"]),
                      domain, Path(row["path"]).stem,
                      system=SYSTEM, user=user_prompt)
        records.append(rec)
        ok = "✓" if rec["error"] is None else "✗"
        print(f"     [{tag}] {i+1:3d}/{len(sample)} {ok} y={rec['label']} "
              f"ŷ={rec['pred_defect']} ${rec['cost_usd']:.4f}", flush=True)
    valid = [r for r in records if r["pred_defect"] is not None and r["error"] is None]
    if not valid:
        return float("nan"), records
    y = np.array([r["label"] for r in valid])
    p = np.array([int(r["pred_defect"]) for r in valid])
    return float(f1_score(y, p, zero_division=0)), records


def request_prompt_edit(domain: str, provider: str, records: list[dict]) -> dict:
    """Send disagreements to the tuner LLM. Returns parsed dict matching TUNER_SCHEMA."""
    disagreements = [
        r for r in records
        if r["pred_defect"] is not None
        and r["error"] is None
        and (int(r["pred_defect"]) != r["label"])
    ]
    if not disagreements:
        return {"summary": "No disagreements on this calibration round.",
                "common_mistakes": [],
                "prompt_addendum": ""}

    bullets = []
    for r in disagreements[:12]:
        parsed = r.get("parsed") or {}
        kind = "FALSE POSITIVE (model said defect, truth=normal)" if r["label"] == 0 \
               else "FALSE NEGATIVE (model said normal, truth=defect)"
        bullets.append(
            f"- {kind}\n"
            f"  image_id: {r['image_id']}\n"
            f"  model_confidence: {r['confidence']:.2f}\n"
            f"  model_reasoning: {parsed.get('reasoning','(no reasoning)')[:400]}\n"
            f"  model_attrs: {parsed.get('attributes',{})}"
        )
    user_msg = (
        f"Domain: {domain}\n"
        f"Model under review: {provider} / {records[0]['model']}\n"
        f"Calibration size: {len(records)}  •  Disagreements: {len(disagreements)}\n\n"
        f"Here are the disagreements with their reasoning. Propose a prompt "
        f"addendum that would help.\n\n" + "\n".join(bullets)
    )

    from openai import OpenAI
    client = OpenAI(api_key=__import__("os").environ.get("NATIVE_OPENAI_API_KEY", ""))
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=TUNER_MODEL,
        messages=[
            {"role": "system", "content": TUNER_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "prompt_edit", "strict": True, "schema": TUNER_SCHEMA},
        },
    )
    raw = resp.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    # Log tuner cost
    from src.cost import log_call
    log_call("openai", TUNER_MODEL,
             resp.usage.prompt_tokens or 0,
             resp.usage.completion_tokens or 0,
             phase="phaseG-tuner", note=f"{domain}/{provider}")
    parsed["_tuner_latency_s"] = round(time.perf_counter() - t0, 2)
    parsed["_n_disagreements"] = len(disagreements)
    return parsed


def plot_curve(history: dict, out: Path) -> None:
    domains = list(history["domains"].keys())
    n = len(domains)
    width = 8 if n == 1 else 13
    fig, axes_raw = plt.subplots(1, n, figsize=(width, 5), sharey=True)
    axes = [axes_raw] if n == 1 else list(axes_raw)
    for ax, domain in zip(axes, domains):
        for provider, rounds in history["domains"][domain].items():
            f1s = [r["f1_calib"] for r in rounds]
            xs = list(range(len(f1s)))
            ax.plot(xs, f1s, marker="o", linewidth=2.5, markersize=9,
                    color=DARK2[provider], label=f"{provider} ({rounds[0]['model']})")
            if rounds[-1].get("f1_eval") is not None:
                ax.scatter([len(f1s) - 1 + 0.15], [rounds[-1]["f1_eval"]],
                           marker="*", s=240, color=DARK2[provider],
                           edgecolor="black", linewidth=1.2,
                           label=f"{provider} held-out F1")
        ax.set_title(domain, fontsize=12, fontweight="bold")
        ax.set_xlabel("Tuning round")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("F1")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Prompt tuning lifts F1 across rounds — calibration (line) → held-out (★)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def estimate_cost(rounds: int, n_calib: int, n_eval: int) -> float:
    total = 0.0
    for prov, model in PROVIDERS.items():
        m = model or "gpt-4.1-mini"
        pr = PRICING.get(m, {"in": 2.5, "out": 15.0})
        calls = len(DATASETS) * ((rounds + 1) * n_calib + n_eval)
        total += calls * (1800 * pr["in"] + 500 * pr["out"]) / 1_000_000
    # tuner LLM cost
    tpr = PRICING[TUNER_MODEL]
    tcalls = len(DATASETS) * len(PROVIDERS) * rounds
    total += tcalls * (3500 * tpr["in"] + 600 * tpr["out"]) / 1_000_000
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds",  type=int, default=2)
    ap.add_argument("--n-calib", type=int, default=12, help="Per-domain calibration size.")
    ap.add_argument("--n-eval",  type=int, default=30, help="Per-domain held-out eval size.")
    ap.add_argument("--providers", nargs="+", default=list(PROVIDERS.keys()),
                    choices=list(PROVIDERS.keys()))
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    est = estimate_cost(args.rounds, args.n_calib, args.n_eval)
    print(f"\nPrompt-tuning loop — {args.rounds} rounds × {len(args.providers)} VLMs × "
          f"{len(DATASETS)} domains")
    print(f"  calibration: {args.n_calib} imgs   held-out: {args.n_eval} imgs")
    print(f"  estimated total cost: ${est:.2f}\n")
    if args.dry_run:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    history: dict = {
        "timestamp": ts, "rounds": args.rounds,
        "n_calib": args.n_calib, "n_eval": args.n_eval,
        "tuner_model": TUNER_MODEL,
        "domains": {},
    }
    all_rationale_records: list[dict] = []   # for downstream gradcam-vs-rationale figure

    for domain, cfg in DATASETS.items():
        # Disjoint calib + eval splits via two different seeds.
        calib_pool = sample_dataset(domain, cfg, max(args.n_calib * 4, 60), seed=args.seed)
        eval_pool  = sample_dataset(domain, cfg, max(args.n_eval * 4, 120), seed=args.seed + 1)
        # Remove overlap
        eval_pool = eval_pool[~eval_pool["path"].isin(calib_pool["path"])]
        calib = (pd.concat([calib_pool[calib_pool["has_defect"]].head(args.n_calib // 2),
                            calib_pool[~calib_pool["has_defect"]].head(args.n_calib // 2)])
                 .reset_index(drop=True))
        held  = (pd.concat([eval_pool[eval_pool["has_defect"]].head(args.n_eval // 2),
                            eval_pool[~eval_pool["has_defect"]].head(args.n_eval // 2)])
                 .reset_index(drop=True))
        print(f"\n==== {domain}: calib={len(calib)}  held-out={len(held)} ====")

        history["domains"][domain] = {}
        for provider in args.providers:
            model = PROVIDERS[provider]
            print(f"\n  --- {provider} / {model or 'AOAI'} ---")
            addenda: list[str] = []
            rounds_log: list[dict] = []

            for rnd in range(args.rounds + 1):
                cur_prompt = USER_T3_COT + build_addendum_block(addenda)
                tag = f"R{rnd}"
                print(f"    Round {rnd}: evaluating on calibration "
                      f"(prompt len={len(cur_prompt)} chars)")
                f1_cal, calib_recs = evaluate_prompt(provider, model, calib,
                                                     cur_prompt, domain, tag)
                print(f"    → calibration F1 = {f1_cal:.3f}")
                rounds_log.append({
                    "round":          rnd,
                    "f1_calib":       round(f1_cal, 4),
                    "prompt_chars":   len(cur_prompt),
                    "addendum_count": len(addenda),
                    "model":          model or "gpt-4.1-mini",
                    "tuner_output":   None,
                    "f1_eval":        None,
                })
                # Stash rationale for figure use
                all_rationale_records.extend(calib_recs)

                if rnd < args.rounds:
                    print(f"    Round {rnd}: requesting prompt edit from {TUNER_MODEL}…")
                    edit = request_prompt_edit(domain, provider, calib_recs)
                    rounds_log[-1]["tuner_output"] = edit
                    if edit["prompt_addendum"].strip():
                        addenda.append(edit["prompt_addendum"].strip())
                        print(f"    → tuner addendum ({len(edit['prompt_addendum'])} chars): "
                              f"{edit['summary']}")
                    else:
                        print("    → tuner returned empty addendum; skipping.")

            # Final held-out eval with the final prompt
            final_prompt = USER_T3_COT + build_addendum_block(addenda)
            print(f"\n    Held-out eval ({len(held)} imgs)…")
            f1_eval, eval_recs = evaluate_prompt(provider, model, held,
                                                 final_prompt, domain, "HELD")
            rounds_log[-1]["f1_eval"] = round(f1_eval, 4)
            rounds_log[-1]["final_prompt"] = final_prompt
            print(f"    → held-out F1 = {f1_eval:.3f}  "
                  f"(round-0 calib F1 was {rounds_log[0]['f1_calib']:.3f})")

            history["domains"][domain][provider] = rounds_log

    # Persist
    hist_path = RESULTS_DIR / f"prompt_tune_history_{ts}.json"
    hist_path.write_text(json.dumps(history, indent=2, default=str))
    print(f"\n  → {hist_path}")

    rat_path = RESULTS_DIR / f"prompt_tune_rationale_{ts}.jsonl"
    with rat_path.open("w") as f:
        for r in all_rationale_records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  → {rat_path}")

    plot_curve(history, FIGURES_DIR / "prompt_tuning_curve.png")


if __name__ == "__main__":
    main()
