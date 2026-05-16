"""Cost tracking ledger with hard budget cap."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .config import COST_LEDGER_PATH, PRICING, VLM_BUDGET_USD


class BudgetExceeded(RuntimeError):
    pass


def _ensure_header(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "provider", "model", "in_tok", "out_tok", "cost_usd", "phase", "note"])


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    rate = PRICING.get(model)
    if rate is None:
        return 0.0
    return (in_tok / 1_000_000) * rate["in"] + (out_tok / 1_000_000) * rate["out"]


def running_total(path: Path = COST_LEDGER_PATH) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    with path.open() as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 6:
                try:
                    total += float(row[5])
                except ValueError:
                    pass
    return total


def log_call(provider: str, model: str, in_tok: int, out_tok: int, phase: str, note: str = "") -> float:
    """Append a call to the ledger. Raises BudgetExceeded if cap is hit."""
    _ensure_header(COST_LEDGER_PATH)
    cost = estimate_cost(model, in_tok, out_tok)
    with COST_LEDGER_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.utcnow().isoformat(), provider, model, in_tok, out_tok, f"{cost:.6f}", phase, note])
    total = running_total()
    if total > VLM_BUDGET_USD:
        raise BudgetExceeded(f"VLM budget ${VLM_BUDGET_USD} exceeded (current: ${total:.2f})")
    return cost


def assert_budget_available(headroom: float = 1.0) -> None:
    total = running_total()
    if total + headroom > VLM_BUDGET_USD:
        raise BudgetExceeded(
            f"Insufficient budget headroom: ${total:.2f} spent + ${headroom:.2f} requested > ${VLM_BUDGET_USD}"
        )
