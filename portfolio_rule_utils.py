from __future__ import annotations

import json
from typing import Any

from config import LIVE_PORTFOLIO_RULES_PATH


DEFAULT_PORTFOLIO_RULES = {
    "portfolio_weight_multiplier": 1.0,
    "total_position_multiplier": 1.0,
    "stop_loss_adjustment": 0.0,
    "reason": "default rules",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_live_portfolio_rules() -> dict[str, Any]:
    if not LIVE_PORTFOLIO_RULES_PATH.exists():
        return DEFAULT_PORTFOLIO_RULES.copy()
    try:
        payload = json.loads(LIVE_PORTFOLIO_RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_PORTFOLIO_RULES.copy()

    rules = DEFAULT_PORTFOLIO_RULES.copy()
    rules.update(payload or {})
    rules["portfolio_weight_multiplier"] = _clamp(
        float(rules.get("portfolio_weight_multiplier") or 1.0),
        0.5,
        1.2,
    )
    rules["total_position_multiplier"] = _clamp(
        float(rules.get("total_position_multiplier") or 1.0),
        0.5,
        1.2,
    )
    rules["stop_loss_adjustment"] = _clamp(
        float(rules.get("stop_loss_adjustment") or 0.0),
        -3.0,
        3.0,
    )
    return rules
