from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    DAILY_SIGNAL_SCORES_PATH,
    GM_ATTRIBUTION_DIR,
    LIVE_ATTRIBUTION_ENABLED,
    LIVE_FACTOR_FEEDBACK_PATH,
    LIVE_PORTFOLIO_RULES_PATH,
    SELECTED_CANDIDATES_PATH,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _latest_attribution_path() -> Path | None:
    if not GM_ATTRIBUTION_DIR.exists():
        return None
    candidates = sorted(GM_ATTRIBUTION_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_signal_snapshot() -> pd.DataFrame:
    if not DAILY_SIGNAL_SCORES_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(DAILY_SIGNAL_SCORES_PATH)
    except Exception:
        return pd.DataFrame()


def _load_selected_candidates() -> pd.DataFrame:
    if not SELECTED_CANDIDATES_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(SELECTED_CANDIDATES_PATH)
    except Exception:
        return pd.DataFrame()


def _symbol_returns(attribution: dict[str, Any]) -> dict[str, float]:
    returns = {}
    for key in ["symbol_returns", "holding_pnl"]:
        payload = attribution.get(key) or {}
        if isinstance(payload, dict):
            for symbol, value in payload.items():
                if isinstance(value, dict):
                    returns[str(symbol).lower()] = _safe_float(
                        value.get("return_percent", value.get("pnl_percent", value.get("return"))),
                        0.0,
                    )
                else:
                    returns[str(symbol).lower()] = _safe_float(value, 0.0)
    return returns


def _update_factor_feedback(attribution: dict[str, Any]) -> dict[str, Any]:
    selected = _load_selected_candidates()
    returns = _symbol_returns(attribution)
    existing = _read_json(LIVE_FACTOR_FEEDBACK_PATH)
    factors = existing.get("factors") or {}

    if selected.empty or not returns:
        feedback = {
            **existing,
            "status": "skipped",
            "reason": "missing selected candidates or live symbol returns",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(LIVE_FACTOR_FEEDBACK_PATH, feedback)
        return feedback

    factor_columns = [
        col
        for col in selected.columns
        if col
        not in {
            "symbol",
            "name",
            "signal",
            "cross_section_score",
            "alpha_cross_section_score",
            "transformer_cross_section_score",
            "risk_liquidity_cross_section_score",
            "base_signal_score",
            "alpha_score",
            "transformer_score",
            "risk_liquidity_score",
            "target_weight",
            "signal_reason",
            "risk_flag",
            "industry",
            "fundamental_available",
        }
        and pd.api.types.is_numeric_dtype(selected[col])
    ]

    updates = {}
    for factor in factor_columns:
        contributions = []
        for _, row in selected.iterrows():
            symbol = str(row.get("symbol", "")).lower()
            live_return = returns.get(symbol)
            if live_return is None:
                continue
            exposure = _safe_float(row.get(factor), 0.0)
            if exposure == 0:
                continue
            contributions.append(1.0 if exposure * live_return > 0 else -1.0)
        if not contributions:
            continue
        avg = sum(contributions) / len(contributions)
        prior = factors.get(factor) or {}
        multiplier = _safe_float(prior.get("feedback_multiplier"), 1.0)
        multiplier = max(0.5, min(1.5, multiplier + max(-0.10, min(0.10, avg * 0.05))))
        updates[factor] = {
            "feedback_multiplier": round(multiplier, 6),
            "last_feedback_score": round(avg, 6),
            "sample_count": len(contributions),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    factors.update(updates)
    feedback = {
        "status": "ok" if updates else "skipped",
        "reason": "live factor feedback updated" if updates else "no factor feedback samples",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "source_generated_at": attribution.get("generated_at"),
        "factors": factors,
    }
    _write_json(LIVE_FACTOR_FEEDBACK_PATH, feedback)
    return feedback


def _update_portfolio_rules(attribution: dict[str, Any]) -> dict[str, Any]:
    existing = _read_json(LIVE_PORTFOLIO_RULES_PATH)
    live_return = _safe_float(attribution.get("portfolio_return_percent"), 0.0)
    benchmark_return = _safe_float(attribution.get("benchmark_return_percent"), 0.0)
    excess = live_return - benchmark_return

    total_multiplier = _safe_float(existing.get("total_position_multiplier"), 1.0)
    weight_multiplier = _safe_float(existing.get("portfolio_weight_multiplier"), 1.0)
    if excess < -2.0:
        total_multiplier *= 0.90
        weight_multiplier *= 0.95
        reason = "live portfolio underperformed benchmark by more than 2pct"
    elif excess > 1.0:
        total_multiplier *= 1.05
        weight_multiplier *= 1.03
        reason = "live portfolio outperformed benchmark"
    else:
        reason = "live portfolio performance within neutral band"

    payload = {
        "status": "ok",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "source_generated_at": attribution.get("generated_at"),
        "portfolio_return_percent": live_return,
        "benchmark_return_percent": benchmark_return,
        "excess_return_percent": round(excess, 6),
        "portfolio_weight_multiplier": round(max(0.5, min(1.2, weight_multiplier)), 6),
        "total_position_multiplier": round(max(0.5, min(1.2, total_multiplier)), 6),
        "stop_loss_adjustment": _safe_float(existing.get("stop_loss_adjustment"), 0.0),
        "reason": reason,
    }
    _write_json(LIVE_PORTFOLIO_RULES_PATH, payload)
    return payload


def run_live_attribution_update() -> dict[str, Any]:
    if not LIVE_ATTRIBUTION_ENABLED:
        return {"status": "skipped", "reason": "LIVE_ATTRIBUTION_ENABLED is False"}

    path = _latest_attribution_path()
    if path is None:
        return {"status": "skipped", "reason": "no GM attribution file found"}

    attribution = _read_json(path)
    if not attribution:
        return {"status": "skipped", "reason": f"failed to read {path}"}

    factor_feedback = _update_factor_feedback(attribution)
    portfolio_rules = _update_portfolio_rules(attribution)
    return {
        "status": "ok",
        "attribution_path": str(path),
        "source_generated_at": attribution.get("generated_at"),
        "factor_feedback": factor_feedback,
        "portfolio_rules": portfolio_rules,
    }
