import pandas as pd

from benchmark_weight_utils import benchmark_weight_map
from config import (
    BENCHMARK_ACTIVE_TILT,
    BENCHMARK_SYMBOL,
    INDUSTRY_WEIGHT_CAP,
    LONG_QUANTILE,
    MAX_POSITION_PER_STOCK,
    MAX_TOTAL_POSITION,
)
from portfolio_rule_utils import load_live_portfolio_rules


def _latest_value(row, column):
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _candidate_from_latest(symbol: str, item: dict, latest) -> dict:
    return {
        "symbol": symbol,
        "name": item["name"],
        "signal": latest.get("signal"),
        "cross_section_score": _latest_value(latest, "cross_section_score"),
        "alpha_cross_section_score": _latest_value(latest, "alpha_cross_section_score"),
        "transformer_cross_section_score": _latest_value(latest, "transformer_cross_section_score"),
        "risk_liquidity_cross_section_score": _latest_value(latest, "risk_liquidity_cross_section_score"),
        "base_signal_score": _latest_value(latest, "base_signal_score"),
        "alpha_score": _latest_value(latest, "alpha_score"),
        "alpha_adjustment": _latest_value(latest, "alpha_adjustment"),
        "transformer_score": _latest_value(latest, "transformer_score"),
        "risk_liquidity_score": _latest_value(latest, "risk_liquidity_score"),
        "cross_section_rank_pct": _latest_value(latest, "cross_section_rank_pct"),
        "long_bucket": bool(latest.get("long_bucket", False)),
        "short_bucket": bool(latest.get("short_bucket", False)),
        "market_regime_adjustment": _latest_value(latest, "market_regime_adjustment"),
        "risk_adjustment": _latest_value(latest, "risk_adjustment"),
        "transformer_prediction": _latest_value(latest, "transformer_prediction"),
        "transformer_prediction_rank": _latest_value(latest, "transformer_prediction_rank"),
        "transformer_prediction_zscore": _latest_value(latest, "transformer_prediction_zscore"),
        "return_20d": _latest_value(latest, "return_20d"),
        "volatility_20d": _latest_value(latest, "volatility_20d"),
        "max_drawdown_20d": _latest_value(latest, "max_drawdown_20d"),
        "trend": latest.get("trend"),
        "rsi_14": _latest_value(latest, "rsi_14"),
        "macd_diff": _latest_value(latest, "macd_diff"),
        "adx_14": _latest_value(latest, "adx_14"),
        "bollinger_width": _latest_value(latest, "bollinger_width"),
        "industry": latest.get("industry"),
        "fundamental_available": bool(latest.get("fundamental_available", False)),
        "pe": _latest_value(latest, "pe"),
        "pb": _latest_value(latest, "pb"),
        "roe": _latest_value(latest, "roe"),
        "total_market_cap": _latest_value(latest, "total_market_cap"),
        "float_market_cap": _latest_value(latest, "float_market_cap"),
        "revenue_yoy": _latest_value(latest, "revenue_yoy"),
        "net_profit_yoy": _latest_value(latest, "net_profit_yoy"),
        "debt_to_asset": _latest_value(latest, "debt_to_asset"),
        "signal_reason": latest.get("signal_reason"),
        "risk_flag": latest.get("risk_flag"),
    }


def select_ranked_portfolio_candidates(all_factor_data: dict, long_quantile: float = LONG_QUANTILE) -> list[dict]:
    candidates = []
    for symbol, item in all_factor_data.items():
        df = item["data"]
        if df.empty:
            continue
        latest = df.iloc[-1]
        rank_pct = _latest_value(latest, "cross_section_rank_pct")
        if rank_pct is None:
            if latest.get("signal") != "BUY":
                continue
        elif rank_pct <= 1.0 - float(long_quantile):
            continue
        candidates.append(_candidate_from_latest(symbol, item, latest))

    candidates.sort(
        key=lambda item: (
            -(
                item.get("cross_section_rank_pct")
                if item.get("cross_section_rank_pct") is not None
                else item.get("cross_section_score")
                or 0
            ),
            -(item.get("cross_section_score") or 0),
            item["volatility_20d"] if item["volatility_20d"] is not None else float("inf"),
        )
    )
    return candidates


def _apply_industry_cap(items: list[dict], cap: float) -> None:
    if not items:
        return
    industry_totals = {}
    for item in items:
        industry = item.get("industry") or "unknown"
        industry_totals[industry] = industry_totals.get(industry, 0.0) + float(item.get("target_weight") or 0.0)
    for industry, total in industry_totals.items():
        if total <= cap or total <= 0:
            continue
        scale = cap / total
        for item in items:
            if (item.get("industry") or "unknown") == industry:
                item["target_weight"] = float(item.get("target_weight") or 0.0) * scale


def allocate_positions(
    candidates: list[dict],
    max_position_per_stock: float,
    max_total_position: float,
) -> list[dict]:
    if not candidates:
        return []

    live_rules = load_live_portfolio_rules()
    portfolio_multiplier = float(live_rules.get("portfolio_weight_multiplier") or 1.0)
    adjusted_max_total = max_total_position * float(live_rules.get("total_position_multiplier") or 1.0)
    benchmark_weights, weight_source = benchmark_weight_map(
        [item["symbol"] for item in candidates],
        benchmark_symbol=BENCHMARK_SYMBOL,
    )
    vols = [
        float(item.get("volatility_20d") or 0.0)
        for item in candidates
        if float(item.get("volatility_20d") or 0.0) > 0
    ]
    median_vol = sorted(vols)[len(vols) // 2] if vols else 1.0

    allocated = []
    for item in candidates:
        candidate = item.copy()
        rank_pct = float(candidate.get("cross_section_rank_pct") or 0.5)
        benchmark_weight = float(benchmark_weights.get(candidate["symbol"], 0.0))
        if benchmark_weight <= 0:
            benchmark_weight = 1.0 / len(candidates)
        active_tilt = max(rank_pct - (1.0 - float(LONG_QUANTILE)), 0.0) / max(float(LONG_QUANTILE), 1e-9)
        volatility = float(candidate.get("volatility_20d") or median_vol)
        volatility_adjustment = min(1.5, max(0.5, median_vol / volatility)) if volatility > 0 else 1.0
        target = benchmark_weight * (1.0 + float(BENCHMARK_ACTIVE_TILT) * active_tilt)
        target *= volatility_adjustment * portfolio_multiplier
        adjusted_weight = min(max(target, 0.0), max_position_per_stock)
        candidate["benchmark_weight"] = round(benchmark_weight, 6)
        candidate["benchmark_weight_source"] = weight_source
        candidate["active_tilt"] = round(active_tilt, 6)
        candidate["volatility_weight_adjustment"] = round(volatility_adjustment, 6)
        candidate["base_target_weight"] = round(benchmark_weight, 6)
        candidate["transformer_weight_adjustment"] = 0.0
        candidate["portfolio_weight_multiplier"] = round(portfolio_multiplier, 6)
        candidate["target_weight"] = adjusted_weight
        allocated.append(candidate)
    _apply_industry_cap(allocated, float(INDUSTRY_WEIGHT_CAP))
    total_weight = sum(item["target_weight"] for item in allocated)
    if total_weight > adjusted_max_total and total_weight > 0:
        scale = adjusted_max_total / total_weight
        for item in allocated:
            item["target_weight"] = item["target_weight"] * scale
    for item in allocated:
        item["target_weight"] = round(float(item["target_weight"]), 4)
    return allocated
