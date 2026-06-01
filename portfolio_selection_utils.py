import pandas as pd


def _latest_value(row, column):
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return value


def select_top_candidates(all_factor_data: dict, top_n: int) -> list[dict]:
    candidates = []
    for symbol, item in all_factor_data.items():
        df = item["data"]
        if df.empty:
            continue
        latest = df.iloc[-1]
        if latest.get("signal") != "BUY":
            continue
        candidates.append(
            {
                "symbol": symbol,
                "name": item["name"],
                "signal": latest.get("signal"),
                "signal_score": _latest_value(latest, "signal_score"),
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
        )

    candidates.sort(
        key=lambda item: (
            -(item["signal_score"] or 0),
            item["volatility_20d"] if item["volatility_20d"] is not None else float("inf"),
        )
    )
    return candidates[:top_n]


def allocate_positions(
    candidates: list[dict],
    max_position_per_stock: float,
    max_total_position: float,
) -> list[dict]:
    if not candidates:
        return []

    equal_weight = max_total_position / len(candidates)
    weight = min(equal_weight, max_position_per_stock)

    allocated = []
    for item in candidates:
        candidate = item.copy()
        candidate["target_weight"] = round(weight, 4)
        allocated.append(candidate)
    return allocated
