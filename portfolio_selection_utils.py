import pandas as pd

from config import MAX_POSITION_PER_STOCK, MAX_TOTAL_POSITION, TRANSFORMER_PREDICTION_PATH


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
                "base_signal_score": _latest_value(latest, "base_signal_score"),
                "alpha_adjustment": _latest_value(latest, "alpha_adjustment"),
                "market_regime_adjustment": _latest_value(latest, "market_regime_adjustment"),
                "risk_adjustment": _latest_value(latest, "risk_adjustment"),
                "final_signal_score": _latest_value(latest, "final_signal_score"),
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
            -(item.get("final_signal_score") or item.get("signal_score") or 0),
            item["volatility_20d"] if item["volatility_20d"] is not None else float("inf"),
        )
    )
    return candidates[:top_n]


def _latest_transformer_predictions() -> dict[str, dict]:
    if not TRANSFORMER_PREDICTION_PATH.exists():
        return {}
    try:
        df = pd.read_csv(TRANSFORMER_PREDICTION_PATH)
    except Exception:
        return {}
    if df.empty or "prediction" not in df.columns:
        return {}
    if "symbol" not in df.columns and "instrument" in df.columns:
        df["symbol"] = df["instrument"].astype(str).str.upper().str.replace(
            r"^(SH|SZ)(\d{6})$",
            lambda match: match.group(1).lower() + match.group(2),
            regex=True,
        )
    if "prediction_zscore" not in df.columns:
        std = pd.to_numeric(df["prediction"], errors="coerce").std()
        mean = pd.to_numeric(df["prediction"], errors="coerce").mean()
        df["prediction_zscore"] = 0.0 if not std or pd.isna(std) else (df["prediction"] - mean) / std
    if "prediction_rank" not in df.columns:
        df["prediction_rank"] = pd.to_numeric(df["prediction"], errors="coerce").rank(
            ascending=False,
            method="dense",
        )
    if "symbol" not in df.columns:
        return {}
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime")
    latest = df.groupby("symbol", as_index=False).tail(1)
    return latest.where(pd.notna(latest), None).set_index("symbol").to_dict("index")


def allocate_positions(
    candidates: list[dict],
    max_position_per_stock: float,
    max_total_position: float,
) -> list[dict]:
    if not candidates:
        return []

    predictions = _latest_transformer_predictions()
    equal_weight = max_total_position / len(candidates)
    weight = min(equal_weight, max_position_per_stock)

    allocated = []
    for item in candidates:
        candidate = item.copy()
        symbol = str(candidate.get("symbol"))
        pred = predictions.get(symbol, {})
        prediction_zscore = pred.get("prediction_zscore")
        adjustment = 0.0
        try:
            zscore = float(prediction_zscore)
            if zscore >= 1.0:
                adjustment = min(0.02, max_position_per_stock * 0.20)
            elif zscore <= -1.0:
                adjustment = -min(0.02, weight * 0.50)
            candidate["transformer_prediction"] = pred.get("prediction")
            candidate["transformer_prediction_rank"] = pred.get("prediction_rank")
            candidate["transformer_prediction_zscore"] = zscore
        except (TypeError, ValueError):
            candidate["transformer_prediction"] = pred.get("prediction")
            candidate["transformer_prediction_rank"] = pred.get("prediction_rank")
            candidate["transformer_prediction_zscore"] = prediction_zscore

        adjusted_weight = min(max(weight + adjustment, 0.0), max_position_per_stock)
        candidate["base_target_weight"] = round(weight, 4)
        candidate["transformer_weight_adjustment"] = round(adjustment, 4)
        candidate["target_weight"] = round(adjusted_weight, 4)
        allocated.append(candidate)
    total_weight = sum(item["target_weight"] for item in allocated)
    if total_weight > max_total_position and total_weight > 0:
        scale = max_total_position / total_weight
        for item in allocated:
            item["target_weight"] = round(item["target_weight"] * scale, 4)
    return allocated
