import json
import logging

import pandas as pd

from config import ALPHA_SIGNAL_WEIGHTING_ENABLED, ALPHA_FACTOR_SELECTION_PATH


logger = logging.getLogger(__name__)


HIGH_VOLATILITY_THRESHOLD = 40.0
DEEP_DRAWDOWN_THRESHOLD = -10.0
HIGH_BOLLINGER_WIDTH_THRESHOLD = 25.0
STRONG_ADX_THRESHOLD = 25.0
OVERHEATED_MFI_THRESHOLD = 80.0

FACTOR_SELECTION_CACHE = None


def _is_missing(value):
    return value is None or pd.isna(value)


def _as_float(row, column):
    value = row.get(column)
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_factor_selection():
    global FACTOR_SELECTION_CACHE
    if FACTOR_SELECTION_CACHE is not None:
        return FACTOR_SELECTION_CACHE
    if not ALPHA_SIGNAL_WEIGHTING_ENABLED or not ALPHA_FACTOR_SELECTION_PATH.exists():
        FACTOR_SELECTION_CACHE = {}
        return FACTOR_SELECTION_CACHE
    try:
        payload = json.loads(ALPHA_FACTOR_SELECTION_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load alpha factor selection: %s", e)
        FACTOR_SELECTION_CACHE = {}
        return FACTOR_SELECTION_CACHE

    FACTOR_SELECTION_CACHE = {
        item.get("factor"): item
        for item in payload.get("factor_selection", [])
        if item.get("factor")
    }
    return FACTOR_SELECTION_CACHE


def _factor_exposure(row, factor):
    value = _as_float(row, factor)
    if value is None:
        return None

    if factor == "volume_ratio_20d":
        return value - 1.0
    if factor == "rsi_14":
        if 40 <= value <= 70:
            return 1.0
        if value < 30 or value > 80:
            return -1.0
        return 0.0
    if factor == "macd_diff":
        return value
    if factor == "adx_14":
        trend = row.get("trend")
        if trend == "uptrend":
            return value - STRONG_ADX_THRESHOLD
        if trend == "downtrend":
            return STRONG_ADX_THRESHOLD - value
        return 0.0
    if factor == "bollinger_width":
        return value - HIGH_BOLLINGER_WIDTH_THRESHOLD
    if factor == "volatility_20d":
        return value - HIGH_VOLATILITY_THRESHOLD
    if factor == "max_drawdown_20d":
        return value - DEEP_DRAWDOWN_THRESHOLD
    if factor == "debt_to_asset":
        return value - 70.0
    return value


def _apply_alpha_factor_weighting(row, score, reasons):
    selections = _load_factor_selection()
    if not selections:
        return score

    adjustment = 0.0
    applied = []
    for factor, meta in selections.items():
        if meta.get("validity") == "剔除":
            continue
        direction = meta.get("direction")
        if direction not in {"正向", "反向"}:
            continue
        weight = float(meta.get("signal_weight") or 0.0)
        if weight <= 0:
            continue

        exposure = _factor_exposure(row, factor)
        if exposure is None or exposure == 0:
            continue

        aligned = exposure > 0 if direction == "正向" else exposure < 0
        delta = weight if aligned else -weight
        adjustment += delta
        action = "+" if delta > 0 else "-"
        applied.append(f"alpha {factor} {action}{abs(delta):.2f}")

    if applied:
        score += adjustment
        reasons.append("alpha factor weighting: " + ", ".join(applied))
    return score


def _score_row(row):
    required = [
        "trend",
        "return_20d",
        "price_vs_ma20",
        "volume_ratio_20d",
        "volatility_20d",
        "max_drawdown_20d",
    ]
    if any(_is_missing(row.get(col)) for col in required):
        return "HOLD", 0, "insufficient core factors", "insufficient_data"

    score = 0
    reasons = []
    risk_flag = "normal"

    trend = row["trend"]
    return_20d = float(row["return_20d"])
    price_vs_ma20 = float(row["price_vs_ma20"])
    volume_ratio_20d = float(row["volume_ratio_20d"])
    volatility_20d = float(row["volatility_20d"])
    max_drawdown_20d = float(row["max_drawdown_20d"])

    if trend == "uptrend":
        score += 2
        reasons.append("trend up")
    if return_20d > 0:
        score += 1
        reasons.append("positive 20d return")
    if price_vs_ma20 > 0:
        score += 1
        reasons.append("price above MA20")
    if volume_ratio_20d > 1:
        score += 1
        reasons.append("volume above 20d average")

    if trend == "downtrend":
        score -= 2
        risk_flag = "weak_trend"
        reasons.append("trend down")
    if return_20d < 0:
        score -= 1
        reasons.append("negative 20d return")
    if max_drawdown_20d < DEEP_DRAWDOWN_THRESHOLD:
        score -= 2
        risk_flag = "deep_drawdown"
        reasons.append("deep 20d drawdown")
    if volatility_20d > HIGH_VOLATILITY_THRESHOLD:
        score -= 1
        if risk_flag == "normal":
            risk_flag = "high_volatility"
        reasons.append("high 20d volatility")

    pe = _as_float(row, "pe")
    pb = _as_float(row, "pb")
    roe = _as_float(row, "roe")
    debt_to_asset = _as_float(row, "debt_to_asset")

    if pe is not None and 0 < pe <= 30:
        score += 1
        reasons.append("reasonable PE")
    if pb is not None and 0 < pb <= 5:
        score += 1
        reasons.append("reasonable PB")
    if roe is not None and roe >= 8:
        score += 1
        reasons.append("high ROE")
    if debt_to_asset is not None and debt_to_asset > 70:
        score -= 1
        reasons.append("high debt ratio")

    rsi_14 = _as_float(row, "rsi_14")
    macd = _as_float(row, "macd")
    macd_signal = _as_float(row, "macd_signal")
    bollinger_width = _as_float(row, "bollinger_width")
    adx_14 = _as_float(row, "adx_14")
    mfi_14 = _as_float(row, "mfi_14")

    if rsi_14 is not None and 40 <= rsi_14 <= 70:
        score += 1
        reasons.append("RSI in healthy range")
    if macd is not None and macd_signal is not None and macd > macd_signal:
        score += 1
        reasons.append("MACD above signal")
    if bollinger_width is not None and bollinger_width > HIGH_BOLLINGER_WIDTH_THRESHOLD:
        score -= 1
        reasons.append("wide Bollinger band")
    if adx_14 is not None and adx_14 >= STRONG_ADX_THRESHOLD and trend == "uptrend":
        score += 1
        reasons.append("strong uptrend ADX")
    if mfi_14 is not None and mfi_14 >= OVERHEATED_MFI_THRESHOLD:
        score -= 1
        reasons.append("MFI overheated")

    score = _apply_alpha_factor_weighting(row, score, reasons)

    if score >= 3 and risk_flag == "normal":
        signal = "BUY"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    if not reasons:
        reasons.append("no clear directional signal")
    return signal, score, "; ".join(reasons), risk_flag


def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Generate deterministic trading signals from factor columns without calling LLMs."""
    df = df.copy()
    signals = []
    scores = []
    reasons = []
    risk_flags = []

    for _, row in df.iterrows():
        signal, score, reason, risk_flag = _score_row(row)
        signals.append(signal)
        scores.append(score)
        reasons.append(reason)
        risk_flags.append(risk_flag)

    df["signal"] = signals
    df["signal_score"] = scores
    df["signal_reason"] = reasons
    df["risk_flag"] = risk_flags
    return df
