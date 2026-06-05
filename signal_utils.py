import json
import logging

import pandas as pd

from config import ALPHA_FACTOR_SELECTION_PATH, ALPHA_SIGNAL_WEIGHTING_ENABLED


logger = logging.getLogger(__name__)

# Structural guard thresholds — Alpha layer handles all factor-level scoring.
# These only gate extreme safety conditions Alpha cannot express.

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
    """Convert raw factor value to directional exposure (deviation from neutral)."""
    value = _as_float(row, factor)
    if value is None:
        return None

    if factor == "volume_ratio_20d":
        return value - 1.0
    if factor == "adx_14":
        trend = row.get("trend")
        if trend == "uptrend":
            return value - 25.0
        if trend == "downtrend":
            return 25.0 - value
        return 0.0
    return value


def _alpha_adjustment(row) -> tuple[float, list[str]]:
    selections = _load_factor_selection()
    if not selections:
        return 0.0, []

    adjustment = 0.0
    applied = []
    for factor, meta in selections.items():
        if meta.get("validity") == "discard":
            continue
        direction = meta.get("direction")
        if direction not in {"positive", "negative"}:
            continue
        weight = float(meta.get("factor_weight", meta.get("signal_weight", 0.0)) or 0.0)
        if weight <= 0:
            continue

        exposure = _factor_exposure(row, factor)
        if exposure is None or exposure == 0:
            continue

        aligned = exposure > 0 if direction == "positive" else exposure < 0
        delta = weight if aligned else -weight
        adjustment += delta
        action = "+" if delta > 0 else "-"
        applied.append(f"alpha {factor} {action}{abs(delta):.2f}")
    return round(adjustment, 6), applied


def _market_regime_adjustment(trend: str, volatility_20d: float) -> tuple[float, list[str]]:
    if trend == "uptrend" and volatility_20d <= 60.0:
        return 0.5, ["market regime supports risk-on"]
    if trend == "downtrend":
        return -0.5, ["market regime weak"]
    return 0.0, []


def _score_row(row):
    required = ["trend", "return_20d", "price_vs_ma20"]
    if any(_is_missing(row.get(col)) for col in required):
        return {
            "signal": "HOLD",
            "base_signal_score": 0.0,
            "alpha_adjustment": 0.0,
            "market_regime_adjustment": 0.0,
            "risk_adjustment": 0.0,
            "final_signal_score": 0.0,
            "signal_reason": "insufficient core factors",
            "risk_flag": "insufficient_data",
        }

    base_score = 0.0
    risk_adjustment = 0.0
    reasons = []
    risk_flag = "normal"

    trend = row["trend"]
    return_20d = float(row["return_20d"])
    price_vs_ma20 = float(row["price_vs_ma20"])
    volatility_20d = float(row["volatility_20d"])

    # ═══ 趋势方向判断 · Alpha 做不到的 ═══
    # trend 是分类值 (uptrend/downtrend/range)，Alpha 无法对此算 IC
    if trend == "uptrend":
        base_score += 2
        reasons.append("trend up")
    if trend == "downtrend":
        risk_adjustment -= 2
        risk_flag = "weak_trend"
        reasons.append("trend down")

    # 基础动量方向 · 二值信号，Alpha 通过连续值间接覆盖
    if return_20d > 0:
        base_score += 1
        reasons.append("positive 20d return")
    else:
        risk_adjustment -= 1
        reasons.append("negative 20d return")

    if price_vs_ma20 > 0:
        base_score += 1
        reasons.append("price above MA20")

    # ═══ 估值安全网 · Alpha 管不到的底线 ═══
    # PE/PB 不是预测因子 (IC 算不出)，它们是"别买太贵"的结构性过滤
    pe = _as_float(row, "pe")
    pb = _as_float(row, "pb")

    if pe is not None and 0 < pe <= 30:
        base_score += 1
        reasons.append("reasonable PE")
    if pb is not None and 0 < pb <= 5:
        base_score += 1
        reasons.append("reasonable PB")

    # ── 技术指标 · 全部交给 Alpha 驱动层 ──
    # RSI / MACD / ATR / Bollinger / MFI / Stochastic / CCI / ROE / debt_to_asset
    # volatility / volume_ratio / max_drawdown / return_1d / return_5d
    # → 全部通过 _alpha_adjustment() 由 factor_selection.json 动态加权

    market_adjustment, market_reasons = _market_regime_adjustment(trend, volatility_20d)
    alpha_adjustment, alpha_reasons = _alpha_adjustment(row)
    reasons.extend(market_reasons)
    if alpha_reasons:
        reasons.append("alpha factor weighting: " + ", ".join(alpha_reasons))

    final_score = base_score + risk_adjustment + market_adjustment + alpha_adjustment
    if final_score >= 3 and risk_flag == "normal":
        signal = "BUY"
    elif final_score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    if not reasons:
        reasons.append("no clear directional signal")
    return {
        "signal": signal,
        "base_signal_score": round(base_score, 6),
        "alpha_adjustment": round(alpha_adjustment, 6),
        "market_regime_adjustment": round(market_adjustment, 6),
        "risk_adjustment": round(risk_adjustment, 6),
        "final_signal_score": round(final_score, 6),
        "signal_reason": "; ".join(reasons),
        "risk_flag": risk_flag,
    }


def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Generate deterministic trading signals from factor columns without calling LLMs."""
    df = df.copy()
    rows = [_score_row(row) for _, row in df.iterrows()]
    out = pd.DataFrame(rows, index=df.index)
    for column in out.columns:
        df[column] = out[column]
    df["signal_score"] = df["final_signal_score"]
    return df
