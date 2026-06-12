import json
import logging

import numpy as np
import pandas as pd

from config import (
    ALPHA_FACTOR_SELECTION_PATH,
    ALPHA_SIGNAL_WEIGHTING_ENABLED,
    FACTOR_WINSORIZE_LOWER,
    FACTOR_WINSORIZE_UPPER,
    LONG_QUANTILE,
    SHORT_QUANTILE,
    SIGNAL_CROSS_WEIGHTS,
    TRANSFORMER_PREDICTION_PATH,
)


logger = logging.getLogger(__name__)

# Structural guard thresholds — Alpha layer handles all factor-level scoring.
# These only gate extreme safety conditions Alpha cannot express.

FACTOR_SELECTION_CACHE = None
TRANSFORMER_PREDICTION_CACHE = None

DEFAULT_ALPHA_FACTORS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "max_drawdown_20d",
    "price_vs_ma20",
    "volume_ratio_20d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bollinger_width",
    "atr_14",
    "stoch_k",
    "stoch_d",
    "mfi_14",
    "adx_14",
    "cci_20",
    "roe",
    "debt_to_asset",
]


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


def clear_signal_caches():
    global FACTOR_SELECTION_CACHE, TRANSFORMER_PREDICTION_CACHE
    FACTOR_SELECTION_CACHE = None
    TRANSFORMER_PREDICTION_CACHE = None


def _load_transformer_predictions():
    global TRANSFORMER_PREDICTION_CACHE
    if TRANSFORMER_PREDICTION_CACHE is not None:
        return TRANSFORMER_PREDICTION_CACHE
    if not TRANSFORMER_PREDICTION_PATH.exists():
        TRANSFORMER_PREDICTION_CACHE = {}
        return TRANSFORMER_PREDICTION_CACHE
    try:
        df = pd.read_csv(TRANSFORMER_PREDICTION_PATH)
    except Exception as e:
        logger.warning("Failed to load transformer predictions: %s", e)
        TRANSFORMER_PREDICTION_CACHE = {}
        return TRANSFORMER_PREDICTION_CACHE
    if df.empty or "prediction" not in df.columns:
        TRANSFORMER_PREDICTION_CACHE = {}
        return TRANSFORMER_PREDICTION_CACHE
    if "symbol" not in df.columns and "instrument" in df.columns:
        df["symbol"] = df["instrument"].astype(str).str.upper().str.replace(
            r"^(SH|SZ)(\d{6})$",
            lambda match: match.group(1).lower() + match.group(2),
            regex=True,
        )
    if "symbol" not in df.columns:
        TRANSFORMER_PREDICTION_CACHE = {}
        return TRANSFORMER_PREDICTION_CACHE
    if "prediction_zscore" not in df.columns:
        pred = pd.to_numeric(df["prediction"], errors="coerce")
        std = pred.std()
        mean = pred.mean()
        df["prediction_zscore"] = 0.0 if not std or pd.isna(std) else (pred - mean) / std
    if "prediction_rank" not in df.columns:
        df["prediction_rank"] = pd.to_numeric(df["prediction"], errors="coerce").rank(
            ascending=False,
            method="dense",
        )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime")
    latest = df.groupby("symbol", as_index=False).tail(1)
    TRANSFORMER_PREDICTION_CACHE = latest.where(pd.notna(latest), None).set_index("symbol").to_dict("index")
    return TRANSFORMER_PREDICTION_CACHE


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


def _transformer_score(row) -> tuple[float, dict, list[str]]:
    symbol = str(row.get("symbol") or "").lower()
    pred = _load_transformer_predictions().get(symbol, {})
    if not pred:
        return 0.0, {}, ["transformer prediction unavailable"]
    try:
        zscore = float(pred.get("prediction_zscore") or 0.0)
    except (TypeError, ValueError):
        zscore = 0.0
    score = max(-3.0, min(3.0, zscore))
    return round(score, 6), pred, [f"transformer zscore {zscore:.2f}"]


def _risk_liquidity_score(row, risk_flag: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []
    volume = _as_float(row, "volume")
    volatility = _as_float(row, "volatility_20d")
    max_drawdown = _as_float(row, "max_drawdown_20d")
    trend = row.get("trend")

    if risk_flag == "normal":
        score += 0.5
        reasons.append("risk flag normal")
    else:
        score -= 1.0
        reasons.append(f"risk flag {risk_flag}")
    if volume is not None and volume > 0:
        score += 0.5
        reasons.append("liquidity available")
    else:
        score -= 1.0
        reasons.append("zero or missing volume")
    if volatility is not None:
        if volatility <= 60:
            score += 0.5
            reasons.append("normal volatility")
        elif volatility > 100:
            score -= 1.0
            reasons.append("high volatility")
    if max_drawdown is not None and max_drawdown < -20:
        score -= 1.0
        reasons.append("large drawdown")
    if trend == "downtrend":
        score -= 1.0
        reasons.append("weak trend")
    return round(max(-3.0, min(3.0, score)), 6), reasons


def _score_row(row):
    required = ["trend", "return_20d", "price_vs_ma20"]
    if any(_is_missing(row.get(col)) for col in required):
        return {
            "signal": "HOLD",
            "base_signal_score": 0.0,
            "alpha_score": 0.0,
            "alpha_adjustment": 0.0,
            "transformer_score": 0.0,
            "risk_liquidity_score": 0.0,
            "market_regime_adjustment": 0.0,
            "risk_adjustment": 0.0,
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
    alpha_score = base_score + market_adjustment + alpha_adjustment
    transformer_score, transformer_pred, transformer_reasons = _transformer_score(row)
    risk_liquidity_score, risk_liquidity_reasons = _risk_liquidity_score(row, risk_flag)
    reasons.extend(transformer_reasons)
    reasons.extend(risk_liquidity_reasons)

    preliminary_score = alpha_score + risk_adjustment
    if preliminary_score >= 2 and risk_flag == "normal":
        signal = "BUY"
    elif preliminary_score <= -1.5:
        signal = "SELL"
    else:
        signal = "HOLD"

    if not reasons:
        reasons.append("no clear directional signal")
    return {
        "signal": signal,
        "base_signal_score": round(base_score, 6),
        "alpha_score": round(alpha_score, 6),
        "alpha_adjustment": round(alpha_adjustment, 6),
        "transformer_score": round(transformer_score, 6),
        "transformer_prediction": transformer_pred.get("prediction"),
        "transformer_prediction_rank": transformer_pred.get("prediction_rank"),
        "transformer_prediction_zscore": transformer_pred.get("prediction_zscore"),
        "risk_liquidity_score": round(risk_liquidity_score, 6),
        "market_regime_adjustment": round(market_adjustment, 6),
        "risk_adjustment": round(risk_adjustment, 6),
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
    df = df.drop(columns=["signal_score", "fusion_score", "raw_fusion_score", "fusion_weights"], errors="ignore")
    return df


def _winsorize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    clean = values.dropna()
    if clean.empty:
        return values
    lower = clean.quantile(float(FACTOR_WINSORIZE_LOWER))
    upper = clean.quantile(float(FACTOR_WINSORIZE_UPPER))
    return values.clip(lower=lower, upper=upper)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std()
    if not std or pd.isna(std):
        return values * 0
    return (values - values.mean()) / std


def _market_cap_residual(values: pd.Series, market_cap: pd.Series) -> pd.Series:
    y = pd.to_numeric(values, errors="coerce")
    x = pd.to_numeric(market_cap, errors="coerce")
    valid = y.notna() & x.notna() & (x > 0)
    residual = y.copy()
    if valid.sum() < 3:
        return residual
    log_cap = np.log(x[valid].astype(float))
    if log_cap.std() == 0 or pd.isna(log_cap.std()):
        return residual
    slope, intercept = np.polyfit(log_cap, y[valid].astype(float), 1)
    residual.loc[valid] = y.loc[valid] - (slope * log_cap + intercept)
    return residual


def _factor_direction_and_weight() -> dict[str, tuple[str, float]]:
    selections = _load_factor_selection()
    if not selections:
        return {factor: ("positive", 1.0) for factor in DEFAULT_ALPHA_FACTORS}
    result = {}
    for factor, meta in selections.items():
        if meta.get("validity") == "discard":
            continue
        direction = meta.get("direction")
        weight = float(meta.get("factor_weight", meta.get("signal_weight", 0.0)) or 0.0)
        if direction in {"positive", "negative"} and weight > 0:
            result[factor] = (direction, weight)
    return result


def _rank_center_score(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if values.nunique(dropna=True) <= 1:
        neutral = pd.Series(0.0, index=values.index)
        rank = pd.Series(0.5, index=values.index)
        return neutral, rank
    rank = values.rank(pct=True, method="average").fillna(0.5)
    return (rank - 0.5).fillna(0.0), rank


def _standardize_day(day_df: pd.DataFrame, factor_meta: dict[str, tuple[str, float]]) -> pd.DataFrame:
    day_df = day_df.copy()
    alpha_score = pd.Series(0.0, index=day_df.index)
    weight_sum = 0.0
    cap = day_df.get("float_market_cap")
    if cap is None:
        cap = day_df.get("total_market_cap", pd.Series(index=day_df.index, dtype=float))
    else:
        fallback = day_df.get("total_market_cap", pd.Series(index=day_df.index, dtype=float))
        cap = cap.fillna(fallback)

    for factor, (direction, weight) in factor_meta.items():
        if factor not in day_df.columns:
            continue
        clipped = _winsorize(day_df[factor])
        z = _zscore(clipped)
        if "industry" in day_df.columns:
            z = z - z.groupby(day_df["industry"].fillna("unknown")).transform("mean")
        neutral = _market_cap_residual(z, cap)
        z_neutral = _zscore(neutral)
        adjusted = z_neutral if direction == "positive" else -z_neutral
        rank_pct = adjusted.rank(pct=True, method="average")

        day_df[f"{factor}_zscore"] = z
        day_df[f"{factor}_neutralized"] = z_neutral
        day_df[f"{factor}_rank_pct"] = rank_pct
        alpha_score = alpha_score.add((rank_pct.fillna(0.5) - 0.5) * weight, fill_value=0.0)
        weight_sum += weight

    if weight_sum > 0:
        alpha_score = alpha_score / weight_sum
    else:
        alpha_score, _ = _rank_center_score(day_df.get("alpha_score", pd.Series(0.0, index=day_df.index)))

    transformer_score, _ = _rank_center_score(day_df.get("transformer_score", pd.Series(0.0, index=day_df.index)))
    risk_liquidity_score, _ = _rank_center_score(day_df.get("risk_liquidity_score", pd.Series(0.0, index=day_df.index)))
    final_score = (
        alpha_score * float(SIGNAL_CROSS_WEIGHTS["alpha"])
        + transformer_score * float(SIGNAL_CROSS_WEIGHTS["transformer"])
        + risk_liquidity_score * float(SIGNAL_CROSS_WEIGHTS["risk_liquidity"])
    )
    final_rank = final_score.rank(pct=True, method="average")

    day_df["alpha_cross_section_score"] = alpha_score.round(6)
    day_df["transformer_cross_section_score"] = transformer_score.round(6)
    day_df["risk_liquidity_cross_section_score"] = risk_liquidity_score.round(6)
    day_df["cross_section_score"] = final_score.round(6)
    day_df["cross_section_rank_pct"] = final_rank.fillna(0.5).round(6)
    day_df["long_bucket"] = day_df["cross_section_rank_pct"] > (1.0 - float(LONG_QUANTILE))
    day_df["short_bucket"] = day_df["cross_section_rank_pct"] <= float(SHORT_QUANTILE)
    day_df["signal"] = np.select(
        [day_df["long_bucket"], day_df["short_bucket"]],
        ["BUY", "SELL"],
        default="HOLD",
    )
    day_df = day_df.drop(
        columns=["signal_score", "fusion_score", "raw_fusion_score", "fusion_weights", "final_signal_score"],
        errors="ignore",
    )
    return day_df


def apply_cross_sectional_signal_scores(all_factor_data: dict) -> dict:
    """Apply daily cross-sectional standardization/ranking across all instruments."""
    frames = []
    for symbol, item in all_factor_data.items():
        df = item.get("data")
        if df is None or df.empty or "date" not in df.columns:
            continue
        panel = df.copy()
        panel["_symbol_key"] = symbol
        panel["_row_id"] = np.arange(len(panel))
        frames.append(panel)
    if not frames:
        return all_factor_data

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])
    if panel.empty:
        return all_factor_data

    factor_meta = _factor_direction_and_weight()
    def _apply_day(day: pd.DataFrame) -> pd.DataFrame:
        out = _standardize_day(day, factor_meta)
        out["date"] = day.name
        return out

    scored = panel.groupby("date", group_keys=False).apply(_apply_day)
    result = {}
    for symbol, item in all_factor_data.items():
        part = scored[scored["_symbol_key"] == symbol].sort_values("_row_id")
        if part.empty:
            result[symbol] = item
            continue
        part = part.drop(columns=["_symbol_key", "_row_id"], errors="ignore")
        part["date"] = part["date"].dt.strftime("%Y-%m-%d")
        result[symbol] = {**item, "data": part.reset_index(drop=True)}
    return result
