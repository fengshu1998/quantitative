import logging
from pathlib import Path

import pandas as pd

try:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import ADXIndicator, CCIIndicator, MACD
    from ta.volatility import AverageTrueRange, BollingerBands
    from ta.volume import MFIIndicator, OnBalanceVolumeIndicator
except ImportError:  # pragma: no cover - optional dependency fallback
    RSIIndicator = StochasticOscillator = None
    ADXIndicator = CCIIndicator = MACD = None
    AverageTrueRange = BollingerBands = None
    MFIIndicator = OnBalanceVolumeIndicator = None

from config import LOOKBACK_DAYS, TRADING_DAYS_PER_YEAR


logger = logging.getLogger(__name__)


def pick_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def safe_round(value, digits=2):
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _max_drawdown(series):
    drawdown = series / series.cummax() - 1
    return drawdown.min()


def classify_trend(close, ma5, ma20, return_20d):
    if pd.isna(ma5) or pd.isna(ma20) or pd.isna(return_20d):
        return "unknown"
    if close > ma5 > ma20 and return_20d > 0:
        return "uptrend"
    if close < ma5 < ma20 and return_20d < 0:
        return "downtrend"
    return "range"


def add_basic_factors(df, label, lookback_days=LOOKBACK_DAYS):
    date_col = pick_column(df, ["日期", "date"])
    close_col = pick_column(df, ["收盘", "close"])
    volume_col = pick_column(df, ["成交量", "volume"])
    if close_col is None:
        raise KeyError(f"{label} 缺少收盘价列，实际列: {list(df.columns)}")
    if volume_col is None:
        raise KeyError(f"{label} 缺少成交量列，实际列: {list(df.columns)}")

    df = df.copy()
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    df = df.tail(lookback_days).copy()
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")

    df["return_1d"] = df[close_col].pct_change(1) * 100
    df["return_5d"] = df[close_col].pct_change(5) * 100
    df["return_20d"] = df[close_col].pct_change(20) * 100
    df["MA5"] = df[close_col].rolling(5).mean()
    df["MA20"] = df[close_col].rolling(20).mean()
    df["volatility_20d"] = (
        df[close_col].pct_change().rolling(20).std() * (TRADING_DAYS_PER_YEAR**0.5) * 100
    )
    df["max_drawdown_20d"] = (
        df[close_col].rolling(20).apply(_max_drawdown, raw=False) * 100
    )
    df["price_vs_ma20"] = (df[close_col] / df["MA20"] - 1) * 100

    df["volume_ma20"] = df[volume_col].rolling(20).mean()
    df["volume_ratio_20d"] = df[volume_col] / df["volume_ma20"]

    df["trend"] = [
        classify_trend(close, ma5, ma20, ret20)
        for close, ma5, ma20, ret20 in zip(
            df[close_col], df["MA5"], df["MA20"], df["return_20d"]
        )
    ]
    return df, close_col


def add_ta_factors(df, label):
    """使用 ta 库补充成熟技术指标；缺少依赖或字段时保持主流程可运行。"""
    if RSIIndicator is None:
        logger.warning("ta 未安装，跳过扩展技术指标: %s", label)
        return df

    high_col = pick_column(df, ["最高", "high"])
    low_col = pick_column(df, ["最低", "low"])
    close_col = pick_column(df, ["收盘", "close"])
    volume_col = pick_column(df, ["成交量", "volume"])
    required = [high_col, low_col, close_col, volume_col]
    if any(col is None for col in required):
        logger.warning("缺少 high/low/close/volume，跳过 ta 指标: %s", label)
        return df

    high = pd.to_numeric(df[high_col], errors="coerce")
    low = pd.to_numeric(df[low_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")
    volume = pd.to_numeric(df[volume_col], errors="coerce")

    try:
        df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

        macd_indicator = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_indicator.macd()
        df["macd_signal"] = macd_indicator.macd_signal()
        df["macd_diff"] = macd_indicator.macd_diff()

        bollinger = BollingerBands(close=close, window=20, window_dev=2)
        df["bollinger_mavg"] = bollinger.bollinger_mavg()
        df["bollinger_high"] = bollinger.bollinger_hband()
        df["bollinger_low"] = bollinger.bollinger_lband()
        df["bollinger_width"] = bollinger.bollinger_wband()

        df["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

        stochastic = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        df["stoch_k"] = stochastic.stoch()
        df["stoch_d"] = stochastic.stoch_signal()

        df["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        df["mfi_14"] = MFIIndicator(high=high, low=low, close=close, volume=volume, window=14).money_flow_index()
        df["adx_14"] = ADXIndicator(high=high, low=low, close=close, window=14).adx()
        df["cci_20"] = CCIIndicator(high=high, low=low, close=close, window=20).cci()
    except Exception as e:
        logger.warning("计算 ta 技术指标失败 %s: %s", label, e)
    return df


def add_cross_sectional_factor_fields(df):
    """预留横截面标准化字段，当前单标的数据中保持原值。"""
    df = df.copy()
    for col in ["return_20d", "volume_ratio_20d", "rsi_14", "macd_diff", "adx_14"]:
        if col in df.columns:
            df[f"{col}_raw"] = df[col]
    return df


def add_market_factors(df, label, lookback_days=LOOKBACK_DAYS):
    df, close_col = add_basic_factors(df, label, lookback_days)
    df = add_ta_factors(df, label)
    df = add_cross_sectional_factor_fields(df)
    return df, close_col


def build_latest_feature(df, close_col, symbol, label, asset_type):
    latest = df.iloc[-1]
    ma5_val = latest["MA5"]
    if pd.isna(ma5_val):
        raise ValueError(f"{label} MA5 计算失败，数据不足5个交易日")

    return {
        "symbol": symbol,
        "label": label,
        "asset_type": asset_type,
        "close": safe_round(latest[close_col]),
        "ma5": safe_round(latest["MA5"]),
        "ma20": safe_round(latest["MA20"]),
        "return_1d": safe_round(latest["return_1d"]),
        "return_5d": safe_round(latest["return_5d"]),
        "return_20d": safe_round(latest["return_20d"]),
        "volatility_20d": safe_round(latest["volatility_20d"]),
        "max_drawdown_20d": safe_round(latest["max_drawdown_20d"]),
        "volume_ratio_20d": safe_round(latest["volume_ratio_20d"]),
        "price_vs_ma20": safe_round(latest["price_vs_ma20"]),
        "trend": latest["trend"],
        "rsi_14": safe_round(latest.get("rsi_14")),
        "macd": safe_round(latest.get("macd")),
        "macd_signal": safe_round(latest.get("macd_signal")),
        "macd_diff": safe_round(latest.get("macd_diff")),
        "bollinger_width": safe_round(latest.get("bollinger_width")),
        "atr_14": safe_round(latest.get("atr_14")),
        "adx_14": safe_round(latest.get("adx_14")),
        "mfi_14": safe_round(latest.get("mfi_14")),
        "cci_20": safe_round(latest.get("cci_20")),
    }


def compute_market_factors(raw_df, symbol, label, asset_type, lookback_days=LOOKBACK_DAYS):
    df, close_col = add_market_factors(raw_df, label, lookback_days)
    feature = build_latest_feature(df, close_col, symbol, label, asset_type)
    return df, feature


def merge_fundamental_features(price_factor_df, fundamental_snapshot):
    """把单只股票的财务快照合并到每一行因子数据中。"""
    df = price_factor_df.copy()
    for key, value in fundamental_snapshot.items():
        if key == "symbol":
            continue
        df[key] = value
    return df


def build_alphalens_factor_data(all_factor_data, factor_name, price_field="close"):
    """构造 Alphalens 所需的 factor Series 和 prices 矩阵。"""
    factor_frames = []
    price_frames = []

    for symbol, item in all_factor_data.items():
        df = item["data"].copy()
        if factor_name not in df.columns or price_field not in df.columns or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        factor_series = df.set_index("date")[factor_name].rename(symbol)
        factor_frames.append(factor_series)

        price_series = df.set_index("date")[price_field].rename(symbol)
        price_frames.append(price_series)

    if not factor_frames or not price_frames:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
        return pd.Series(index=empty_index, dtype=float, name=factor_name), pd.DataFrame()

    factor_matrix = pd.concat(factor_frames, axis=1)
    prices = pd.concat(price_frames, axis=1).sort_index()
    factor = factor_matrix.stack(future_stack=True).dropna()
    factor.index = factor.index.set_names(["date", "asset"])
    factor.name = factor_name
    return factor, prices


def run_alphalens_report(all_factor_data, factor_name, output_dir="data/factor_reports"):
    """可选因子评估入口；失败或依赖不可用时不影响主流程。"""
    try:
        import matplotlib.pyplot as plt
        from alphalens import plotting, tears, utils
    except Exception as e:
        logger.warning("Alphalens 不可用，跳过因子报告 %s: %s", factor_name, e)
        return {"status": "skipped", "reason": f"alphalens unavailable: {e}"}

    factor, prices = build_alphalens_factor_data(all_factor_data, factor_name)
    if factor.empty or prices.empty:
        return {"status": "skipped", "reason": "empty factor or prices"}

    try:
        clean_factor_data = utils.get_clean_factor_and_forward_returns(
            factor=factor,
            prices=prices,
            periods=(1, 5, 10),
            quantiles=5,
            max_loss=0.5,
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        tears.create_full_tear_sheet(clean_factor_data, long_short=True, group_neutral=False)
        report_path = output_path / f"{factor_name}_alphalens.png"
        plt.savefig(report_path, bbox_inches="tight")
        plt.close("all")
        return {
            "status": "ok",
            "factor_name": factor_name,
            "rows": int(len(clean_factor_data)),
            "report_path": str(report_path),
        }
    except Exception as e:
        logger.warning("Alphalens 因子评估失败 %s: %s", factor_name, e)
        return {"status": "skipped", "reason": str(e)}
