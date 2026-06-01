import logging

import akshare as ak
import pandas as pd


logger = logging.getLogger(__name__)


def _plain_code(symbol: str) -> str:
    symbol = str(symbol).lower().strip()
    if symbol.startswith(("sh", "sz")):
        return symbol[2:]
    return symbol


def _em_code(symbol: str) -> str:
    code = _plain_code(symbol)
    if str(symbol).lower().startswith("sh") or code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


def _safe_number(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_first(row: pd.Series, columns: list[str]):
    for col in columns:
        if col in row.index:
            value = _safe_number(row[col])
            if value is not None:
                return value
    return None


def _empty_snapshot(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "fundamental_available": False,
        "pe": None,
        "pb": None,
        "roe": None,
        "total_market_cap": None,
        "float_market_cap": None,
        "revenue_yoy": None,
        "net_profit_yoy": None,
        "debt_to_asset": None,
    }


def fetch_fundamental_data(symbol: str, spot_df: pd.DataFrame | None = None) -> dict:
    """获取单只股票的财务摘要；失败时返回 fundamental_available=False。"""
    snapshot = _empty_snapshot(symbol)
    code = _plain_code(symbol)

    try:
        if spot_df is not None and not spot_df.empty and "代码" in spot_df.columns:
            row_df = spot_df[spot_df["代码"].astype(str).str.zfill(6) == code]
            if not row_df.empty:
                row = row_df.iloc[0]
                snapshot["pe"] = _pick_first(row, ["市盈率-动态", "市盈率", "PE"])
                snapshot["pb"] = _pick_first(row, ["市净率", "PB"])
                snapshot["total_market_cap"] = _pick_first(row, ["总市值"])
                snapshot["float_market_cap"] = _pick_first(row, ["流通市值"])
    except Exception as e:
        logger.warning("提取实时估值失败 %s: %s", symbol, e)

    try:
        indicators = ak.stock_financial_analysis_indicator_em(symbol=_em_code(symbol))
        if indicators is not None and not indicators.empty:
            latest = indicators.iloc[0]
            snapshot["roe"] = _pick_first(latest, ["ROEJQ", "ROEKCJQ"])
            snapshot["revenue_yoy"] = _pick_first(latest, ["TOTALOPERATEREVETZ", "DJD_TOI_YOY"])
            snapshot["net_profit_yoy"] = _pick_first(latest, ["PARENTNETPROFITTZ", "DJD_DPNP_YOY"])
            snapshot["debt_to_asset"] = _pick_first(latest, ["ZCFZL"])
    except Exception as e:
        logger.warning("获取财务指标失败 %s: %s", symbol, e)

    snapshot["fundamental_available"] = any(
        snapshot[field] is not None
        for field in ["pe", "pb", "roe", "total_market_cap", "float_market_cap", "revenue_yoy", "net_profit_yoy", "debt_to_asset"]
    )
    return snapshot


def fundamental_snapshot_to_frame(snapshot: dict) -> pd.DataFrame:
    return pd.DataFrame([snapshot])
