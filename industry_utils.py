import logging

import akshare as ak
import pandas as pd


logger = logging.getLogger(__name__)


def _normalize_stock_symbol(code: str) -> str | None:
    raw = str(code).strip().lower()
    if raw.startswith(("sh", "sz")) and raw[2:].isdigit():
        return raw
    digits = "".join(ch for ch in raw if ch.isdigit())
    if 0 < len(digits) < 6:
        digits = digits.zfill(6)
    if len(digits) < 6:
        return None
    digits = digits[-6:]
    if digits.startswith("6"):
        return f"sh{digits}"
    if digits.startswith(("0", "3")):
        return f"sz{digits}"
    return None


def _pick_column(df: pd.DataFrame, candidates: list[str]):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def fetch_industry_map(symbols: list[str] | None = None) -> pd.DataFrame:
    """构造 symbol -> industry_name 映射；行业接口失败时返回空表。"""
    try:
        industries = ak.stock_board_industry_name_em()
    except Exception as e:
        logger.warning("获取行业列表失败: %s", e)
        return pd.DataFrame(columns=["symbol", "industry"])

    name_col = _pick_column(industries, ["板块名称", "名称", "name"])
    if name_col is None:
        logger.warning("无法识别行业名称字段，实际列: %s", list(industries.columns))
        return pd.DataFrame(columns=["symbol", "industry"])

    wanted = set(symbols or [])
    rows = []
    for industry in industries[name_col].dropna().astype(str):
        try:
            cons = ak.stock_board_industry_cons_em(symbol=industry)
        except Exception as e:
            logger.warning("获取行业成分失败 %s: %s", industry, e)
            continue
        code_col = _pick_column(cons, ["代码", "证券代码", "品种代码"])
        if code_col is None:
            continue
        for code in cons[code_col]:
            symbol = _normalize_stock_symbol(code)
            if symbol is None:
                continue
            if wanted and symbol not in wanted:
                continue
            rows.append({"symbol": symbol, "industry": industry})

    if not rows:
        return pd.DataFrame(columns=["symbol", "industry"])
    return pd.DataFrame(rows).drop_duplicates(subset=["symbol"], keep="first")
