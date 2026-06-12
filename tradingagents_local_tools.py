from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from config import INDUSTRY_DATA_DIR
from storage_utils import load_dataframe, load_selected_candidates


logger = logging.getLogger(__name__)


def _json_default(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _to_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=_json_default, indent=2)


def _normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().lower()
    if value.startswith(("sh", "sz")) and len(value) == 8:
        return value
    if value.isdigit() and len(value) == 6:
        if value.startswith("6"):
            return f"sh{value}"
        if value.startswith(("0", "3")):
            return f"sz{value}"
    return value


def _safe_records(df: pd.DataFrame, rows: int = 10) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.tail(rows).where(pd.notna(df), None).to_dict("records")


def _load_candidate(symbol: str) -> dict[str, Any]:
    try:
        df = load_selected_candidates()
    except Exception:
        return {}
    if df.empty or "symbol" not in df.columns:
        return {}
    normalized = _normalize_symbol(symbol)
    matched = df[df["symbol"].astype(str).str.lower() == normalized]
    if matched.empty:
        return {}
    return matched.iloc[0].where(pd.notna(matched.iloc[0]), None).to_dict()


def _load_latest_row(dataset_type: str, symbol: str) -> dict[str, Any]:
    try:
        df = load_dataframe(dataset_type, _normalize_symbol(symbol))
    except Exception as e:
        logger.warning("Failed to load %s for %s: %s", dataset_type, symbol, e)
        return {}
    if df.empty:
        return {}
    return df.iloc[-1].where(pd.notna(df.iloc[-1]), None).to_dict()


def _load_industry(symbol: str) -> str | None:
    path = Path(INDUSTRY_DATA_DIR) / "stock_industry_map.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "symbol" not in df.columns:
        return None
    matched = df[df["symbol"].astype(str).str.lower() == _normalize_symbol(symbol)]
    if matched.empty:
        return None
    for column in ("industry", "industry_name", "name"):
        if column in matched.columns:
            value = matched.iloc[0].get(column)
            if pd.notna(value):
                return str(value)
    return None


@tool
def get_stock_data(symbol: str, rows: int = 20) -> str:
    """Return recent local A-share OHLCV price rows for a sh/sz symbol."""
    symbol = _normalize_symbol(symbol)
    try:
        df = load_dataframe("prices", symbol)
    except Exception as e:
        return _to_json({"symbol": symbol, "available": False, "error": str(e)})
    columns = [col for col in ["date", "open", "high", "low", "close", "volume", "amount"] if col in df.columns]
    payload = {
        "symbol": symbol,
        "available": not df.empty,
        "rows": _safe_records(df[columns] if columns else df, rows),
    }
    return _to_json(payload)


@tool
def get_indicators(symbol: str, indicators: str = "", rows: int = 5) -> str:
    """Return local technical, fundamental, and signal factor rows for a sh/sz symbol."""
    symbol = _normalize_symbol(symbol)
    try:
        df = load_dataframe("factors", symbol)
    except Exception as e:
        return _to_json({"symbol": symbol, "available": False, "error": str(e)})
    requested = [item.strip() for item in indicators.split(",") if item.strip()]
    base_columns = [
        "date",
        "close",
        "return_20d",
        "volatility_20d",
        "max_drawdown_20d",
        "volume_ratio_20d",
        "price_vs_ma20",
        "trend",
        "rsi_14",
        "macd_diff",
        "adx_14",
        "bollinger_width",
        "pe",
        "pb",
        "roe",
        "debt_to_asset",
        "signal",
        "cross_section_score",
        "alpha_cross_section_score",
        "transformer_cross_section_score",
        "risk_liquidity_cross_section_score",
        "signal_reason",
        "risk_flag",
    ]
    columns = [col for col in base_columns + requested if col in df.columns]
    payload = {
        "symbol": symbol,
        "available": not df.empty,
        "requested_indicators": requested,
        "rows": _safe_records(df[columns] if columns else df, rows),
    }
    return _to_json(payload)


@tool
def get_verified_market_snapshot(symbol: str, trade_date: str = "") -> str:
    """Return a deterministic local market snapshot for exact value checks."""
    symbol = _normalize_symbol(symbol)
    latest = _load_latest_row("factors", symbol)
    candidate = _load_candidate(symbol)
    payload = {
        "symbol": symbol,
        "trade_date": trade_date,
        "available": bool(latest),
        "candidate": candidate,
        "industry": candidate.get("industry") or _load_industry(symbol),
        "latest": latest,
    }
    return _to_json(payload)


@tool
def get_fundamentals(symbol: str) -> str:
    """Return the local A-share fundamental snapshot for a sh/sz symbol."""
    symbol = _normalize_symbol(symbol)
    candidate = _load_candidate(symbol)
    latest = _load_latest_row("fundamentals", symbol)
    factor_latest = _load_latest_row("factors", symbol)
    payload = {
        "symbol": symbol,
        "available": bool(latest) or bool(factor_latest),
        "candidate_fundamentals": {
            key: candidate.get(key)
            for key in [
                "pe",
                "pb",
                "roe",
                "total_market_cap",
                "float_market_cap",
                "revenue_yoy",
                "net_profit_yoy",
                "debt_to_asset",
                "fundamental_available",
            ]
            if key in candidate
        },
        "fundamental_snapshot": latest,
        "factor_snapshot": {
            key: factor_latest.get(key)
            for key in [
                "pe",
                "pb",
                "roe",
                "total_market_cap",
                "float_market_cap",
                "revenue_yoy",
                "net_profit_yoy",
                "debt_to_asset",
                "fundamental_available",
            ]
            if key in factor_latest
        },
    }
    return _to_json(payload)


@tool
def get_balance_sheet(symbol: str) -> str:
    """Return local balance-sheet style fields if available."""
    symbol = _normalize_symbol(symbol)
    latest = _load_latest_row("fundamentals", symbol)
    payload = {
        "symbol": symbol,
        "available": bool(latest),
        "debt_to_asset": latest.get("debt_to_asset"),
        "note": "Detailed balance sheet statements are not connected; using local snapshot fields.",
    }
    return _to_json(payload)


@tool
def get_cashflow(symbol: str) -> str:
    """Return local cashflow data status for a sh/sz symbol."""
    return _to_json(
        {
            "symbol": _normalize_symbol(symbol),
            "available": False,
            "note": "Cashflow statement data is not connected in this project phase.",
        }
    )


@tool
def get_income_statement(symbol: str) -> str:
    """Return local income-statement style fields if available."""
    symbol = _normalize_symbol(symbol)
    latest = _load_latest_row("fundamentals", symbol)
    payload = {
        "symbol": symbol,
        "available": bool(latest),
        "revenue_yoy": latest.get("revenue_yoy"),
        "net_profit_yoy": latest.get("net_profit_yoy"),
        "roe": latest.get("roe"),
        "note": "Detailed income statements are not connected; using local snapshot fields.",
    }
    return _to_json(payload)


@tool
def get_news(symbol: str, lookback_days: int = 7) -> str:
    """Return local news data status for a sh/sz symbol."""
    return _to_json(
        {
            "symbol": _normalize_symbol(symbol),
            "lookback_days": lookback_days,
            "available": False,
            "note": "News/RAG data is not connected in this project phase.",
        }
    )


@tool
def get_global_news(query: str = "", lookback_days: int = 7) -> str:
    """Return local global-news data status."""
    return _to_json(
        {
            "query": query,
            "lookback_days": lookback_days,
            "available": False,
            "note": "Macro/news RAG data is not connected in this project phase.",
        }
    )


@tool
def get_insider_transactions(symbol: str) -> str:
    """Return local insider-transaction data status for a sh/sz symbol."""
    return _to_json(
        {
            "symbol": _normalize_symbol(symbol),
            "available": False,
            "note": "Insider transaction data is not connected for A-share candidates.",
        }
    )
