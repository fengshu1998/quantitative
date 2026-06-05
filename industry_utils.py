import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

from config import (
    INDUSTRY_CACHE_ENABLED,
    INDUSTRY_DATA_DIR,
    INDUSTRY_FORCE_REFRESH,
    INDUSTRY_REFRESH_DAYS,
)


logger = logging.getLogger(__name__)

INDUSTRY_CACHE_PATH = Path(INDUSTRY_DATA_DIR) / "stock_industry_map.csv"
INDUSTRY_CACHE_META_PATH = Path(INDUSTRY_DATA_DIR) / "stock_industry_map.meta.json"


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


def _empty_industry_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "industry"])


def _filter_symbols(df: pd.DataFrame, symbols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return _empty_industry_frame()
    df = df.reindex(columns=["symbol", "industry"]).dropna(subset=["symbol"])
    df["symbol"] = df["symbol"].astype(str).str.lower()
    if symbols:
        wanted = set(symbols)
        df = df[df["symbol"].isin(wanted)]
    return df.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)


def _read_cached_industry_map(symbols: list[str] | None = None) -> pd.DataFrame:
    if not INDUSTRY_CACHE_PATH.exists():
        return _empty_industry_frame()
    try:
        df = pd.read_csv(INDUSTRY_CACHE_PATH)
    except Exception as e:
        logger.warning("Failed to read cached industry map: %s", e)
        return _empty_industry_frame()
    return _filter_symbols(df, symbols)


def _cache_updated_at() -> datetime | None:
    if INDUSTRY_CACHE_META_PATH.exists():
        try:
            payload = json.loads(INDUSTRY_CACHE_META_PATH.read_text(encoding="utf-8"))
            value = payload.get("updated_at")
            if value:
                return datetime.fromisoformat(value)
        except Exception:
            pass
    if INDUSTRY_CACHE_PATH.exists():
        return datetime.fromtimestamp(INDUSTRY_CACHE_PATH.stat().st_mtime)
    return None


def _industry_cache_is_fresh() -> bool:
    if not INDUSTRY_CACHE_ENABLED or INDUSTRY_FORCE_REFRESH:
        return False
    updated_at = _cache_updated_at()
    if updated_at is None:
        return False
    return datetime.now() - updated_at < timedelta(days=INDUSTRY_REFRESH_DAYS)


def _write_industry_cache(df: pd.DataFrame):
    if df.empty:
        return
    INDUSTRY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDUSTRY_CACHE_PATH, index=False, encoding="utf-8-sig")
    INDUSTRY_CACHE_META_PATH.write_text(
        json.dumps(
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "row_count": int(len(df)),
                "refresh_days": INDUSTRY_REFRESH_DAYS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _fetch_industry_map_remote(symbols: list[str] | None = None) -> pd.DataFrame:
    try:
        industries = ak.stock_board_industry_name_em()
    except Exception as e:
        logger.warning("Fetch industry list failed: %s", e)
        return _empty_industry_frame()

    name_col = _pick_column(industries, ["板块名称", "名称", "name"])
    if name_col is None:
        logger.warning("Cannot identify industry name column; columns=%s", list(industries.columns))
        return _empty_industry_frame()

    wanted = set(symbols or [])
    rows = []
    for industry in industries[name_col].dropna().astype(str):
        try:
            cons = ak.stock_board_industry_cons_em(symbol=industry)
        except Exception as e:
            logger.warning("Fetch industry constituents failed %s: %s", industry, e)
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
        return _empty_industry_frame()
    return _filter_symbols(pd.DataFrame(rows), symbols)


def fetch_industry_map(symbols: list[str] | None = None) -> pd.DataFrame:
    """Build symbol -> industry map with local cache fallback."""
    cached = _read_cached_industry_map(symbols)
    if INDUSTRY_CACHE_ENABLED and _industry_cache_is_fresh() and not cached.empty:
        logger.info("Using cached industry map: %s", INDUSTRY_CACHE_PATH)
        return cached

    refreshed = _fetch_industry_map_remote(symbols)
    if not refreshed.empty:
        _write_industry_cache(refreshed)
        return refreshed

    if not cached.empty:
        logger.warning("Industry refresh failed; using stale local cache: %s", INDUSTRY_CACHE_PATH)
        return cached

    logger.warning("Industry refresh failed and no cache is available; industry will be unknown.")
    return _empty_industry_frame()
