import logging
import re

import akshare as ak
import pandas as pd
import requests

from config import (
    DATA_DIR,
    DAILY_SIGNAL_SCORES_PATH,
    EASTMONEY_LIGHTWEIGHT_SPOT_ENABLED,
    EASTMONEY_REQUEST_TIMEOUT,
    FILTER_ST,
    FILTER_SUSPENDED,
    LOOKBACK_DAYS,
    MAX_POSITION_PER_STOCK,
    MAX_TOTAL_POSITION,
    MIN_REQUIRED_DAYS,
    TOP_N,
    UNIVERSE_INDEX,
    UNIVERSE_INDEX_SYMBOL,
    UNIVERSE_SIZE_LIMIT,
    UNIVERSE_SOURCE,
)
from factor_utils import compute_market_factors, merge_fundamental_features
from fundamental_utils import fetch_fundamental_data, fundamental_snapshot_to_frame
from industry_utils import fetch_industry_map
from market_summary_utils import build_market_summary
from portfolio_selection_utils import allocate_positions, select_top_candidates
from signal_utils import generate_signal
from storage_utils import save_dataframe, save_selected_candidates


logger = logging.getLogger(__name__)


EASTMONEY_SPOT_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_SPOT_FIELDS = "f12,f14,f9,f20,f21,f23"
EASTMONEY_SPOT_COLUMN_MAP = {
    "f12": "symbol",
    "f14": "name",
    "f9": "pe",
    "f20": "total_market_cap",
    "f21": "float_market_cap",
    "f23": "pb",
}


STANDARD_COLUMNS = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "换手率": "turnover",
}


def normalize_stock_symbol(code: str) -> str:
    raw = str(code).strip().lower()
    if raw.startswith(("sh", "sz")) and raw[2:].isdigit():
        return raw

    digits = "".join(re.findall(r"\d", raw))
    if 0 < len(digits) < 6:
        digits = digits.zfill(6)
    if len(digits) < 6:
        raise ValueError(f"无法识别股票代码: {code}")
    digits = digits[-6:]

    if digits.startswith("6"):
        return f"sh{digits}"
    if digits.startswith(("0", "3")):
        return f"sz{digits}"
    raise ValueError(f"无法根据代码判断交易所: {code}")


def _pick_column(df: pd.DataFrame, candidates: list[str]):
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        normalized = str(col).lower()
        if any(candidate.lower() in normalized for candidate in candidates):
            return col
    return None


_CONSTITUENT_FETCHERS = {
    "csindex": lambda sym: ak.index_stock_cons_csindex(symbol=sym),
    "eastmoney": lambda sym: ak.index_stock_cons(symbol=sym),
}


def fetch_index_constituents() -> list[dict]:
    fetcher = _CONSTITUENT_FETCHERS.get(UNIVERSE_SOURCE)
    if fetcher is None:
        raise ValueError(f"未知指数数据源类型: {UNIVERSE_SOURCE}，可选: {list(_CONSTITUENT_FETCHERS)}")

    cached = _cached_constituents_from_factor_files()
    if cached and (UNIVERSE_SIZE_LIMIT is None or len(cached) >= UNIVERSE_SIZE_LIMIT):
        logger.info("使用本地因子文件缓存股票池，避免成分股接口阻塞")
        return cached

    try:
        raw = fetcher(UNIVERSE_INDEX_SYMBOL)
    except Exception as e:
        if cached:
            logger.warning("获取指数成分股失败，使用本地因子文件缓存股票池: %s", e)
            return cached
        raise
    if raw is None or raw.empty:
        raise RuntimeError(f"{UNIVERSE_INDEX}成分股列表为空，请检查 AkShare 接口")

    code_col = _pick_column(raw, ["成分券代码", "品种代码", "证券代码", "代码", "code"])
    name_col = _pick_column(raw, ["成分券名称", "品种名称", "证券简称", "名称", "name"])
    if code_col is None or name_col is None:
        raise KeyError(f"无法识别成分股代码/名称字段，实际列: {list(raw.columns)}")

    constituents = []
    seen = set()
    for _, row in raw.iterrows():
        try:
            symbol = normalize_stock_symbol(row[code_col])
        except ValueError as e:
            logger.warning("跳过无法识别的成分股代码: %s", e)
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        constituents.append({"symbol": symbol, "name": str(row[name_col]).strip()})
    return constituents


def _cached_constituents_from_factor_files() -> list[dict]:
    factor_dir = DATA_DIR / "factors"
    if not factor_dir.exists():
        return []
    return [
        {"symbol": path.stem, "name": path.stem}
        for path in sorted(factor_dir.glob("*.csv"))
    ]


def _standardize_ohlcv(raw_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df = raw_df.rename(columns={k: v for k, v in STANDARD_COLUMNS.items() if k in raw_df.columns})
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"行情数据缺少必要字段: {missing}, 实际列: {list(raw_df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount", "turnover", "amplitude", "pct_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").tail(lookback).copy()
    preferred = [col for col in ["date", "open", "high", "low", "close", "volume", "amount", "turnover", "amplitude", "pct_change"] if col in df.columns]
    others = [col for col in df.columns if col not in preferred]
    return df[preferred + others]


def fetch_akshare_data(symbol: str, name: str, lookback: int) -> pd.DataFrame:
    """拉取单只 A 股历史行情，并统一字段为 date/open/high/low/close/volume。"""
    raw_df = ak.stock_zh_a_daily(symbol=symbol, adjust="qfq")
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    return _standardize_ohlcv(raw_df, lookback)


def is_valid_stock_data(df: pd.DataFrame, name: str, min_required_days: int) -> tuple[bool, str]:
    if FILTER_ST and ("ST" in name.upper() or "*ST" in name.upper()):
        return False, "ST stock"
    if df is None or df.empty:
        return False, "empty data"
    if len(df.dropna(subset=["close"])) < min_required_days:
        return False, "insufficient data"
    if "close" not in df.columns or "volume" not in df.columns:
        return False, "missing close/volume"

    latest = df.iloc[-1]
    if pd.isna(latest["close"]):
        return False, "latest close is NaN"
    if FILTER_SUSPENDED and (pd.isna(latest["volume"]) or latest["volume"] <= 0):
        return False, "suspended or zero volume"
    return True, "ok"


def _save_selected_candidates(selected_candidates: list[dict]):
    columns = [
        "symbol",
        "name",
        "industry",
        "signal",
        "signal_score",
        "base_signal_score",
        "alpha_adjustment",
        "market_regime_adjustment",
        "risk_adjustment",
        "final_signal_score",
        "target_weight",
        "base_target_weight",
        "transformer_weight_adjustment",
        "transformer_prediction",
        "transformer_prediction_rank",
        "transformer_prediction_zscore",
        "return_20d",
        "volatility_20d",
        "max_drawdown_20d",
        "trend",
        "rsi_14",
        "macd_diff",
        "adx_14",
        "bollinger_width",
        "pe",
        "pb",
        "roe",
        "total_market_cap",
        "float_market_cap",
        "revenue_yoy",
        "net_profit_yoy",
        "debt_to_asset",
        "fundamental_available",
        "signal_reason",
        "risk_flag",
    ]
    df = pd.DataFrame(selected_candidates)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        df = df.reindex(columns=columns)
    save_selected_candidates(df)


def _save_daily_signal_scores(all_factor_data: dict):
    rows = []
    for symbol, item in all_factor_data.items():
        df = item.get("data")
        if df is None or df.empty:
            continue
        latest = df.iloc[-1].copy()
        row = {
            "date": latest.get("date"),
            "symbol": symbol,
            "name": item.get("name"),
            "signal": latest.get("signal"),
            "signal_score": latest.get("signal_score"),
            "base_signal_score": latest.get("base_signal_score"),
            "alpha_adjustment": latest.get("alpha_adjustment"),
            "market_regime_adjustment": latest.get("market_regime_adjustment"),
            "risk_adjustment": latest.get("risk_adjustment"),
            "final_signal_score": latest.get("final_signal_score"),
            "risk_flag": latest.get("risk_flag"),
            "signal_reason": latest.get("signal_reason"),
        }
        rows.append(row)
    DAILY_SIGNAL_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DAILY_SIGNAL_SCORES_PATH, index=False, encoding="utf-8-sig")


def _eastmoney_spot_params(page: int = 1, page_size: int = 100) -> dict:
    return {
        "pn": page,
        "pz": page_size,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f12",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": EASTMONEY_SPOT_FIELDS,
    }


def _eastmoney_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://quote.eastmoney.com/",
        }
    )
    return session


def _fetch_lightweight_eastmoney_spot() -> pd.DataFrame:
    columns = list(EASTMONEY_SPOT_COLUMN_MAP.values())
    if not EASTMONEY_LIGHTWEIGHT_SPOT_ENABLED:
        return pd.DataFrame(columns=columns)

    rows = []
    page_size = 100
    try:
        with _eastmoney_session() as session:
            response = session.get(
                EASTMONEY_SPOT_URL,
                params=_eastmoney_spot_params(page=1, page_size=page_size),
                timeout=EASTMONEY_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") or {}
            diff = data.get("diff") or []
            rows.extend(diff)

            total = int(data.get("total") or len(diff))
            total_pages = max(1, (total + page_size - 1) // page_size)
            for page in range(2, total_pages + 1):
                response = session.get(
                    EASTMONEY_SPOT_URL,
                    params=_eastmoney_spot_params(page=page, page_size=page_size),
                    timeout=EASTMONEY_REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                page_payload = response.json()
                rows.extend((page_payload.get("data") or {}).get("diff") or [])
    except Exception as e:
        logger.warning(
            "Lightweight EastMoney valuation snapshot failed; PE/PB/market-cap will be skipped: %s",
            e,
        )
        return pd.DataFrame(columns=columns)

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows).rename(columns=EASTMONEY_SPOT_COLUMN_MAP)
    df = df.reindex(columns=columns)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    for column in ["pe", "pb", "total_market_cap", "float_market_cap"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _fetch_spot_snapshot():
    return _fetch_lightweight_eastmoney_spot()


def _industry_lookup(industry_map: pd.DataFrame) -> dict:
    if industry_map is None or industry_map.empty:
        return {}
    return dict(zip(industry_map["symbol"], industry_map["industry"]))


def _process_one_stock(stock: dict, spot_df: pd.DataFrame, industry_by_symbol: dict):
    symbol = stock["symbol"]
    name = stock["name"]
    df = fetch_akshare_data(symbol, name, LOOKBACK_DAYS)
    valid, reason = is_valid_stock_data(df, name, MIN_REQUIRED_DAYS)
    if not valid:
        return None, reason

    save_dataframe(df, "prices", symbol)

    factor_df, _ = compute_market_factors(
        raw_df=df,
        symbol=symbol,
        label=name,
        asset_type="stock",
        lookback_days=LOOKBACK_DAYS,
    )
    fundamental_snapshot = fetch_fundamental_data(symbol, spot_df)
    save_dataframe(fundamental_snapshot_to_frame(fundamental_snapshot), "fundamentals", symbol)
    factor_df = merge_fundamental_features(factor_df, fundamental_snapshot)
    factor_df["industry"] = industry_by_symbol.get(symbol) or "unknown"
    factor_df = generate_signal(factor_df)
    save_dataframe(factor_df, "factors", symbol)
    return {"name": name, "data": factor_df}, "ok"


def get_market_data():
    """获取指数成分股批量扫描摘要，并将带因子和信号的数据落盘。"""
    return get_market_snapshot()["summary_text"]


def get_market_snapshot():
    """获取指数成分股、生成信号、选股、分配仓位，并返回结构化快照。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    constituents = fetch_index_constituents()
    universe_count = len(constituents)
    if UNIVERSE_SIZE_LIMIT is not None:
        constituents = constituents[:UNIVERSE_SIZE_LIMIT]

    symbols = [stock["symbol"] for stock in constituents]
    spot_df = _fetch_spot_snapshot()
    industry_map = fetch_industry_map(symbols=symbols)
    save_dataframe(industry_map, "industries", "stock_industry_map")
    industry_by_symbol = _industry_lookup(industry_map)

    all_factor_data = {}
    fetched_count = 0
    filtered_count = 0
    failed_count = 0
    filter_reasons = {}

    for stock in constituents:
        symbol = stock["symbol"]
        name = stock["name"]
        try:
            result, reason = _process_one_stock(stock, spot_df, industry_by_symbol)
            if result is None:
                filtered_count += 1
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                logger.warning("过滤股票 %s %s: %s", symbol, name, reason)
                continue
            fetched_count += 1
            all_factor_data[symbol] = result
        except Exception as e:
            failed_count += 1
            logger.warning("处理股票失败 %s %s: %s", symbol, name, e)
            continue

    selected_candidates = select_top_candidates(all_factor_data, TOP_N)
    selected_candidates = allocate_positions(
        selected_candidates,
        max_position_per_stock=MAX_POSITION_PER_STOCK,
        max_total_position=MAX_TOTAL_POSITION,
    )
    _save_daily_signal_scores(all_factor_data)
    _save_selected_candidates(selected_candidates)

    stats = {
        "universe_count": universe_count,
        "processed_count": len(constituents),
        "fetched_count": fetched_count,
        "filtered_count": filtered_count,
        "failed_count": failed_count,
        "valid_count": len(all_factor_data),
        "selected_count": len(selected_candidates),
        "filter_reasons": filter_reasons,
    }
    logger.info(
        "%s成分股数量: %s, 本次处理: %s, 成功拉取: %s, 过滤: %s, 失败: %s, 有效: %s, 最终候选: %s",
        UNIVERSE_INDEX,
        universe_count,
        len(constituents),
        fetched_count,
        filtered_count,
        failed_count,
        len(all_factor_data),
        len(selected_candidates),
    )

    return {
        "summary_text": build_market_summary(all_factor_data, selected_candidates, stats),
        "features": all_factor_data,
        "selected_candidates": selected_candidates,
        "stats": stats,
    }



if __name__ == "__main__":
    summary = get_market_data()
    print(summary)
