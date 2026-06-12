import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import akshare as ak
except Exception:  # pragma: no cover - optional in lightweight test environments
    ak = None

from config import BENCHMARK_SYMBOL, BENCHMARK_WEIGHT_DIR, BENCHMARK_WEIGHT_SOURCE


logger = logging.getLogger(__name__)


def _normalize_symbol(value) -> str:
    text = str(value or "").strip().lower()
    if len(text) == 6 and text.isdigit():
        if text.startswith("6"):
            return f"sh{text}"
        if text.startswith(("0", "3")):
            return f"sz{text}"
    if len(text) >= 8 and text[:2] in {"sh", "sz"}:
        return text[:8]
    return text


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    lowered = {str(col).lower(): col for col in df.columns}
    for name in candidates:
        col = lowered.get(name.lower())
        if col is not None:
            return col
    return None


def _weight_path(benchmark_symbol: str) -> Path:
    return Path(BENCHMARK_WEIGHT_DIR) / f"{benchmark_symbol}_weights.csv"


def _read_cached_weights(benchmark_symbol: str = BENCHMARK_SYMBOL) -> pd.DataFrame:
    path = _weight_path(benchmark_symbol)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "benchmark_weight", "benchmark_weight_source"])
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning("Failed to read benchmark weight cache %s: %s", path, e)
        return pd.DataFrame(columns=["symbol", "benchmark_weight", "benchmark_weight_source"])
    if "symbol" not in df.columns or "benchmark_weight" not in df.columns:
        return pd.DataFrame(columns=["symbol", "benchmark_weight", "benchmark_weight_source"])
    df["symbol"] = df["symbol"].map(_normalize_symbol)
    df["benchmark_weight"] = pd.to_numeric(df["benchmark_weight"], errors="coerce")
    return df.dropna(subset=["symbol", "benchmark_weight"])


def _fetch_akshare_weights(benchmark_symbol: str) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("akshare unavailable")
    fetchers = [
        lambda: ak.index_stock_cons_weight_csindex(symbol=benchmark_symbol),
        lambda: ak.index_stock_cons_weight(symbol=benchmark_symbol),
    ]
    last_error = None
    for fetcher in fetchers:
        try:
            raw = fetcher()
        except Exception as e:
            last_error = e
            continue
        if raw is None or raw.empty:
            continue
        code_col = _pick_column(raw, ["成分券代码", "品种代码", "证券代码", "代码", "con_code", "code", "symbol"])
        weight_col = _pick_column(raw, ["权重", "权重(%)", "weight", "i_weight"])
        if code_col is None or weight_col is None:
            continue
        df = raw[[code_col, weight_col]].copy()
        df.columns = ["symbol", "benchmark_weight"]
        df["symbol"] = df["symbol"].map(_normalize_symbol)
        df["benchmark_weight"] = pd.to_numeric(df["benchmark_weight"], errors="coerce")
        df = df.dropna(subset=["symbol", "benchmark_weight"])
        if df.empty:
            continue
        if df["benchmark_weight"].sum() > 1.5:
            df["benchmark_weight"] = df["benchmark_weight"] / 100.0
        df["benchmark_weight_source"] = BENCHMARK_WEIGHT_SOURCE
        df["benchmark_symbol"] = benchmark_symbol
        df["updated_at"] = datetime.now().isoformat(timespec="seconds")
        return df
    if last_error is not None:
        raise last_error
    raise RuntimeError("benchmark weight data unavailable")


def fetch_benchmark_weights(benchmark_symbol: str = BENCHMARK_SYMBOL) -> pd.DataFrame:
    """Fetch and cache benchmark constituent weights.

    Falls back to the existing cache if the remote source is unavailable.
    """
    try:
        df = _fetch_akshare_weights(benchmark_symbol)
    except Exception as e:
        logger.warning("Benchmark weight fetch failed for %s: %s", benchmark_symbol, e)
        return _read_cached_weights(benchmark_symbol)

    Path(BENCHMARK_WEIGHT_DIR).mkdir(parents=True, exist_ok=True)
    df.to_csv(_weight_path(benchmark_symbol), index=False, encoding="utf-8-sig")
    return df


def benchmark_weight_map(
    symbols: Iterable[str],
    benchmark_symbol: str = BENCHMARK_SYMBOL,
) -> tuple[dict[str, float], str]:
    normalized = [_normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)]
    if not normalized:
        return {}, "empty_universe"
    if len(normalized) < 20:
        equal = 1.0 / len(normalized)
        return {symbol: equal for symbol in normalized}, "fallback_equal_weight"

    df = fetch_benchmark_weights(benchmark_symbol)
    if not df.empty:
        weights = {
            str(row["symbol"]): float(row["benchmark_weight"])
            for _, row in df.iterrows()
            if str(row["symbol"]) in normalized
        }
        total = sum(weights.values())
        if total > 0:
            return {symbol: weight / total for symbol, weight in weights.items()}, str(
                df.get("benchmark_weight_source", pd.Series([BENCHMARK_WEIGHT_SOURCE])).iloc[0]
            )

    equal = 1.0 / len(normalized)
    return {symbol: equal for symbol in normalized}, "fallback_equal_weight"
