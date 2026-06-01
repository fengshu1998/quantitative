from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd

from config import (
    FACTOR_DATA_DIR,
    MAX_TOTAL_POSITION,
    QLIB_ACCOUNT,
    QLIB_BACKTEST_WINDOW,
    QLIB_BENCHMARK,
    QLIB_CLOSE_COST,
    QLIB_ENABLED,
    QLIB_MIN_COST,
    QLIB_OPEN_COST,
    QLIB_PROVIDER_URI,
    TOP_N,
)


logger = logging.getLogger(__name__)

LOW_SIGNAL_SCORE = -1_000_000.0
REQUIRED_PRICE_FIELDS = ["open", "high", "low", "close", "volume"]


def to_qlib_instrument(symbol: str) -> str:
    symbol = str(symbol).strip()
    if len(symbol) >= 8 and symbol[:2].lower() in {"sh", "sz"}:
        return f"{symbol[:2].upper()}{symbol[2:]}"
    return symbol.upper()


def from_qlib_instrument(instrument: str) -> str:
    instrument = str(instrument).strip()
    if len(instrument) >= 8 and instrument[:2].upper() in {"SH", "SZ"}:
        return f"{instrument[:2].lower()}{instrument[2:]}"
    return instrument.lower()


def _read_factor_files() -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(Path(FACTOR_DATA_DIR).glob("*.csv")):
        symbol = path.stem
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("Failed to read factor file %s: %s", path, e)
            continue
        if df.empty or "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        if all(col in df.columns for col in REQUIRED_PRICE_FIELDS):
            frames[to_qlib_instrument(symbol)] = df
    return frames


def _calendar_from_frames(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    dates = sorted({date for df in frames.values() for date in df["date"].dropna().tolist()})
    if not dates:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(dates)


def _write_feature_bin(path: Path, values: pd.Series, calendar: pd.DatetimeIndex):
    values = values.reindex(calendar)
    first_valid = values.first_valid_index()
    if first_valid is None:
        return
    start_index = int(calendar.get_loc(first_valid))
    data = values.iloc[start_index:].astype("float32").to_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.hstack([[start_index], data]).astype("<f").tofile(path)


def _feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["factor"] = 1.0
    data["change"] = pd.to_numeric(data["close"], errors="coerce").pct_change().fillna(0.0)
    return data


def _fetch_hs300_benchmark(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    try:
        raw = ak.stock_zh_index_daily(symbol="sh000300")
    except Exception as e:
        logger.warning("Failed to fetch HS300 benchmark from AkShare: %s", e)
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    raw = raw.rename(
        columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    if not all(col in raw.columns for col in ["date", "open", "high", "low", "close"]):
        return pd.DataFrame()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date")
    raw = raw[raw["date"].isin(calendar)]
    if "volume" not in raw.columns:
        raw["volume"] = 0
    return raw[["date", "open", "high", "low", "close", "volume"]]


def _synthetic_benchmark(frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    closes = []
    for instrument, df in frames.items():
        series = df.set_index("date")["close"].reindex(calendar).astype(float)
        closes.append(series.rename(instrument))
    if not closes:
        return pd.DataFrame()
    close = pd.concat(closes, axis=1).pct_change().mean(axis=1).fillna(0)
    close = 1000 * (1 + close).cumprod()
    return pd.DataFrame(
        {
            "date": calendar,
            "open": close.values,
            "high": close.values,
            "low": close.values,
            "close": close.values,
            "volume": 0,
        }
    )


def prepare_qlib_data() -> dict[str, Any]:
    frames = _read_factor_files()
    if not frames:
        return {"status": "skipped", "reason": "no factor csv files available"}

    calendar = _calendar_from_frames(frames)
    if calendar.empty:
        return {"status": "skipped", "reason": "empty qlib calendar"}

    provider = Path(QLIB_PROVIDER_URI)
    calendars_dir = provider / "calendars"
    instruments_dir = provider / "instruments"
    features_dir = provider / "features"
    calendars_dir.mkdir(parents=True, exist_ok=True)
    instruments_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    (calendars_dir / "day.txt").write_text(
        "\n".join(date.strftime("%Y-%m-%d") for date in calendar) + "\n",
        encoding="utf-8",
    )
    future_calendar = calendar.append(pd.DatetimeIndex([calendar[-1] + pd.offsets.BDay(1)]))
    (calendars_dir / "day_future.txt").write_text(
        "\n".join(date.strftime("%Y-%m-%d") for date in future_calendar) + "\n",
        encoding="utf-8",
    )

    benchmark_df = _fetch_hs300_benchmark(calendar)
    benchmark_source = "akshare"
    if benchmark_df.empty:
        benchmark_df = _synthetic_benchmark(frames, calendar)
        benchmark_source = "synthetic_equal_weight"
    if not benchmark_df.empty:
        frames[QLIB_BENCHMARK] = benchmark_df

    instrument_lines = []
    for instrument, df in frames.items():
        start = pd.to_datetime(df["date"].min()).strftime("%Y-%m-%d")
        end = pd.to_datetime(df["date"].max()).strftime("%Y-%m-%d")
        instrument_lines.append(f"{instrument}\t{start}\t{end}")

        feature_df = _feature_columns(df).set_index("date")
        for field in ["open", "high", "low", "close", "volume", "factor", "change"]:
            series = pd.to_numeric(feature_df[field], errors="coerce")
            _write_feature_bin(features_dir / instrument.lower() / f"{field}.day.bin", series, calendar)

    instrument_text = "\n".join(sorted(instrument_lines)) + "\n"
    for name in ["all", "csi300"]:
        (instruments_dir / f"{name}.txt").write_text(instrument_text, encoding="utf-8")

    return {
        "status": "ok",
        "provider_uri": str(provider),
        "instrument_count": len(frames),
        "calendar_count": len(calendar),
        "start_date": calendar.min().date().isoformat(),
        "end_date": calendar.max().date().isoformat(),
        "benchmark": QLIB_BENCHMARK,
        "benchmark_source": benchmark_source,
    }


def build_daily_signal_series(window: int = QLIB_BACKTEST_WINDOW) -> pd.Series:
    frames = _read_factor_files()
    records = []
    for instrument, df in frames.items():
        if "signal_score" not in df.columns:
            continue
        latest_dates = pd.DatetimeIndex(df["date"].dropna().sort_values().unique())[-window:]
        df = df[df["date"].isin(latest_dates)].copy()
        signal_ok = (
            df.get("signal", pd.Series(index=df.index, dtype=object)).astype(str).str.upper().eq("BUY")
            & df.get("risk_flag", pd.Series(index=df.index, dtype=object)).astype(str).eq("normal")
        )
        scores = pd.to_numeric(df["signal_score"], errors="coerce").where(signal_ok, LOW_SIGNAL_SCORE)
        for trade_date, score in zip(df["date"], scores):
            records.append((instrument, pd.Timestamp(trade_date), float(score)))

    if not records:
        return pd.Series(dtype=float, name="score")
    signal = pd.Series(
        [item[2] for item in records],
        index=pd.MultiIndex.from_tuples(
            [(item[0], item[1]) for item in records],
            names=["instrument", "datetime"],
        ),
        name="score",
    ).sort_index()
    return signal


def _flatten_metric_frame(metric: Any) -> dict[str, Any]:
    if isinstance(metric, tuple):
        metric = metric[0]
    if isinstance(metric, pd.DataFrame):
        if metric.empty:
            return {}
        row = metric.iloc[-1]
        return {str(k): _safe_number(v) for k, v in row.items()}
    if isinstance(metric, pd.Series):
        return {str(k): _safe_number(v) for k, v in metric.items()}
    if isinstance(metric, dict):
        return {str(k): _safe_number(v) for k, v in metric.items()}
    return {}


def _safe_number(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return round(float(value), 6)
    return str(value)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return _safe_number(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _metric_value(metrics: dict[str, Any], names: list[str], default=None):
    lower = {str(k).lower(): v for k, v in metrics.items()}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return default


def _extract_qlib_report(report: Any) -> tuple[pd.DataFrame | None, Any]:
    if isinstance(report, dict) and "1day" in report:
        value = report["1day"]
        if isinstance(value, tuple) and len(value) >= 2:
            return value[0], value[1]
        if isinstance(value, pd.DataFrame):
            return value, None
    if isinstance(report, tuple) and len(report) >= 2 and isinstance(report[0], pd.DataFrame):
        return report[0], report[1]
    if isinstance(report, pd.DataFrame):
        return report, None
    return None, None


def _extract_indicator_frame(indicators: Any) -> pd.DataFrame | None:
    if isinstance(indicators, dict) and "1day" in indicators:
        value = indicators["1day"]
        if isinstance(value, tuple) and len(value) >= 1 and isinstance(value[0], pd.DataFrame):
            return value[0]
        if isinstance(value, pd.DataFrame):
            return value
    if isinstance(indicators, tuple) and len(indicators) >= 1 and isinstance(indicators[0], pd.DataFrame):
        return indicators[0]
    if isinstance(indicators, pd.DataFrame):
        return indicators
    return None


def _performance_from_account(report_df: pd.DataFrame | None) -> dict[str, Any]:
    if report_df is None or report_df.empty or "account" not in report_df.columns:
        return {}

    account = pd.to_numeric(report_df["account"], errors="coerce").dropna()
    returns = pd.to_numeric(report_df.get("return"), errors="coerce").dropna()
    if account.empty:
        return {}

    total_return = account.iloc[-1] / account.iloc[0] - 1
    annualized_return = (1 + total_return) ** (252 / max(len(account), 1)) - 1
    annualized_volatility = returns.std() * (252**0.5) if not returns.empty else None
    sharpe = annualized_return / annualized_volatility if annualized_volatility and annualized_volatility > 0 else None
    drawdown = account / account.cummax() - 1

    bench_return = None
    if "bench" in report_df.columns:
        bench_daily = pd.to_numeric(report_df["bench"], errors="coerce").fillna(0)
        bench_return = (1 + bench_daily).prod() - 1

    return {
        "total_return_percent": round(total_return * 100, 2),
        "annualized_return_percent": round(annualized_return * 100, 2),
        "annualized_volatility_percent": round(annualized_volatility * 100, 2)
        if annualized_volatility is not None
        else None,
        "max_drawdown_percent": round(drawdown.min() * 100, 2),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "benchmark_return_percent": round(bench_return * 100, 2) if bench_return is not None else None,
        "excess_return_percent": round((total_return - bench_return) * 100, 2)
        if bench_return is not None
        else None,
    }


def summarize_qlib_backtest(
    report: Any,
    positions: Any,
    indicators: Any,
    data_info: dict[str, Any],
) -> dict[str, Any]:
    report_df, report_positions = _extract_qlib_report(report)
    indicator_df = _extract_indicator_frame(indicators)
    if positions is None:
        positions = report_positions

    perf = _performance_from_account(report_df)
    indicator_metrics = _indicator_summary(indicator_df)

    return {
        "status": "ok",
        "engine": "qlib",
        "provider_uri": data_info.get("provider_uri"),
        "benchmark": data_info.get("benchmark", QLIB_BENCHMARK),
        "benchmark_source": data_info.get("benchmark_source"),
        "backtest_window_days": QLIB_BACKTEST_WINDOW,
        "start_date": data_info.get("backtest_start_date"),
        "end_date": data_info.get("backtest_end_date"),
        "account": QLIB_ACCOUNT,
        "total_return_percent": perf.get("total_return_percent"),
        "annualized_return_percent": perf.get("annualized_return_percent"),
        "annualized_volatility_percent": perf.get("annualized_volatility_percent"),
        "max_drawdown_percent": perf.get("max_drawdown_percent"),
        "sharpe": perf.get("sharpe"),
        "turnover_rate": indicator_metrics.get("turnover_rate"),
        "raw_portfolio_metrics": _report_tail_records(report_df),
        "raw_indicator_metrics": indicator_metrics,
        "positions": _positions_to_records(positions),
        "trade_records": _indicator_to_records(indicator_df),
        "benchmark_comparison": {
            "benchmark": data_info.get("benchmark", QLIB_BENCHMARK),
            "benchmark_return_percent": perf.get("benchmark_return_percent"),
            "excess_return_percent": perf.get("excess_return_percent"),
        },
    }


def _indicator_summary(indicator_df: pd.DataFrame | None) -> dict[str, Any]:
    if indicator_df is None or indicator_df.empty:
        return {}
    summary = {}
    if "deal_amount" in indicator_df.columns and "value" in indicator_df.columns:
        value = pd.to_numeric(indicator_df["value"], errors="coerce").replace(0, np.nan)
        deal = pd.to_numeric(indicator_df["deal_amount"], errors="coerce")
        summary["turnover_rate"] = round(float((deal / value).replace([np.inf, -np.inf], np.nan).mean()), 6)
    for col in ["deal_amount", "value", "count"]:
        if col in indicator_df.columns:
            summary[f"avg_{col}"] = round(float(pd.to_numeric(indicator_df[col], errors="coerce").mean()), 6)
    return summary


def _report_tail_records(report_df: pd.DataFrame | None, limit: int = 5) -> list[dict]:
    if report_df is None or report_df.empty:
        return []
    return _json_ready(report_df.tail(limit).reset_index().to_dict("records"))


def _as_percent(value):
    if value is None:
        return None
    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return None


def _positions_to_records(positions: Any, limit: int = 10) -> list[dict]:
    if isinstance(positions, pd.DataFrame):
        return _json_ready(positions.tail(limit).reset_index().to_dict("records"))
    if isinstance(positions, pd.Series):
        return _json_ready(positions.tail(limit).reset_index().to_dict("records"))
    if isinstance(positions, dict):
        records = []
        for key, value in list(positions.items())[-limit:]:
            records.append({"datetime": str(key), "position": str(value)})
        return records
    return []


def _indicator_to_records(indicators: Any, limit: int = 10) -> list[dict]:
    if isinstance(indicators, pd.DataFrame):
        return _json_ready(indicators.tail(limit).reset_index().to_dict("records"))
    if isinstance(indicators, pd.Series):
        return _json_ready(indicators.tail(limit).reset_index().to_dict("records"))
    return []


def build_qlib_report_rows(qlib_report: dict[str, Any]) -> list[list[Any]]:
    if qlib_report.get("status") != "ok":
        return [["status", qlib_report.get("status", "unknown")], ["reason", qlib_report.get("reason", "")]]
    return [
        ["benchmark", qlib_report.get("benchmark")],
        ["benchmark_source", qlib_report.get("benchmark_source")],
        ["window", qlib_report.get("backtest_window_days")],
        ["total_return", _fmt_percent(qlib_report.get("total_return_percent"))],
        ["annualized_return", _fmt_percent(qlib_report.get("annualized_return_percent"))],
        ["max_drawdown", _fmt_percent(qlib_report.get("max_drawdown_percent"))],
        ["sharpe", qlib_report.get("sharpe")],
        ["turnover_rate", qlib_report.get("turnover_rate")],
    ]


def _fmt_percent(value):
    if value is None:
        return "N/A"
    return f"{float(value):.2f}%"


def run_qlib_backtest() -> dict[str, Any]:
    if not QLIB_ENABLED:
        return {"status": "skipped", "reason": "QLIB_ENABLED is False"}

    try:
        import qlib
        from qlib.backtest import backtest
        from qlib.backtest.executor import SimulatorExecutor
        from qlib.constant import REG_CN
        from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    except Exception as e:
        return {"status": "skipped", "reason": f"qlib unavailable: {e}"}

    data_info = prepare_qlib_data()
    if data_info.get("status") != "ok":
        return data_info

    signal = build_daily_signal_series()
    if signal.empty:
        return {"status": "skipped", "reason": "empty daily qlib signal"}

    dates = signal.index.get_level_values("datetime").unique().sort_values()
    if len(dates) < 2:
        return {"status": "skipped", "reason": "not enough dates for qlib backtest"}
    start_date = dates[-QLIB_BACKTEST_WINDOW].date().isoformat() if len(dates) >= QLIB_BACKTEST_WINDOW else dates[0].date().isoformat()
    end_date = dates[-1].date().isoformat()
    data_info["backtest_start_date"] = start_date
    data_info["backtest_end_date"] = end_date

    try:
        qlib.init(provider_uri=str(QLIB_PROVIDER_URI), region=REG_CN, skip_if_reg=False)
        strategy = TopkDropoutStrategy(
            signal=signal,
            topk=TOP_N,
            n_drop=TOP_N,
            risk_degree=MAX_TOTAL_POSITION,
            only_tradable=False,
            forbid_all_trade_at_limit=False,
        )
        executor = SimulatorExecutor(
            time_per_step="day",
            generate_portfolio_metrics=True,
            verbose=False,
        )
        report, indicators = backtest(
            start_time=start_date,
            end_time=end_date,
            strategy=strategy,
            executor=executor,
            benchmark=QLIB_BENCHMARK,
            account=QLIB_ACCOUNT,
            exchange_kwargs={
                "freq": "day",
                "deal_price": "close",
                "open_cost": QLIB_OPEN_COST,
                "close_cost": QLIB_CLOSE_COST,
                "min_cost": QLIB_MIN_COST,
                "limit_threshold": None,
                "trade_unit": None,
            },
        )
        return summarize_qlib_backtest(report, None, indicators, data_info)
    except Exception as e:
        logger.exception("Qlib backtest failed: %s", e)
        return {
            "status": "skipped",
            "reason": str(e),
            "engine": "qlib",
            "provider_uri": str(QLIB_PROVIDER_URI),
        }
