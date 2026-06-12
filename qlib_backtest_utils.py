from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd

from config import (
    AGENT_SIGNAL_DIR,
    FACTOR_DATA_DIR,
    LONG_QUANTILE,
    MAX_TOTAL_POSITION,
    QLIB_ACCOUNT,
    QLIB_BACKTEST_WINDOW,
    QLIB_BENCHMARK,
    QLIB_CLOSE_COST,
    QLIB_ENABLED,
    QLIB_MIN_COST,
    QLIB_OPEN_COST,
    QLIB_BACKTEST_COMPARISON_PATH,
    QLIB_PROVIDER_URI,
    QLIB_REPORT_DIR,
    TRANSFORMER_PREDICTION_PATH,
    TRANSFORMER_LIVE_MIN_SNAPSHOT_DAYS,
    TRANSFORMER_LIVE_PREDICTION_DIR,
    TRANSFORMER_WALK_FORWARD_ENABLED,
    TRANSFORMER_WALK_FORWARD_PREDICTION_PATH,
)


logger = logging.getLogger(__name__)

LOW_SIGNAL_SCORE = -1_000_000.0
REQUIRED_PRICE_FIELDS = ["open", "high", "low", "close", "volume"]
LAST_TRANSFORMER_SIGNAL_METADATA: dict[str, Any] = {}
QLIB_COST_PROFILES = {
    "low": {"open_cost": 0.0002, "close_cost": 0.0007, "min_cost": QLIB_MIN_COST},
    "current": {"open_cost": QLIB_OPEN_COST, "close_cost": QLIB_CLOSE_COST, "min_cost": QLIB_MIN_COST},
    "high": {"open_cost": 0.0010, "close_cost": 0.0025, "min_cost": QLIB_MIN_COST},
}
QLIB_REBALANCE_FREQUENCIES = ["daily", "weekly", "monthly"]


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


def _select_transformer_prediction_file() -> tuple[Path, str]:
    if TRANSFORMER_WALK_FORWARD_ENABLED and TRANSFORMER_WALK_FORWARD_PREDICTION_PATH.exists():
        return TRANSFORMER_WALK_FORWARD_PREDICTION_PATH, "walk_forward"
    return TRANSFORMER_PREDICTION_PATH, "static"


def _prediction_file_metadata(path: Path, mode: str, pred: pd.DataFrame | None = None) -> dict[str, Any]:
    metadata = {
        "prediction_mode": mode,
        "prediction_path": str(path),
        "walk_forward_enabled": bool(TRANSFORMER_WALK_FORWARD_ENABLED),
    }
    if pred is None or pred.empty:
        return metadata
    if mode == "walk_forward":
        metadata["leakage_guard"] = "train_valid_before_test"
    if "fold_id" in pred.columns:
        metadata["fold_count"] = int(pred["fold_id"].nunique())
    if "test_start" in pred.columns:
        years = pd.to_datetime(pred["test_start"], errors="coerce").dt.year.dropna()
        if not years.empty:
            metadata["first_test_year"] = int(years.min())
            metadata["last_test_year"] = int(years.max())
    return metadata


def _transformer_prediction_series() -> pd.Series:
    global LAST_TRANSFORMER_SIGNAL_METADATA
    path, mode = _select_transformer_prediction_file()
    LAST_TRANSFORMER_SIGNAL_METADATA = _prediction_file_metadata(path, mode)
    if not path.exists():
        return pd.Series(dtype=float, name="score")
    try:
        pred = pd.read_csv(path)
    except Exception as e:
        logger.warning("Failed to read transformer predictions: %s", e)
        return pd.Series(dtype=float, name="score")
    if pred.empty or "prediction" not in pred.columns:
        return pd.Series(dtype=float, name="score")
    LAST_TRANSFORMER_SIGNAL_METADATA = _prediction_file_metadata(path, mode, pred)
    if {"date", "symbol"}.issubset(pred.columns):
        pred["datetime"] = pd.to_datetime(pred["date"], errors="coerce")
        pred["instrument"] = pred["symbol"].map(to_qlib_instrument)
    elif {"datetime", "instrument"}.issubset(pred.columns):
        # Backward compatibility for predictions generated before the stable
        # date/symbol/prediction_rank/prediction_zscore schema was introduced.
        pred["datetime"] = pd.to_datetime(pred["datetime"], errors="coerce")
        pred["instrument"] = pred["instrument"].astype(str).str.upper()
    else:
        return pd.Series(dtype=float, name="score")
    pred = pred.dropna(subset=["datetime"])
    return pd.Series(
        pd.to_numeric(pred["prediction"], errors="coerce").fillna(LOW_SIGNAL_SCORE).to_numpy(),
        index=pd.MultiIndex.from_frame(pred[["instrument", "datetime"]]),
        name="score",
    ).sort_index()


def _transformer_live_prediction_series() -> pd.Series:
    global LAST_TRANSFORMER_SIGNAL_METADATA
    prediction_dir = Path(TRANSFORMER_LIVE_PREDICTION_DIR)
    paths = sorted(prediction_dir.glob("*.csv"))
    LAST_TRANSFORMER_SIGNAL_METADATA = {
        "prediction_mode": "live_snapshot",
        "prediction_dir": str(prediction_dir),
        "snapshot_day_count": 0,
        "live_min_snapshot_days": int(TRANSFORMER_LIVE_MIN_SNAPSHOT_DAYS),
    }
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("Failed to read live transformer snapshot %s: %s", path, e)
            continue
        if df.empty or "prediction" not in df.columns:
            continue
        if "date" not in df.columns:
            df["date"] = path.stem
        frames.append(df)
    if not frames:
        return pd.Series(dtype=float, name="score")

    pred = pd.concat(frames, ignore_index=True)
    if {"date", "symbol"}.issubset(pred.columns):
        pred["datetime"] = pd.to_datetime(pred["date"], errors="coerce")
        pred["instrument"] = pred["symbol"].map(to_qlib_instrument)
    elif {"datetime", "instrument"}.issubset(pred.columns):
        pred["datetime"] = pd.to_datetime(pred["datetime"], errors="coerce")
        pred["instrument"] = pred["instrument"].astype(str).str.upper()
    else:
        return pd.Series(dtype=float, name="score")
    pred = pred.dropna(subset=["datetime", "instrument"])
    if pred.empty:
        return pd.Series(dtype=float, name="score")
    pred = pred.sort_values(["datetime", "instrument", "generated_at"] if "generated_at" in pred.columns else ["datetime", "instrument"])
    pred = pred.drop_duplicates(["datetime", "instrument"], keep="last")
    dates = pd.DatetimeIndex(pred["datetime"].dropna().unique()).sort_values()
    LAST_TRANSFORMER_SIGNAL_METADATA = {
        "prediction_mode": "live_snapshot",
        "prediction_dir": str(prediction_dir),
        "snapshot_day_count": int(len(dates)),
        "first_snapshot_date": dates[0].date().isoformat() if len(dates) else None,
        "last_snapshot_date": dates[-1].date().isoformat() if len(dates) else None,
        "live_min_snapshot_days": int(TRANSFORMER_LIVE_MIN_SNAPSHOT_DAYS),
    }
    if len(dates) < int(TRANSFORMER_LIVE_MIN_SNAPSHOT_DAYS):
        return pd.Series(dtype=float, name="score")
    return pd.Series(
        pd.to_numeric(pred["prediction"], errors="coerce").fillna(LOW_SIGNAL_SCORE).to_numpy(),
        index=pd.MultiIndex.from_frame(pred[["instrument", "datetime"]]),
        name="score",
    ).sort_index()


def _hybrid_signal(rule_signal: pd.Series, transformer_signal: pd.Series) -> pd.Series:
    combined = pd.concat(
        [rule_signal.rename("rule"), transformer_signal.rename("transformer")],
        axis=1,
    ).dropna()
    if combined.empty:
        return pd.Series(dtype=float, name="score")
    rule = combined["rule"].where(combined["rule"] > LOW_SIGNAL_SCORE / 2, np.nan)
    rule_z = (rule - rule.mean()) / rule.std() if rule.std() and not pd.isna(rule.std()) else rule * 0
    transformer_z = (combined["transformer"] - combined["transformer"].mean()) / combined["transformer"].std()
    transformer_z = transformer_z.replace([np.inf, -np.inf], 0).fillna(0)
    return (rule_z.fillna(LOW_SIGNAL_SCORE) * 0.6 + transformer_z * 0.4).rename("score").sort_index()


def _agent_signal_series() -> pd.Series:
    records = []
    paths = sorted(Path(AGENT_SIGNAL_DIR).glob("*.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read agent signal snapshot %s: %s", path, e)
            continue
        snapshot_date = payload.get("date") or path.stem
        trade_date = pd.to_datetime(snapshot_date, errors="coerce")
        if pd.isna(trade_date):
            continue
        signals = payload.get("signals")
        if signals is None and isinstance(payload, list):
            signals = payload
        if not isinstance(signals, list):
            continue
        for item in signals:
            if not isinstance(item, dict):
                continue
            symbol = item.get("stock_code") or item.get("symbol")
            if not symbol:
                continue
            action = str(item.get("action") or "hold").lower()
            weight = pd.to_numeric(item.get("weight"), errors="coerce")
            score = float(weight) if action == "buy" and pd.notna(weight) and float(weight) > 0 else LOW_SIGNAL_SCORE
            records.append((to_qlib_instrument(str(symbol)), pd.Timestamp(trade_date), score))

    if not records:
        return pd.Series(dtype=float, name="score")
    dates = {item[1] for item in records}
    if len(dates) < 2:
        return pd.Series(dtype=float, name="score")
    return pd.Series(
        [item[2] for item in records],
        index=pd.MultiIndex.from_tuples(
            [(item[0], item[1]) for item in records],
            names=["instrument", "datetime"],
        ),
        name="score",
    ).sort_index()


def build_daily_signal_series(window: int = QLIB_BACKTEST_WINDOW, signal_source: str = "rule") -> pd.Series:
    if signal_source == "agents":
        return _agent_signal_series()

    frames = _read_factor_files()
    records = []
    for instrument, df in frames.items():
        score_column = "cross_section_score" if "cross_section_score" in df.columns else "signal_score"
        if score_column not in df.columns:
            continue
        latest_dates = pd.DatetimeIndex(df["date"].dropna().sort_values().unique())[-window:]
        df = df[df["date"].isin(latest_dates)].copy()
        rule_scores = pd.to_numeric(df[score_column], errors="coerce")
        if "risk_flag" in df.columns:
            normal_risk = df["risk_flag"].astype(str).eq("normal")
            rule_scores = rule_scores.where(normal_risk, LOW_SIGNAL_SCORE)
        for trade_date, score in zip(df["date"], rule_scores.fillna(LOW_SIGNAL_SCORE)):
            records.append((instrument, pd.Timestamp(trade_date), float(score)))

    if not records:
        return pd.Series(dtype=float, name="score")
    rule_signal = pd.Series(
        [item[2] for item in records],
        index=pd.MultiIndex.from_tuples(
            [(item[0], item[1]) for item in records],
            names=["instrument", "datetime"],
        ),
        name="score",
    ).sort_index()
    if signal_source == "rule":
        return rule_signal

    if signal_source in {"transformer_live", "hybrid_live"}:
        live_signal = _transformer_live_prediction_series()
        if live_signal.empty:
            return pd.Series(dtype=float, name="score")
        if signal_source == "transformer_live":
            return live_signal
        return _hybrid_signal(rule_signal, live_signal)

    transformer_signal = _transformer_prediction_series()
    if transformer_signal.empty:
        return pd.Series(dtype=float, name="score")
    if signal_source == "transformer":
        return transformer_signal
    if signal_source == "hybrid":
        return _hybrid_signal(rule_signal, transformer_signal)
    raise ValueError(f"Unsupported qlib signal_source: {signal_source}")


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
    if qlib_report.get("status") == "ok" and qlib_report.get("engine") == "qlib_comparison":
        rows = [
            ["mode", "comparison"],
            ["prediction_mode", qlib_report.get("prediction_mode")],
            ["walk_forward_enabled", qlib_report.get("walk_forward_enabled")],
            ["leakage_guard", qlib_report.get("leakage_guard")],
            ["fold_count", qlib_report.get("fold_count")],
            ["first_test_year", qlib_report.get("first_test_year")],
            ["last_test_year", qlib_report.get("last_test_year")],
            ["snapshot_day_count", qlib_report.get("snapshot_day_count")],
            ["first_snapshot_date", qlib_report.get("first_snapshot_date")],
            ["last_snapshot_date", qlib_report.get("last_snapshot_date")],
            ["live_min_snapshot_days", qlib_report.get("live_min_snapshot_days")],
        ]
        for source, report in qlib_report.get("signal_reports", {}).items():
            rows.extend(
                [
                    [f"{source}_status", report.get("status")],
                    [f"{source}_reason", report.get("reason", "")],
                    [f"{source}_total_return", _fmt_percent(report.get("total_return_percent"))],
                    [f"{source}_max_drawdown", _fmt_percent(report.get("max_drawdown_percent"))],
                    [f"{source}_sharpe", report.get("sharpe")],
                ]
            )
        return rows
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


def _save_qlib_report(report: dict[str, Any]) -> dict[str, Any]:
    QLIB_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = QLIB_BACKTEST_COMPARISON_PATH if report.get("engine") == "qlib_comparison" else QLIB_REPORT_DIR / f"qlib_backtest_{report.get('signal_source', 'rule')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(path)
    return report


def _portfolio_topk(signal: pd.Series) -> int:
    dates = signal.index.get_level_values("datetime")
    latest_date = dates.max()
    count = int((dates == latest_date).sum())
    return max(1, int(round(count * float(LONG_QUANTILE))))


def _apply_rebalance_frequency(signal: pd.Series, frequency: str) -> pd.Series:
    if frequency == "daily" or signal.empty:
        return signal
    frame = signal.rename("score").reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values(["datetime", "instrument"])
    if frame.empty:
        return signal
    dates = pd.DatetimeIndex(frame["datetime"].drop_duplicates().sort_values())
    if frequency == "weekly":
        periods = dates.to_period("W")
    elif frequency == "monthly":
        periods = dates.to_period("M")
    else:
        raise ValueError(f"Unsupported rebalance_frequency: {frequency}")
    rebalance_dates = []
    seen = set()
    for date, period in zip(dates, periods):
        if period in seen:
            continue
        seen.add(period)
        rebalance_dates.append(date)
    pivot = frame.pivot_table(index="datetime", columns="instrument", values="score", aggfunc="last")
    pivot = pivot.reindex(dates)
    rebalanced = pivot.loc[rebalance_dates].reindex(dates).ffill()
    out = rebalanced.stack(dropna=True).rename("score")
    out.index = out.index.set_names(["datetime", "instrument"])
    return out.reorder_levels(["instrument", "datetime"]).sort_index()


def _run_single_qlib_backtest(
    signal_source: str = "rule",
    rebalance_frequency: str = "daily",
    cost_profile: str = "current",
    save_report: bool = True,
) -> dict[str, Any]:
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

    signal = build_daily_signal_series(signal_source=signal_source)
    if signal.empty:
        result = {"status": "skipped", "reason": f"empty daily qlib signal for {signal_source}", "signal_source": signal_source}
        if signal_source in {"transformer_live", "hybrid_live"}:
            result.update(LAST_TRANSFORMER_SIGNAL_METADATA)
            result["reason"] = (
                f"not enough live transformer prediction snapshots "
                f"({LAST_TRANSFORMER_SIGNAL_METADATA.get('snapshot_day_count', 0)}/"
                f"{TRANSFORMER_LIVE_MIN_SNAPSHOT_DAYS})"
            )
        return result
    signal = _apply_rebalance_frequency(signal, rebalance_frequency)
    topk = _portfolio_topk(signal)
    costs = QLIB_COST_PROFILES.get(cost_profile, QLIB_COST_PROFILES["current"])

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
            topk=topk,
            n_drop=topk,
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
                "open_cost": costs["open_cost"],
                "close_cost": costs["close_cost"],
                "min_cost": costs["min_cost"],
                "limit_threshold": None,
                "trade_unit": None,
            },
        )
        result = summarize_qlib_backtest(report, None, indicators, data_info)
        result["signal_source"] = signal_source
        result["rebalance_frequency"] = rebalance_frequency
        result["cost_profile"] = cost_profile
        result["costs"] = costs
        result["topk"] = topk
        if signal_source in {"transformer", "hybrid", "transformer_live", "hybrid_live"}:
            result.update(LAST_TRANSFORMER_SIGNAL_METADATA)
        return _save_qlib_report(result) if save_report else result
    except Exception as e:
        logger.exception("Qlib backtest failed: %s", e)
        return {
            "status": "skipped",
            "reason": str(e),
            "engine": "qlib",
            "signal_source": signal_source,
            "provider_uri": str(QLIB_PROVIDER_URI),
            "rebalance_frequency": rebalance_frequency,
            "cost_profile": cost_profile,
        }


def run_rebalance_cost_comparison(signal_source: str = "rule") -> list[dict[str, Any]]:
    rows = []
    for frequency in QLIB_REBALANCE_FREQUENCIES:
        for profile in ["low", "current", "high"]:
            report = _run_single_qlib_backtest(
                signal_source=signal_source,
                rebalance_frequency=frequency,
                cost_profile=profile,
                save_report=False,
            )
            rows.append(
                {
                    "signal_source": signal_source,
                    "rebalance_frequency": frequency,
                    "cost_profile": profile,
                    "status": report.get("status"),
                    "reason": report.get("reason"),
                    "total_return_percent": report.get("total_return_percent"),
                    "annualized_return_percent": report.get("annualized_return_percent"),
                    "max_drawdown_percent": report.get("max_drawdown_percent"),
                    "sharpe": report.get("sharpe"),
                    "turnover_rate": report.get("turnover_rate"),
                    "benchmark_comparison": report.get("benchmark_comparison"),
                    "costs": report.get("costs"),
                    "topk": report.get("topk"),
                }
            )
    return rows


def run_qlib_backtest(signal_source: str = "comparison") -> dict[str, Any]:
    if signal_source != "comparison":
        return _run_single_qlib_backtest(signal_source)

    reports = {
        source: _run_single_qlib_backtest(source)
        for source in ["rule", "transformer", "hybrid", "transformer_live", "hybrid_live", "agents"]
    }
    ok_reports = {source: report for source, report in reports.items() if report.get("status") == "ok"}
    if not ok_reports:
        return {
            "status": "skipped",
            "engine": "qlib_comparison",
            "reason": "all qlib signal-source backtests skipped",
            "signal_reports": reports,
        }
    report = {
        "status": "ok",
        "engine": "qlib_comparison",
        "signal_reports": reports,
        "primary_signal_source": "hybrid_live"
        if reports.get("hybrid_live", {}).get("status") == "ok"
        else "hybrid"
        if reports.get("hybrid", {}).get("status") == "ok"
        else "rule",
        "rebalance_cost_comparison": run_rebalance_cost_comparison("rule"),
    }
    primary = reports.get(report["primary_signal_source"], {})
    for key in [
        "benchmark",
        "benchmark_source",
        "backtest_window_days",
        "start_date",
        "end_date",
        "account",
        "total_return_percent",
        "annualized_return_percent",
        "annualized_volatility_percent",
        "max_drawdown_percent",
        "sharpe",
        "turnover_rate",
        "positions",
        "trade_records",
        "benchmark_comparison",
        "prediction_mode",
        "prediction_path",
        "prediction_dir",
        "walk_forward_enabled",
        "leakage_guard",
        "fold_count",
        "first_test_year",
        "last_test_year",
        "snapshot_day_count",
        "first_snapshot_date",
        "last_snapshot_date",
        "live_min_snapshot_days",
    ]:
        if key in primary:
            report[key] = primary[key]
    return _save_qlib_report(report)
