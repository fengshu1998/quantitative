from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except Exception:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

import pandas as pd

from config import (
    FACTOR_DATA_DIR,
    TRAINING_LOOKBACK_DAYS,
    TRANSFORMER_FEATURE_MIN_ABS_RANKIC,
    TRANSFORMER_FEATURE_MIN_STD,
    TRANSFORMER_FEATURE_MISSING_MAX,
    TRANSFORMER_FEATURE_NORMALIZATION,
    TRANSFORMER_FEATURE_WINSORIZE_LOWER,
    TRANSFORMER_FEATURE_WINSORIZE_UPPER,
    TRANSFORMER_LABEL_HORIZON,
    TRANSFORMER_LABEL_MODE,
    TRANSFORMER_MODEL_PATH,
    TRANSFORMER_MODEL_DIR,
    TRANSFORMER_PREDICTION_PATH,
    TRANSFORMER_INFERENCE_ENABLED,
    TRANSFORMER_LIVE_PREDICTION_DIR,
    TRANSFORMER_LIVE_MODEL_PATH,
    TRANSFORMER_LIVE_TRAINING_ENABLED,
    TRANSFORMER_LIVE_TRAIN_YEARS,
    TRANSFORMER_LIVE_VALID_YEARS,
    TRANSFORMER_RETRAIN_ON_DAILY_RUN,
    TRANSFORMER_WALK_FORWARD_END_YEAR,
    TRANSFORMER_WALK_FORWARD_MODEL_DIR,
    TRANSFORMER_WALK_FORWARD_PREDICTION_PATH,
    TRANSFORMER_WALK_FORWARD_START_YEAR,
    TRANSFORMER_WALK_FORWARD_TEST_YEARS,
    TRANSFORMER_WALK_FORWARD_TRAIN_YEARS,
    TRANSFORMER_WALK_FORWARD_VALID_YEARS,
)
from data_utils import fetch_akshare_data, fetch_index_constituents
from factor_utils import compute_market_factors
from qlib_backtest_utils import from_qlib_instrument, to_qlib_instrument
from signal_utils import generate_signal
from storage_utils import save_dataframe


logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
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


def _raw_label_col(horizon: int = TRANSFORMER_LABEL_HORIZON) -> str:
    return f"label_{horizon}d_forward_return"


def _label_end_col(horizon: int = TRANSFORMER_LABEL_HORIZON) -> str:
    return f"label_{horizon}d_end_datetime"


def _cs_excess_label_col(horizon: int = TRANSFORMER_LABEL_HORIZON) -> str:
    return f"label_{horizon}d_cs_excess_return"


def _rank_label_col(horizon: int = TRANSFORMER_LABEL_HORIZON) -> str:
    return f"label_{horizon}d_rank_pct"


def _training_label_col() -> str:
    if TRANSFORMER_LABEL_MODE == "raw_return":
        return _raw_label_col()
    if TRANSFORMER_LABEL_MODE == "rank_pct":
        return _rank_label_col()
    return _cs_excess_label_col()


def _feature_name(column: str) -> str:
    prefix = "cs_rank" if TRANSFORMER_FEATURE_NORMALIZATION == "cross_sectional_rank" else "cs_z"
    return f"{prefix}_{column}"


def _date_text(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def ensure_training_factor_data() -> dict[str, Any]:
    """Ensure factor CSVs have enough rows for model research."""
    factor_dir = Path(FACTOR_DATA_DIR)
    factor_dir.mkdir(parents=True, exist_ok=True)
    existing = list(factor_dir.glob("*.csv"))
    min_rows = None
    for path in existing:
        try:
            rows = len(pd.read_csv(path, usecols=["date"]))
        except Exception:
            continue
        min_rows = rows if min_rows is None else min(min_rows, rows)

    if existing and min_rows is not None and min_rows >= TRAINING_LOOKBACK_DAYS:
        return {
            "status": "ok",
            "source": "existing_factor_files",
            "file_count": len(existing),
            "min_rows": min_rows,
        }

    updated = 0
    failed = []
    for stock in fetch_index_constituents():
        symbol = stock["symbol"]
        name = stock.get("name") or symbol
        try:
            raw_df = fetch_akshare_data(symbol, name, TRAINING_LOOKBACK_DAYS)
            if raw_df.empty:
                failed.append({"symbol": symbol, "reason": "empty data"})
                continue
            factor_df, _ = compute_market_factors(raw_df, symbol, name, "stock", TRAINING_LOOKBACK_DAYS)
            factor_df["fundamental_available"] = False
            factor_df["industry"] = "unknown"
            factor_df = generate_signal(factor_df)
            save_dataframe(factor_df, "factors", symbol)
            updated += 1
        except Exception as e:
            logger.warning("Failed to refresh training factors for %s: %s", symbol, e)
            failed.append({"symbol": symbol, "reason": str(e)})

    return {
        "status": "ok" if updated else "skipped",
        "source": "refreshed_from_akshare",
        "updated": updated,
        "failed": failed[:10],
    }


def _load_factor_panel() -> pd.DataFrame:
    frames = []
    for path in sorted(Path(FACTOR_DATA_DIR).glob("*.csv")):
        symbol = path.stem
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("Failed to read %s for training: %s", path, e)
            continue
        if df.empty or "date" not in df.columns or "close" not in df.columns:
            continue
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        df["instrument"] = to_qlib_instrument(symbol)
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
        close = pd.to_numeric(df["close"], errors="coerce")
        for horizon in [1, TRANSFORMER_LABEL_HORIZON, 10]:
            df[f"label_{horizon}d_forward_return"] = close.shift(-horizon) / close - 1
            df[f"label_{horizon}d_end_datetime"] = df["datetime"].shift(-horizon)
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = _add_cross_sectional_labels(panel)
    return panel.sort_values(["datetime", "instrument"])


def _add_cross_sectional_labels(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    raw_col = _raw_label_col()
    cs_col = _cs_excess_label_col()
    rank_col = _rank_label_col()
    if raw_col not in panel.columns:
        return panel

    raw = pd.to_numeric(panel[raw_col], errors="coerce")
    panel[raw_col] = raw
    daily_mean = raw.groupby(panel["datetime"]).transform("mean")
    panel[cs_col] = raw - daily_mean
    panel[rank_col] = raw.groupby(panel["datetime"]).rank(pct=True, method="average")
    return panel


def _winsorize_by_date(values: pd.Series, dates: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    lower = values.groupby(dates).transform(lambda x: x.quantile(TRANSFORMER_FEATURE_WINSORIZE_LOWER))
    upper = values.groupby(dates).transform(lambda x: x.quantile(TRANSFORMER_FEATURE_WINSORIZE_UPPER))
    return values.clip(lower=lower, upper=upper)


def _build_cross_sectional_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    index = pd.MultiIndex.from_frame(panel[["datetime", "instrument"]])
    out = pd.DataFrame(index=index)
    dates = panel["datetime"]
    for col in FEATURE_COLUMNS:
        values = pd.to_numeric(panel[col], errors="coerce") if col in panel.columns else pd.Series(np.nan, index=panel.index)
        clipped = _winsorize_by_date(values, dates)
        if TRANSFORMER_FEATURE_NORMALIZATION == "cross_sectional_rank":
            transformed = clipped.groupby(dates).rank(pct=True, method="average") - 0.5
        else:
            mean = clipped.groupby(dates).transform("mean")
            std = clipped.groupby(dates).transform("std").replace(0, np.nan)
            transformed = (clipped - mean) / std
        out[_feature_name(col)] = transformed.to_numpy()
    return out.replace([np.inf, -np.inf], np.nan)


def _mean_daily_rank_ic(feature: pd.Series, label: pd.Series) -> float | None:
    aligned = pd.concat([feature.rename("feature"), label.rename("label")], axis=1).dropna()
    if aligned.empty:
        return None
    values = []
    for _, group in aligned.groupby(level=0):
        if len(group) < 3 or group["feature"].nunique() <= 1 or group["label"].nunique() <= 1:
            continue
        corr = group["feature"].corr(group["label"], method="spearman")
        if not pd.isna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else None


def _select_transformer_features(
    feature_df: pd.DataFrame,
    label_series: pd.Series,
    train_index,
) -> tuple[list[str], dict[str, Any]]:
    train_mask = feature_df.index.isin(train_index)
    reference = feature_df.loc[train_mask]
    reference_label = label_series.loc[label_series.index.isin(train_index)]
    if reference.empty:
        selected = list(feature_df.columns)
        return selected, {"selected_features": selected, "fallback_reason": "empty_train_reference"}

    missing_rate = reference.isna().mean()
    std = reference.std(skipna=True)
    missing_pass = missing_rate[missing_rate <= TRANSFORMER_FEATURE_MISSING_MAX].index.tolist()
    std_pass = std[std.abs() > TRANSFORMER_FEATURE_MIN_STD].index.tolist()
    base_pass = [col for col in feature_df.columns if col in missing_pass and col in std_pass]

    rank_ic = {}
    rank_pass = []
    for col in base_pass:
        value = _mean_daily_rank_ic(reference[col], reference_label)
        rank_ic[col] = value
        if value is not None and abs(value) >= TRANSFORMER_FEATURE_MIN_ABS_RANKIC:
            rank_pass.append(col)

    selected = rank_pass
    fallback_reason = None
    if not selected:
        selected = base_pass or list(feature_df.columns)
        fallback_reason = "rank_ic_filter_removed_all_features" if base_pass else "missing_or_std_filter_removed_all_features"

    diagnostics = {
        "selected_features": selected,
        "selected_feature_count": len(selected),
        "missing_rate_filter": {
            "threshold": TRANSFORMER_FEATURE_MISSING_MAX,
            "removed": [col for col in feature_df.columns if col not in missing_pass],
        },
        "zero_std_filter": {
            "threshold": TRANSFORMER_FEATURE_MIN_STD,
            "removed": [col for col in feature_df.columns if col not in std_pass],
        },
        "rank_ic_filter": {
            "threshold": TRANSFORMER_FEATURE_MIN_ABS_RANKIC,
            "rank_ic": {key: _round_or_none(value) for key, value in rank_ic.items()},
            "removed": [col for col in base_pass if col not in rank_pass],
        },
    }
    if fallback_reason:
        diagnostics["fallback_reason"] = fallback_reason
    return selected, diagnostics


def _build_transformer_frames(
    panel: pd.DataFrame,
    train_index=None,
    selected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    label_col = _training_label_col()
    if label_col not in panel.columns:
        panel = _add_cross_sectional_labels(panel)

    index = pd.MultiIndex.from_frame(panel[["datetime", "instrument"]])
    feature_df = _build_cross_sectional_feature_frame(panel)
    label_df = pd.DataFrame(panel[[label_col]].to_numpy(), index=index, columns=["label"])
    label_series = label_df["label"]

    if selected_features is None:
        reference_index = train_index if train_index is not None else index
        selected_features, diagnostics = _select_transformer_features(feature_df, label_series, reference_index)
    else:
        missing = [col for col in selected_features if col not in feature_df.columns]
        for col in missing:
            feature_df[col] = np.nan
        diagnostics = {
            "selected_features": selected_features,
            "selected_feature_count": len(selected_features),
            "missing_selected_features": missing,
        }

    feature_df = feature_df.reindex(columns=selected_features)
    feature_scaler = _fit_feature_scaler(feature_df, reference_index=train_index)
    feature_df = _apply_feature_scaler(feature_df, feature_scaler)
    feature_scaler["selected_features"] = list(selected_features)
    feature_scaler["base_features"] = list(FEATURE_COLUMNS)
    feature_scaler["label_mode"] = TRANSFORMER_LABEL_MODE
    feature_scaler["label"] = label_col
    feature_scaler["feature_normalization"] = TRANSFORMER_FEATURE_NORMALIZATION
    feature_scaler["feature_filter"] = diagnostics

    metadata = {
        "label": label_col,
        "label_mode": TRANSFORMER_LABEL_MODE,
        "raw_label": _raw_label_col(),
        "rank_label": _rank_label_col(),
        "feature_normalization": TRANSFORMER_FEATURE_NORMALIZATION,
        "features": list(selected_features),
        "feature_filter": diagnostics,
        "feature_scaler": feature_scaler,
    }
    return feature_df, label_df, metadata


def _fit_feature_scaler(feature_df: pd.DataFrame, reference_index=None) -> dict[str, dict[str, float]]:
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    reference = feature_df if reference_index is None else feature_df.loc[feature_df.index.isin(reference_index)]
    if reference.empty:
        reference = feature_df
    med = reference.median()
    std = reference.std().replace(0, np.nan)
    mean = reference.mean()
    return {
        "median": {key: float(value) for key, value in med.fillna(0.0).items()},
        "mean": {key: float(value) for key, value in mean.fillna(0.0).items()},
        "std": {key: float(value) for key, value in std.fillna(1.0).items()},
    }


def _apply_feature_scaler(feature_df: pd.DataFrame, scaler: dict[str, dict[str, float]]) -> pd.DataFrame:
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    median = pd.Series(scaler.get("median", {})).reindex(feature_df.columns).fillna(0.0)
    mean = pd.Series(scaler.get("mean", {})).reindex(feature_df.columns).fillna(0.0)
    std = pd.Series(scaler.get("std", {})).reindex(feature_df.columns).replace(0, np.nan).fillna(1.0)
    feature_df = feature_df.fillna(median).fillna(0.0)
    feature_df = (feature_df - mean) / std
    return feature_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _standardize_features(feature_df: pd.DataFrame, reference_index=None) -> pd.DataFrame:
    return _apply_feature_scaler(feature_df, _fit_feature_scaler(feature_df, reference_index=reference_index))


def _live_feature_scaler_path() -> Path:
    return TRANSFORMER_MODEL_DIR / "transformer_live_feature_scaler.json"


def _segments_from_dates(dates: pd.DatetimeIndex) -> dict[str, tuple[str, str]]:
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    n = len(dates)
    train_end = max(int(n * 0.70) - 1, 0)
    valid_end = max(int(n * 0.85) - 1, train_end + 1)
    valid_end = min(valid_end, n - 2)
    return {
        "train": (dates[0].date().isoformat(), dates[train_end].date().isoformat()),
        "valid": (dates[train_end + 1].date().isoformat(), dates[valid_end].date().isoformat()),
        "test": (dates[valid_end + 1].date().isoformat(), dates[-1].date().isoformat()),
    }


def build_walk_forward_year_windows(
    start_year: int = TRANSFORMER_WALK_FORWARD_START_YEAR,
    end_year: int | None = TRANSFORMER_WALK_FORWARD_END_YEAR,
    train_years: int = TRANSFORMER_WALK_FORWARD_TRAIN_YEARS,
    valid_years: int = TRANSFORMER_WALK_FORWARD_VALID_YEARS,
    test_years: int = TRANSFORMER_WALK_FORWARD_TEST_YEARS,
    available_end_year: int | None = None,
) -> list[dict[str, Any]]:
    if train_years <= 0 or valid_years <= 0 or test_years <= 0:
        raise ValueError("walk-forward train/valid/test years must be positive")
    if end_year is None:
        end_year = available_end_year
    if end_year is None:
        return []

    windows = []
    train_start = int(start_year)
    while True:
        train_end = train_start + train_years - 1
        valid_start = train_end + 1
        valid_end = valid_start + valid_years - 1
        test_start = valid_end + 1
        test_end = test_start + test_years - 1
        if test_end > int(end_year):
            break
        fold_id = f"test_{test_start}" if test_years == 1 else f"test_{test_start}_{test_end}"
        windows.append(
            {
                "fold_id": fold_id,
                "train_start": f"{train_start}-01-01",
                "train_end": f"{train_end}-12-31",
                "valid_start": f"{valid_start}-01-01",
                "valid_end": f"{valid_end}-12-31",
                "test_start": f"{test_start}-01-01",
                "test_end": f"{test_end}-12-31",
                "train_start_year": train_start,
                "train_end_year": train_end,
                "valid_start_year": valid_start,
                "valid_end_year": valid_end,
                "test_start_year": test_start,
                "test_end_year": test_end,
            }
        )
        train_start += test_years
    return windows


def _segment_mask(panel: pd.DataFrame, start: str, end: str) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return panel["datetime"].between(start_ts, end_ts, inclusive="both")


def build_transformer_dataset_for_segments(
    segments: dict[str, tuple[str, str]],
    panel: pd.DataFrame | None = None,
) -> tuple[Any, dict[str, Any]]:
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    panel = _load_factor_panel() if panel is None else panel.copy()
    if panel.empty:
        return None, {"status": "skipped", "reason": "empty factor panel"}

    label_col = _training_label_col()
    label_end_col = _label_end_col()
    required = {"train", "valid"}
    optional = {"test"}
    if not required.issubset(segments) or not set(segments).issubset(required | optional):
        return None, {"status": "skipped", "reason": "segments must include train/valid and optional test"}

    panel = _add_cross_sectional_labels(panel)
    panel = panel.dropna(subset=[label_col])
    if panel.empty:
        return None, {"status": "skipped", "reason": "empty labels"}

    train_mask = _segment_mask(panel, *segments["train"])
    valid_mask = _segment_mask(panel, *segments["valid"])
    test_mask = _segment_mask(panel, *segments["test"]) if "test" in segments else pd.Series(False, index=panel.index)
    if label_end_col in panel.columns:
        train_mask &= pd.to_datetime(panel[label_end_col], errors="coerce") <= pd.Timestamp(segments["train"][1])
        valid_mask &= pd.to_datetime(panel[label_end_col], errors="coerce") <= pd.Timestamp(segments["valid"][1])
    panel = panel[train_mask | valid_mask | test_mask].copy()
    if panel.empty:
        return None, {"status": "skipped", "reason": "empty walk-forward segment panel", "segments": segments}

    counts = {
        "train": int(train_mask.sum()),
        "valid": int(valid_mask.sum()),
    }
    if "test" in segments:
        counts["test"] = int(test_mask.sum())
    if min(counts.values()) <= 0:
        return None, {
            "status": "skipped",
            "reason": "one or more walk-forward segments are empty",
            "segments": segments,
            "segment_sample_counts": counts,
        }

    train_index = pd.MultiIndex.from_frame(panel.loc[_segment_mask(panel, *segments["train"]), ["datetime", "instrument"]])
    feature_df, label_df, frame_info = _build_transformer_frames(panel, train_index=train_index)

    handler = DataHandlerLP(data_loader=StaticDataLoader({"feature": feature_df, "label": label_df}))
    dataset = DatasetH(handler=handler, segments=segments)
    dates = panel["datetime"].dropna()
    info = {
        "status": "ok",
        "sample_count": int(len(panel)),
        "date_count": int(dates.nunique()),
        "instrument_count": int(panel["instrument"].nunique()),
        "features": frame_info["features"],
        "base_features": FEATURE_COLUMNS,
        "label": frame_info["label"],
        "label_mode": frame_info["label_mode"],
        "raw_label": frame_info["raw_label"],
        "rank_label": frame_info["rank_label"],
        "feature_normalization": frame_info["feature_normalization"],
        "feature_filter": frame_info["feature_filter"],
        "segments": segments,
        "segment_sample_counts": counts,
        "leakage_guard": "train_valid_labels_end_before_segment_end",
        "feature_scaler": frame_info["feature_scaler"],
    }
    return dataset, info


def build_transformer_dataset():
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    panel = _load_factor_panel()
    if panel.empty:
        return None, {"status": "skipped", "reason": "empty factor panel"}

    label_col = _training_label_col()
    panel = _add_cross_sectional_labels(panel)
    panel = panel.dropna(subset=[label_col])
    if panel.empty:
        return None, {"status": "skipped", "reason": "empty labels"}

    dates = panel["datetime"].dropna()
    if dates.nunique() < 30 or len(panel) < 200:
        return None, {
            "status": "skipped",
            "reason": "not enough samples for Transformer training",
            "sample_count": int(len(panel)),
            "date_count": int(dates.nunique()),
        }

    segments = _segments_from_dates(pd.DatetimeIndex(dates))
    train_index = pd.MultiIndex.from_frame(panel.loc[_segment_mask(panel, *segments["train"]), ["datetime", "instrument"]])
    feature_df, label_df, frame_info = _build_transformer_frames(panel, train_index=train_index)
    handler = DataHandlerLP(data_loader=StaticDataLoader({"feature": feature_df, "label": label_df}))
    dataset = DatasetH(handler=handler, segments=segments)
    info = {
        "status": "ok",
        "sample_count": int(len(panel)),
        "date_count": int(dates.nunique()),
        "instrument_count": int(panel["instrument"].nunique()),
        "features": frame_info["features"],
        "base_features": FEATURE_COLUMNS,
        "label": frame_info["label"],
        "label_mode": frame_info["label_mode"],
        "raw_label": frame_info["raw_label"],
        "rank_label": frame_info["rank_label"],
        "feature_normalization": frame_info["feature_normalization"],
        "feature_filter": frame_info["feature_filter"],
        "feature_scaler": frame_info["feature_scaler"],
        "segments": segments,
    }
    return dataset, info


def _prediction_metrics(pred: pd.Series, dataset, prediction_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    from qlib.data.dataset.handler import DataHandlerLP

    test_df = dataset.prepare("test", col_set=["label"], data_key=DataHandlerLP.DK_L)
    label = test_df["label"].iloc[:, 0] if isinstance(test_df["label"], pd.DataFrame) else test_df["label"]
    aligned = pd.concat([pred.rename("prediction"), label.rename("label")], axis=1).dropna()
    if aligned.empty:
        return {"test_ic": None, "test_rank_ic": None, "test_mse": None}
    metrics = {
        "test_ic": _round_or_none(aligned["prediction"].corr(aligned["label"], method="pearson")),
        "test_rank_ic": _round_or_none(aligned["prediction"].corr(aligned["label"], method="spearman")),
        "test_mse": _round_or_none(((aligned["prediction"] - aligned["label"]) ** 2).mean()),
    }
    if prediction_frame is not None:
        metrics.update(_portfolio_prediction_metrics(prediction_frame))
    return metrics


def _append_prediction_label_columns(pred_frame: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    label_col = _training_label_col()
    raw_col = _raw_label_col()
    rank_col = _rank_label_col()
    cols = ["datetime", "instrument", label_col, raw_col, rank_col]
    available = [col for col in cols if col in panel.columns]
    labels = panel[available].copy()
    labels = labels.rename(
        columns={
            "datetime": "date",
            "instrument": "symbol",
            label_col: "label",
            raw_col: "raw_forward_return",
            rank_col: "label_rank_pct",
        }
    )
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.date.astype(str)
    labels["symbol"] = labels["symbol"].map(from_qlib_instrument)
    out = pred_frame.merge(labels, on=["date", "symbol"], how="left")
    return out


def _portfolio_prediction_metrics(prediction_frame: pd.DataFrame) -> dict[str, Any]:
    frame = prediction_frame.copy()
    if "date" not in frame.columns or "prediction" not in frame.columns or "label" not in frame.columns:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    value_col = "raw_forward_return" if "raw_forward_return" in frame.columns else "label"
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame = frame.dropna(subset=["date", "prediction", "label", value_col])
    if frame.empty:
        return {
            "top20_bottom20_return": None,
            "top20_excess_return": None,
            "monthly_rank_ic_mean": None,
            "monthly_rank_ic_ir": None,
            "monthly_rank_ic_positive_rate": None,
        }

    long_short = []
    top_excess = []
    daily_rank_ic = []
    for date, group in frame.groupby("date"):
        if len(group) < 2:
            continue
        group = group.sort_values("prediction", ascending=False)
        bucket = max(1, int(np.ceil(len(group) * 0.20)))
        top = group.head(bucket)
        bottom = group.tail(bucket)
        long_short.append(float(top[value_col].mean() - bottom[value_col].mean()))
        top_excess.append(float(top["label"].mean()))
        if group["prediction"].nunique() > 1 and group["label"].nunique() > 1:
            corr = group["prediction"].corr(group["label"], method="spearman")
            if not pd.isna(corr):
                daily_rank_ic.append({"date": date, "rank_ic": float(corr)})

    monthly = pd.DataFrame(daily_rank_ic)
    if not monthly.empty:
        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
        monthly_rank_ic = monthly.groupby("month")["rank_ic"].mean()
        monthly_mean = monthly_rank_ic.mean()
        monthly_std = monthly_rank_ic.std()
    else:
        monthly_rank_ic = pd.Series(dtype=float)
        monthly_mean = np.nan
        monthly_std = np.nan

    return {
        "top20_bottom20_return": _round_or_none(np.mean(long_short) if long_short else None),
        "top20_excess_return": _round_or_none(np.mean(top_excess) if top_excess else None),
        "monthly_rank_ic_mean": _round_or_none(monthly_mean),
        "monthly_rank_ic_ir": _round_or_none(monthly_mean / monthly_std if monthly_std and not pd.isna(monthly_std) else None),
        "monthly_rank_ic_positive_rate": _round_or_none((monthly_rank_ic > 0).mean() if not monthly_rank_ic.empty else None),
    }


def _aggregate_fold_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "test_ic",
        "test_rank_ic",
        "test_mse",
        "top20_bottom20_return",
        "top20_excess_return",
        "monthly_rank_ic_mean",
        "monthly_rank_ic_ir",
        "monthly_rank_ic_positive_rate",
    ]
    summary = {}
    for key in keys:
        values = [fold.get("metrics", {}).get(key) for fold in folds]
        values = [float(value) for value in values if value is not None and not pd.isna(value)]
        summary[f"{key}_mean"] = _round_or_none(np.mean(values) if values else None)
        summary[f"{key}_median"] = _round_or_none(np.median(values) if values else None)
    return summary


def _round_or_none(value, digits=6):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _create_transformer_model(torch, feature_count: int):
    from qlib.contrib.model.pytorch_transformer import TransformerModel

    return TransformerModel(
        d_feat=feature_count,
        d_model=64,
        batch_size=2048,
        nhead=2,
        num_layers=2,
        dropout=0.0,
        n_epochs=20,
        lr=0.0001,
        early_stop=5,
        GPU=0 if torch.cuda.is_available() else -1,
        seed=42,
    )


def _predictions_to_frame(pred: pd.Series) -> pd.DataFrame:
    out = pred.rename("prediction").reset_index()
    rename_map = {}
    if "datetime" in out.columns:
        rename_map["datetime"] = "date"
    if "instrument" in out.columns:
        rename_map["instrument"] = "symbol"
    out = out.rename(columns=rename_map)
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(from_qlib_instrument)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    out["prediction_rank"] = out.groupby("date")["prediction"].rank(ascending=False, method="first") if "date" in out.columns else out["prediction"].rank(ascending=False, method="first")
    std = out["prediction"].std()
    out["prediction_zscore"] = 0.0 if std == 0 or pd.isna(std) else (out["prediction"] - out["prediction"].mean()) / std
    return out


def save_transformer_predictions(pred: pd.Series) -> Path:
    TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSFORMER_PREDICTION_PATH
    out = _predictions_to_frame(pred)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_transformer_live_prediction_snapshot(
    pred: pd.Series,
    dataset_info: dict[str, Any],
    generated_at: str,
) -> Path:
    TRANSFORMER_LIVE_PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    out = _predictions_to_frame(pred)
    if "date" in out.columns and out["date"].notna().any():
        snapshot_date = str(pd.to_datetime(out["date"], errors="coerce").dropna().max().date())
    else:
        snapshot_date = pd.Timestamp(generated_at).date().isoformat()
    out["generated_at"] = generated_at
    out["prediction_mode"] = "live_snapshot"
    out["model_path"] = str(TRANSFORMER_MODEL_PATH)
    out["feature_scaler_path"] = str(dataset_info.get("feature_scaler_path") or _live_feature_scaler_path())
    path = TRANSFORMER_LIVE_PREDICTION_DIR / f"{snapshot_date}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_latest_inference_dataset(panel: pd.DataFrame | None = None) -> tuple[Any, dict[str, Any]]:
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    scaler_path = _live_feature_scaler_path()
    if not scaler_path.exists():
        return None, {"status": "skipped", "reason": f"live feature scaler not found: {scaler_path}"}
    try:
        scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, {"status": "skipped", "reason": f"failed to load live feature scaler: {e}"}
    if not scaler.get("selected_features"):
        return None, {
            "status": "skipped",
            "reason": "live feature scaler is missing selected_features; rerun live production Transformer training",
            "feature_scaler_path": str(scaler_path),
        }

    panel = _load_factor_panel() if panel is None else panel.copy()
    if panel.empty:
        return None, {"status": "skipped", "reason": "empty factor panel"}
    latest = panel.dropna(subset=["datetime"]).sort_values("datetime").groupby("instrument", as_index=False).tail(1)
    if latest.empty:
        return None, {"status": "skipped", "reason": "empty latest factor rows"}

    selected_features = list(scaler["selected_features"])
    index = pd.MultiIndex.from_frame(latest[["datetime", "instrument"]])
    feature_df = _build_cross_sectional_feature_frame(latest)
    missing_selected_features = [col for col in selected_features if col not in feature_df.columns]
    for col in missing_selected_features:
        feature_df[col] = np.nan
    feature_df = feature_df.reindex(columns=list(selected_features))
    feature_df = _apply_feature_scaler(feature_df, scaler)
    label_df = pd.DataFrame(0.0, index=index, columns=["label"])

    dates = pd.DatetimeIndex(latest["datetime"].dropna().unique()).sort_values()
    start = _date_text(dates.min())
    end = _date_text(dates.max())
    handler = DataHandlerLP(data_loader=StaticDataLoader({"feature": feature_df, "label": label_df}))
    dataset = DatasetH(handler=handler, segments={"test": (start, end)})
    info = {
        "status": "ok",
        "inference_mode": "latest_live",
        "sample_count": int(len(latest)),
        "date_count": int(dates.nunique()),
        "instrument_count": int(latest["instrument"].nunique()),
        "features": list(selected_features),
        "base_features": FEATURE_COLUMNS,
        "label_mode": scaler.get("label_mode", TRANSFORMER_LABEL_MODE),
        "feature_normalization": scaler.get("feature_normalization", TRANSFORMER_FEATURE_NORMALIZATION),
        "missing_selected_features": missing_selected_features,
        "segments": {"test": (start, end)},
        "feature_scaler_path": str(scaler_path),
    }
    return dataset, info


def _factor_data_coverage(panel: pd.DataFrame | None = None) -> dict[str, Any]:
    panel = _load_factor_panel() if panel is None else panel
    if panel.empty:
        return {"status": "skipped", "reason": "empty factor panel"}
    dates = pd.to_datetime(panel["datetime"], errors="coerce").dropna()
    if dates.empty:
        return {"status": "skipped", "reason": "empty factor dates"}
    return {
        "status": "ok",
        "start_date": _date_text(dates.min()),
        "end_date": _date_text(dates.max()),
        "start_year": int(dates.min().year),
        "end_year": int(dates.max().year),
        "date_count": int(dates.nunique()),
        "instrument_count": int(panel["instrument"].nunique()),
    }


def ensure_walk_forward_factor_data(windows: list[dict[str, Any]], force: bool = False) -> dict[str, Any]:
    panel = _load_factor_panel()
    coverage = _factor_data_coverage(panel)
    if not windows:
        return {"status": "skipped", "reason": "no walk-forward windows", "coverage": coverage}
    required_start = pd.Timestamp(windows[0]["train_start"])
    required_end = pd.Timestamp(windows[-1]["test_end"])
    if coverage.get("status") == "ok":
        start = pd.Timestamp(coverage["start_date"])
        end = pd.Timestamp(coverage["end_date"])
        if start <= required_start and end >= required_end:
            return {
                "status": "ok",
                "source": "existing_factor_files",
                "coverage": coverage,
                "required_start": _date_text(required_start),
                "required_end": _date_text(required_end),
            }
    if not force:
        return {
            "status": "skipped",
            "reason": "factor history does not cover walk-forward windows",
            "coverage": coverage,
            "required_start": _date_text(required_start),
            "required_end": _date_text(required_end),
        }

    lookback_days = max((required_end.year - required_start.year + 1) * 260 + 120, TRAINING_LOOKBACK_DAYS)
    updated = 0
    failed = []
    for stock in fetch_index_constituents():
        symbol = stock["symbol"]
        name = stock.get("name") or symbol
        try:
            raw_df = fetch_akshare_data(symbol, name, lookback_days)
            if raw_df.empty:
                failed.append({"symbol": symbol, "reason": "empty data"})
                continue
            factor_df, _ = compute_market_factors(raw_df, symbol, name, "stock", lookback_days)
            factor_df["fundamental_available"] = False
            factor_df["industry"] = "unknown"
            factor_df = generate_signal(factor_df)
            save_dataframe(factor_df, "factors", symbol)
            updated += 1
        except Exception as e:
            logger.warning("Failed to refresh walk-forward factors for %s: %s", symbol, e)
            failed.append({"symbol": symbol, "reason": str(e)})
    refreshed_coverage = _factor_data_coverage()
    return {
        "status": "ok" if updated else "skipped",
        "source": "refreshed_from_akshare",
        "lookback_days": lookback_days,
        "updated": updated,
        "failed": failed[:10],
        "coverage": refreshed_coverage,
        "required_start": _date_text(required_start),
        "required_end": _date_text(required_end),
    }


def build_live_production_segments(
    latest_date=None,
    train_years: int = TRANSFORMER_LIVE_TRAIN_YEARS,
    valid_years: int = TRANSFORMER_LIVE_VALID_YEARS,
) -> dict[str, tuple[str, str]]:
    if train_years <= 0 or valid_years <= 0:
        raise ValueError("live production train/valid years must be positive")
    latest_ts = pd.Timestamp(latest_date) if latest_date is not None else None
    if latest_ts is None or pd.isna(latest_ts):
        panel = _load_factor_panel()
        coverage = _factor_data_coverage(panel)
        if coverage.get("status") != "ok":
            return {}
        latest_ts = pd.Timestamp(coverage["end_date"])

    valid_end_year = int(latest_ts.year) - 1
    valid_start_year = valid_end_year - valid_years + 1
    train_end_year = valid_start_year - 1
    train_start_year = train_end_year - train_years + 1
    return {
        "train": (f"{train_start_year}-01-01", f"{train_end_year}-12-31"),
        "valid": (f"{valid_start_year}-01-01", f"{valid_end_year}-12-31"),
    }


def ensure_live_production_factor_data(segments: dict[str, tuple[str, str]], force: bool = False) -> dict[str, Any]:
    if not segments:
        return {"status": "skipped", "reason": "empty live production segments"}
    panel = _load_factor_panel()
    coverage = _factor_data_coverage(panel)
    required_start = pd.Timestamp(segments["train"][0])
    required_end = pd.Timestamp(segments["valid"][1])
    if coverage.get("status") == "ok":
        start = pd.Timestamp(coverage["start_date"])
        end = pd.Timestamp(coverage["end_date"])
        if start <= required_start and end >= required_end:
            return {
                "status": "ok",
                "source": "existing_factor_files",
                "coverage": coverage,
                "required_start": _date_text(required_start),
                "required_end": _date_text(required_end),
            }
    if not force:
        return {
            "status": "skipped",
            "reason": "factor history does not cover live production training window",
            "coverage": coverage,
            "required_start": _date_text(required_start),
            "required_end": _date_text(required_end),
            "segments": segments,
        }

    lookback_days = max((required_end.year - required_start.year + 1) * 260 + 120, TRAINING_LOOKBACK_DAYS)
    updated = 0
    failed = []
    for stock in fetch_index_constituents():
        symbol = stock["symbol"]
        name = stock.get("name") or symbol
        try:
            raw_df = fetch_akshare_data(symbol, name, lookback_days)
            if raw_df.empty:
                failed.append({"symbol": symbol, "reason": "empty data"})
                continue
            factor_df, _ = compute_market_factors(raw_df, symbol, name, "stock", lookback_days)
            factor_df["fundamental_available"] = False
            factor_df["industry"] = "unknown"
            factor_df = generate_signal(factor_df)
            save_dataframe(factor_df, "factors", symbol)
            updated += 1
        except Exception as e:
            logger.warning("Failed to refresh live production factors for %s: %s", symbol, e)
            failed.append({"symbol": symbol, "reason": str(e)})
    return {
        "status": "ok" if updated else "skipped",
        "source": "refreshed_from_akshare",
        "lookback_days": lookback_days,
        "updated": updated,
        "failed": failed[:10],
        "coverage": _factor_data_coverage(),
        "required_start": _date_text(required_start),
        "required_end": _date_text(required_end),
        "segments": segments,
    }


def run_transformer_live_production_training(force: bool = False) -> dict[str, Any]:
    if not TRANSFORMER_LIVE_TRAINING_ENABLED and not force:
        return {"status": "skipped", "reason": "TRANSFORMER_LIVE_TRAINING_ENABLED is False"}

    if not _TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "Transformer dependencies unavailable"}

    torch = _torch
    panel = _load_factor_panel()
    coverage = _factor_data_coverage(panel)
    if coverage.get("status") != "ok":
        return {"status": "skipped", "reason": coverage.get("reason", "factor data unavailable"), "coverage": coverage}

    segments = build_live_production_segments(latest_date=coverage["end_date"])
    data_info = ensure_live_production_factor_data(segments, force=force)
    if data_info.get("status") != "ok":
        return {
            "status": "skipped",
            "reason": data_info.get("reason", "live production factor data unavailable"),
            "data_info": data_info,
            "segments": segments,
        }

    panel = _load_factor_panel()
    dataset, dataset_info = build_transformer_dataset_for_segments(segments, panel=panel)
    if dataset is None:
        dataset_info["data_info"] = data_info
        return dataset_info

    TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = TRANSFORMER_LIVE_MODEL_PATH
    summary_path = TRANSFORMER_MODEL_DIR / "transformer_live_training_summary.json"
    scaler_path = _live_feature_scaler_path()

    try:
        model = _create_transformer_model(torch, len(dataset_info.get("features", FEATURE_COLUMNS)))
        evals_result = {}
        model.fit(dataset, evals_result=evals_result, save_path=str(model_path))
        scaler_path.write_text(
            json.dumps(dataset_info.get("feature_scaler", {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report = {
            "status": "ok",
            "model": "Qlib TransformerModel",
            "training_mode": "live_production",
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "model_path": str(model_path),
            "summary_path": str(summary_path),
            "feature_scaler_path": str(scaler_path),
            "data_info": data_info,
            "dataset": dataset_info,
            "evals_result": evals_result,
        }
    except Exception as e:
        logger.exception("Transformer training failed: %s", e)
        report = {
            "status": "failed",
            "reason": str(e),
            "training_mode": "live_production",
            "data_info": data_info,
            "dataset": dataset_info,
        }

    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_transformer_training(force: bool = False) -> dict[str, Any]:
    report = run_transformer_live_production_training(force=force or TRANSFORMER_RETRAIN_ON_DAILY_RUN)
    report["deprecated_alias"] = True
    report["alias_for"] = "run_transformer_live_production_training"
    return report


def run_transformer_walk_forward_training(force: bool = False) -> dict[str, Any]:
    if not _TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "Transformer dependencies unavailable"}

    torch = _torch
    panel = _load_factor_panel()
    coverage = _factor_data_coverage(panel)
    available_end_year = coverage.get("end_year") if coverage.get("status") == "ok" else None
    windows = build_walk_forward_year_windows(available_end_year=available_end_year)
    data_info = ensure_walk_forward_factor_data(windows, force=force)
    if data_info.get("status") != "ok":
        return {
            "status": "skipped",
            "reason": data_info.get("reason", "walk-forward factor data unavailable"),
            "data_info": data_info,
            "windows": windows,
        }

    panel = _load_factor_panel()
    coverage = _factor_data_coverage(panel)
    available_end_year = coverage.get("end_year") if coverage.get("status") == "ok" else None
    windows = build_walk_forward_year_windows(available_end_year=available_end_year)
    if not windows:
        return {"status": "skipped", "reason": "no walk-forward windows", "data_info": data_info}

    TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMER_WALK_FORWARD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_predictions = []
    fold_reports = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for window in windows:
        segments = {
            "train": (window["train_start"], window["train_end"]),
            "valid": (window["valid_start"], window["valid_end"]),
            "test": (window["test_start"], window["test_end"]),
        }
        dataset, dataset_info = build_transformer_dataset_for_segments(segments, panel=panel)
        model_path = TRANSFORMER_WALK_FORWARD_MODEL_DIR / f"model_{window['fold_id']}.pth"
        if dataset is None:
            fold_reports.append({"status": "skipped", **window, "dataset": dataset_info})
            continue
        try:
            model = _create_transformer_model(torch, len(dataset_info.get("features", FEATURE_COLUMNS)))
            evals_result = {}
            model.fit(dataset, evals_result=evals_result, save_path=str(model_path))
            pred = model.predict(dataset, segment="test")
            pred_frame = _predictions_to_frame(pred)
            pred_frame = _append_prediction_label_columns(pred_frame, panel)
            for key in [
                "fold_id",
                "train_start",
                "train_end",
                "valid_start",
                "valid_end",
                "test_start",
                "test_end",
            ]:
                pred_frame[key] = window[key]
            pred_frame["model_path"] = str(model_path)
            all_predictions.append(pred_frame)
            fold_reports.append(
                {
                    "status": "ok",
                    **window,
                    "model_path": str(model_path),
                    "prediction_count": int(len(pred_frame)),
                    "dataset": dataset_info,
                    "metrics": _prediction_metrics(pred, dataset, pred_frame),
                    "evals_result": evals_result,
                }
            )
        except Exception as e:
            logger.exception("Transformer walk-forward fold failed: %s", e)
            fold_reports.append({"status": "failed", **window, "reason": str(e), "dataset": dataset_info})

    ok_folds = [fold for fold in fold_reports if fold.get("status") == "ok"]
    summary_path = TRANSFORMER_MODEL_DIR / "transformer_walk_forward_summary.json"
    if not ok_folds or not all_predictions:
        report = {
            "status": "skipped",
            "reason": "no successful walk-forward folds",
            "model": "Qlib TransformerModel",
            "device": device,
            "data_info": data_info,
            "coverage": coverage,
            "fold_count": 0,
            "folds": fold_reports,
            "summary_path": str(summary_path),
        }
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.sort_values(["date", "symbol"]).reset_index(drop=True)
    predictions.to_csv(TRANSFORMER_WALK_FORWARD_PREDICTION_PATH, index=False, encoding="utf-8-sig")
    test_years = [fold["test_start_year"] for fold in ok_folds]
    report = {
        "status": "ok",
        "model": "Qlib TransformerModel",
        "device": device,
        "prediction_mode": "walk_forward",
        "prediction_path": str(TRANSFORMER_WALK_FORWARD_PREDICTION_PATH),
        "model_dir": str(TRANSFORMER_WALK_FORWARD_MODEL_DIR),
        "data_info": data_info,
        "coverage": coverage,
        "fold_count": len(ok_folds),
        "label_mode": TRANSFORMER_LABEL_MODE,
        "feature_normalization": TRANSFORMER_FEATURE_NORMALIZATION,
        "first_test_year": min(test_years),
        "last_test_year": max(test_years),
        "metrics_summary": _aggregate_fold_metrics(ok_folds),
        "leakage_guard": "train_valid_before_test",
        "folds": fold_reports,
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_transformer_inference() -> dict[str, Any]:
    if not TRANSFORMER_INFERENCE_ENABLED:
        return {"status": "skipped", "reason": "TRANSFORMER_INFERENCE_ENABLED is False"}
    if not TRANSFORMER_MODEL_PATH.exists():
        return {"status": "skipped", "reason": f"model file not found: {TRANSFORMER_MODEL_PATH}"}

    if not _TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "Transformer dependencies unavailable"}

    torch = _torch
    dataset, dataset_info = build_latest_inference_dataset()
    if dataset is None:
        return dataset_info

    try:
        model = _create_transformer_model(torch, len(dataset_info.get("features", FEATURE_COLUMNS)))
        if not hasattr(model, "load"):
            return {"status": "skipped", "reason": "Qlib TransformerModel does not expose load() in this environment"}
        model.load(str(TRANSFORMER_MODEL_PATH))
        pred = model.predict(dataset, segment="test")
        pred_path = save_transformer_predictions(pred)
        generated_at = pd.Timestamp.now().isoformat(timespec="seconds")
        snapshot_path = save_transformer_live_prediction_snapshot(pred, dataset_info, generated_at)
        report = {
            "status": "ok",
            "model": "Qlib TransformerModel",
            "inference_mode": "latest_live",
            "model_path": str(TRANSFORMER_MODEL_PATH),
            "prediction_path": str(pred_path),
            "live_prediction_snapshot_path": str(snapshot_path),
            "generated_at": generated_at,
            "dataset": dataset_info,
        }
    except Exception as e:
        logger.exception("Transformer inference failed: %s", e)
        report = {
            "status": "failed",
            "reason": str(e),
            "model_path": str(TRANSFORMER_MODEL_PATH),
            "dataset": dataset_info,
        }

    inference_path = TRANSFORMER_MODEL_DIR / "transformer_inference_summary.json"
    TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    inference_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["summary_path"] = str(inference_path)
    return report
