from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    FACTOR_DATA_DIR,
    TRAINING_LOOKBACK_DAYS,
    TRANSFORMER_LABEL_HORIZON,
    TRANSFORMER_MODEL_PATH,
    TRANSFORMER_MODEL_DIR,
    TRANSFORMER_PREDICTION_PATH,
    TRANSFORMER_INFERENCE_ENABLED,
    TRANSFORMER_RETRAIN_ON_DAILY_RUN,
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
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["datetime", "instrument"])


def _standardize_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    med = feature_df.median()
    feature_df = feature_df.fillna(med).fillna(0.0)
    std = feature_df.std().replace(0, np.nan)
    feature_df = (feature_df - feature_df.mean()) / std
    return feature_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


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


def build_transformer_dataset():
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    panel = _load_factor_panel()
    if panel.empty:
        return None, {"status": "skipped", "reason": "empty factor panel"}

    label_col = f"label_{TRANSFORMER_LABEL_HORIZON}d_forward_return"
    panel = panel.dropna(subset=[label_col])
    if panel.empty:
        return None, {"status": "skipped", "reason": "empty labels"}

    index = pd.MultiIndex.from_frame(panel[["datetime", "instrument"]])
    feature_df = pd.DataFrame(panel[FEATURE_COLUMNS].to_numpy(), index=index, columns=FEATURE_COLUMNS)
    label_df = pd.DataFrame(panel[[label_col]].to_numpy(), index=index, columns=["label"])
    feature_df = _standardize_features(feature_df)

    dates = panel["datetime"].dropna()
    if dates.nunique() < 30 or len(panel) < 200:
        return None, {
            "status": "skipped",
            "reason": "not enough samples for Transformer training",
            "sample_count": int(len(panel)),
            "date_count": int(dates.nunique()),
        }

    segments = _segments_from_dates(pd.DatetimeIndex(dates))
    handler = DataHandlerLP(data_loader=StaticDataLoader({"feature": feature_df, "label": label_df}))
    dataset = DatasetH(handler=handler, segments=segments)
    info = {
        "status": "ok",
        "sample_count": int(len(panel)),
        "date_count": int(dates.nunique()),
        "instrument_count": int(panel["instrument"].nunique()),
        "features": FEATURE_COLUMNS,
        "label": label_col,
        "segments": segments,
    }
    return dataset, info


def _prediction_metrics(pred: pd.Series, dataset) -> dict[str, Any]:
    from qlib.data.dataset.handler import DataHandlerLP

    test_df = dataset.prepare("test", col_set=["label"], data_key=DataHandlerLP.DK_L)
    label = test_df["label"].iloc[:, 0] if isinstance(test_df["label"], pd.DataFrame) else test_df["label"]
    aligned = pd.concat([pred.rename("prediction"), label.rename("label")], axis=1).dropna()
    if aligned.empty:
        return {"test_ic": None, "test_rank_ic": None, "test_mse": None}
    return {
        "test_ic": _round_or_none(aligned["prediction"].corr(aligned["label"], method="pearson")),
        "test_rank_ic": _round_or_none(aligned["prediction"].corr(aligned["label"], method="spearman")),
        "test_mse": _round_or_none(((aligned["prediction"] - aligned["label"]) ** 2).mean()),
    }


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


def save_transformer_predictions(pred: pd.Series) -> Path:
    TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSFORMER_PREDICTION_PATH
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
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_transformer_training(force: bool = False) -> dict[str, Any]:
    if not force and not TRANSFORMER_RETRAIN_ON_DAILY_RUN:
        return {"status": "skipped", "reason": "TRANSFORMER_RETRAIN_ON_DAILY_RUN is False"}

    try:
        import torch
    except Exception as e:
        return {"status": "skipped", "reason": f"Transformer dependencies unavailable: {e}"}

    data_info = ensure_training_factor_data()
    dataset, dataset_info = build_transformer_dataset()
    if dataset is None:
        dataset_info["data_info"] = data_info
        return dataset_info

    TRANSFORMER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = TRANSFORMER_MODEL_PATH
    summary_path = TRANSFORMER_MODEL_DIR / "transformer_training_summary.json"

    try:
        model = _create_transformer_model(torch, len(FEATURE_COLUMNS))
        evals_result = {}
        model.fit(dataset, evals_result=evals_result, save_path=str(model_path))
        report = {
            "status": "ok",
            "model": "Qlib TransformerModel",
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "model_path": str(model_path),
            "summary_path": str(summary_path),
            "data_info": data_info,
            "dataset": dataset_info,
            "evals_result": evals_result,
        }
    except Exception as e:
        logger.exception("Transformer training failed: %s", e)
        report = {
            "status": "failed",
            "reason": str(e),
            "data_info": data_info,
            "dataset": dataset_info,
        }

    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_transformer_inference() -> dict[str, Any]:
    if not TRANSFORMER_INFERENCE_ENABLED:
        return {"status": "skipped", "reason": "TRANSFORMER_INFERENCE_ENABLED is False"}
    if not TRANSFORMER_MODEL_PATH.exists():
        return {"status": "skipped", "reason": f"model file not found: {TRANSFORMER_MODEL_PATH}"}

    try:
        import torch
    except Exception as e:
        return {"status": "skipped", "reason": f"Transformer dependencies unavailable: {e}"}

    dataset, dataset_info = build_transformer_dataset()
    if dataset is None:
        return dataset_info

    try:
        model = _create_transformer_model(torch, len(FEATURE_COLUMNS))
        if not hasattr(model, "load"):
            return {"status": "skipped", "reason": "Qlib TransformerModel does not expose load() in this environment"}
        model.load(str(TRANSFORMER_MODEL_PATH))
        pred = model.predict(dataset, segment="test")
        pred_path = save_transformer_predictions(pred)
        metrics = _prediction_metrics(pred, dataset)
        report = {
            "status": "ok",
            "model": "Qlib TransformerModel",
            "model_path": str(TRANSFORMER_MODEL_PATH),
            "prediction_path": str(pred_path),
            "dataset": dataset_info,
            "metrics": metrics,
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
