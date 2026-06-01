from pathlib import Path

import pandas as pd

from config import (
    FACTOR_DATA_DIR,
    FUNDAMENTAL_DATA_DIR,
    INDUSTRY_DATA_DIR,
    PRICE_DATA_DIR,
    SELECTED_CANDIDATES_PATH,
    STORAGE_FORMAT,
)


DATASET_DIRS = {
    "prices": PRICE_DATA_DIR,
    "factors": FACTOR_DATA_DIR,
    "fundamentals": FUNDAMENTAL_DATA_DIR,
    "industries": INDUSTRY_DATA_DIR,
}


def _dataset_dir(dataset_type: str) -> Path:
    if dataset_type not in DATASET_DIRS:
        raise ValueError(f"未知数据集类型: {dataset_type}")
    path = DATASET_DIRS[dataset_type]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extension() -> str:
    if STORAGE_FORMAT == "csv":
        return "csv"
    if STORAGE_FORMAT == "parquet":
        return "parquet"
    raise ValueError(f"不支持的 STORAGE_FORMAT: {STORAGE_FORMAT}")


def _write_dataframe(df: pd.DataFrame, path: Path):
    if STORAGE_FORMAT == "csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if STORAGE_FORMAT == "parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"不支持的 STORAGE_FORMAT: {STORAGE_FORMAT}")


def _read_dataframe(path: Path) -> pd.DataFrame:
    if STORAGE_FORMAT == "csv":
        return pd.read_csv(path)
    if STORAGE_FORMAT == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"不支持的 STORAGE_FORMAT: {STORAGE_FORMAT}")


def save_dataframe(df: pd.DataFrame, dataset_type: str, symbol_or_name: str) -> Path:
    path = _dataset_dir(dataset_type) / f"{symbol_or_name}.{_extension()}"
    _write_dataframe(df, path)
    return path


def load_dataframe(dataset_type: str, symbol_or_name: str) -> pd.DataFrame:
    path = _dataset_dir(dataset_type) / f"{symbol_or_name}.{_extension()}"
    return _read_dataframe(path)


def save_selected_candidates(df: pd.DataFrame) -> Path:
    SELECTED_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SELECTED_CANDIDATES_PATH, index=False, encoding="utf-8-sig")
    return SELECTED_CANDIDATES_PATH


def load_selected_candidates() -> pd.DataFrame:
    return pd.read_csv(SELECTED_CANDIDATES_PATH)
