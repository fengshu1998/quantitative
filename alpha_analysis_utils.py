from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import ALPHA_ANALYSIS_ENABLED, ALPHA_REPORT_DIR, FACTOR_DATA_DIR


logger = logging.getLogger(__name__)

ALPHA_FACTORS = [
    "return_20d",
    "volume_ratio_20d",
    "rsi_14",
    "macd_diff",
    "adx_14",
    "bollinger_width",
    "volatility_20d",
    "max_drawdown_20d",
    "roe",
    "debt_to_asset",
]


def _load_alpha_panel() -> pd.DataFrame:
    frames = []
    for path in sorted(Path(FACTOR_DATA_DIR).glob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("Failed to read alpha file %s: %s", path, e)
            continue
        if df.empty or "date" not in df.columns or "close" not in df.columns:
            continue
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        df["instrument"] = path.stem
        close = pd.to_numeric(df["close"], errors="coerce")
        df["forward_return_5d"] = close.shift(-5) / close - 1
        for col in ALPHA_FACTORS:
            if col not in df.columns:
                df[col] = np.nan
        frames.append(df[["datetime", "instrument", "forward_return_5d"] + ALPHA_FACTORS])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=["datetime"])


def _factor_stats(panel: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for factor in ALPHA_FACTORS:
        sub = panel[["datetime", factor, "forward_return_5d"]].copy()
        missing_rate = float(sub[factor].isna().mean()) if len(sub) else 1.0
        sub = sub.dropna()
        if sub.empty:
            rows.append(
                {
                    "factor": factor,
                    "sample_count": 0,
                    "missing_rate": round(missing_rate, 6),
                    "ic": None,
                    "rank_ic": None,
                    "long_short_return": None,
                }
            )
            continue

        daily = []
        quantile_returns = []
        for _, day_df in sub.groupby("datetime"):
            if len(day_df) < 5:
                continue
            ic = day_df[factor].corr(day_df["forward_return_5d"], method="pearson")
            rank_ic = day_df[factor].corr(day_df["forward_return_5d"], method="spearman")
            daily.append({"ic": ic, "rank_ic": rank_ic})
            try:
                q = pd.qcut(day_df[factor].rank(method="first"), 5, labels=False)
                top = day_df.loc[q == 4, "forward_return_5d"].mean()
                bottom = day_df.loc[q == 0, "forward_return_5d"].mean()
                quantile_returns.append(top - bottom)
            except ValueError:
                continue

        daily_df = pd.DataFrame(daily)
        rows.append(
            {
                "factor": factor,
                "sample_count": int(len(sub)),
                "missing_rate": round(missing_rate, 6),
                "ic": _round_or_none(daily_df["ic"].mean() if "ic" in daily_df else None),
                "rank_ic": _round_or_none(daily_df["rank_ic"].mean() if "rank_ic" in daily_df else None),
                "long_short_return": _round_or_none(np.nanmean(quantile_returns) if quantile_returns else None),
            }
        )
    return rows


def _round_or_none(value, digits=6):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _factor_correlation(panel: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    corr = panel[ALPHA_FACTORS].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    return {
        row: {col: _round_or_none(corr.loc[row, col]) for col in corr.columns}
        for row in corr.index
    }


def build_alpha_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Alpha Analysis Summary",
        "",
        f"- status: {report.get('status')}",
        f"- sample_count: {report.get('sample_count')}",
        f"- date_count: {report.get('date_count')}",
        f"- instrument_count: {report.get('instrument_count')}",
        "",
        "| factor | IC | RankIC | Long-Short Return | Missing Rate | Samples |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("factor_stats", []):
        lines.append(
            "| {factor} | {ic} | {rank_ic} | {ls} | {miss} | {n} |".format(
                factor=row["factor"],
                ic=row["ic"],
                rank_ic=row["rank_ic"],
                ls=row["long_short_return"],
                miss=row["missing_rate"],
                n=row["sample_count"],
            )
        )
    return "\n".join(lines) + "\n"


def run_alpha_analysis() -> dict[str, Any]:
    if not ALPHA_ANALYSIS_ENABLED:
        return {"status": "skipped", "reason": "ALPHA_ANALYSIS_ENABLED is False"}

    panel = _load_alpha_panel()
    if panel.empty:
        return {"status": "skipped", "reason": "empty alpha panel"}

    panel = panel.dropna(subset=["forward_return_5d"])
    if panel.empty:
        return {"status": "skipped", "reason": "empty forward returns"}

    ALPHA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "sample_count": int(len(panel)),
        "date_count": int(panel["datetime"].nunique()),
        "instrument_count": int(panel["instrument"].nunique()),
        "target": "forward_return_5d",
        "factor_stats": _factor_stats(panel),
        "factor_correlation": _factor_correlation(panel),
        "json_path": str(ALPHA_REPORT_DIR / "alpha_summary.json"),
        "markdown_path": str(ALPHA_REPORT_DIR / "alpha_summary.md"),
    }
    (ALPHA_REPORT_DIR / "alpha_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ALPHA_REPORT_DIR / "alpha_summary.md").write_text(
        build_alpha_summary_markdown(report),
        encoding="utf-8",
    )
    return report
