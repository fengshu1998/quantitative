from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ALPHA_ANALYSIS_ENABLED,
    ALPHA_FACTOR_SELECTION_PATH,
    ALPHA_REPORT_DIR,
    FACTOR_DATA_DIR,
)


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

MIN_EFFECTIVE_SAMPLES = 1000
MAX_EFFECTIVE_MISSING_RATE = 0.35
MIN_OBSERVE_SAMPLES = 300
MAX_OBSERVE_MISSING_RATE = 0.60
EFFECTIVE_RANK_IC_THRESHOLD = 0.02
EFFECTIVE_IC_THRESHOLD = 0.04
OBSERVE_RANK_IC_THRESHOLD = 0.005
OBSERVE_IC_THRESHOLD = 0.015
HIGH_WEIGHT_THRESHOLD = 0.05
MEDIUM_WEIGHT_THRESHOLD = 0.025


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


def _abs_or_zero(value) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return abs(float(value))


def _sign(value) -> int:
    if value is None or pd.isna(value):
        return 0
    value = float(value)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _direction_from_stats(row: dict[str, Any]) -> str:
    ic_sign = _sign(row.get("ic"))
    rank_ic_sign = _sign(row.get("rank_ic"))
    long_short_sign = _sign(row.get("long_short_return"))
    votes = [item for item in [ic_sign, rank_ic_sign, long_short_sign] if item != 0]
    if not votes:
        return "不稳定"
    positive_votes = sum(1 for item in votes if item > 0)
    negative_votes = sum(1 for item in votes if item < 0)
    if positive_votes >= 2:
        return "正向"
    if negative_votes >= 2:
        return "反向"
    return "不稳定"


def _validity_from_stats(row: dict[str, Any], direction: str) -> str:
    sample_count = int(row.get("sample_count") or 0)
    missing_rate = float(row.get("missing_rate") if row.get("missing_rate") is not None else 1.0)
    rank_ic_abs = _abs_or_zero(row.get("rank_ic"))
    ic_abs = _abs_or_zero(row.get("ic"))
    long_short_abs = _abs_or_zero(row.get("long_short_return"))

    if (
        direction != "不稳定"
        and sample_count >= MIN_EFFECTIVE_SAMPLES
        and missing_rate <= MAX_EFFECTIVE_MISSING_RATE
        and (
            rank_ic_abs >= EFFECTIVE_RANK_IC_THRESHOLD
            or ic_abs >= EFFECTIVE_IC_THRESHOLD
            or long_short_abs >= EFFECTIVE_RANK_IC_THRESHOLD
        )
    ):
        return "有效"

    if (
        direction != "不稳定"
        and sample_count >= MIN_OBSERVE_SAMPLES
        and missing_rate <= MAX_OBSERVE_MISSING_RATE
        and (
            rank_ic_abs >= OBSERVE_RANK_IC_THRESHOLD
            or ic_abs >= OBSERVE_IC_THRESHOLD
            or long_short_abs >= OBSERVE_RANK_IC_THRESHOLD / 2
        )
    ):
        return "观察"

    return "剔除"


def _weight_from_stats(row: dict[str, Any], validity: str) -> tuple[str, float]:
    if validity == "剔除":
        return "0", 0.0

    strength = max(
        _abs_or_zero(row.get("rank_ic")),
        _abs_or_zero(row.get("ic")),
        _abs_or_zero(row.get("long_short_return")),
    )
    if validity == "有效" and strength >= HIGH_WEIGHT_THRESHOLD:
        return "高", 1.5
    if validity == "有效" and strength >= MEDIUM_WEIGHT_THRESHOLD:
        return "中", 1.0
    if validity == "有效":
        return "低", 0.5
    return "低", 0.25


def build_factor_selection(factor_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selections = []
    for row in factor_stats:
        direction = _direction_from_stats(row)
        validity = _validity_from_stats(row, direction)
        weight_level, signal_weight = _weight_from_stats(row, validity)
        selections.append(
            {
                **row,
                "validity": validity,
                "direction": direction,
                "weight_level": weight_level,
                "signal_weight": signal_weight,
            }
        )
    return selections


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
    lines.extend(
        [
            "",
            "## Factor Selection",
            "",
            "| factor | validity | direction | weight_level | signal_weight |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("factor_selection", []):
        lines.append(
            "| {factor} | {validity} | {direction} | {weight_level} | {signal_weight} |".format(
                factor=row["factor"],
                validity=row["validity"],
                direction=row["direction"],
                weight_level=row["weight_level"],
                signal_weight=row["signal_weight"],
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
    factor_stats = _factor_stats(panel)
    factor_selection = build_factor_selection(factor_stats)
    report = {
        "status": "ok",
        "sample_count": int(len(panel)),
        "date_count": int(panel["datetime"].nunique()),
        "instrument_count": int(panel["instrument"].nunique()),
        "target": "forward_return_5d",
        "factor_stats": factor_stats,
        "factor_selection": factor_selection,
        "factor_correlation": _factor_correlation(panel),
        "json_path": str(ALPHA_REPORT_DIR / "alpha_summary.json"),
        "markdown_path": str(ALPHA_REPORT_DIR / "alpha_summary.md"),
        "factor_selection_path": str(ALPHA_FACTOR_SELECTION_PATH),
    }
    (ALPHA_REPORT_DIR / "alpha_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ALPHA_REPORT_DIR / "alpha_summary.md").write_text(
        build_alpha_summary_markdown(report),
        encoding="utf-8",
    )
    ALPHA_FACTOR_SELECTION_PATH.write_text(
        json.dumps({"factor_selection": factor_selection}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
