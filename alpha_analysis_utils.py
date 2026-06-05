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

MIN_EFFECTIVE_SAMPLES = 1000
MAX_EFFECTIVE_MISSING_RATE = 0.35
MIN_OBSERVE_SAMPLES = 300
MAX_OBSERVE_MISSING_RATE = 0.60
EFFECTIVE_RANK_IC_THRESHOLD = 0.02
EFFECTIVE_IC_THRESHOLD = 0.04
EFFECTIVE_IR_THRESHOLD = 0.20
OBSERVE_RANK_IC_THRESHOLD = 0.005
OBSERVE_IC_THRESHOLD = 0.015
HIGH_WEIGHT_THRESHOLD = 0.05
MEDIUM_WEIGHT_THRESHOLD = 0.025
CORRELATION_PENALTY_THRESHOLD = 0.80

CN_LABELS = {
    "effective": "有效",
    "watch": "观察",
    "discard": "剔除",
    "positive": "正向",
    "negative": "反向",
    "unstable": "不稳定",
    "high": "高",
    "medium": "中",
    "low": "低",
    "zero": "0",
}


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


def _round_or_none(value, digits=6):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _safe_ir(mean_value, std_value):
    if mean_value is None or pd.isna(mean_value) or std_value is None or pd.isna(std_value) or std_value == 0:
        return None
    return float(mean_value) / float(std_value)


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
                    "ic_ir": None,
                    "rank_ic_ir": None,
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
        ic_mean = daily_df["ic"].mean() if "ic" in daily_df else None
        rank_ic_mean = daily_df["rank_ic"].mean() if "rank_ic" in daily_df else None
        ic_std = daily_df["ic"].std() if "ic" in daily_df else None
        rank_ic_std = daily_df["rank_ic"].std() if "rank_ic" in daily_df else None
        rows.append(
            {
                "factor": factor,
                "sample_count": int(len(sub)),
                "missing_rate": round(missing_rate, 6),
                "ic": _round_or_none(ic_mean),
                "rank_ic": _round_or_none(rank_ic_mean),
                "ic_ir": _round_or_none(_safe_ir(ic_mean, ic_std)),
                "rank_ic_ir": _round_or_none(_safe_ir(rank_ic_mean, rank_ic_std)),
                "long_short_return": _round_or_none(np.nanmean(quantile_returns) if quantile_returns else None),
            }
        )
    return rows


def _factor_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[ALPHA_FACTORS].apply(pd.to_numeric, errors="coerce").corr(method="spearman")


def _factor_correlation_json(corr: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    return {
        row: {col: _round_or_none(corr.loc[row, col]) for col in corr.columns}
        for row in corr.index
    }


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
    votes = [_sign(row.get("ic")), _sign(row.get("rank_ic")), _sign(row.get("long_short_return"))]
    positive_votes = sum(1 for item in votes if item > 0)
    negative_votes = sum(1 for item in votes if item < 0)
    if positive_votes >= 2:
        return "positive"
    if negative_votes >= 2:
        return "negative"
    return "unstable"


def _validity_from_stats(row: dict[str, Any], direction: str) -> str:
    sample_count = int(row.get("sample_count") or 0)
    missing_rate = float(row.get("missing_rate") if row.get("missing_rate") is not None else 1.0)
    rank_ic_abs = _abs_or_zero(row.get("rank_ic"))
    ic_abs = _abs_or_zero(row.get("ic"))
    rank_ic_ir_abs = _abs_or_zero(row.get("rank_ic_ir"))
    ic_ir_abs = _abs_or_zero(row.get("ic_ir"))

    if (
        direction != "unstable"
        and sample_count >= MIN_EFFECTIVE_SAMPLES
        and missing_rate <= MAX_EFFECTIVE_MISSING_RATE
        and (
            rank_ic_abs >= EFFECTIVE_RANK_IC_THRESHOLD
            or ic_abs >= EFFECTIVE_IC_THRESHOLD
            or rank_ic_ir_abs >= EFFECTIVE_IR_THRESHOLD
            or ic_ir_abs >= EFFECTIVE_IR_THRESHOLD
        )
    ):
        return "effective"

    if (
        direction != "unstable"
        and sample_count >= MIN_OBSERVE_SAMPLES
        and missing_rate <= MAX_OBSERVE_MISSING_RATE
        and (rank_ic_abs >= OBSERVE_RANK_IC_THRESHOLD or ic_abs >= OBSERVE_IC_THRESHOLD)
    ):
        return "watch"

    return "discard"


def _correlation_penalty(factor: str, corr: pd.DataFrame) -> float:
    if factor not in corr.index:
        return 0.0
    values = corr.loc[factor].drop(labels=[factor], errors="ignore").abs().dropna()
    if values.empty:
        return 0.0
    max_corr = float(values.max())
    if max_corr <= CORRELATION_PENALTY_THRESHOLD:
        return 0.0
    return min(0.50, round((max_corr - CORRELATION_PENALTY_THRESHOLD) / (1 - CORRELATION_PENALTY_THRESHOLD) * 0.50, 6))


def _weight_from_stats(row: dict[str, Any], validity: str, penalty: float) -> tuple[str, float]:
    if validity == "discard":
        return "zero", 0.0

    strength = max(
        _abs_or_zero(row.get("rank_ic")),
        _abs_or_zero(row.get("ic")),
        _abs_or_zero(row.get("rank_ic_ir")) / 10,
        _abs_or_zero(row.get("ic_ir")) / 10,
    )
    if validity == "effective" and strength >= HIGH_WEIGHT_THRESHOLD:
        level, weight = "high", 1.5
    elif validity == "effective" and strength >= MEDIUM_WEIGHT_THRESHOLD:
        level, weight = "medium", 1.0
    elif validity == "effective":
        level, weight = "low", 0.5
    else:
        level, weight = "low", 0.25
    return level, round(max(weight * (1 - penalty), 0.0), 6)


def build_factor_selection(factor_stats: list[dict[str, Any]], corr: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    corr = corr if corr is not None else pd.DataFrame()
    selections = []
    for row in factor_stats:
        direction = _direction_from_stats(row)
        validity = _validity_from_stats(row, direction)
        penalty = _correlation_penalty(row["factor"], corr)
        weight_level, factor_weight = _weight_from_stats(row, validity, penalty)
        selections.append(
            {
                **row,
                "validity": validity,
                "direction": direction,
                "weight_level": weight_level,
                "correlation_penalty": penalty,
                "factor_weight": factor_weight,
                "signal_weight": factor_weight,
            }
        )
    return selections


def build_alpha_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Alpha Analysis Summary",
        "",
        f"- status: {report.get('status')}",
        f"- sample_count: {report.get('sample_count')}",
        f"- date_count: {report.get('date_count')}",
        f"- instrument_count: {report.get('instrument_count')}",
        "",
        "| factor | IC | RankIC | IC_IR | RankIC_IR | Long-Short | Missing | Samples |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("factor_stats", []):
        lines.append(
            "| {factor} | {ic} | {rank_ic} | {ic_ir} | {rank_ic_ir} | {ls} | {miss} | {n} |".format(
                factor=row["factor"],
                ic=row["ic"],
                rank_ic=row["rank_ic"],
                ic_ir=row["ic_ir"],
                rank_ic_ir=row["rank_ic_ir"],
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
            "| factor | validity | direction | weight_level | factor_weight | corr_penalty | 中文判断 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("factor_selection", []):
        cn = "/".join(
            [
                CN_LABELS.get(row["validity"], row["validity"]),
                CN_LABELS.get(row["direction"], row["direction"]),
                CN_LABELS.get(row["weight_level"], row["weight_level"]),
            ]
        )
        lines.append(
            "| {factor} | {validity} | {direction} | {weight_level} | {factor_weight} | {penalty} | {cn} |".format(
                factor=row["factor"],
                validity=row["validity"],
                direction=row["direction"],
                weight_level=row["weight_level"],
                factor_weight=row["factor_weight"],
                penalty=row["correlation_penalty"],
                cn=cn,
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
    corr = _factor_correlation(panel)
    factor_selection = build_factor_selection(factor_stats, corr)
    report = {
        "status": "ok",
        "sample_count": int(len(panel)),
        "date_count": int(panel["datetime"].nunique()),
        "instrument_count": int(panel["instrument"].nunique()),
        "target": "forward_return_5d",
        "factor_stats": factor_stats,
        "factor_selection": factor_selection,
        "factor_correlation": _factor_correlation_json(corr),
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
