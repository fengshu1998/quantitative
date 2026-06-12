from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import AGENT_PERFORMANCE_FEEDBACK_PATH, QLIB_BACKTEST_COMPARISON_PATH


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _save_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    AGENT_PERFORMANCE_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    AGENT_PERFORMANCE_FEEDBACK_PATH.write_text(
        json.dumps(feedback, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feedback["feedback_path"] = str(AGENT_PERFORMANCE_FEEDBACK_PATH)
    return feedback


def run_agent_feedback_update(
    qlib_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write structured Agents performance feedback for the next research run."""

    comparison = qlib_comparison or _read_json(Path(QLIB_BACKTEST_COMPARISON_PATH))
    generated_at = datetime.now().isoformat(timespec="seconds")
    reports = comparison.get("signal_reports") or {}
    agent_report = reports.get("agents") or {}
    hybrid_report = reports.get("hybrid") or {}

    if agent_report.get("status") != "ok":
        return _save_feedback(
            {
                "status": "skipped",
                "generated_at": generated_at,
                "reason": agent_report.get("reason") or "agents qlib backtest unavailable",
                "feedback_summary": "Agents 回测历史不足或不可用，暂不反馈给研究偏好。",
                "recommended_constraints": [],
            }
        )

    if hybrid_report.get("status") != "ok":
        return _save_feedback(
            {
                "status": "skipped",
                "generated_at": generated_at,
                "reason": hybrid_report.get("reason") or "hybrid qlib backtest unavailable",
                "feedback_summary": "Hybrid 对照回测不可用，暂不生成 Agents 相对表现反馈。",
                "recommended_constraints": [],
            }
        )

    agent_total = _num(agent_report.get("total_return_percent"), 0.0) or 0.0
    hybrid_total = _num(hybrid_report.get("total_return_percent"), 0.0) or 0.0
    agent_drawdown = _num(agent_report.get("max_drawdown_percent"))
    hybrid_drawdown = _num(hybrid_report.get("max_drawdown_percent"))
    agent_sharpe = _num(agent_report.get("sharpe"))
    hybrid_sharpe = _num(hybrid_report.get("sharpe"))
    excess_vs_hybrid = round(agent_total - hybrid_total, 2)

    constraints: list[str] = []
    if excess_vs_hybrid < -1.0:
        constraints.append("prefer_hybrid_score_rank_when_agent_underperforms")
    if agent_drawdown is not None and hybrid_drawdown is not None and agent_drawdown < hybrid_drawdown - 2.0:
        constraints.append("tighten_buy_confidence_when_agent_drawdown_worse")
    if agent_sharpe is not None and hybrid_sharpe is not None and agent_sharpe < hybrid_sharpe:
        constraints.append("require_clearer_risk_reward_for_agent_buy")

    if excess_vs_hybrid >= 0:
        summary = f"Agents 回测收益高于或持平 hybrid，超额 {excess_vs_hybrid:.2f}pct。"
    else:
        summary = f"Agents 回测收益低于 hybrid，落后 {abs(excess_vs_hybrid):.2f}pct；下轮研究需更保守引用该反馈。"

    return _save_feedback(
        {
            "status": "ok",
            "generated_at": generated_at,
            "agent_total_return_percent": round(agent_total, 2),
            "hybrid_total_return_percent": round(hybrid_total, 2),
            "agent_excess_vs_hybrid_percent": excess_vs_hybrid,
            "agent_max_drawdown_percent": agent_drawdown,
            "hybrid_max_drawdown_percent": hybrid_drawdown,
            "agent_sharpe": agent_sharpe,
            "hybrid_sharpe": hybrid_sharpe,
            "feedback_summary": summary,
            "recommended_constraints": constraints,
        }
    )
