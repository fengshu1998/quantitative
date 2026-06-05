import json
from datetime import datetime
from pathlib import Path

from qlib_backtest_utils import build_qlib_report_rows


REPORTS_DIR = Path("reports")
CN_ALPHA_LABELS = {
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


def _safe_text(value):
    return str(value).replace("\r\n", "\n").strip()


def _format_percent(value):
    return f"{float(value):.2f}%"


def _markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _backtest_rows(backtest_report):
    if backtest_report.get("status") != "ok":
        return [["status", backtest_report.get("status", "unknown")], ["reason", backtest_report.get("reason", "")]]

    return [
        ["backtest_window", f"{backtest_report['start_date']} to {backtest_report['end_date']}"],
        ["trading_days", backtest_report["trading_days"]],
        ["total_return", _format_percent(backtest_report["total_return_percent"])],
        ["annualized_return", _format_percent(backtest_report["annualized_return_percent"])],
        ["annualized_volatility", _format_percent(backtest_report["annualized_volatility_percent"])],
        ["sharpe", backtest_report["sharpe"]],
        ["max_drawdown", _format_percent(backtest_report["max_drawdown_percent"])],
    ]


def _alpha_factor_selection_rows(alpha_report):
    rows = []
    for item in alpha_report.get("factor_selection", []):
        rows.append(
            [
                item.get("factor"),
                f"{item.get('validity')} ({CN_ALPHA_LABELS.get(item.get('validity'), item.get('validity'))})",
                f"{item.get('direction')} ({CN_ALPHA_LABELS.get(item.get('direction'), item.get('direction'))})",
                f"{item.get('weight_level')} ({CN_ALPHA_LABELS.get(item.get('weight_level'), item.get('weight_level'))})",
                item.get("factor_weight", item.get("signal_weight")),
                item.get("rank_ic"),
                item.get("long_short_return"),
            ]
        )
    if not rows:
        rows = [["None", "None", "None", "0", "0", "None", "None"]]
    return rows


def build_report_payload(market_summary, messages, agent_outputs, portfolio_result, generated_at):
    qlib_report = getattr(
        portfolio_result,
        "qlib_backtest_report",
        {"status": "skipped", "reason": "not available"},
    )
    transformer_report = getattr(
        portfolio_result,
        "transformer_training_report",
        {"status": "skipped", "reason": "not available"},
    )
    transformer_inference_report = getattr(
        portfolio_result,
        "transformer_inference_report",
        {"status": "skipped", "reason": "not available"},
    )
    alpha_report = getattr(
        portfolio_result,
        "alpha_analysis_report",
        {"status": "skipped", "reason": "not available"},
    )
    return {
        "run_id": generated_at.strftime("%Y%m%d_%H%M%S"),
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "market_summary": market_summary,
        "market_state": agent_outputs.macro.model_dump(),
        "stock_signals": [item.model_dump() for item in agent_outputs.stocks],
        "model_risk": agent_outputs.model_risk.model_dump(),
        "deterministic_risk": portfolio_result.risk.model_dump(),
        "backtest_report": portfolio_result.backtest_report,
        "qlib_backtest_report": qlib_report,
        "qlib_positions": qlib_report.get("positions", []),
        "qlib_trade_records": qlib_report.get("trade_records", []),
        "benchmark_comparison": qlib_report.get("benchmark_comparison", {}),
        "transformer_training_report": transformer_report,
        "transformer_inference_report": transformer_inference_report,
        "alpha_analysis_report": alpha_report,
        "llm_explanation": {
            "macro": agent_outputs.macro.reason,
            "stocks": [
                {
                    "stock_code": item.stock_code,
                    "action": item.action,
                    "reason": item.reason,
                }
                for item in agent_outputs.stocks
            ],
            "risk_adjustments_from_model": agent_outputs.model_risk.adjustments,
            "risk_adjustments_from_rules": portfolio_result.risk.adjustments,
        },
        "raw_agent_outputs": {
            "macro": agent_outputs.macro_raw,
            "stock": agent_outputs.stock_raw,
            "risk": agent_outputs.risk_raw,
        },
        "conversation_history": [
            {"name": item.get("name", "System"), "content": item.get("content", "")}
            for item in messages
        ],
    }


def _json_block(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def _parse_json_text(text):
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None


def _role_text(value):
    if isinstance(value, dict):
        return _json_block(value)
    return _safe_text(value)


def _strategy_validation_rows(strategy_validation):
    strategy_validation = strategy_validation or {}
    return [
        ["validation_available", strategy_validation.get("validation_available")],
        ["rule_backtest_support", strategy_validation.get("rule_backtest_support")],
        ["transformer_backtest_support", strategy_validation.get("transformer_backtest_support")],
        ["hybrid_improvement", strategy_validation.get("hybrid_improvement")],
        ["benchmark_outperformance", strategy_validation.get("benchmark_outperformance")],
        ["drawdown_warning", strategy_validation.get("drawdown_warning")],
        ["rule_weakness", strategy_validation.get("rule_weakness")],
        ["transformer_return_advantage_pct", strategy_validation.get("transformer_return_advantage_pct")],
        ["transformer_offsets_rule_weakness", strategy_validation.get("transformer_offsets_rule_weakness")],
        ["exposure_adjustment", strategy_validation.get("exposure_adjustment")],
        ["exposure_multiplier", strategy_validation.get("exposure_multiplier")],
        ["agent_research_conclusion", strategy_validation.get("agent_research_conclusion")],
        ["validation_reason", strategy_validation.get("validation_reason")],
    ]


def _tradingagents_role_sections(conversation_history):
    sections = []
    role_specs = [
        ("Market Analyst 技术/市场分析", lambda state: state.get("market_report", "")),
        ("Fundamentals Analyst 基本面分析", lambda state: state.get("fundamentals_report", "")),
        ("News Analyst 新闻分析", lambda state: state.get("news_report", "")),
        ("Sentiment Analyst 情绪分析", lambda state: state.get("sentiment_report", "")),
        (
            "Bull Researcher 看多观点",
            lambda state: (state.get("investment_debate_state") or {}).get("bull_history", ""),
        ),
        (
            "Bear Researcher 看空观点",
            lambda state: (state.get("investment_debate_state") or {}).get("bear_history", ""),
        ),
        (
            "Research Manager 多空裁决",
            lambda state: state.get("investment_plan")
            or (state.get("investment_debate_state") or {}).get("judge_decision", ""),
        ),
        ("Trader 交易员建议", lambda state: state.get("trader_investment_plan", "")),
        (
            "Aggressive Risk Analyst 激进风险观点",
            lambda state: (state.get("risk_debate_state") or {}).get("aggressive_history", ""),
        ),
        (
            "Conservative Risk Analyst 保守风险观点",
            lambda state: (state.get("risk_debate_state") or {}).get("conservative_history", ""),
        ),
        (
            "Neutral Risk Analyst 中性风险观点",
            lambda state: (state.get("risk_debate_state") or {}).get("neutral_history", ""),
        ),
        ("Portfolio Manager 最终组合决策", lambda state: state.get("final_trade_decision", "")),
    ]

    for message in conversation_history:
        name = message.get("name", "")
        if not name.startswith("TradingAgentsLangGraph:"):
            continue
        symbol = name.split(":", 1)[1]
        payload = _parse_json_text(message.get("content", ""))
        if not payload:
            continue
        state = payload.get("state") or {}
        strategy_validation = payload.get("strategy_validation") or {}
        sections.append(f"### {symbol}")
        sections.append(
            _markdown_table(
                ["Field", "Value"],
                [
                    ["source", payload.get("source")],
                    ["action", payload.get("action")],
                    ["rating", payload.get("rating")],
                    ["weight", payload.get("weight")],
                ],
            )
        )
        sections.append("#### 策略验证字段")
        sections.append(_markdown_table(["Field", "Value"], _strategy_validation_rows(strategy_validation)))
        for title, getter in role_specs:
            text = _role_text(getter(state))
            if not text:
                text = "No output captured for this role."
            sections.extend([f"#### {title}", "```text", text, "```"])

    if not sections:
        return "No TradingAgents LangGraph role reports captured."
    return "\n\n".join(sections)


def render_markdown_report(payload):
    market_state = payload["market_state"]
    stock_signals = payload["stock_signals"]
    risk_result = payload["deterministic_risk"]
    backtest_report = payload["backtest_report"]
    qlib_backtest_report = payload.get("qlib_backtest_report", {})
    transformer_report = payload.get("transformer_training_report", {})
    transformer_inference_report = payload.get("transformer_inference_report", {})
    alpha_report = payload.get("alpha_analysis_report", {})
    llm_explanation = payload["llm_explanation"]

    stock_rows = [
        [
            item["stock_code"],
            item["action"],
            _format_percent(item["weight"]),
            _safe_text(item["reason"]),
        ]
        for item in stock_signals
    ]
    if not stock_rows:
        stock_rows = [["None", "None", "0.00%", "No stock signal generated"]]

    order_rows = [
        [
            item["stock_code"],
            item["direction"],
            _format_percent(item["quantity_percent"]),
            _format_percent(item["stop_loss_percent"]),
        ]
        for item in risk_result["final_orders"]
    ]
    if not order_rows:
        order_rows = [["None", "None", "0.00%", "0.00%"]]

    rule_adjustments = risk_result["adjustments"] or ["No deterministic risk adjustment"]
    model_adjustments = payload["model_risk"]["adjustments"] or ["No model risk adjustment"]
    stock_explanations = [
        f"- {item['stock_code']}: {item['action']}, {_safe_text(item['reason'])}"
        for item in llm_explanation["stocks"]
    ]

    sections = [
        f"# Daily Quant Research Report - {payload['generated_at'][:10]}",
        f"Generated at: {payload['generated_at']}",
        "## Market State",
        f"- Market environment: {market_state['market_environment']}",
        f"- Suggested total position: {_format_percent(market_state['suggested_position'])}",
        f"- LLM explanation: {_safe_text(market_state['reason'])}",
        "### Market Data Summary",
        "```text",
        _safe_text(payload["market_summary"]),
        "```",
        "## Stock Signals",
        _markdown_table(["Stock", "Action", "Suggested Weight", "Reason"], stock_rows),
        "## Deterministic Risk Result",
        f"- Approved: {risk_result['approved']}",
        "- Rule adjustments:",
        "\n".join(f"  - {item}" for item in rule_adjustments),
        "### Final Orders",
        _markdown_table(["Stock", "Direction", "Weight", "Stop Loss"], order_rows),
        "## Lightweight Backtest",
        _markdown_table(["Metric", "Value"], _backtest_rows(backtest_report)),
        "## Qlib Professional Backtest",
        _markdown_table(["Metric", "Value"], build_qlib_report_rows(qlib_backtest_report)),
        "### Qlib Positions",
        "```json",
        _json_block(payload.get("qlib_positions", [])[:10]),
        "```",
        "### Qlib Trade Records",
        "```json",
        _json_block(payload.get("qlib_trade_records", [])[:10]),
        "```",
        "### Benchmark Comparison",
        "```json",
        _json_block(payload.get("benchmark_comparison", {})),
        "```",
        "## Transformer Research",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["status", transformer_report.get("status")],
                ["model", transformer_report.get("model")],
                ["device", transformer_report.get("device")],
                ["sample_count", transformer_report.get("dataset", {}).get("sample_count")],
                ["date_count", transformer_report.get("dataset", {}).get("date_count")],
                ["model_path", transformer_report.get("model_path")],
            ],
        ),
        "### Transformer Inference",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["status", transformer_inference_report.get("status")],
                ["test_ic", transformer_inference_report.get("metrics", {}).get("test_ic")],
                ["test_rank_ic", transformer_inference_report.get("metrics", {}).get("test_rank_ic")],
                ["prediction_path", transformer_inference_report.get("prediction_path")],
            ],
        ),
        "## Alpha Analysis",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["status", alpha_report.get("status")],
                ["sample_count", alpha_report.get("sample_count")],
                ["date_count", alpha_report.get("date_count")],
                ["instrument_count", alpha_report.get("instrument_count")],
                ["json_path", alpha_report.get("json_path")],
                ["markdown_path", alpha_report.get("markdown_path")],
                ["factor_selection_path", alpha_report.get("factor_selection_path")],
            ],
        ),
        "### Factor Selection",
        _markdown_table(
            ["Factor", "Validity", "Direction", "Weight", "Signal Weight", "RankIC", "Long-Short"],
            _alpha_factor_selection_rows(alpha_report),
        ),
        "## LLM Explanation",
        f"- Macro: {_safe_text(llm_explanation['macro'])}",
        "### Stocks",
        "\n".join(stock_explanations) if stock_explanations else "- None",
        "### Model Risk Notes",
        "\n".join(f"- {item}" for item in model_adjustments),
        "## TradingAgents 多角色研究过程",
        _tradingagents_role_sections(payload.get("conversation_history", [])),
        "## Raw Agent Outputs",
        "### Macro",
        "```json",
        _safe_text(payload["raw_agent_outputs"]["macro"]),
        "```",
        "### Stock",
        "```json",
        _safe_text(payload["raw_agent_outputs"]["stock"]),
        "```",
        "### Risk",
        "```json",
        _safe_text(payload["raw_agent_outputs"]["risk"]),
        "```",
    ]
    return "\n\n".join(sections) + "\n"


def save_daily_reports(market_summary, messages, agent_outputs, portfolio_result):
    generated_at = datetime.now()
    payload = build_report_payload(
        market_summary=market_summary,
        messages=messages,
        agent_outputs=agent_outputs,
        portfolio_result=portfolio_result,
        generated_at=generated_at,
    )

    report_dir = REPORTS_DIR / generated_at.strftime("%Y-%m-%d")
    report_dir.mkdir(parents=True, exist_ok=True)

    base_name = payload["run_id"]
    json_path = report_dir / f"{base_name}.json"
    markdown_path = report_dir / f"{base_name}.md"

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(payload), encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
