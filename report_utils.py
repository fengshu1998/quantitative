import json
from datetime import datetime
from pathlib import Path

from qlib_backtest_utils import build_qlib_report_rows


REPORTS_DIR = Path("reports")


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


def render_markdown_report(payload):
    market_state = payload["market_state"]
    stock_signals = payload["stock_signals"]
    risk_result = payload["deterministic_risk"]
    backtest_report = payload["backtest_report"]
    qlib_backtest_report = payload.get("qlib_backtest_report", {})
    transformer_report = payload.get("transformer_training_report", {})
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
                ["test_ic", transformer_report.get("metrics", {}).get("test_ic")],
                ["test_rank_ic", transformer_report.get("metrics", {}).get("test_rank_ic")],
                ["prediction_path", transformer_report.get("prediction_path")],
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
            ],
        ),
        "## LLM Explanation",
        f"- Macro: {_safe_text(llm_explanation['macro'])}",
        "### Stocks",
        "\n".join(stock_explanations) if stock_explanations else "- None",
        "### Model Risk Notes",
        "\n".join(f"- {item}" for item in model_adjustments),
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
