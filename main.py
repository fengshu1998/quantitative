import json
import logging
import sys
import warnings
from dataclasses import dataclass

# pyqlib 0.9.7 depends on gym (unmaintained since 2022); the RL module imports it
# but this project only uses backtest/training, so the deprecation warning is noise.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

from agents import get_latest_agent_payloads, run_tradingagents_research
from agent_feedback_utils import run_agent_feedback_update
from alpha_analysis_utils import run_alpha_analysis
from data_utils import build_market_snapshot_from_factors, refresh_factor_data
from gm_executor import execute_final_orders
from live_attribution_utils import run_live_attribution_update
from portfolio_utils import apply_deterministic_risk_rules
from qlib_backtest_utils import run_qlib_backtest
from qlib_training_utils import run_transformer_inference
from report_utils import save_daily_reports
from schemas import MacroAnalysis, RiskReview, StockRecommendation


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentOutputs:
    macro_raw: str
    macro: MacroAnalysis
    stock_raw: str
    stocks: list[StockRecommendation]
    risk_raw: str
    model_risk: RiskReview
    agent_signal_snapshot: dict


@dataclass(frozen=True)
class PortfolioResult:
    risk: RiskReview
    qlib_backtest_report: dict
    transformer_training_report: dict
    transformer_inference_report: dict
    alpha_analysis_report: dict
    live_attribution_report: dict
    agent_feedback_report: dict


@dataclass(frozen=True)
class ReportFiles:
    json: str
    markdown: str


@dataclass(frozen=True)
class MarketPreparation:
    market_summary: str
    alpha_analysis_report: dict
    transformer_training_report: dict
    transformer_inference_report: dict
    live_attribution_report: dict


def prepare_market_context():
    logger.info("读取实盘归因反馈")
    live_attribution_report = run_live_attribution_update()
    logger.info("实盘归因反馈: %s", json.dumps(live_attribution_report, ensure_ascii=False, indent=2))

    logger.info("刷新行情、财务和因子数据")
    refresh_result = refresh_factor_data()

    logger.info("前置 Alpha 分析并生成本轮 factor_selection")
    alpha_analysis_report = run_alpha_analysis()
    logger.info("Alpha 分析报告")
    print(json.dumps(alpha_analysis_report, ensure_ascii=False, indent=2))

    logger.info("Transformer 训练/推理")
    transformer_training_report = {
        "status": "skipped",
        "reason": "daily pipeline does not train Transformer; run live production training offline",
        "training_mode": "daily_disabled",
    }
    transformer_inference_report = run_transformer_inference()
    logger.info("Transformer 训练研究报告")
    print(json.dumps(transformer_training_report, ensure_ascii=False, indent=2))
    logger.info("Transformer 推理研究报告")
    print(json.dumps(transformer_inference_report, ensure_ascii=False, indent=2))

    logger.info("用最新 Alpha + Transformer 融合信号生成候选池")
    market_snapshot = build_market_snapshot_from_factors(refresh_result)
    market_summary = market_snapshot["summary_text"]
    print(market_summary)
    return MarketPreparation(
        market_summary=market_summary,
        alpha_analysis_report=alpha_analysis_report,
        transformer_training_report=transformer_training_report,
        transformer_inference_report=transformer_inference_report,
        live_attribution_report=live_attribution_report,
    )


def run_agent_research(market_summary):
    logger.info("启动 TradingAgents 研究层：前20%指数增强候选股 -> 研究图 -> schema 映射")
    return run_tradingagents_research(market_summary)


def print_conversation_history(messages):
    logger.info("完整研究过程")
    for i, message in enumerate(messages):
        speaker = message.get("name", "System")
        body = message.get("content", "")
        print(f"\n--- [{i + 1}] {speaker} ---")
        print(body[:500] + ("..." if len(body) > 500 else ""))


def collect_agent_outputs():
    payloads = get_latest_agent_payloads()

    macro_raw = payloads["macro_raw"]
    macro_result = payloads["macro"]
    logger.info("TradingAgents 宏观映射输出")
    print(macro_raw)

    stock_raw = payloads["stock_raw"]
    stock_results = payloads["stocks"]
    logger.info("TradingAgents 个股研究输出")
    print(stock_raw)

    risk_raw = payloads["risk_raw"]
    model_risk_result = payloads["model_risk"]
    logger.info("TradingAgents 风控映射输出")
    print(risk_raw)

    return AgentOutputs(
        macro_raw=macro_raw,
        macro=macro_result,
        stock_raw=stock_raw,
        stocks=stock_results,
        risk_raw=risk_raw,
        model_risk=model_risk_result,
        agent_signal_snapshot=payloads.get("agent_signal_snapshot", {}),
    )


def construct_portfolio(agent_outputs, preparation: MarketPreparation):
    risk_result = apply_deterministic_risk_rules(
        agent_outputs.macro,
        agent_outputs.stocks,
    )
    qlib_backtest_report = run_qlib_backtest()
    agent_feedback_report = run_agent_feedback_update(qlib_backtest_report)
    return PortfolioResult(
        risk=risk_result,
        qlib_backtest_report=qlib_backtest_report,
        transformer_training_report=preparation.transformer_training_report,
        transformer_inference_report=preparation.transformer_inference_report,
        alpha_analysis_report=preparation.alpha_analysis_report,
        live_attribution_report=preparation.live_attribution_report,
        agent_feedback_report=agent_feedback_report,
    )


def print_portfolio_result(portfolio_result):
    logger.info("Qlib 专业回测报告")
    print(json.dumps(portfolio_result.qlib_backtest_report, ensure_ascii=False, indent=2))

    logger.info("Transformer 训练研究报告")
    print(json.dumps(portfolio_result.transformer_training_report, ensure_ascii=False, indent=2))

    logger.info("Transformer 推理研究报告")
    print(json.dumps(portfolio_result.transformer_inference_report, ensure_ascii=False, indent=2))

    logger.info("Alpha 分析报告")
    print(json.dumps(portfolio_result.alpha_analysis_report, ensure_ascii=False, indent=2))
    logger.info("Agents Qlib feedback report")
    print(json.dumps(portfolio_result.agent_feedback_report, ensure_ascii=False, indent=2))

    logger.info("确定性风控后的最终交易指令")
    print(portfolio_result.risk.model_dump_json(indent=2))


def save_report_files(market_summary, messages, agent_outputs, portfolio_result):
    logger.info("保存 JSON 结果文件和 Markdown 报告")
    report_files = save_daily_reports(
        market_summary=market_summary,
        messages=messages,
        agent_outputs=agent_outputs,
        portfolio_result=portfolio_result,
    )
    logger.info("JSON 结果文件: %s", report_files["json"])
    logger.info("Markdown 报告: %s", report_files["markdown"])
    return ReportFiles(**report_files)


def build_pipeline_result(agent_outputs, portfolio_result, report_files):
    return {
        "macro": agent_outputs.macro.model_dump(),
        "stock": [item.model_dump() for item in agent_outputs.stocks],
        "model_risk": agent_outputs.model_risk.model_dump(),
        "risk": portfolio_result.risk.model_dump(),
        "qlib_backtest_report": portfolio_result.qlib_backtest_report,
        "transformer_training_report": portfolio_result.transformer_training_report,
        "transformer_inference_report": portfolio_result.transformer_inference_report,
        "alpha_analysis_report": portfolio_result.alpha_analysis_report,
        "agent_feedback_report": portfolio_result.agent_feedback_report,
        "report_files": {
            "json": report_files.json,
            "markdown": report_files.markdown,
        },
    }


def run_pipeline():
    preparation = prepare_market_context()
    market_summary = preparation.market_summary
    messages = run_agent_research(market_summary)
    print_conversation_history(messages)
    agent_outputs = collect_agent_outputs()
    portfolio_result = construct_portfolio(agent_outputs, preparation)
    print_portfolio_result(portfolio_result)
    report_files = save_report_files(
        market_summary,
        messages,
        agent_outputs,
        portfolio_result,
    )

    # 推送交易指令到掘金仿真/实盘账户
    execution_report = execute_final_orders(portfolio_result.risk.final_orders)
    logger.info("掘金执行结果: %s", json.dumps(execution_report, ensure_ascii=False, indent=2))

    pipeline_result = build_pipeline_result(agent_outputs, portfolio_result, report_files)
    pipeline_result["gm_execution_report"] = execution_report
    return pipeline_result


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        logger.exception("流水线执行失败: %s", e)
