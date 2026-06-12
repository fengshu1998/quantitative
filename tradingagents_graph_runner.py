from __future__ import annotations

import json
import logging
import os
import re
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from config import (
    AGENT_PERFORMANCE_FEEDBACK_PATH,
    AGENT_SIGNAL_DIR,
    ALPHA_FACTOR_SELECTION_PATH,
    DATA_DIR,
    MAX_TOTAL_POSITION,
    QLIB_BACKTEST_COMPARISON_PATH,
    TRADINGAGENTS_CACHE_DIR,
    TRADINGAGENTS_GRAPH_ENABLED,
    TRADINGAGENTS_MAX_DEBATE_ROUNDS,
    TRADINGAGENTS_MAX_RISK_ROUNDS,
    TRADINGAGENTS_OUTPUT_LANGUAGE,
    TRADINGAGENTS_RESEARCH_TOP_N,
    TRADINGAGENTS_RESULTS_DIR,
    TRANSFORMER_PREDICTION_PATH,
)
from schemas import MacroAnalysis, RiskOrder, RiskReview, StockRecommendation
from storage_utils import load_dataframe, load_selected_candidates
from tradingagents_local_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_news,
    get_stock_data,
    get_verified_market_snapshot,
)


logger = logging.getLogger(__name__)

VENDOR_ROOT = Path(__file__).resolve().parent / "third_party" / "tradingagents"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager  # noqa: E402
from tradingagents.agents.managers.research_manager import create_research_manager  # noqa: E402
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher  # noqa: E402
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher  # noqa: E402
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator  # noqa: E402
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator  # noqa: E402
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator  # noqa: E402
from tradingagents.agents.trader.trader import create_trader  # noqa: E402
from tradingagents.agents.utils.agent_states import AgentState  # noqa: E402
from tradingagents.agents.utils.agent_utils import create_msg_delete  # noqa: E402
from tradingagents.agents.utils.rating import parse_rating  # noqa: E402
from tradingagents.dataflows.config import set_config  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.graph.conditional_logic import ConditionalLogic  # noqa: E402
from tradingagents.graph.propagation import Propagator  # noqa: E402
from tradingagents.llm_clients import create_llm_client  # noqa: E402


load_dotenv("environment.env")

_LATEST_RESULT: dict[str, Any] | None = None
_GRAPH_CACHE: Any | None = None
_LLM_CACHE: tuple[Any, Any] | None = None


def _json_default(value):
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _to_float(value, default=0.0):
    if value is None or pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _target_weight_percent(candidate: dict[str, Any]) -> float:
    value = _to_float(candidate.get("target_weight"), 0.0)
    if value <= 1:
        return round(value * 100, 2)
    return round(value, 2)


def _limit_candidates(candidates: list[dict]) -> list[dict]:
    if TRADINGAGENTS_RESEARCH_TOP_N is None:
        return candidates
    return candidates[: int(TRADINGAGENTS_RESEARCH_TOP_N)]


def load_tradingagents_candidates() -> list[dict]:
    try:
        df = load_selected_candidates()
    except FileNotFoundError:
        logger.warning("selected_candidates.csv not found; TradingAgents has no candidates.")
        return []
    except Exception as e:
        logger.warning("Failed to load selected candidates: %s", e)
        return []

    if df.empty:
        return []
    candidates = df.where(pd.notna(df), None).to_dict("records")
    return _limit_candidates(candidates)


def _tradingagents_config() -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "results_dir": str(TRADINGAGENTS_RESULTS_DIR),
            "data_cache_dir": str(TRADINGAGENTS_CACHE_DIR),
            "memory_log_path": str(DATA_DIR / "tradingagents_memory" / "trading_memory.md"),
            "llm_provider": "deepseek",
            "deep_think_llm": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "quick_think_llm": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "backend_url": os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            "output_language": TRADINGAGENTS_OUTPUT_LANGUAGE,
            "max_debate_rounds": TRADINGAGENTS_MAX_DEBATE_ROUNDS,
            "max_risk_discuss_rounds": TRADINGAGENTS_MAX_RISK_ROUNDS,
            "max_recur_limit": 80,
            "checkpoint_enabled": False,
            "temperature": 0.1,
        }
    )
    Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["data_cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["memory_log_path"]).parent.mkdir(parents=True, exist_ok=True)
    set_config(config)
    return config


def _get_llms() -> tuple[Any, Any]:
    global _LLM_CACHE
    if _LLM_CACHE is not None:
        return _LLM_CACHE

    config = _tradingagents_config()
    quick_client = create_llm_client(
        provider=config["llm_provider"],
        model=config["quick_think_llm"],
        base_url=config.get("backend_url"),
        temperature=config.get("temperature"),
    )
    deep_client = create_llm_client(
        provider=config["llm_provider"],
        model=config["deep_think_llm"],
        base_url=config.get("backend_url"),
        temperature=config.get("temperature"),
    )
    _LLM_CACHE = (quick_client.get_llm(), deep_client.get_llm())
    return _LLM_CACHE


def _latest_factor_snapshot(symbol: str) -> dict[str, Any]:
    try:
        df = load_dataframe("factors", symbol)
    except Exception:
        return {}
    if df.empty:
        return {}
    return df.iloc[-1].where(pd.notna(df.iloc[-1]), None).to_dict()


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _snapshot_date_from_candidates(candidates: list[dict]) -> str:
    dates = []
    for candidate in candidates:
        value = candidate.get("date") or candidate.get("datetime") or candidate.get("trade_date")
        if value is None:
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            dates.append(pd.Timestamp(parsed))
    if dates:
        return max(dates).date().isoformat()
    return date.today().isoformat()


def save_agent_signal_snapshot(
    stock_recommendations: list[StockRecommendation],
    agent_results: list[dict[str, Any]] | None = None,
    candidates: list[dict] | None = None,
    snapshot_date: str | None = None,
) -> dict[str, Any]:
    """Persist final agent-adjusted actions for forward-only Qlib backtests."""

    candidates = candidates or []
    agent_results = agent_results or []
    trade_date = snapshot_date or _snapshot_date_from_candidates(candidates)
    results_by_symbol = {
        str(result.get("ticker") or result.get("symbol") or "").lower(): result
        for result in agent_results
    }
    signals = []
    for recommendation in stock_recommendations:
        symbol = recommendation.stock_code.lower()
        result = results_by_symbol.get(symbol, {})
        signals.append(
            {
                "date": trade_date,
                "stock_code": recommendation.stock_code,
                "action": recommendation.action,
                "weight": recommendation.weight,
                "reason": recommendation.reason,
                "strategy_validation": result.get("strategy_validation"),
                "strategy_validation_effect": result.get("strategy_validation_effect"),
            }
        )

    AGENT_SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    path = AGENT_SIGNAL_DIR / f"{trade_date}.json"
    payload = {
        "date": trade_date,
        "generated_at": date.today().isoformat(),
        "signal_count": len(signals),
        "signals": signals,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, default=_json_default, indent=2), encoding="utf-8")
    return {"status": "ok", "path": str(path), "date": trade_date, "signal_count": len(signals)}


def _latest_transformer_prediction(symbol: str) -> dict[str, Any]:
    if not TRANSFORMER_PREDICTION_PATH.exists():
        return {}
    try:
        df = pd.read_csv(TRANSFORMER_PREDICTION_PATH)
    except Exception:
        return {}
    if df.empty:
        return {}
    if "symbol" not in df.columns and "instrument" in df.columns:
        df["symbol"] = df["instrument"].astype(str).str.upper().str.replace(
            r"^(SH|SZ)(\d{6})$",
            lambda match: match.group(1).lower() + match.group(2),
            regex=True,
        )
    if "symbol" not in df.columns:
        return {}
    matched = df[df["symbol"].astype(str).str.lower() == str(symbol).lower()]
    if matched.empty:
        return {}
    if "date" in matched.columns:
        matched = matched.sort_values("date")
    elif "datetime" in matched.columns:
        matched = matched.sort_values("datetime")
    return matched.iloc[-1].where(pd.notna(matched.iloc[-1]), None).to_dict()


def _metric_float(report: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = report.get(key, default)
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _benchmark_excess_return(report: dict[str, Any]) -> float:
    benchmark = report.get("benchmark_comparison") or {}
    return _metric_float(benchmark, "excess_return_percent", 0.0)


def _backtest_support(report: dict[str, Any]) -> str:
    if not report or report.get("status") != "ok":
        return "neutral"
    total_return = _metric_float(report, "total_return_percent", 0.0)
    sharpe = _metric_float(report, "sharpe", 0.0)
    max_drawdown = abs(_metric_float(report, "max_drawdown_percent", 0.0))
    excess_return = _benchmark_excess_return(report)

    if total_return > 0 and sharpe > 0.5 and excess_return > 0:
        return "positive"
    if total_return < 0 or excess_return < -3.0 or max_drawdown > 15.0:
        return "negative"
    return "neutral"


TRANSFORMER_RETURN_ADVANTAGE_THRESHOLD = 3.0
REDUCED_EXPOSURE_MULTIPLIER = 0.5


def _valid_report(report: dict[str, Any]) -> bool:
    return bool(report) and report.get("status") == "ok"


def _live_first_report(
    signal_reports: dict[str, Any],
    live_key: str,
    fallback_key: str,
) -> tuple[dict[str, Any], str]:
    live = signal_reports.get(live_key) or {}
    if _valid_report(live):
        return live, "live_snapshot"
    fallback = signal_reports.get(fallback_key) or {}
    return fallback, "walk_forward_fallback"


def build_strategy_validation(qlib_comparison: dict[str, Any] | None = None) -> dict[str, Any]:
    if qlib_comparison is None:
        qlib_comparison = _read_json_file(Path(QLIB_BACKTEST_COMPARISON_PATH))
    signal_reports = (qlib_comparison or {}).get("signal_reports") or {}
    rule = signal_reports.get("rule") or {}
    transformer, transformer_scope = _live_first_report(signal_reports, "transformer_live", "transformer")
    hybrid, hybrid_scope = _live_first_report(signal_reports, "hybrid_live", "hybrid")
    validation_signal_scope = "live_snapshot" if transformer_scope == "live_snapshot" or hybrid_scope == "live_snapshot" else "walk_forward_fallback"
    validation_available = bool(rule)

    rule_support = _backtest_support(rule)
    transformer_support = _backtest_support(transformer)
    rule_return = _metric_float(rule, "total_return_percent", 0.0)
    transformer_return = _metric_float(transformer, "total_return_percent", 0.0)
    hybrid_return = _metric_float(hybrid, "total_return_percent", 0.0)
    rule_drawdown = abs(_metric_float(rule, "max_drawdown_percent", 0.0))
    transformer_drawdown = abs(_metric_float(transformer, "max_drawdown_percent", 0.0))
    hybrid_drawdown = abs(_metric_float(hybrid, "max_drawdown_percent", 0.0))

    hybrid_improvement = bool(hybrid) and hybrid_return > rule_return and hybrid_drawdown <= rule_drawdown + 5.0
    benchmark_outperformance = bool(rule) and _benchmark_excess_return(rule) > 0
    rule_drawdown_warning = bool(rule) and rule_drawdown > 15.0
    transformer_drawdown_warning = bool(transformer) and transformer_drawdown > 15.0
    hybrid_drawdown_warning = bool(hybrid) and hybrid_drawdown > 15.0
    hybrid_vs_rule_drawdown_warning = bool(hybrid) and hybrid_drawdown > rule_drawdown + 5.0
    drawdown_warning = (
        rule_drawdown_warning
        or transformer_drawdown_warning
        or hybrid_drawdown_warning
        or hybrid_vs_rule_drawdown_warning
    )
    rule_weakness = rule_support == "negative" or rule_drawdown_warning
    transformer_return_advantage_pct = round(transformer_return - rule_return, 2)
    transformer_offsets_rule_weakness = (
        rule_weakness
        and transformer_support == "positive"
        and transformer_return_advantage_pct >= TRANSFORMER_RETURN_ADVANTAGE_THRESHOLD
    )

    if rule_support == "positive" and benchmark_outperformance and not drawdown_warning:
        conclusion = "support"
        exposure_adjustment = "none"
        exposure_multiplier = 1.0
    elif transformer_offsets_rule_weakness:
        conclusion = "cautious"
        exposure_adjustment = "reduce"
        exposure_multiplier = REDUCED_EXPOSURE_MULTIPLIER
    elif rule_support == "negative" or drawdown_warning:
        conclusion = "downgrade"
        exposure_adjustment = "zero"
        exposure_multiplier = 0.0
    else:
        conclusion = "cautious"
        exposure_adjustment = "none"
        exposure_multiplier = 1.0

    reason_parts = [
        f"validation_signal_scope={validation_signal_scope}",
        f"rule={rule_support}",
        f"transformer={transformer_support}",
        f"transformer_source={transformer_scope}",
        f"hybrid_source={hybrid_scope}",
        f"hybrid_improvement={hybrid_improvement}",
        f"benchmark_outperformance={benchmark_outperformance}",
        f"drawdown_warning={drawdown_warning}",
        f"rule_weakness={rule_weakness}",
        f"transformer_return_advantage_pct={transformer_return_advantage_pct}",
        f"transformer_offsets_rule_weakness={transformer_offsets_rule_weakness}",
        f"exposure_adjustment={exposure_adjustment}",
    ]
    if not validation_available:
        reason_parts.append("qlib comparison report unavailable")

    return {
        "validation_available": validation_available,
        "validation_signal_scope": validation_signal_scope,
        "transformer_validation_source": transformer_scope,
        "hybrid_validation_source": hybrid_scope,
        "rule_backtest_support": rule_support,
        "transformer_backtest_support": transformer_support,
        "hybrid_improvement": hybrid_improvement,
        "benchmark_outperformance": benchmark_outperformance,
        "drawdown_warning": drawdown_warning,
        "rule_weakness": rule_weakness,
        "transformer_return_advantage_pct": transformer_return_advantage_pct,
        "transformer_offsets_rule_weakness": transformer_offsets_rule_weakness,
        "exposure_adjustment": exposure_adjustment,
        "exposure_multiplier": exposure_multiplier,
        "agent_research_conclusion": conclusion,
        "validation_reason": "; ".join(reason_parts),
    }


def build_tradingagents_context(market_summary: str, candidate: dict) -> str:
    symbol = str(candidate.get("symbol") or "")
    snapshot = _latest_factor_snapshot(symbol)
    factor_selection = _read_json_file(Path(ALPHA_FACTOR_SELECTION_PATH))
    qlib_comparison = _read_json_file(Path(QLIB_BACKTEST_COMPARISON_PATH))
    agent_feedback = _read_json_file(Path(AGENT_PERFORMANCE_FEEDBACK_PATH))
    transformer_prediction = _latest_transformer_prediction(symbol)
    strategy_validation = build_strategy_validation(qlib_comparison)
    context = {
        "market_summary": market_summary,
        "candidate": candidate,
        "latest_factor_snapshot": snapshot,
        "alpha_factor_selection": factor_selection,
        "latest_transformer_prediction": transformer_prediction,
        "latest_qlib_backtest_comparison": qlib_comparison,
        "agent_performance_feedback": agent_feedback,
        "strategy_validation": strategy_validation,
        "data_policy": "Use only local A-share project data returned by tools. News and sentiment may be unavailable.",
    }
    return json.dumps(context, ensure_ascii=False, default=_json_default, indent=2)


def _instrument_context(candidate: dict, market_summary: str) -> str:
    symbol = str(candidate.get("symbol") or "")
    name = candidate.get("name") or symbol
    industry = candidate.get("industry") or "unknown"
    target_weight = _target_weight_percent(candidate)
    return (
        f"The A-share instrument to analyze is `{symbol}`. "
        f"Company/name: {name}. Industry: {industry}. "
        "Use this exact sh/sz symbol in all local tool calls. "
        f"The deterministic candidate target weight is {target_weight:.2f}%. "
        "The local project context is the source of truth; do not use yfinance, US ticker data, "
        "or external news assumptions. "
        f"Local market summary:\n{market_summary}"
    )


def _create_tool_analyst(llm, name: str, report_key: str, tools: list, system_message: str):
    def analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = state.get("instrument_context", "")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a TradingAgents analyst node in a LangGraph workflow. "
                    "Use the available local A-share tools when needed, and write the final report in Chinese. "
                    "Never rely on yfinance or US-market assumptions. "
                    "You must explicitly cite strategy_validation fields when they are present: "
                    "rule_backtest_support, transformer_backtest_support, hybrid_improvement, "
                    "benchmark_outperformance, drawdown_warning, and agent_research_conclusion. "
                    "If agent_performance_feedback is present, treat it as historical performance context "
                    "for risk discipline only; it must not override strategy_validation or deterministic risk rules. "
                    "Available tools: {tool_names}.\n\n"
                    "{system_message}\n\n"
                    "Current date: {current_date}.\n"
                    "{instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result = (prompt | llm.bind_tools(tools)).invoke(state["messages"])
        report = "" if getattr(result, "tool_calls", None) else str(result.content)
        return {
            "messages": [result],
            report_key: report,
            "sender": name,
        }

    return analyst_node


def _create_news_node():
    def news_node(state):
        report = (
            "新闻节点结论：当前项目尚未接入新闻/RAG 数据。"
            "本轮 TradingAgents 研究仅使用本地行情、因子、财务、行业和候选池信息，"
            "因此不得基于未提供的新闻催化作出判断。"
        )
        return {"messages": [AIMessage(content=report)], "news_report": report, "sender": "News Analyst"}

    return news_node


def _create_sentiment_node():
    def sentiment_node(state):
        report = (
            "情绪节点结论：当前项目尚未接入社媒/舆情数据。"
            "本轮情绪信号标记为 unavailable，后续多空辩论不得臆造市场情绪证据。"
        )
        return {
            "messages": [AIMessage(content=report)],
            "sentiment_report": report,
            "sender": "Sentiment Analyst",
        }

    return sentiment_node


def _build_graph():
    quick_llm, deep_llm = _get_llms()
    conditional_logic = ConditionalLogic(
        max_debate_rounds=TRADINGAGENTS_MAX_DEBATE_ROUNDS,
        max_risk_discuss_rounds=TRADINGAGENTS_MAX_RISK_ROUNDS,
    )

    market_tools = [get_stock_data, get_indicators, get_verified_market_snapshot]
    fundamentals_tools = [get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement]

    workflow = StateGraph(AgentState)
    workflow.add_node(
        "Market Analyst",
        _create_tool_analyst(
            quick_llm,
            "Market Analyst",
            "market_report",
            market_tools,
            (
                "Analyze local A-share price action, technical factors, deterministic cross_section_score, "
                "risk_flag, trend, volatility, drawdown, and target weight. Call get_stock_data, "
                "get_indicators, and get_verified_market_snapshot before writing the final report."
            ),
        ),
    )
    workflow.add_node("tools_market", ToolNode(market_tools))
    workflow.add_node("Msg Clear Market", create_msg_delete())

    workflow.add_node(
        "Fundamentals Analyst",
        _create_tool_analyst(
            quick_llm,
            "Fundamentals Analyst",
            "fundamentals_report",
            fundamentals_tools,
            (
                "Analyze local A-share fundamentals including PE, PB, ROE, market cap, revenue growth, "
                "profit growth, debt_to_asset, and fundamental_available. If detailed statements are "
                "unavailable, explicitly say so instead of inventing data."
            ),
        ),
    )
    workflow.add_node("tools_fundamentals", ToolNode(fundamentals_tools))
    workflow.add_node("Msg Clear Fundamentals", create_msg_delete())

    workflow.add_node("News Analyst", _create_news_node())
    workflow.add_node("Msg Clear News", create_msg_delete())
    workflow.add_node("Sentiment Analyst", _create_sentiment_node())
    workflow.add_node("Msg Clear Sentiment", create_msg_delete())

    workflow.add_node("Bull Researcher", create_bull_researcher(quick_llm))
    workflow.add_node("Bear Researcher", create_bear_researcher(quick_llm))
    workflow.add_node("Research Manager", create_research_manager(deep_llm))
    workflow.add_node("Trader", create_trader(quick_llm))
    workflow.add_node("Aggressive Analyst", create_aggressive_debator(quick_llm))
    workflow.add_node("Conservative Analyst", create_conservative_debator(quick_llm))
    workflow.add_node("Neutral Analyst", create_neutral_debator(quick_llm))
    workflow.add_node("Portfolio Manager", create_portfolio_manager(deep_llm))

    workflow.add_edge(START, "Market Analyst")
    workflow.add_conditional_edges(
        "Market Analyst",
        conditional_logic.should_continue_market,
        ["tools_market", "Msg Clear Market"],
    )
    workflow.add_edge("tools_market", "Market Analyst")
    workflow.add_edge("Msg Clear Market", "Fundamentals Analyst")

    workflow.add_conditional_edges(
        "Fundamentals Analyst",
        conditional_logic.should_continue_fundamentals,
        ["tools_fundamentals", "Msg Clear Fundamentals"],
    )
    workflow.add_edge("tools_fundamentals", "Fundamentals Analyst")
    workflow.add_edge("Msg Clear Fundamentals", "News Analyst")

    workflow.add_edge("News Analyst", "Msg Clear News")
    workflow.add_edge("Msg Clear News", "Sentiment Analyst")
    workflow.add_edge("Sentiment Analyst", "Msg Clear Sentiment")
    workflow.add_edge("Msg Clear Sentiment", "Bull Researcher")

    workflow.add_conditional_edges(
        "Bull Researcher",
        conditional_logic.should_continue_debate,
        {"Bear Researcher": "Bear Researcher", "Research Manager": "Research Manager"},
    )
    workflow.add_conditional_edges(
        "Bear Researcher",
        conditional_logic.should_continue_debate,
        {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"},
    )
    workflow.add_edge("Research Manager", "Trader")
    workflow.add_edge("Trader", "Aggressive Analyst")
    workflow.add_conditional_edges(
        "Aggressive Analyst",
        conditional_logic.should_continue_risk_analysis,
        {"Conservative Analyst": "Conservative Analyst", "Portfolio Manager": "Portfolio Manager"},
    )
    workflow.add_conditional_edges(
        "Conservative Analyst",
        conditional_logic.should_continue_risk_analysis,
        {"Neutral Analyst": "Neutral Analyst", "Portfolio Manager": "Portfolio Manager"},
    )
    workflow.add_conditional_edges(
        "Neutral Analyst",
        conditional_logic.should_continue_risk_analysis,
        {"Aggressive Analyst": "Aggressive Analyst", "Portfolio Manager": "Portfolio Manager"},
    )
    workflow.add_edge("Portfolio Manager", END)
    return workflow.compile()


def _get_graph():
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        _GRAPH_CACHE = _build_graph()
    return _GRAPH_CACHE


_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def _load_skill(name: str) -> str:
    path = _SKILLS_DIR / name / "SKILL.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _initial_state(candidate: dict, market_summary: str) -> dict[str, Any]:
    symbol = str(candidate.get("symbol") or "")
    context = build_tradingagents_context(market_summary, candidate)
    buffett = _load_skill("buffett")
    skill_block = ""
    if buffett:
        skill_block = (
            "\n\n## Investment Framework\n"
            "All agents must incorporate the following Buffett investment framework "
            "into their analysis and final decision:\n\n"
            f"{buffett}\n"
        )
    state = Propagator().create_initial_state(
        company_name=symbol,
        trade_date=date.today().isoformat(),
        asset_type="stock",
        past_context="",
        instrument_context=_instrument_context(candidate, market_summary),
    )
    state["messages"] = [
        HumanMessage(
            content=(
                f"Run the full TradingAgents LangGraph research process for {symbol}.\n\n"
                "Every analyst, researcher, trader, risk debater, and portfolio manager must explicitly "
                "reference the strategy_validation fields if they are available. The Portfolio Manager "
                "must state whether agent_research_conclusion is support, cautious, or downgrade.\n\n"
                f"Local context:\n{context}{skill_block}"
            )
        )
    ]
    return state


def _fallback_research(candidate: dict, reason: str) -> dict[str, Any]:
    signal = str(candidate.get("signal") or "HOLD").upper()
    action = {"BUY": "buy", "SELL": "sell"}.get(signal, "hold")
    strategy_validation = build_strategy_validation()
    weight = _target_weight_percent(candidate) if action == "buy" else 0.0
    action, weight, validation_effect = _apply_strategy_validation_to_action(action, weight, strategy_validation)
    validation_note = f" strategy_validation={strategy_validation.get('validation_reason')}"
    return {
        "ticker": candidate.get("symbol"),
        "action": action,
        "weight": weight,
        "reason": (
            f"TradingAgents LangGraph failed, fallback to deterministic signal. "
            f"signal={signal}; reason={candidate.get('signal_reason') or reason}."
            f"{validation_note}"
            + _strategy_validation_action_note(validation_effect)
        ),
        "risk_notes": [f"fallback: {reason}"],
        "confidence": 0.0,
        "source": "fallback",
        "state": {},
        "strategy_validation": strategy_validation,
        "strategy_validation_effect": validation_effect,
    }


def _parse_pm_weight(text: str) -> float | None:
    candidates = []
    for pattern in (
        r"position sizing.*?(\d+(?:\.\d+)?)\s*%",
        r"position.*?(\d+(?:\.\d+)?)\s*%",
        r"仓位.*?(\d+(?:\.\d+)?)\s*%",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            candidates.append(float(match.group(1)))
    if not candidates:
        return None
    return min(candidates)


def _rating_to_action(rating: str) -> str:
    normalized = str(rating or "").lower()
    if normalized in {"buy", "overweight"}:
        return "buy"
    if normalized == "sell":
        return "sell"
    return "hold"


def _state_reason(final_state: dict[str, Any]) -> str:
    final_decision = str(final_state.get("final_trade_decision") or "")
    trader_plan = str(final_state.get("trader_investment_plan") or "")
    investment_plan = str(final_state.get("investment_plan") or "")
    parts = []
    if final_decision:
        parts.append(f"Portfolio Manager: {final_decision}")
    if trader_plan:
        parts.append(f"Trader: {trader_plan}")
    if investment_plan:
        parts.append(f"Research Manager: {investment_plan}")
    return "\n\n".join(parts) or "No TradingAgents state reason available."


def _apply_strategy_validation_to_action(
    action: str,
    weight: float,
    strategy_validation: dict[str, Any],
) -> tuple[str, float, str]:
    validation_conclusion = strategy_validation.get("agent_research_conclusion")
    exposure_adjustment = strategy_validation.get("exposure_adjustment")
    exposure_multiplier = _to_float(strategy_validation.get("exposure_multiplier"), 1.0)

    if validation_conclusion == "downgrade" and action == "buy":
        return "hold", 0.0, "downgraded"
    if exposure_adjustment == "reduce" and action == "buy":
        return action, max(weight * exposure_multiplier, 0.0), "reduced"
    return action, weight, "none"


def _strategy_validation_action_note(effect: str) -> str:
    if effect == "downgraded":
        return " Strategy validation downgraded original buy decision to hold and zero exposure."
    if effect == "reduced":
        return " Strategy validation kept buy exposure but reduced position size by 50%."
    return ""


def _state_to_result(candidate: dict, final_state: dict[str, Any]) -> dict[str, Any]:
    final_decision = str(final_state.get("final_trade_decision") or "")
    rating = parse_rating(final_decision, default="Hold")
    action = _rating_to_action(rating)
    deterministic_weight = _target_weight_percent(candidate)
    pm_weight = _parse_pm_weight(final_decision)
    weight = deterministic_weight if action == "buy" else 0.0
    if action == "buy" and pm_weight is not None:
        weight = min(weight, pm_weight)
    strategy_validation = build_strategy_validation()
    action, weight, validation_effect = _apply_strategy_validation_to_action(action, weight, strategy_validation)
    validation_conclusion = strategy_validation.get("agent_research_conclusion")

    risk_state = final_state.get("risk_debate_state", {}) or {}
    risk_notes = []
    if risk_state.get("history"):
        risk_notes.append(str(risk_state["history"]))
    if risk_state.get("judge_decision"):
        risk_notes.append(str(risk_state["judge_decision"]))

    return {
        "ticker": candidate.get("symbol"),
        "action": action,
        "rating": rating,
        "weight": round(weight, 2),
        "reason": _state_reason(final_state)
        + f"\n\nStrategy validation: {strategy_validation.get('validation_reason')}"
        + _strategy_validation_action_note(validation_effect)
        + (" Cautious validation, no action upgrade applied." if validation_conclusion == "cautious" else ""),
        "risk_notes": risk_notes,
        "confidence": 0.7 if action == "buy" else 0.5,
        "source": "tradingagents_langgraph",
        "state": final_state,
        "strategy_validation": strategy_validation,
        "strategy_validation_effect": validation_effect,
    }


def run_tradingagents_for_candidate(candidate: dict, context: str | None = None) -> dict:
    if not TRADINGAGENTS_GRAPH_ENABLED:
        return _fallback_research(candidate, "TRADINGAGENTS_GRAPH_ENABLED is False")
    symbol = candidate.get("symbol")
    try:
        graph = _get_graph()
        state = _initial_state(candidate, context or "")
        final_state = graph.invoke(state, config={"recursion_limit": 80})
        return _state_to_result(candidate, final_state)
    except Exception as e:
        logger.warning("TradingAgents LangGraph failed for %s: %s", symbol, e)
        return _fallback_research(candidate, str(e))


def map_tradingagents_to_stock_recommendation(
    result: dict,
    candidate: dict,
) -> StockRecommendation:
    action = str(result.get("action") or "hold").lower()
    if action not in {"buy", "sell", "hold"}:
        action = "hold"
    strategy_validation = result.get("strategy_validation") or build_strategy_validation()
    weight = _to_float(result.get("weight"), 0.0)
    validation_effect = str(result.get("strategy_validation_effect") or "")
    if validation_effect not in {"downgraded", "reduced", "none"}:
        action, weight, validation_effect = _apply_strategy_validation_to_action(action, weight, strategy_validation)
    if action != "buy":
        weight = 0.0
    reason = str(result.get("reason") or candidate.get("signal_reason") or "No reason provided.")
    if result.get("source") == "fallback":
        reason = f"[fallback] {reason}"
    if validation_effect == "downgraded" and "downgraded" not in reason.lower():
        reason = f"{reason}\n\nStrategy validation downgraded buy exposure to hold and zero exposure."
    elif validation_effect == "reduced" and "reduced" not in reason.lower():
        reason = f"{reason}\n\nStrategy validation kept buy exposure but reduced position size by 50%."
    elif strategy_validation.get("agent_research_conclusion") == "cautious" and "Strategy validation" not in reason:
        reason = f"{reason}\n\nStrategy validation is cautious; no action upgrade applied."
    return StockRecommendation(
        stock_code=str(candidate.get("symbol")),
        action=action,
        weight=round(weight, 2),
        reason=reason,
    )


def build_macro_from_candidate_set(
    market_summary: str,
    candidates: list[dict],
    agent_results: list[dict],
) -> MacroAnalysis:
    if not candidates:
        return MacroAnalysis(
            market_environment="defensive",
            suggested_position=0,
            reason="No ranked index-enhancement candidates are available; keep cash and skip new positions.",
        )

    buy_results = [item for item in agent_results if str(item.get("action")).lower() == "buy"]
    total_buy_weight = sum(_to_float(item.get("weight"), 0.0) for item in buy_results)
    suggested_position = min(total_buy_weight, MAX_TOTAL_POSITION * 100)
    buy_ratio = len(buy_results) / max(len(candidates), 1)

    if buy_ratio >= 0.6 and suggested_position > 0:
        environment = "offensive"
    elif buy_ratio == 0:
        environment = "defensive"
    else:
        environment = "neutral"

    return MacroAnalysis(
        market_environment=environment,
        suggested_position=round(suggested_position, 2),
        reason=(
            f"TradingAgents LangGraph analyzed {len(candidates)} ranked index-enhancement candidates; "
            f"{len(buy_results)} are buy-rated. Suggested position is capped by deterministic limits."
        ),
    )


def build_model_risk_from_agent_results(agent_results: list[dict]) -> RiskReview:
    final_orders = []
    adjustments = []
    for item in agent_results:
        action = str(item.get("action") or "hold").lower()
        symbol = str(item.get("ticker") or item.get("symbol") or "")
        if action not in {"buy", "sell"} or not symbol:
            continue
        weight = _to_float(item.get("weight"), 0.0)
        if action == "buy" and weight <= 0:
            continue
        if action == "sell" and weight <= 0:
            weight = 100.0
        final_orders.append(
            RiskOrder(
                stock_code=symbol,
                direction=action,
                quantity_percent=round(weight, 2),
                stop_loss_percent=8.0 if action == "buy" else 0.0,
            )
        )
        for note in item.get("risk_notes") or []:
            adjustments.append(str(note))
    return RiskReview(
        approved=bool(final_orders),
        adjustments=adjustments,
        final_orders=final_orders,
    )


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return {}
    return {
        "market_report": state.get("market_report", ""),
        "fundamentals_report": state.get("fundamentals_report", ""),
        "news_report": state.get("news_report", ""),
        "sentiment_report": state.get("sentiment_report", ""),
        "investment_debate_state": state.get("investment_debate_state", {}),
        "trader_investment_plan": state.get("trader_investment_plan", ""),
        "risk_debate_state": state.get("risk_debate_state", {}),
        "final_trade_decision": state.get("final_trade_decision", ""),
    }


def _research_messages(market_summary: str, agent_results: list[dict], macro, stock_recommendations, risk) -> list[dict]:
    messages = [{"name": "User", "content": market_summary}]
    for result in agent_results:
        symbol = result.get("ticker")
        source = result.get("source")
        messages.append(
            {
                "name": f"TradingAgentsLangGraph:{symbol}",
                "content": json.dumps(
                    {
                        "source": source,
                        "action": result.get("action"),
                        "rating": result.get("rating"),
                        "weight": result.get("weight"),
                        "reason": result.get("reason"),
                        "strategy_validation": result.get("strategy_validation"),
                        "state": _state_summary(result.get("state", {})),
                    },
                    ensure_ascii=False,
                    default=_json_default,
                    indent=2,
                ),
            }
        )
    messages.extend(
        [
            {"name": "TradingAgentsMacro", "content": macro.model_dump_json(indent=2)},
            {
                "name": "TradingAgentsMappedStocks",
                "content": json.dumps(
                    [item.model_dump() for item in stock_recommendations],
                    ensure_ascii=False,
                    indent=2,
                ),
            },
            {"name": "TradingAgentsRiskManager", "content": risk.model_dump_json(indent=2)},
        ]
    )
    return messages


def run_tradingagents_research(market_summary: str) -> list[dict]:
    candidates = load_tradingagents_candidates()
    agent_results = []

    for candidate in candidates:
        result = run_tradingagents_for_candidate(candidate, market_summary)
        agent_results.append(result)

    stock_recommendations = [
        map_tradingagents_to_stock_recommendation(result, candidate)
        for result, candidate in zip(agent_results, candidates)
    ]
    agent_signal_snapshot = save_agent_signal_snapshot(stock_recommendations, agent_results, candidates)
    macro = build_macro_from_candidate_set(market_summary, candidates, agent_results)
    risk = build_model_risk_from_agent_results(agent_results)

    macro_raw = macro.model_dump_json(indent=2)
    stock_raw = json.dumps(
        [item.model_dump() for item in stock_recommendations],
        ensure_ascii=False,
        indent=2,
    )
    risk_raw = risk.model_dump_json(indent=2)
    messages = _research_messages(market_summary, agent_results, macro, stock_recommendations, risk)

    global _LATEST_RESULT
    _LATEST_RESULT = {
        "messages": messages,
        "macro_raw": macro_raw,
        "macro": macro,
        "stock_raw": stock_raw,
        "stocks": stock_recommendations,
        "risk_raw": risk_raw,
        "model_risk": risk,
        "agent_results": agent_results,
        "agent_signal_snapshot": agent_signal_snapshot,
    }
    return messages


def get_latest_agent_payloads() -> dict[str, Any]:
    if _LATEST_RESULT is None:
        raise RuntimeError("TradingAgents LangGraph research has not been run yet.")
    return _LATEST_RESULT
