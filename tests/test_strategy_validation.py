import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import pandas as pd


def _install_external_stubs():
    sys.modules.setdefault("akshare", types.SimpleNamespace())

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)

    messages = types.ModuleType("langchain_core.messages")

    class _Message:
        def __init__(self, content=""):
            self.content = content

    messages.AIMessage = _Message
    messages.HumanMessage = _Message
    sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
    sys.modules.setdefault("langchain_core.messages", messages)

    prompts = types.ModuleType("langchain_core.prompts")
    prompts.ChatPromptTemplate = type("ChatPromptTemplate", (), {"from_messages": classmethod(lambda cls, *a, **k: cls())})
    prompts.MessagesPlaceholder = lambda *args, **kwargs: None
    sys.modules.setdefault("langchain_core.prompts", prompts)

    graph = types.ModuleType("langgraph.graph")
    graph.END = "__end__"
    graph.START = "__start__"
    graph.StateGraph = type("StateGraph", (), {})
    sys.modules.setdefault("langgraph", types.ModuleType("langgraph"))
    sys.modules.setdefault("langgraph.graph", graph)

    prebuilt = types.ModuleType("langgraph.prebuilt")
    prebuilt.ToolNode = type("ToolNode", (), {})
    sys.modules.setdefault("langgraph.prebuilt", prebuilt)

    tools = types.ModuleType("tradingagents_local_tools")
    for name in [
        "get_balance_sheet",
        "get_cashflow",
        "get_fundamentals",
        "get_global_news",
        "get_income_statement",
        "get_indicators",
        "get_insider_transactions",
        "get_news",
        "get_stock_data",
        "get_verified_market_snapshot",
    ]:
        setattr(tools, name, lambda *args, **kwargs: None)
    sys.modules.setdefault("tradingagents_local_tools", tools)

    def module(name):
        mod = types.ModuleType(name)
        sys.modules.setdefault(name, mod)
        return mod

    for name in [
        "tradingagents",
        "tradingagents.agents",
        "tradingagents.agents.managers",
        "tradingagents.agents.researchers",
        "tradingagents.agents.risk_mgmt",
        "tradingagents.agents.trader",
        "tradingagents.agents.utils",
        "tradingagents.dataflows",
        "tradingagents.graph",
    ]:
        module(name)

    factories = {
        "tradingagents.agents.managers.portfolio_manager": "create_portfolio_manager",
        "tradingagents.agents.managers.research_manager": "create_research_manager",
        "tradingagents.agents.researchers.bear_researcher": "create_bear_researcher",
        "tradingagents.agents.researchers.bull_researcher": "create_bull_researcher",
        "tradingagents.agents.risk_mgmt.aggressive_debator": "create_aggressive_debator",
        "tradingagents.agents.risk_mgmt.conservative_debator": "create_conservative_debator",
        "tradingagents.agents.risk_mgmt.neutral_debator": "create_neutral_debator",
        "tradingagents.agents.trader.trader": "create_trader",
    }
    for mod_name, fn_name in factories.items():
        mod = module(mod_name)
        setattr(mod, fn_name, lambda *args, **kwargs: None)

    agent_states = module("tradingagents.agents.utils.agent_states")
    agent_states.AgentState = dict
    agent_utils = module("tradingagents.agents.utils.agent_utils")
    agent_utils.create_msg_delete = lambda *args, **kwargs: None
    rating = module("tradingagents.agents.utils.rating")
    rating.parse_rating = lambda text, default="Hold": "Buy" if "buy" in str(text).lower() else default
    data_config = module("tradingagents.dataflows.config")
    data_config.set_config = lambda *args, **kwargs: None
    default_config = module("tradingagents.default_config")
    default_config.DEFAULT_CONFIG = {}
    conditional_logic = module("tradingagents.graph.conditional_logic")
    conditional_logic.ConditionalLogic = type("ConditionalLogic", (), {})
    propagation = module("tradingagents.graph.propagation")
    propagation.Propagator = type("Propagator", (), {})
    llm_clients = module("tradingagents.llm_clients")
    llm_clients.create_llm_client = lambda *args, **kwargs: None


_install_external_stubs()

import tradingagents_graph_runner as runner  # noqa: E402
import qlib_backtest_utils  # noqa: E402
import agent_feedback_utils  # noqa: E402
from schemas import StockRecommendation  # noqa: E402


def _report(total_return, sharpe=1.0, excess=1.0, drawdown=5.0):
    return {
        "status": "ok",
        "total_return_percent": total_return,
        "sharpe": sharpe,
        "max_drawdown_percent": -abs(drawdown),
        "benchmark_comparison": {"excess_return_percent": excess},
    }


def _comparison(rule, transformer, hybrid=None, transformer_live=None, hybrid_live=None):
    reports = {"rule": rule, "transformer": transformer, "hybrid": hybrid or {}}
    if transformer_live is not None:
        reports["transformer_live"] = transformer_live
    if hybrid_live is not None:
        reports["hybrid_live"] = hybrid_live
    return {"signal_reports": reports}


class StrategyValidationTest(unittest.TestCase):
    def test_transformer_offsets_rule_weakness_and_reduces_exposure(self):
        validation = runner.build_strategy_validation(_comparison(_report(-1, excess=-4), _report(5)))

        self.assertEqual(validation["agent_research_conclusion"], "cautious")
        self.assertEqual(validation["exposure_adjustment"], "reduce")
        self.assertEqual(validation["exposure_multiplier"], 0.5)
        self.assertEqual(validation["transformer_return_advantage_pct"], 6)
        self.assertIs(validation["transformer_offsets_rule_weakness"], True)

        rec = runner.map_tradingagents_to_stock_recommendation(
            {"action": "buy", "weight": 10, "reason": "agent buy", "strategy_validation": validation},
            {"symbol": "sh600000"},
        )
        self.assertEqual(rec.action, "buy")
        self.assertEqual(rec.weight, 5)

    def test_transformer_small_advantage_still_downgrades_to_hold(self):
        validation = runner.build_strategy_validation(_comparison(_report(-1, excess=-4), _report(1.5)))

        self.assertEqual(validation["agent_research_conclusion"], "downgrade")
        self.assertEqual(validation["exposure_adjustment"], "zero")

        rec = runner.map_tradingagents_to_stock_recommendation(
            {"action": "buy", "weight": 10, "reason": "agent buy", "strategy_validation": validation},
            {"symbol": "sh600000"},
        )
        self.assertEqual(rec.action, "hold")
        self.assertEqual(rec.weight, 0)

    def test_transformer_not_positive_still_downgrades_to_hold(self):
        validation = runner.build_strategy_validation(
            _comparison(_report(-1, excess=-4), _report(5, sharpe=0.1, excess=1))
        )

        self.assertEqual(validation["transformer_backtest_support"], "neutral")
        self.assertEqual(validation["agent_research_conclusion"], "downgrade")
        self.assertEqual(validation["exposure_adjustment"], "zero")

    def test_rule_positive_support_does_not_reduce_exposure(self):
        validation = runner.build_strategy_validation(_comparison(_report(4), _report(2)))

        self.assertEqual(validation["agent_research_conclusion"], "support")
        self.assertEqual(validation["exposure_adjustment"], "none")
        self.assertEqual(validation["exposure_multiplier"], 1.0)

        rec = runner.map_tradingagents_to_stock_recommendation(
            {"action": "buy", "weight": 10, "reason": "agent buy", "strategy_validation": validation},
            {"symbol": "sh600000"},
        )
        self.assertEqual(rec.action, "buy")
        self.assertEqual(rec.weight, 10)

    def test_strategy_validation_prefers_live_transformer_reports(self):
        validation = runner.build_strategy_validation(
            _comparison(
                _report(-1, excess=-4),
                _report(-2),
                hybrid=_report(-2),
                transformer_live=_report(5),
                hybrid_live=_report(6),
            )
        )

        self.assertEqual(validation["validation_signal_scope"], "live_snapshot")
        self.assertEqual(validation["transformer_validation_source"], "live_snapshot")
        self.assertEqual(validation["hybrid_validation_source"], "live_snapshot")
        self.assertEqual(validation["agent_research_conclusion"], "cautious")
        self.assertEqual(validation["exposure_adjustment"], "reduce")

    def test_strategy_validation_falls_back_when_live_skipped(self):
        validation = runner.build_strategy_validation(
            _comparison(
                _report(-1, excess=-4),
                _report(5),
                hybrid=_report(5),
                transformer_live={"status": "skipped", "reason": "not enough snapshots"},
                hybrid_live={"status": "skipped", "reason": "not enough snapshots"},
            )
        )

        self.assertEqual(validation["validation_signal_scope"], "walk_forward_fallback")
        self.assertEqual(validation["transformer_validation_source"], "walk_forward_fallback")
        self.assertEqual(validation["hybrid_validation_source"], "walk_forward_fallback")
        self.assertEqual(validation["agent_research_conclusion"], "cautious")


class AgentSignalSnapshotTest(unittest.TestCase):
    def test_agent_snapshot_persists_final_adjusted_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = runner.AGENT_SIGNAL_DIR
            runner.AGENT_SIGNAL_DIR = Path(tmp)
            try:
                report = runner.save_agent_signal_snapshot(
                    [
                        StockRecommendation(
                            stock_code="sh600000",
                            action="buy",
                            weight=5,
                            reason="reduced exposure",
                        )
                    ],
                    [
                        {
                            "ticker": "sh600000",
                            "strategy_validation": {"agent_research_conclusion": "cautious"},
                            "strategy_validation_effect": "reduced",
                        }
                    ],
                    snapshot_date="2026-06-08",
                )

                payload = json.loads(Path(report["path"]).read_text(encoding="utf-8"))

                self.assertEqual(payload["date"], "2026-06-08")
                self.assertEqual(payload["signals"][0]["stock_code"], "sh600000")
                self.assertEqual(payload["signals"][0]["action"], "buy")
                self.assertEqual(payload["signals"][0]["weight"], 5)
                self.assertEqual(payload["signals"][0]["strategy_validation_effect"], "reduced")
            finally:
                runner.AGENT_SIGNAL_DIR = old_dir

    def test_tradingagents_context_includes_agent_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_path = runner.AGENT_PERFORMANCE_FEEDBACK_PATH
            feedback_path = Path(tmp) / "agent_performance_feedback.json"
            runner.AGENT_PERFORMANCE_FEEDBACK_PATH = feedback_path
            feedback_path.write_text(
                json.dumps({"status": "ok", "agent_excess_vs_hybrid_percent": -1.2}),
                encoding="utf-8",
            )
            try:
                context = json.loads(runner.build_tradingagents_context("summary", {"symbol": "sh600000"}))
                self.assertEqual(context["agent_performance_feedback"]["status"], "ok")
                self.assertEqual(context["agent_performance_feedback"]["agent_excess_vs_hybrid_percent"], -1.2)
            finally:
                runner.AGENT_PERFORMANCE_FEEDBACK_PATH = old_path


class QlibAgentSignalTest(unittest.TestCase):
    def test_agent_signal_series_reads_forward_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = qlib_backtest_utils.AGENT_SIGNAL_DIR
            qlib_backtest_utils.AGENT_SIGNAL_DIR = Path(tmp)
            try:
                for day, action, weight in [
                    ("2026-06-05", "buy", 5),
                    ("2026-06-08", "hold", 0),
                ]:
                    (Path(tmp) / f"{day}.json").write_text(
                        json.dumps(
                            {
                                "date": day,
                                "signals": [
                                    {
                                        "stock_code": "sh600000",
                                        "action": action,
                                        "weight": weight,
                                        "reason": "test",
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )

                signal = qlib_backtest_utils.build_daily_signal_series(signal_source="agents")

                self.assertEqual(float(signal.loc[("SH600000", pd.Timestamp("2026-06-05"))]), 5.0)
                self.assertEqual(
                    float(signal.loc[("SH600000", pd.Timestamp("2026-06-08"))]),
                    qlib_backtest_utils.LOW_SIGNAL_SCORE,
                )
            finally:
                qlib_backtest_utils.AGENT_SIGNAL_DIR = old_dir

    def test_agent_signal_requires_at_least_two_snapshot_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = qlib_backtest_utils.AGENT_SIGNAL_DIR
            qlib_backtest_utils.AGENT_SIGNAL_DIR = Path(tmp)
            try:
                (Path(tmp) / "2026-06-08.json").write_text(
                    json.dumps(
                        {
                            "date": "2026-06-08",
                            "signals": [{"stock_code": "sh600000", "action": "buy", "weight": 5}],
                        }
                    ),
                    encoding="utf-8",
                )

                signal = qlib_backtest_utils.build_daily_signal_series(signal_source="agents")

                self.assertTrue(signal.empty)
            finally:
                qlib_backtest_utils.AGENT_SIGNAL_DIR = old_dir

    def test_comparison_includes_agents_signal_source(self):
        old_runner = qlib_backtest_utils._run_single_qlib_backtest
        seen = []
        try:
            def fake_run(source=None, **kwargs):
                source = source or kwargs.get("signal_source")
                seen.append(source)
                return {"status": "ok", "signal_source": source, **kwargs}

            qlib_backtest_utils._run_single_qlib_backtest = fake_run
            report = qlib_backtest_utils.run_qlib_backtest("comparison")

            self.assertEqual(seen[:6], ["rule", "transformer", "hybrid", "transformer_live", "hybrid_live", "agents"])
            self.assertIn("agents", report["signal_reports"])
            self.assertIn("transformer_live", report["signal_reports"])
            self.assertIn("hybrid_live", report["signal_reports"])
            self.assertEqual(report["primary_signal_source"], "hybrid_live")
            self.assertEqual(len(report["rebalance_cost_comparison"]), 9)
            self.assertIn("rebalance_frequency", report["rebalance_cost_comparison"][0])
        finally:
            qlib_backtest_utils._run_single_qlib_backtest = old_runner


class AgentFeedbackUtilsTest(unittest.TestCase):
    def test_feedback_compares_agents_against_hybrid(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_path = agent_feedback_utils.AGENT_PERFORMANCE_FEEDBACK_PATH
            agent_feedback_utils.AGENT_PERFORMANCE_FEEDBACK_PATH = Path(tmp) / "feedback.json"
            try:
                feedback = agent_feedback_utils.run_agent_feedback_update(
                    {
                        "signal_reports": {
                            "agents": {
                                "status": "ok",
                                "total_return_percent": 2.0,
                                "max_drawdown_percent": -4.0,
                                "sharpe": 0.8,
                            },
                            "hybrid": {
                                "status": "ok",
                                "total_return_percent": 3.5,
                                "max_drawdown_percent": -3.0,
                                "sharpe": 1.1,
                            },
                        }
                    }
                )

                self.assertEqual(feedback["status"], "ok")
                self.assertEqual(feedback["agent_excess_vs_hybrid_percent"], -1.5)
                self.assertIn("require_clearer_risk_reward_for_agent_buy", feedback["recommended_constraints"])
                self.assertTrue(agent_feedback_utils.AGENT_PERFORMANCE_FEEDBACK_PATH.exists())
            finally:
                agent_feedback_utils.AGENT_PERFORMANCE_FEEDBACK_PATH = old_path

    def test_feedback_skips_when_agents_backtest_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_path = agent_feedback_utils.AGENT_PERFORMANCE_FEEDBACK_PATH
            agent_feedback_utils.AGENT_PERFORMANCE_FEEDBACK_PATH = Path(tmp) / "feedback.json"
            try:
                feedback = agent_feedback_utils.run_agent_feedback_update(
                    {"signal_reports": {"agents": {"status": "skipped", "reason": "not enough dates"}}}
                )

                self.assertEqual(feedback["status"], "skipped")
                self.assertEqual(feedback["reason"], "not enough dates")
            finally:
                agent_feedback_utils.AGENT_PERFORMANCE_FEEDBACK_PATH = old_path


if __name__ == "__main__":
    unittest.main()
