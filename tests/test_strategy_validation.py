import sys
import types
import unittest


def _install_external_stubs():
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


def _report(total_return, sharpe=1.0, excess=1.0, drawdown=5.0):
    return {
        "status": "ok",
        "total_return_percent": total_return,
        "sharpe": sharpe,
        "max_drawdown_percent": -abs(drawdown),
        "benchmark_comparison": {"excess_return_percent": excess},
    }


def _comparison(rule, transformer, hybrid=None):
    return {"signal_reports": {"rule": rule, "transformer": transformer, "hybrid": hybrid or {}}}


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


if __name__ == "__main__":
    unittest.main()
