"""TradingAgents research layer entry points.

The project previously used an AutoGen group chat here. The third-layer
research flow now routes through a TradingAgents-compatible adapter while
preserving the rest of the pipeline's public functions and schemas.
"""

from tradingagents_adapter import (
    build_macro_from_candidate_set,
    build_model_risk_from_agent_results,
    build_tradingagents_context,
    get_latest_agent_payloads,
    load_tradingagents_candidates,
    map_tradingagents_to_stock_recommendation,
    run_tradingagents_for_candidate,
    run_tradingagents_research,
)


__all__ = [
    "build_macro_from_candidate_set",
    "build_model_risk_from_agent_results",
    "build_tradingagents_context",
    "get_latest_agent_payloads",
    "load_tradingagents_candidates",
    "map_tradingagents_to_stock_recommendation",
    "run_tradingagents_for_candidate",
    "run_tradingagents_research",
]
