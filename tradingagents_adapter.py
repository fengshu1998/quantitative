from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from config import MAX_TOTAL_POSITION, TOP_N
from schemas import MacroAnalysis, RiskOrder, RiskReview, StockRecommendation
from storage_utils import load_dataframe, load_selected_candidates


logger = logging.getLogger(__name__)

VENDOR_ROOT = Path(__file__).resolve().parent / "third_party" / "tradingagents"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402


load_dotenv("environment.env")

_LATEST_RESULT: dict[str, Any] | None = None


def _json_default(value):
    if pd.isna(value):
        return None
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


def _extract_json_object(text: str) -> dict[str, Any]:
    normalized = re.sub(r",\s*([}\]])", r"\1", text.strip())
    try:
        payload = json.loads(normalized)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(normalized):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(normalized[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"No JSON object found in TradingAgents output: {text[:300]}")


def load_tradingagents_candidates() -> list[dict]:
    try:
        df = load_selected_candidates()
    except FileNotFoundError:
        logger.warning("selected_candidates.csv not found; TradingAgents has no candidates.")
        return []

    if df.empty:
        return []
    return df.head(TOP_N).where(pd.notna(df), None).to_dict("records")


def _tail_records(dataset_type: str, symbol: str, rows: int = 5) -> list[dict]:
    try:
        df = load_dataframe(dataset_type, symbol)
    except Exception as e:
        logger.warning("Failed to load %s data for %s: %s", dataset_type, symbol, e)
        return []
    if df.empty:
        return []
    return df.tail(rows).where(pd.notna(df), None).to_dict("records")


def build_tradingagents_context(market_summary: str, candidate: dict) -> str:
    symbol = candidate.get("symbol", "")
    context = {
        "task": "Analyze this A-share candidate using a TradingAgents-style research process.",
        "market_summary": market_summary,
        "candidate": candidate,
        "recent_factor_rows": _tail_records("factors", symbol),
        "fundamental_snapshot": _tail_records("fundamentals", symbol, rows=1),
        "news": {"available": False, "note": "News/RAG is not connected in this phase."},
        "sentiment": {"available": False, "note": "Sentiment data is not connected in this phase."},
    }
    return json.dumps(context, ensure_ascii=False, default=_json_default, indent=2)


def _deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _run_llm_research(ticker: str, run_date: str, context: str) -> dict[str, Any]:
    client = _deepseek_client()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a TradingAgents-style multi-agent trading research graph. "
                    "Internally simulate technical, fundamental, bull/bear, trader, "
                    "risk manager, and portfolio manager roles. Use only the provided "
                    "A-share local project context. Return pure JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Ticker: {ticker}\nDate: {run_date}\n\n"
                    f"Context:\n{context}\n\n"
                    "Return JSON schema exactly:\n"
                    "{\n"
                    '  "action": "buy|hold|sell",\n'
                    '  "weight": 0-100,\n'
                    '  "reason": "concise Chinese explanation",\n'
                    '  "risk_notes": ["risk note"],\n'
                    '  "confidence": 0-1\n'
                    "}\n"
                    "Do not use markdown fences."
                ),
            },
        ],
    )
    content = response.choices[0].message.content or ""
    payload = _extract_json_object(content)
    payload["raw_content"] = content
    return payload


def _fallback_research(candidate: dict, reason: str) -> dict[str, Any]:
    signal = str(candidate.get("signal") or "HOLD").upper()
    action = {"BUY": "buy", "SELL": "sell"}.get(signal, "hold")
    weight = _target_weight_percent(candidate) if action == "buy" else 0
    return {
        "ticker": candidate.get("symbol"),
        "action": action,
        "weight": weight,
        "reason": (
            f"TradingAgents failed, fallback to deterministic signal. "
            f"signal={signal}; reason={candidate.get('signal_reason') or reason}"
        ),
        "risk_notes": [f"fallback: {reason}"],
        "confidence": 0,
        "source": "fallback",
    }


def run_tradingagents_for_candidate(candidate: dict, context: str) -> dict:
    symbol = candidate.get("symbol")
    try:
        result = _run_llm_research(str(symbol), date.today().isoformat(), context)
        result.setdefault("ticker", symbol)
        result.setdefault("date", date.today().isoformat())
        result.setdefault("source", "tradingagents_local_adapter")
        return result
    except Exception as e:
        logger.warning("TradingAgents failed for %s: %s", symbol, e)
        return _fallback_research(candidate, str(e))


def map_tradingagents_to_stock_recommendation(
    result: dict,
    candidate: dict,
) -> StockRecommendation:
    action = str(result.get("action") or "hold").lower()
    if action not in {"buy", "sell", "hold"}:
        action = "hold"

    weight = _to_float(result.get("weight"), 0.0)
    if action == "buy" and weight <= 0:
        weight = _target_weight_percent(candidate)
    if action == "hold":
        weight = 0.0

    reason = str(result.get("reason") or candidate.get("signal_reason") or "No reason provided.")
    source = result.get("source", "tradingagents")
    if source == "fallback":
        reason = f"[fallback] {reason}"

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
            reason="No TOP_N candidates are available; keep cash and skip new positions.",
        )

    buy_results = [item for item in agent_results if str(item.get("action")).lower() == "buy"]
    total_candidate_weight = sum(_target_weight_percent(item) for item in candidates)
    suggested_position = min(max(total_candidate_weight, 20.0 if buy_results else 0.0), MAX_TOTAL_POSITION * 100)

    buy_ratio = len(buy_results) / max(len(candidates), 1)
    avg_score = sum(_to_float(item.get("signal_score"), 0.0) for item in candidates) / len(candidates)
    if buy_ratio >= 0.6 and avg_score >= 5:
        environment = "offensive"
    elif buy_ratio == 0 or len(candidates) < max(2, TOP_N // 3):
        environment = "defensive"
    else:
        environment = "neutral"

    return MacroAnalysis(
        market_environment=environment,
        suggested_position=round(suggested_position, 2),
        reason=(
            f"TradingAgents analyzed {len(candidates)} TOP_N candidates; "
            f"{len(buy_results)} are buy-rated. Average deterministic score={avg_score:.2f}. "
            "Macro position is capped by configured portfolio risk limits."
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
        if item.get("risk_notes"):
            adjustments.extend(str(note) for note in item["risk_notes"])

    return RiskReview(
        approved=bool(final_orders),
        adjustments=adjustments,
        final_orders=final_orders,
    )


def run_tradingagents_research(market_summary: str) -> list[dict]:
    candidates = load_tradingagents_candidates()
    agent_results = []

    for candidate in candidates:
        context = build_tradingagents_context(market_summary, candidate)
        result = run_tradingagents_for_candidate(candidate, context)
        agent_results.append(result)

    stock_recommendations = [
        map_tradingagents_to_stock_recommendation(result, candidate)
        for result, candidate in zip(agent_results, candidates)
    ]
    macro = build_macro_from_candidate_set(market_summary, candidates, agent_results)
    risk = build_model_risk_from_agent_results(agent_results)

    macro_raw = macro.model_dump_json(indent=2)
    stock_raw = json.dumps(
        [item.model_dump() for item in stock_recommendations],
        ensure_ascii=False,
        indent=2,
    )
    risk_raw = risk.model_dump_json(indent=2)

    messages = [
        {"name": "User", "content": market_summary},
        {"name": "TradingAgentsMacro", "content": macro_raw},
        {
            "name": "TradingAgentsStockResearch",
            "content": json.dumps(agent_results, ensure_ascii=False, default=_json_default, indent=2),
        },
        {"name": "TradingAgentsMappedStocks", "content": stock_raw},
        {"name": "TradingAgentsRiskManager", "content": risk_raw},
    ]

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
    }
    return messages


def get_latest_agent_payloads() -> dict[str, Any]:
    if _LATEST_RESULT is None:
        raise RuntimeError("TradingAgents research has not been run yet.")
    return _LATEST_RESULT
