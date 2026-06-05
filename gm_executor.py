"""Goldminer/MyQuant execution bridge.

The research pipeline writes final orders to ``data/gm_orders.json``.
The script pasted into the GM terminal reads that file and submits orders
inside GM's event-driven runtime.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config import GM_EXECUTION_ENABLED

logger = logging.getLogger(__name__)

ORDERS_PATH = Path(__file__).resolve().parent / "data" / "gm_orders.json"


def _write_orders_payload(
    orders_data: list[dict[str, Any]],
    status: str,
    reason: str,
) -> dict[str, Any]:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "reason": reason,
        "total_orders": len(orders_data),
        "orders": orders_data,
    }
    ORDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ORDERS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _execution_result(
    payload: dict[str, Any],
    submitted_count: int,
    skipped_count: int,
) -> dict[str, Any]:
    return {
        "status": payload["status"],
        "engine": "gm",
        "reason": payload["reason"],
        "orders_path": str(ORDERS_PATH),
        "submitted_count": submitted_count,
        "skipped_count": skipped_count,
        "orders": payload["orders"],
    }


def execute_final_orders(final_orders: list[Any]) -> dict[str, Any]:
    """Write final orders for the GM terminal runner.

    Even when there is no valid buy order, this function writes an empty
    ``gm_orders.json`` so the GM side can distinguish "no trade today" from
    "the pipeline did not run or the file path is wrong".
    """
    if not GM_EXECUTION_ENABLED:
        return {"status": "skipped", "reason": "GM_EXECUTION_ENABLED is False"}

    final_orders = final_orders or []
    if not final_orders:
        payload = _write_orders_payload([], "skipped", "no trading instructions")
        logger.info("GM empty order file written: %s", ORDERS_PATH)
        return _execution_result(payload, submitted_count=0, skipped_count=0)

    orders_data = []
    for order in final_orders:
        direction = str(getattr(order, "direction", "") or "").lower()
        stock_code = str(getattr(order, "stock_code", "") or "")
        quantity_pct = float(getattr(order, "quantity_percent", 0) or 0)

        if direction != "buy" or quantity_pct <= 0:
            continue

        orders_data.append(
            {
                "stock_code": stock_code,
                "direction": direction,
                "quantity_percent": quantity_pct,
                "stop_loss_percent": float(getattr(order, "stop_loss_percent", 0) or 0),
            }
        )

    if not orders_data:
        payload = _write_orders_payload([], "skipped", "no valid buy instructions")
        logger.info("GM empty order file written: %s", ORDERS_PATH)
        return _execution_result(payload, submitted_count=0, skipped_count=len(final_orders))

    payload = _write_orders_payload(orders_data, "ok", "orders generated")
    logger.info("GM order file written: %s (%s buy orders)", ORDERS_PATH, len(orders_data))
    return _execution_result(
        payload,
        submitted_count=len(orders_data),
        skipped_count=len(final_orders) - len(orders_data),
    )
