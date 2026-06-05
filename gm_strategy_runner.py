"""掘金策略执行脚本。

在掘金终端策略编辑器里粘贴此文件全部内容，模式选"仿真交易"，启动。
"""

import json
from pathlib import Path


ORDERS_PATH = Path(r"D:\quantitative\data\gm_orders.json")


def _to_gm_symbol(project_symbol: str) -> str:
    symbol = str(project_symbol).strip().lower()
    if symbol.startswith("sh"):
        return f"SHSE.{symbol[2:]}"
    if symbol.startswith("sz"):
        return f"SZSE.{symbol[2:]}"
    return symbol


def _read_orders() -> dict | None:
    if not ORDERS_PATH.exists():
        return None
    try:
        return json.loads(ORDERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def init(context):
    context.orders_executed = False
    context.last_checked = None
    context.shutdown_count = 0
    context.log_count = 0
    print("[GM Runner] 策略初始化完成，等待交易指令...")
    schedule(schedule_func=_check_and_execute, date_rule="0/30 * * * * *")
    print("[GM Runner] 已注册 30 秒定时器")


def _check_and_execute(context):
    context.log_count = (getattr(context, "log_count", 0) or 0) + 1
    if context.log_count <= 2 or context.log_count % 120 == 0:
        print(f"[GM Runner] 定时检查 #{context.log_count} ...")

    payload = _read_orders()
    if payload is None:
        return

    orders = payload.get("orders") or []
    if not orders:
        return

    generated_at = payload.get("generated_at", "")
    if context.last_checked and context.last_checked >= generated_at:
        return

    print(f"[GM Runner] 检测到新订单 ({len(orders)} 条)，开始调仓...")

    for order in orders:
        stock_code = order.get("stock_code", "")
        quantity_pct = float(order.get("quantity_percent", 0))
        if quantity_pct <= 0:
            continue

        try:
            gm_symbol = _to_gm_symbol(stock_code)
            target = quantity_pct / 100.0
            order_target_percent(
                symbol=gm_symbol,
                percent=target,
                order_type=OrderType_Market,
                position_side=PositionSide_Long,
            )
            print(f"  [GM Runner] 下单: {gm_symbol} → 目标仓位 {target:.2%}")
        except Exception as e:
            print(f"  [GM Runner] 下单失败 {stock_code}: {e}")

    context.last_checked = generated_at
    print(f"[GM Runner] 调仓完成 ({len(orders)} 条)，等待下次指令...")


def on_shutdown(context):
    print("[GM Runner] 策略关闭")
