r"""掘金策略执行脚本。

在掘金终端策略编辑器里粘贴此文件全部内容，模式选"仿真交易"，启动。

功能：
- 读取 D:\quantitative\data\gm_orders.json，显示并执行最新组合建议。
- 读取 D:\quantitative\reports\*\*.json，显示项目内轻量回测和 Qlib 回测摘要。
- 启动时立即检查一次；运行中通过 60 秒 bar 和日内定时点继续检查。
"""

from gm.api import *

import json
import os
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(r"D:\quantitative")
ORDERS_PATH = PROJECT_ROOT / "data" / "gm_orders.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
ATTRIBUTION_DIR = PROJECT_ROOT / "data" / "gm_attribution"
HEARTBEAT_SYMBOL = "SHSE.000300"
MIN_BUY_LOT_SHARES = 100


def _to_gm_symbol(project_symbol):
    symbol = str(project_symbol or "").strip().lower()
    if symbol.startswith("sh"):
        return "SHSE." + symbol[2:]
    if symbol.startswith("sz"):
        return "SZSE." + symbol[2:]
    return symbol


def _read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print("[GM Runner] 读取 JSON 失败: {} ({})".format(path, e))
        return None


def _json_ready(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_ready(value.__dict__)
    return str(value)


def _latest_report_path():
    if not REPORTS_DIR.exists():
        return None
    candidates = sorted(REPORTS_DIR.glob("*/*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _fmt_percent(value):
    if value is None or value == "":
        return "N/A"
    try:
        return "{:.2f}%".format(float(value))
    except Exception:
        return str(value)


def _fmt_value(value):
    if value is None or value == "":
        return "N/A"
    return str(value)


def _print_orders(payload):
    status = payload.get("status", "unknown")
    generated_at = payload.get("generated_at", "")
    reason = payload.get("reason", "")
    orders = payload.get("orders") or []

    print("")
    print("[GM Runner] ===== 最新交易组合建议 =====")
    print("[GM Runner] 生成时间: {}".format(generated_at or "N/A"))
    print("[GM Runner] 状态: {} | 原因: {} | 订单数: {}".format(status, reason, len(orders)))

    if not orders:
        print("[GM Runner] 今日无有效买入订单")
        return

    for order in orders:
        stock_code = order.get("stock_code", "")
        direction = order.get("direction", "")
        quantity_pct = order.get("quantity_percent", 0)
        stop_loss = order.get("stop_loss_percent", 0)
        print(
            "[GM Runner] 组合: {} {} 仓位 {} 止损 {}".format(
                stock_code,
                direction,
                _fmt_percent(quantity_pct),
                _fmt_percent(stop_loss),
            )
        )


def _print_stock_signals(report):
    signals = report.get("stock_signals") or []
    if not signals:
        return

    print("[GM Runner] ----- Agents 股票建议 -----")
    for item in signals:
        print(
            "[GM Runner] {} {} 建议仓位 {} | {}".format(
                item.get("stock_code", ""),
                item.get("action", ""),
                _fmt_percent(item.get("weight")),
                str(item.get("reason", ""))[:120],
            )
        )


def _print_lightweight_backtest(report):
    backtest = report.get("backtest_report") or {}
    print("[GM Runner] ----- 项目轻量回测 -----")
    if backtest.get("status") != "ok":
        print("[GM Runner] 状态: {} | 原因: {}".format(backtest.get("status", "unknown"), backtest.get("reason", "")))
        return

    print(
        "[GM Runner] 区间: {} -> {} | 交易日: {}".format(
            backtest.get("start_date", "N/A"),
            backtest.get("end_date", "N/A"),
            backtest.get("trading_days", "N/A"),
        )
    )
    print(
        "[GM Runner] 总收益 {} | 年化 {} | Sharpe {} | 最大回撤 {}".format(
            _fmt_percent(backtest.get("total_return_percent")),
            _fmt_percent(backtest.get("annualized_return_percent")),
            _fmt_value(backtest.get("sharpe")),
            _fmt_percent(backtest.get("max_drawdown_percent")),
        )
    )


def _print_qlib_source(name, source_report):
    if not source_report:
        print("[GM Runner] Qlib {}: N/A".format(name))
        return
    if source_report.get("status") != "ok":
        print("[GM Runner] Qlib {}: {} | {}".format(name, source_report.get("status"), source_report.get("reason", "")))
        return
    benchmark = source_report.get("benchmark_comparison") or {}
    print(
        "[GM Runner] Qlib {}: 总收益 {} | 年化 {} | Sharpe {} | 最大回撤 {} | 超额 {}".format(
            name,
            _fmt_percent(source_report.get("total_return_percent")),
            _fmt_percent(source_report.get("annualized_return_percent")),
            _fmt_value(source_report.get("sharpe")),
            _fmt_percent(source_report.get("max_drawdown_percent")),
            _fmt_percent(benchmark.get("excess_return_percent")),
        )
    )


def _print_qlib_backtest(report):
    qlib = report.get("qlib_backtest_report") or {}
    print("[GM Runner] ----- Qlib 专业回测 -----")
    if qlib.get("status") != "ok":
        print("[GM Runner] 状态: {} | 原因: {}".format(qlib.get("status", "unknown"), qlib.get("reason", "")))
        return

    if qlib.get("engine") == "qlib_comparison":
        print("[GM Runner] 模式: comparison | primary: {}".format(qlib.get("primary_signal_source", "N/A")))
        signal_reports = qlib.get("signal_reports") or {}
        _print_qlib_source("rule", signal_reports.get("rule"))
        _print_qlib_source("transformer", signal_reports.get("transformer"))
        _print_qlib_source("hybrid", signal_reports.get("hybrid"))
        return

    _print_qlib_source("portfolio", qlib)


def _print_report_summary(context, force):
    report_path = _latest_report_path()
    if report_path is None:
        if force:
            print("[GM Runner] 未找到项目日报 JSON: {}".format(REPORTS_DIR))
        return

    report = _read_json(report_path)
    if not report:
        return

    generated_at = report.get("generated_at", "")
    if not force and getattr(context, "last_report_generated_at", None) == generated_at:
        return

    context.last_report_generated_at = generated_at
    print("")
    print("[GM Runner] ===== 最新项目回测报告 =====")
    print("[GM Runner] 报告文件: {}".format(report_path))
    print("[GM Runner] 生成时间: {}".format(generated_at or "N/A"))
    _print_stock_signals(report)
    _print_lightweight_backtest(report)
    _print_qlib_backtest(report)
    print("[GM Runner] ===== 报告摘要结束 =====")


def _submit_orders(context, payload):
    orders = payload.get("orders") or []
    generated_at = payload.get("generated_at", "")
    if not orders:
        return
    if getattr(context, "last_order_generated_at", None) == generated_at:
        return

    print("[GM Runner] 检测到新订单 {} 条，开始调仓...".format(len(orders)))
    explicit_symbols = set()
    current_positions = _current_long_positions()
    account_asset = _current_account_asset()
    if current_positions:
        holding_text = ", ".join(
            "{}:{:.0f}".format(symbol, volume) for symbol, volume in sorted(current_positions.items())
        )
        print("[GM Runner] 当前持仓: {}".format(holding_text))
    else:
        print("[GM Runner] 当前无多头持仓，sell 指令将自动跳过")
    for order in orders:
        stock_code = order.get("stock_code", "")
        direction = str(order.get("direction", "") or "").lower()
        quantity_pct = float(order.get("quantity_percent", 0) or 0)
        if not stock_code or direction in ("hold", "持有"):
            continue
        if direction == "buy" and quantity_pct <= 0:
            continue
        if direction not in ("buy", "sell", "卖出"):
            continue

        try:
            gm_symbol = _to_gm_symbol(stock_code)
            explicit_symbols.add(gm_symbol)
            if direction in ("sell", "卖出") and current_positions.get(gm_symbol, 0.0) <= 0:
                print("[GM Runner] 跳过 sell: {} 当前无持仓".format(gm_symbol))
                continue
            target = quantity_pct / 100.0 if direction == "buy" else 0.0
            if direction == "buy" and _should_skip_small_buy(gm_symbol, target, account_asset):
                continue
            order_target_percent(
                symbol=gm_symbol,
                percent=target,
                order_type=OrderType_Market,
                position_side=PositionSide_Long,
            )
            print("[GM Runner] 下单: {} {} -> 目标仓位 {:.2%}".format(gm_symbol, direction, target))
        except Exception as e:
            print("[GM Runner] 下单失败 {}: {}".format(stock_code, e))

    unmentioned_action = str(payload.get("unmentioned_position_action", "hold") or "hold").lower()
    if unmentioned_action in ("clear", "sell", "liquidate", "reduce_to_zero", "reduce"):
        unmentioned_target = 0.0
        if unmentioned_action == "reduce":
            unmentioned_target = float(payload.get("unmentioned_position_target_percent", 0) or 0) / 100.0
        positions = _as_list(_safe_gm_call("get_position"))
        for position in positions:
            item = _as_mapping(position)
            symbol = item.get("symbol") or item.get("sec_id") or item.get("stock_code")
            if not symbol:
                continue
            gm_symbol = _to_gm_symbol(symbol)
            if gm_symbol in explicit_symbols:
                continue
            try:
                order_target_percent(
                    symbol=gm_symbol,
                    percent=unmentioned_target,
                    order_type=OrderType_Market,
                    position_side=PositionSide_Long,
                )
                print("[GM Runner] 未入选持仓调整: {} -> 目标仓位 {:.2%}".format(gm_symbol, unmentioned_target))
            except Exception as e:
                print("[GM Runner] 未入选持仓调整失败 {}: {}".format(gm_symbol, e))

    context.last_order_generated_at = generated_at
    print("[GM Runner] 调仓完成，等待下次指令...")


def _safe_gm_call(name, *args, **kwargs):
    fn = globals().get(name)
    if fn is None:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print("[GM Runner] {} 调用失败: {}".format(name, e))
        return None


def _as_mapping(value):
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    result = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            field = getattr(value, name)
        except Exception:
            continue
        if not callable(field):
            result[name] = field
    return result


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _position_symbol(position):
    item = _as_mapping(position)
    symbol = item.get("symbol") or item.get("sec_id") or item.get("stock_code")
    return _to_gm_symbol(symbol) if symbol else ""


def _position_long_volume(position):
    item = _as_mapping(position)
    for key in (
        "volume",
        "available",
        "available_volume",
        "available_today",
        "current_amount",
        "amount",
        "position",
        "quantity",
    ):
        if key not in item:
            continue
        try:
            return float(item.get(key) or 0)
        except Exception:
            pass
    return 0.0


def _current_long_positions():
    positions = _as_list(_safe_gm_call("get_position"))
    result = {}
    for position in positions:
        symbol = _position_symbol(position)
        if not symbol:
            continue
        result[symbol] = result.get(symbol, 0.0) + _position_long_volume(position)
    return result


def _first_number(mapping, keys):
    for key in keys:
        try:
            value = mapping.get(key)
        except Exception:
            continue
        try:
            number = float(value)
        except Exception:
            continue
        if number > 0:
            return number
    return None


def _current_account_asset():
    cash = _as_mapping(_safe_gm_call("get_cash"))
    return _first_number(
        cash,
        (
            "nav",
            "net_asset",
            "total_asset",
            "asset",
            "equity",
            "market_value",
            "balance",
            "available",
        ),
    )


def _extract_price(value):
    item = _as_mapping(value)
    return _first_number(
        item,
        (
            "price",
            "last_price",
            "last",
            "close",
            "current",
            "new_price",
            "trade_price",
        ),
    )


def _current_price(symbol):
    for kwargs in (
        {"symbols": symbol},
        {"symbol": symbol},
        {"symbols": symbol, "fields": "symbol,price,last_price,close"},
    ):
        value = _safe_gm_call("current", **kwargs)
        for item in _as_list(value):
            price = _extract_price(item)
            if price:
                return price
    return None


def _should_skip_small_buy(symbol, target_percent, account_asset):
    if target_percent <= 0:
        return True
    if not account_asset:
        print("[GM Runner] 无法读取账户资产，保留 buy 下单: {}".format(symbol))
        return False
    price = _current_price(symbol)
    if not price:
        print("[GM Runner] 无法读取现价，保留 buy 下单: {}".format(symbol))
        return False
    estimated_shares = int((float(account_asset) * float(target_percent)) / float(price))
    if estimated_shares < MIN_BUY_LOT_SHARES:
        print(
            "[GM Runner] 跳过 buy: {} 目标仓位 {:.2%} 预计 {:.0f} 股，不足 {} 股一手".format(
                symbol,
                target_percent,
                estimated_shares,
                MIN_BUY_LOT_SHARES,
            )
        )
        return True
    return False


def _normalize_return_percent(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) <= 1:
        number *= 100.0
    return round(number, 6)


def _extract_return_percent(mapping):
    for key in (
        "portfolio_return_percent",
        "return_percent",
        "daily_return_percent",
        "profit_rate_percent",
        "pnl_ratio_percent",
        "daily_return",
        "return_rate",
        "profit_rate",
        "pnl_ratio",
    ):
        normalized = _normalize_return_percent(mapping.get(key))
        if normalized is not None:
            return normalized
    current_nav = mapping.get("nav") or mapping.get("net_asset") or mapping.get("total_asset")
    previous_nav = mapping.get("pre_nav") or mapping.get("previous_nav") or mapping.get("yesterday_asset")
    try:
        current_nav = float(current_nav)
        previous_nav = float(previous_nav)
    except (TypeError, ValueError):
        return None
    if previous_nav:
        return round((current_nav / previous_nav - 1.0) * 100.0, 6)
    return None


def _extract_position_maps(positions):
    symbol_returns = {}
    holding_pnl = {}
    for position in _as_list(positions):
        item = _as_mapping(position)
        symbol = item.get("symbol") or item.get("sec_id") or item.get("stock_code")
        if not symbol:
            continue
        return_percent = _extract_return_percent(item)
        if return_percent is not None:
            symbol_returns[str(symbol)] = return_percent
        for key in ("pnl", "profit", "floating_pnl", "unrealized_pnl", "position_profit"):
            if key not in item:
                continue
            try:
                holding_pnl[str(symbol)] = round(float(item.get(key)), 4)
            except (TypeError, ValueError):
                pass
            break
    return symbol_returns, holding_pnl


def _write_live_attribution(context):
    now = datetime.now()
    orders_payload = _read_json(ORDERS_PATH) or {}
    cash = _safe_gm_call("get_cash")
    positions = _safe_gm_call("get_position")
    executions = _safe_gm_call("get_execution_reports")
    account_map = _as_mapping(cash)
    symbol_returns, holding_pnl = _extract_position_maps(positions)

    payload = {
        "generated_at": now.isoformat(timespec="seconds"),
        "trading_date": now.date().isoformat(),
        "source": "gm_strategy_runner",
        "orders_generated_at": orders_payload.get("generated_at"),
        "orders": orders_payload.get("orders", []),
        "account": _json_ready(cash),
        "positions": _json_ready(positions),
        "fills": _json_ready(executions),
        "portfolio_return_percent": _extract_return_percent(account_map),
        "benchmark_return_percent": None,
        "symbol_returns": symbol_returns,
        "holding_pnl": holding_pnl,
        "note": "GM API account fields differ by terminal version; project attribution consumes available fields.",
    }
    ATTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)
    path = ATTRIBUTION_DIR / "{}.json".format(now.date().isoformat())
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    context.last_attribution_written_at = payload["generated_at"]
    print("[GM Runner] 已写入实盘归因文件: {}".format(path))


def _check_and_execute(context, force=False):
    context.log_count = (getattr(context, "log_count", 0) or 0) + 1
    if force or context.log_count <= 2 or context.log_count % 30 == 0:
        print("[GM Runner] 检查项目输出 #{} ...".format(context.log_count))

    payload = _read_json(ORDERS_PATH)
    if payload is None:
        if force:
            print("[GM Runner] 未找到订单文件: {}".format(ORDERS_PATH))
        _print_report_summary(context, force=force)
        return

    generated_at = payload.get("generated_at", "")
    if force or getattr(context, "last_order_seen_at", None) != generated_at:
        context.last_order_seen_at = generated_at
        _print_orders(payload)

    _print_report_summary(context, force=force)
    _submit_orders(context, payload)


def init(context):
    context.last_order_seen_at = None
    context.last_order_generated_at = None
    context.last_report_generated_at = None
    context.log_count = 0

    print("[GM Runner] 策略初始化完成，等待 main.py 生成交易指令和日报...")
    _check_and_execute(context, force=True)

    try:
        subscribe(symbols=HEARTBEAT_SYMBOL, frequency="60s", count=1)
        print("[GM Runner] 已订阅 {} 60s bar，用于定期检查项目输出".format(HEARTBEAT_SYMBOL))
    except Exception as e:
        print("[GM Runner] 订阅 60s bar 失败: {}".format(e))

    for time_rule in ["09:35:00", "10:30:00", "13:30:00", "14:30:00", "14:55:00"]:
        try:
            schedule(schedule_func=_scheduled_check, date_rule="1d", time_rule=time_rule)
        except Exception as e:
            print("[GM Runner] 注册定时检查失败 {}: {}".format(time_rule, e))
    try:
        schedule(schedule_func=_scheduled_attribution, date_rule="1d", time_rule="15:05:00")
    except Exception as e:
        print("[GM Runner] 注册收盘归因失败: {}".format(e))

    print("[GM Runner] 已注册日内定时检查")


def on_bar(context, bars):
    _check_and_execute(context, force=False)


def _scheduled_check(context):
    _check_and_execute(context, force=True)


def _scheduled_attribution(context):
    _write_live_attribution(context)


def on_shutdown(context):
    try:
        _write_live_attribution(context)
    except Exception as e:
        print("[GM Runner] 关闭时写归因失败: {}".format(e))
    print("[GM Runner] 策略关闭")


if __name__ == "__main__":
    # 粘贴到掘金终端后，请替换为当前策略的策略 ID 和系统设置里的 token。
    GM_STRATEGY_ID = os.getenv("GM_STRATEGY_ID", "62ea5cb4-60a3-11f1-bcbc-c4bde5334c67")
    GM_TOKEN = os.getenv("GM_TOKEN", "")
    GM_FILENAME = "main.py"
    GM_MODE = MODE_LIVE

    if not GM_STRATEGY_ID or not GM_TOKEN:
        print("[GM Runner] 未启动掘金事件循环：请先填写 GM_STRATEGY_ID 和 GM_TOKEN。")
        print("[GM Runner] 策略 ID 在掘金策略页面查看；token 在终端右上角用户头像 -> 系统设置中复制。")
        print("[GM Runner] 填好后再次启动，run(...) 会阻塞运行并触发 init/on_bar/schedule。")
    else:
        run(
            strategy_id=GM_STRATEGY_ID,
            filename=GM_FILENAME,
            mode=GM_MODE,
            token=GM_TOKEN,
        )
