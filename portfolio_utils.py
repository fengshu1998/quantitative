from schemas import MacroAnalysis, RiskOrder, RiskReview, StockRecommendation
from portfolio_rule_utils import load_live_portfolio_rules


SINGLE_STOCK_LIMIT = 20.0
DEFAULT_STOP_LOSS = 8.0


def _direction_from_action(action: str):
    mapping = {
        "买入": "buy",
        "buy": "buy",
        "卖出": "sell",
        "sell": "sell",
        "持有": None,
        "hold": None,
    }
    return mapping[action]


def apply_deterministic_risk_rules(
    macro: MacroAnalysis,
    stocks: list[StockRecommendation],
    single_stock_limit: float = SINGLE_STOCK_LIMIT,
    default_stop_loss: float = DEFAULT_STOP_LOSS,
) -> RiskReview:
    live_rules = load_live_portfolio_rules()
    total_position_multiplier = float(live_rules.get("total_position_multiplier") or 1.0)
    stop_loss_adjustment = float(live_rules.get("stop_loss_adjustment") or 0.0)
    adjustments = []
    buy_weights = []
    sell_weights = []

    for rec in stocks:
        direction = _direction_from_action(rec.action)
        if direction is None:
            continue

        if direction == "sell":
            # 卖出信号 weight=0 表示“建议清仓”，视为卖出 100%
            sell_weight = 100.0 if rec.weight <= 0 else rec.weight
            sell_weights.append((rec, direction, sell_weight))
        elif rec.weight > 0:
            capped_weight = min(rec.weight, single_stock_limit)
            if capped_weight < rec.weight:
                adjustments.append(
                    f"{rec.stock_code} 单票仓位从 {rec.weight:.2f}% 裁剪到 {capped_weight:.2f}%"
                )
            buy_weights.append((rec, direction, capped_weight))

    # 买入仓位按宏观上限缩放
    total_buy_weight = sum(weight for _, _, weight in buy_weights)
    macro_limit = macro.suggested_position * total_position_multiplier
    if total_position_multiplier != 1.0:
        adjustments.append(
            f"实盘归因总仓位乘数 {total_position_multiplier:.2f}，宏观上限调整为 {macro_limit:.2f}%"
        )
    scale = 1.0
    if total_buy_weight > macro_limit and total_buy_weight > 0:
        scale = macro_limit / total_buy_weight
        adjustments.append(
            f"买入总仓位从 {total_buy_weight:.2f}% 按比例缩放到宏观上限 {macro_limit:.2f}%"
        )

    final_orders = []
    for rec, direction, weight in buy_weights:
        final_weight = round(weight * scale, 2)
        if final_weight <= 0:
            continue
        final_orders.append(
            RiskOrder(
                stock_code=rec.stock_code,
                direction=direction,
                quantity_percent=final_weight,
                stop_loss_percent=max(default_stop_loss + stop_loss_adjustment, 0.0),
            )
        )

    for rec, direction, weight in sell_weights:
        final_weight = round(weight, 2)
        if final_weight <= 0:
            continue
        adjustments.append(
            f"{rec.stock_code} 卖出信号，建议卖出 {final_weight:.2f}% 仓位"
        )
        final_orders.append(
            RiskOrder(
                stock_code=rec.stock_code,
                direction=direction,
                quantity_percent=final_weight,
                stop_loss_percent=0,
            )
        )

    return RiskReview(
        approved=bool(final_orders),
        adjustments=adjustments,
        final_orders=final_orders,
    )
