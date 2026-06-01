from pathlib import Path

import pandas as pd

from schemas import MacroAnalysis, RiskOrder, RiskReview, StockRecommendation


SINGLE_STOCK_LIMIT = 20.0
DEFAULT_STOP_LOSS = 8.0
BACKTEST_WINDOW = 60


def _round_float(value, digits=2):
    return round(float(value), digits)


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
    macro_limit = macro.suggested_position
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
                stop_loss_percent=default_stop_loss,
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


def build_backtest_report(
    orders: list[RiskOrder],
    data_dir: str | Path = "data",
    initial_cash: float = 1.0,
) -> dict:
    data_dir = Path(data_dir)
    buy_orders = [order for order in orders if order.direction == "buy" and order.quantity_percent > 0]
    if not buy_orders:
        return {"status": "skipped", "reason": "没有买入订单，跳过历史复盘"}

    returns = []
    used_weights = {}
    for order in buy_orders:
        path = data_dir / f"{order.stock_code}.csv"
        if not path.exists():
            path = data_dir / "stocks" / f"{order.stock_code}.csv"
        if not path.exists():
            path = data_dir / "factors" / f"{order.stock_code}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        close_col = "close" if "close" in df.columns else "收盘"
        date_col = "date" if "date" in df.columns else "日期"
        if close_col not in df.columns or date_col not in df.columns:
            continue
        series = df[[date_col, close_col]].copy()
        series[date_col] = pd.to_datetime(series[date_col])
        series = series.sort_values(date_col).tail(BACKTEST_WINDOW)
        series = series.set_index(date_col)[close_col].pct_change()
        returns.append(series.rename(order.stock_code) * (order.quantity_percent / 100.0))
        used_weights[order.stock_code] = order.quantity_percent

    if not returns:
        return {"status": "skipped", "reason": "未找到可用的历史行情 CSV"}

    portfolio_returns = pd.concat(returns, axis=1).dropna(how="all").fillna(0).sum(axis=1)
    if portfolio_returns.empty:
        return {"status": "skipped", "reason": "历史行情不足，无法计算收益"}

    equity = initial_cash * (1 + portfolio_returns).cumprod()
    total_return = equity.iloc[-1] / initial_cash - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    annualized_volatility = portfolio_returns.std() * (252**0.5)
    sharpe = (
        annualized_return / annualized_volatility
        if annualized_volatility and annualized_volatility > 0
        else None
    )
    drawdown = equity / equity.cummax() - 1

    return {
        "status": "ok",
        "backtest_window_days": BACKTEST_WINDOW,
        "start_date": portfolio_returns.index.min().date().isoformat(),
        "end_date": portfolio_returns.index.max().date().isoformat(),
        "trading_days": int(len(portfolio_returns)),
        "weights": used_weights,
        "total_return_percent": _round_float(total_return * 100),
        "annualized_return_percent": _round_float(annualized_return * 100),
        "annualized_volatility_percent": _round_float(annualized_volatility * 100),
        "sharpe": _round_float(sharpe) if sharpe is not None else None,
        "max_drawdown_percent": _round_float(drawdown.min() * 100),
    }
