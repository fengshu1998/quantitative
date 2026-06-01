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


MA_FAST = 5
MA_SLOW = 20


def _load_price_series(order, data_dir, warmup_days):
    """Load historical close prices for a single stock, keeping warmup + backtest window."""
    data_dir = Path(data_dir)
    for subpath in [
        data_dir / f"{order.stock_code}.csv",
        data_dir / "stocks" / f"{order.stock_code}.csv",
        data_dir / "factors" / f"{order.stock_code}.csv",
    ]:
        if subpath.exists():
            df = pd.read_csv(subpath)
            break
    else:
        return None, None

    date_col = "date" if "date" in df.columns else "日期"
    close_col = "close" if "close" in df.columns else "收盘"
    if date_col not in df.columns or close_col not in df.columns:
        return None, None

    series = df[[date_col, close_col]].copy()
    series[date_col] = pd.to_datetime(series[date_col])
    series = series.dropna(subset=[date_col, close_col]).sort_values(date_col)
    # keep warmup days for MA calculation, then trim to window
    full = series.set_index(date_col)[close_col].astype(float)
    window = full.iloc[-(BACKTEST_WINDOW + warmup_days):]
    return window, close_col


def _crossover_signals(close, ma_fast=MA_FAST, ma_slow=MA_SLOW):
    """Return a boolean Series: True = position on (MA_fast > MA_slow), False = out."""
    fast = close.rolling(ma_fast).mean()
    slow = close.rolling(ma_slow).mean()
    # first day when fast crosses above slow → enter
    in_position = fast > slow
    # first ma_slow days have no signal (NaN)
    in_position = in_position.fillna(False)
    return in_position


def _single_stock_returns(close, weight_pct, ma_fast=MA_FAST, ma_slow=MA_SLOW):
    """Simulate MA crossover on one stock and return daily contribution Series."""
    in_position = _crossover_signals(close, ma_fast, ma_slow)
    # daily return of the stock itself
    daily_ret = close.pct_change().fillna(0.0)
    # weight contribution: only when in position
    contribution = daily_ret * in_position.astype(float) * (weight_pct / 100.0)
    # count trades
    signal_change = in_position.astype(int).diff()
    trades = int((signal_change.abs() == 1).sum())
    return contribution, trades


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
    total_trades = 0
    for order in buy_orders:
        close, _ = _load_price_series(order, data_dir, warmup_days=MA_SLOW)
        if close is None or len(close) < MA_SLOW + 5:
            continue
        contribution, trades = _single_stock_returns(close, order.quantity_percent)
        # trim warmup period
        contribution = contribution.iloc[-BACKTEST_WINDOW:]
        returns.append(contribution.rename(order.stock_code))
        used_weights[order.stock_code] = order.quantity_percent
        total_trades += trades

    if not returns:
        return {"status": "skipped", "reason": "未找到可用的历史行情 CSV"}

    portfolio_returns = pd.concat(returns, axis=1).dropna(how="all").fillna(0).sum(axis=1)
    portfolio_returns = portfolio_returns[portfolio_returns != 0]
    if portfolio_returns.empty:
        return {"status": "skipped", "reason": "历史行情不足，无法计算收益"}

    equity = initial_cash * (1 + portfolio_returns).cumprod()
    total_return = equity.iloc[-1] / initial_cash - 1
    annualized_return = (1 + total_return) ** (252 / max(len(portfolio_returns), 1)) - 1
    annualized_volatility = portfolio_returns.std() * (252**0.5)
    sharpe = (
        annualized_return / annualized_volatility
        if annualized_volatility and annualized_volatility > 0
        else None
    )
    drawdown = equity / equity.cummax() - 1

    return {
        "status": "ok",
        "strategy": "ma_crossover",
        "ma_fast": MA_FAST,
        "ma_slow": MA_SLOW,
        "backtest_window_days": BACKTEST_WINDOW,
        "start_date": portfolio_returns.index.min().date().isoformat(),
        "end_date": portfolio_returns.index.max().date().isoformat(),
        "trading_days": int(len(portfolio_returns)),
        "weights": used_weights,
        "total_trades": total_trades,
        "total_return_percent": _round_float(total_return * 100),
        "annualized_return_percent": _round_float(annualized_return * 100),
        "annualized_volatility_percent": _round_float(annualized_volatility * 100),
        "sharpe": _round_float(sharpe) if sharpe is not None else None,
        "max_drawdown_percent": _round_float(drawdown.min() * 100),
    }
