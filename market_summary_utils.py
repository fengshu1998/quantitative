import math

from config import LONG_QUANTILE, LOOKBACK_DAYS, UNIVERSE_INDEX


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def format_factor_percent(value):
    if _is_missing(value):
        return "N/A"
    return f"{float(value) / 100:.2%}"


def format_weight(value):
    if _is_missing(value):
        return "N/A"
    return f"{float(value):.2%}"


def format_number(value):
    if _is_missing(value):
        return "N/A"
    return f"{float(value):.2f}"


def _latest_rows(all_factor_data):
    rows = []
    for symbol, item in all_factor_data.items():
        df = item["data"]
        if df.empty:
            continue
        latest = df.iloc[-1]
        rows.append((symbol, item["name"], latest))
    return rows


def _count_by_latest(rows, column):
    counts = {}
    for _, _, latest in rows:
        value = latest.get(column, "unknown")
        if _is_missing(value):
            value = "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _format_counts(counts, keys):
    return "，".join(f"{key}: {counts.get(key, 0)}" for key in keys)


def _candidate_line(candidate, rank):
    return (
        f"{rank}. {candidate['symbol']} {candidate['name']} | "
        f"行业={candidate.get('industry') or 'N/A'} | "
        f"signal={candidate['signal']} | cross_section_score={candidate.get('cross_section_score')} | "
        f"target_weight={format_weight(candidate.get('target_weight'))} | "
        f"20日收益={format_factor_percent(candidate.get('return_20d'))} | "
        f"20日波动率={format_factor_percent(candidate.get('volatility_20d'))} | "
        f"20日最大回撤={format_factor_percent(candidate.get('max_drawdown_20d'))} | "
        f"RSI14={format_number(candidate.get('rsi_14'))} | "
        f"MACD_diff={format_number(candidate.get('macd_diff'))} | "
        f"ADX14={format_number(candidate.get('adx_14'))} | "
        f"Boll_width={format_number(candidate.get('bollinger_width'))} | "
        f"PE={format_number(candidate.get('pe'))} | PB={format_number(candidate.get('pb'))} | "
        f"ROE={format_factor_percent(candidate.get('roe'))} | "
        f"财务可用={candidate.get('fundamental_available', False)} | "
        f"趋势={candidate.get('trend', 'N/A')} | "
        f"原因={candidate.get('signal_reason', 'N/A')}"
    )


def build_market_summary(
    all_factor_data: dict,
    selected_candidates: list[dict] | None = None,
    stats: dict | None = None,
) -> str:
    """生成给 LLM Agent 阅读的股票池扫描摘要，不重新计算因子。"""
    selected_candidates = selected_candidates or []
    stats = stats or {}
    rows = _latest_rows(all_factor_data)
    trend_counts = _count_by_latest(rows, "trend")
    signal_counts = _count_by_latest(rows, "signal")
    industry_counts = _count_by_latest(rows, "industry")
    top_industries = sorted(industry_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    top_industry_text = "，".join(f"{name}: {count}" for name, count in top_industries) or "N/A"

    lines = [
        f"市场数据摘要（{UNIVERSE_INDEX}成分股，最近{LOOKBACK_DAYS}个交易日）：",
        "",
        "一、股票池概况",
        f"- 本次股票池：{UNIVERSE_INDEX}成分股",
        f"- 成分股数量：{stats.get('universe_count', 'N/A')}",
        f"- 成功拉取行情数量：{stats.get('fetched_count', 'N/A')}",
        f"- 过滤数量：{stats.get('filtered_count', 'N/A')}",
        f"- 有效股票数量：{stats.get('valid_count', len(all_factor_data))}",
        f"- 最终候选股票数量：{len(selected_candidates)}",
        "",
        "二、市场整体技术状态",
        f"- 趋势分布：{_format_counts(trend_counts, ['uptrend', 'downtrend', 'range', 'unknown'])}",
        f"- 信号分布：{_format_counts(signal_counts, ['BUY', 'HOLD', 'SELL'])}",
        "",
        "三、行业分布",
        f"- 有效股票行业分布 Top10：{top_industry_text}",
        "",
        f"四、最终指数增强候选股票（前{int(float(LONG_QUANTILE) * 100)}%）",
    ]

    if selected_candidates:
        for rank, candidate in enumerate(selected_candidates, start=1):
            lines.append(_candidate_line(candidate, rank))
    else:
        lines.append("暂无满足 BUY 条件的候选股票。")

    lines.extend(
        [
            "",
            "五、候选股技术信号 + 财务摘要 + 行业信息",
            "- 以上候选股已合并行业、财务可用性、PE/PB/ROE 等字段；N/A 表示当前数据源缺失或接口失败。",
        ]
    )
    return "\n".join(lines)
