# Quantitative Research Pipeline

基于 AkShare + Qlib + TradingAgents 的 A 股量化投研流水线，覆盖 **数据采集 → 因子计算 → 信号生成 → 精选候选 → LLM 多角色研究 → 风控裁剪 → 双重回测 → 报告输出** 的完整闭环。

## 架构概览

```
                          config.py（统一配置中心）
                               │
┌─────────────────┐     ┌──────┴──────┐     ┌─────────────────┐
│   Data Layer     │────▶│ Factor Layer │────▶│ Selection Layer  │
│                  │     │              │     │                  │
│ data_utils.py    │     │ factor_utils │     │ portfolio_sel..  │
│ storage_utils.py │     │ signal_utils │     │ market_summary.. │
│ fundamental_utils│     │              │     │                  │
│ industry_utils   │     │ ta / RSI /   │     │ TOP_N + alloc    │
│                  │     │ MACD / ADX.. │     │                  │
└─────────────────┘     └──────────────┘     └────────┬─────────┘
                                                      │
                                              candidate list
                                                      │
┌─────────────────┐     ┌────────────────┐     ┌──────┴──────────┐
│  Report Layer    │◀────│  Portfolio +   │◀────│  Research Layer  │
│                  │     │  Backtest      │     │                  │
│ report_utils.py  │     │                │     │ tradingagents_..  │
│                  │     │ portfolio_utils│     │ agents.py        │
│ JSON + MD 日报   │     │ qlib_backtest..│     │ schemas.py       │
│                  │     │                │     │                  │
└─────────────────┘     │ deterministic  │     │ DeepSeek LLM     │
                        │ + Qlib 专业回测 │     │ 多角色研究        │
                        └────────────────┘     └──────────────────┘
```

## 项目结构

```
D:\quantitative\
  main.py                       # 入口：编排每日研究流水线
  config.py                     # 统一配置中心（股票池、因子、风控、Qlib）
  agents.py                     # TradingAgents 适配层
  schemas.py                    # Pydantic 输出结构校验

  # 数据层
  data_utils.py                 # 指数成分股抓取、行情拉取、批量处理、流水线编排
  storage_utils.py              # 通用持久化（prices / factors / fundamentals / industries）
  fundamental_utils.py          # 财务数据抓取（PE/PB/ROE/资产负债率/营收利润增速）
  industry_utils.py             # 行业分类映射（申万行业）

  # 因子层
  factor_utils.py               # 量价因子计算（基础 + ta 库高级指标）+ Alphalens 因子评估
  signal_utils.py               # 确定性 BUY/HOLD/SELL 信号生成（因子加权打分）
  portfolio_selection_utils.py  # TOP_N 精选 + 等权仓位分配
  market_summary_utils.py       # 生成给 LLM 阅读的市场数据摘要

  # 研究与风控
  portfolio_utils.py            # 确定性风控裁剪 + 轻量历史回测
  qlib_backtest_utils.py        # Qlib 专业回测（TopkDropoutStrategy + 基准对比 + 交易成本）

  # 报告与第三方
  report_utils.py               # JSON + Markdown 日报生成
  tradingagents_adapter.py      # TradingAgents 研究图适配器（DeepSeek LLM）
  market_report.ipynb           # Jupyter Notebook：可视化图表
  run.bat                       # Windows 一键启动
  environment.env               # API Key 配置（不入库）

  # 目录
  data/prices/                  # 行情 CSV（每只股票一个文件）
  data/factors/                 # 因子 + 信号 CSV
  data/fundamentals/            # 财务快照 CSV
  data/industries/              # 行业映射 CSV
  data/qlib/                    # Qlib 二进制数据（日历/标的/特征）
  reports/YYYY-MM-DD/           # 每日 JSON + Markdown 报告
  daily_report/                 # 可视化图表 PNG + 汇总 CSV
  third_party/tradingagents/    # TradingAgents 研究图框架
  .claude/skills/buffett/       # 巴菲特投资思维 skill
```

## 流水线执行流程

```
python main.py（或双击 run.bat）
  │
  ├─ 1. 数据层
  │    get_market_snapshot()
  │    ├─ fetch_index_constituents()    ← ak.index_stock_cons_csindex
  │    ├─ 逐只拉取 OHLCV                 ← ak.stock_zh_a_daily
  │    ├─ ST/停牌/数据不足过滤
  │    └─ 落盘 prices / fundamentals / industries
  │
  ├─ 2. 因子 + 信号
  │    _process_one_stock() 依次执行：
  │    ├─ compute_market_factors()       ← 基础因子 + RSI/MACD/Bollinger/ADX 等 14 项
  │    ├─ fetch_fundamental_data()       ← PE/PB/ROE/营收增速/利润增速/负债率
  │    ├─ merge_fundamental_features()   ← 合并为扩展因子表
  │    ├─ generate_signal()              ← 多因子加权打分 → BUY/HOLD/SELL
  │    └─ 落盘 factors
  │
  ├─ 3. 精选候选
  │    ├─ select_top_candidates()        ← BUY 信号按 score 排序取 TOP_N
  │    └─ allocate_positions()           ← 等权分配 target_weight
  │
  ├─ 4. 研究层（TradingAgents）
  │    run_tradingagents_research()
  │    ├─ 逐只候选构建 context（因子 + 财务 + 市场摘要）
  │    ├─ DeepSeek LLM 模拟多角色研究   ← 技术/基本面/多空/风控/组合管理
  │    └─ 映射回 StockRecommendation + MacroAnalysis + RiskReview
  │
  ├─ 5. 风控 + 回测
  │    ├─ apply_deterministic_risk_rules() ← 单票上限 20% / 总仓位上限 / 止损
  │    ├─ build_backtest_report()          ← 本地 CSV 轻量回测
  │    └─ run_qlib_backtest()              ← Qlib TopkDropoutStrategy 专业回测
  │
  └─ 6. 报告输出
       save_daily_reports()
       ├─ reports/YYYY-MM-DD/HHMMSS.json  ← 结构化全量数据
       └─ reports/YYYY-MM-DD/HHMMSS.md    ← Markdown 可读日报
```

## 因子体系

### 基础因子（每只股票自动计算）

| 类别 | 因子 | 说明 |
|------|------|------|
| 收益 | return_1d / 5d / 20d | 1/5/20 日涨跌幅 |
| 趋势 | MA5 / MA20 / trend | 双均线 + 趋势分类（uptrend/downtrend/range） |
| 波动 | volatility_20d | 20 日滚动年化波动率 |
| 回撤 | max_drawdown_20d | 20 日滚动最大回撤 |
| 量价 | volume_ratio_20d / price_vs_ma20 | 量比 / 价格偏离度 |

### 高级指标（ta 库，可选依赖）

| 指标 | 字段 | 窗口 |
|------|------|------|
| RSI | rsi_14 | 14 |
| MACD | macd / macd_signal / macd_diff | 12/26/9 |
| Bollinger | bollinger_mavg / high / low / width | 20/2σ |
| ATR | atr_14 | 14 |
| Stochastic | stoch_k / stoch_d | 14/3 |
| OBV | obv | — |
| MFI | mfi_14 | 14 |
| ADX | adx_14 | 14 |
| CCI | cci_20 | 20 |

### 信号打分逻辑

`signal_utils.py` 采用多因子加权打分：

- **正向加分**: 趋势上行、正收益、价格高于 MA20、放量、PE 合理、PB 合理、ROE 高、RSI 40–70、MACD 金叉、ADX 强势多头
- **负向减分**: 趋势下行、负收益、深回撤、高波动、高负债、Bollinger 宽幅、MFI 过热
- **输出**: signal=BUY (score≥3 且正常) / SELL (score≤-2) / HOLD (其他)

## 研究层

### TradingAgents 适配器

替代原来的 AutoGen GroupChat，采用 TradingAgents 研究图范式：

1. 从 `selected_candidates.csv` 加载 TOP_N 候选股票
2. 为每只候选构建研究 context（因子快照 + 财务快照 + 市场摘要）
3. 调用 DeepSeek LLM 模拟多角色研究：技术分析师 / 基本面分析师 / 多头 / 空头 / 交易员 / 风控官 / 组合经理
4. LLM 失败时自动 fallback 到确定性信号
5. 结果映射为 Pydantic schema（MacroAnalysis / StockRecommendation / RiskReview）

### 巴菲特投资框架（Claude Code Skill）

项目安装了 `.claude/skills/buffett/`，在 Claude Code 中分析股票时会自动触发：
- 护城河五类型 + 趋势判断
- 管理层三维评估（诚信/资本配置/主人心态）
- 财务指标（ROIC、所有者收益、现金转化率）
- 估值与安全边际
- 四条卖出标准 + 价值陷阱识别

## 双重回测

### 轻量回测（portfolio_utils.py）

- 基于本地 CSV 数据，纯 Python 计算
- 输出：区间收益、年化收益、年化波动率、Sharpe、最大回撤

### Qlib 专业回测（qlib_backtest_utils.py）

- 将因子 CSV 转为 Qlib 二进制格式
- 使用 `TopkDropoutStrategy` + `SimulatorExecutor`
- 按日调仓，考虑交易成本（开仓 0.05%、平仓 0.15%、最低 5 元）
- 输出：账户权益曲线、年化收益、Sharpe、回撤、换手率、基准对比（沪深300）
- 配置开关：`config.py` → `QLIB_ENABLED = True/False`

## 环境准备

### 1. Python 环境

```bash
conda create -n quant_env python=3.10
conda activate quant_env
```

### 2. 安装依赖

```bash
pip install akshare pandas python-dotenv openai pyqlib
# 可选：高级技术指标
pip install ta
# 可选：因子评估
pip install alphalens
```

### 3. 配置 API Key

编辑 `environment.env`：

```
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## 运行

### 完整流水线

```bash
python main.py
# 或双击 run.bat
```

### 单独测试数据/因子层

```bash
python data_utils.py       # 拉取行情、计算因子信号、生成候选
```

### 生成可视化图表

运行 `market_report.ipynb`（Jupyter Notebook），输出收益曲线、均线、波动率、回撤、Sharpe 等图表到 `daily_report/`。

## 配置说明

`config.py` 是单文件配置中心：

| 配置组 | 关键字段 | 说明 |
|--------|---------|------|
| 存储 | STORAGE_FORMAT | csv / parquet |
| 股票池 | UNIVERSE_INDEX / UNIVERSE_INDEX_SYMBOL / UNIVERSE_SOURCE | 指数名称/代码/数据源（csindex/eastmoney） |
| 选股 | UNIVERSE_SIZE_LIMIT / TOP_N | 处理上限 / 最终候选数 |
| 风控 | MAX_POSITION_PER_STOCK / MAX_TOTAL_POSITION | 单票上限 10% / 总仓位上限 60% |
| 过滤 | FILTER_ST / FILTER_SUSPENDED | 过滤 ST 股 / 停牌股 |
| Qlib | QLIB_ENABLED / QLIB_ACCOUNT / QLIB_BENCHMARK | 回测开关 / 初始资金 100 万 / 基准 SH000300 |

切换指数示例（沪深300 → 中证500）：

```python
UNIVERSE_INDEX = "中证500"
UNIVERSE_INDEX_SYMBOL = "000905"
UNIVERSE_SOURCE = "csindex"
```

## 扩展方向

- **接入更多标的**: 修改 config 中的 `UNIVERSE_*` 配置即可切换指数
- **新增因子**: 在 `factor_utils.py` 中添加计算逻辑，在 `signal_utils.py` 中配置打分权重
- **接入实盘**: 解析 `final_orders`，对接券商 API
- **定时运行**: 配合 Windows 任务计划 / cron / GitHub Actions 自动执行
- **Qlib 深度使用**: 切换为更复杂的策略模型（LightGBM、LSTM 等）
