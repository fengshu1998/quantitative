# Quantitative Research Pipeline

这是一个面向 A 股的量化投研、模型验证、日报生成和掘金终端执行项目。当前版本采用生产化三段 Transformer 架构，并把 Alpha、Transformer、风控与流动性统一到信号融合层中。

核心流程：

```text
实盘归因反馈
-> 行情/财务/行业/因子刷新
-> Alpha 前置分析
-> Transformer daily inference
-> Signal Fusion
-> 候选池与仓位分配
-> TradingAgents LangGraph 多角色研究
-> deterministic risk rules
-> Qlib walk-forward 专业回测
-> JSON/Markdown 日报
-> 掘金 gm_orders.json
```

## 当前生产架构

```text
main.py
  |
  |-- live_attribution_utils.py
  |     读取掘金归因，更新因子反馈和组合规则
  |
  |-- data_utils.py
  |     刷新行情、财务、行业和因子
  |
  |-- alpha_analysis_utils.py
  |     生成 factor_selection.json，给 Alpha score 提供方向和权重
  |
  |-- qlib_training_utils.py
  |     daily inference 只加载 transformer_model.pth
  |
  |-- signal_utils.py
  |     Alpha 70% / Transformer 20% / 风控流动性 10%
  |
  |-- portfolio_selection_utils.py
  |     按 cross_section_rank_pct 选前20%指数增强候选并分配约束仓位
  |
  |-- tradingagents_graph_runner.py
  |     LangGraph 多角色研究、策略验证、降级/降仓
  |
  |-- portfolio_utils.py
  |     deterministic risk rules
  |
  |-- qlib_backtest_utils.py
  |     Qlib comparison 回测，正式口径优先 walk-forward score
  |
  |-- report_utils.py
  |     生成日报
  |
  |-- gm_executor.py
        输出 data/gm_orders.json
```

## 目录与模块

```text
D:\quantitative\
  main.py                         # 每日流水线入口：只推理，不训练 Transformer
  config.py                       # 单文件配置中心
  run.bat                         # Windows 一键运行 main.py

  data_utils.py                   # 股票池、行情、财务、行业、因子刷新和市场快照
  factor_utils.py                 # 量价/技术/财务因子计算
  fundamental_utils.py            # 财务快照抓取和标准化
  industry_utils.py               # 行业映射缓存和刷新
  storage_utils.py                # CSV/parquet 持久化工具
  market_summary_utils.py         # 给 LLM 读取的市场摘要

  alpha_analysis_utils.py         # Alpha IC/RankIC/分层收益/因子权重分析
  signal_utils.py                 # 统一 signal fusion
  benchmark_weight_utils.py       # 指数成分权重获取、缓存和等权 fallback
  portfolio_selection_utils.py    # 前20%指数增强候选排序和约束仓位
  portfolio_rule_utils.py         # live portfolio rules 读取和约束

  qlib_training_utils.py          # Transformer walk-forward/live/daily 三段架构
  qlib_backtest_utils.py          # Qlib 数据准备、TopkDropoutStrategy、调仓频率/成本对比

  agents.py                       # TradingAgents 对外入口 re-export
  tradingagents_graph_runner.py   # 本项目 LangGraph 研究图和策略验证
  tradingagents_local_tools.py    # TradingAgents 可调用的本地数据工具
  third_party/tradingagents/      # 内置 TradingAgents 代码
  skills/buffett/                 # Buffett 投研 skill 与参考资料

  portfolio_utils.py              # deterministic 风控和最终订单裁剪
  schemas.py                      # Pydantic 输出结构
  report_utils.py                 # JSON/Markdown 日报

  gm_executor.py                  # main.py 写 gm_orders.json
  gm_strategy_runner.py           # 粘贴到掘金终端的执行脚本
  live_attribution_utils.py       # 读取掘金归因并反馈因子/组合规则

  tests/                          # 单元测试
  market_report.ipynb             # 可视化分析 Notebook
```

## 数据与产物

```text
data/prices/                              # 原始 OHLCV
data/factors/                             # 因子、cross-section sub-scores / final score / rank
data/fundamentals/                        # 财务快照
data/industries/                          # 行业映射
data/signals/daily_signal_scores.csv      # 每日全股票信号快照
data/benchmark_weights/                   # 指数成分权重缓存
data/selected_candidates.csv              # 前20%指数增强候选池

data/model_reports/transformer_model.pth
data/model_reports/transformer_live_feature_scaler.json
data/model_reports/transformer_predictions.csv
data/model_reports/transformer_inference_summary.json
data/model_reports/transformer_live_training_summary.json
data/model_reports/transformer_walk_forward_predictions.csv
data/model_reports/live_predictions/YYYY-MM-DD.csv
data/model_reports/walk_forward_models/

data/factor_reports/alpha_summary.json
data/factor_reports/alpha_summary.md
data/factor_reports/factor_selection.json
data/factor_reports/live_factor_feedback.json

data/portfolio_rules/live_portfolio_rules.json
data/qlib/                                # Qlib provider 数据
data/qlib_reports/                        # Qlib 回测报告
data/agent_signals/YYYY-MM-DD.json        # Agents 最终 action/weight 前向快照
data/agent_feedback/agent_performance_feedback.json
data/gm_orders.json                       # 给掘金 runner 的交易指令
data/gm_attribution/YYYY-MM-DD.json       # 掘金收盘归因回写

reports/YYYY-MM-DD/HHMMSS.json
reports/YYYY-MM-DD/HHMMSS.md
```

## 每日运行

```bash
python main.py
```

每日流水线不会训练 Transformer。它只会加载 live production training 生成的当前生产模型：

```text
data/model_reports/transformer_model.pth
data/model_reports/transformer_live_feature_scaler.json
```

如果模型或 scaler 缺失，`run_transformer_inference()` 会返回 `skipped`，主流程继续运行，Transformer 分数按 0 处理。

## 每日流水线细节

```text
1. run_live_attribution_update()
   读取 data/gm_attribution 最新 JSON
   更新 live_factor_feedback.json 和 live_portfolio_rules.json

2. refresh_factor_data()
   拉取指数成分股
   拉取 OHLCV、财务快照、行业映射
   计算因子并保存 data/factors/*.csv

3. run_alpha_analysis()
   从 data/factors 构建 alpha panel
   生成 factor_selection.json

4. run_transformer_inference()
   读取 transformer_model.pth 和 transformer_live_feature_scaler.json
   对每只股票最新一行因子做推理
   写 transformer_predictions.csv
   同时覆盖写 live_predictions/YYYY-MM-DD.csv，用于 transformer_live / hybrid_live 回测

5. build_market_snapshot_from_factors()
   对每只股票运行 generate_signal()
   生成 alpha_score、transformer_score、risk_liquidity_score
   执行 alpha / transformer / risk-liquidity 三路横截面标准化和统一融合
   生成 cross_section_score、cross_section_rank_pct、long_bucket、short_bucket
   读取 benchmark_weights，构建沪深300指数增强前20%候选
   输出 daily_signal_scores.csv 和 selected_candidates.csv

6. run_tradingagents_research()
   对候选池运行 LangGraph 多角色研究
   输出 MacroAnalysis、StockRecommendation、RiskReview
   保存 data/agent_signals/YYYY-MM-DD.json

7. apply_deterministic_risk_rules()
   单票上限、总仓位、止损、实盘组合规则 multiplier

8. run_qlib_backtest()
   Qlib comparison 回测：rule / transformer / hybrid / transformer_live / hybrid_live / agents

9. run_agent_feedback_update()
   根据 agents vs hybrid 回测表现写 agent_performance_feedback.json
   该反馈在下一次 TradingAgents 研究时进入上下文

10. save_daily_reports()
   生成 JSON 和 Markdown 日报

11. execute_final_orders()
    写 data/gm_orders.json
```

## Signal Fusion

`signal_utils.py` 是当前候选生成的唯一打分层。默认权重：

```python
SIGNAL_CROSS_WEIGHTS = {
    "alpha": 0.70,
    "transformer": 0.20,
    "risk_liquidity": 0.10,
}
```

输出字段：

```text
alpha_score
transformer_score
risk_liquidity_score
alpha_cross_section_score
transformer_cross_section_score
risk_liquidity_cross_section_score
cross_section_score
cross_section_rank_pct
long_bucket
short_bucket
signal
signal_reason
cross_section_score       # 当前主交易排序分数
```

说明：

- Alpha 分数来自 `factor_selection.json` 中的因子有效性、方向和权重。
- Transformer 分数来自 `transformer_predictions.csv` 的 `prediction_zscore`。
- 风控流动性分数基于因子表中的 `volume`、`volatility_20d`、`max_drawdown_20d`、`trend` 和 `risk_flag`。
- 三路子信号分别转成 `alpha_cross_section_score`、`transformer_cross_section_score`、`risk_liquidity_cross_section_score` 后，按 `SIGNAL_CROSS_WEIGHTS` 融合为唯一最终交易分 `cross_section_score`。
- Alpha 因子横截面处理包括 winsorize / clip、z-score、rank percentile、行业中性化和市值中性化；Transformer 和 risk/liquidity 子信号按当日股票池横截面 rank 中心化。
- `long_bucket=True` 表示进入前20%指数增强候选池。
- `short_bucket=True` 表示进入后20% alpha long-short 研究组；该组不进入 GM 股票实盘开空。
- 候选池不再使用 `TOP_N`，主流程调用 `select_ranked_portfolio_candidates()` 选择前20%候选。
- 仓位分配基于 `benchmark_weight + active_tilt`，并叠加波动率调整、行业权重上限、单票上限和总仓位上限。

## Alpha 分析

`alpha_analysis_utils.py` 在因子刷新后、信号生成前运行。

分析指标：

```text
IC
Rank IC
IC IR
Rank IC IR
分层多空收益
缺失率
因子方向
因子权重
```

默认参与分析的核心因子包括：

```text
return_20d
volume_ratio_20d
rsi_14
macd_diff
adx_14
bollinger_width
volatility_20d
max_drawdown_20d
roe
debt_to_asset
```

输出：

```text
data/factor_reports/alpha_summary.json
data/factor_reports/alpha_summary.md
data/factor_reports/factor_selection.json
```

如果存在 `live_factor_feedback.json`，Alpha 会在统计型因子权重上叠加温和 multiplier；不会因为单日实盘表现直接反转因子方向。

## Transformer 三段架构

### 1. Walk-Forward Training

用途：正式历史回测和模型可信评估。

默认窗口：

```text
6 年 train
1 年 valid
1 年 test
```

示例：

```text
2010-2015 train, 2016 valid, 2017 test
2011-2016 train, 2017 valid, 2018 test
2012-2017 train, 2018 valid, 2019 test
```

运行：

```bash
python -c "from qlib_training_utils import run_transformer_walk_forward_training; print(run_transformer_walk_forward_training(force=True))"
```

输出：

```text
data/model_reports/walk_forward_models/model_test_YYYY.pth
data/model_reports/transformer_walk_forward_predictions.csv
data/model_reports/transformer_walk_forward_summary.json
```

Qlib 正式 `transformer` 和 `hybrid` 回测优先使用 `transformer_walk_forward_predictions.csv`。

Transformer walk-forward 现在以横截面 alpha 为主目标，而不是直接预测 raw 5 日收益：

```text
raw 5d forward return
-> label_5d_cs_excess_return = raw return - same-day universe mean
-> label_5d_rank_pct         = same-day percentile rank
```

训练主 label 默认是 `cross_sectional_excess`。raw return 和 rank label 会保留在 walk-forward prediction CSV 中作为诊断列：

```text
label
raw_forward_return
label_rank_pct
```

Transformer 输入特征会先按交易日做横截面 winsorize / z-score，并在每个 fold 的 train 段剔除高缺失率、零方差和 RankIC 弱的特征。筛选结果写入 summary 的 `dataset.feature_filter`，避免 valid/test 信息泄漏。

Walk-forward fold metrics 额外包含：

```text
top20_bottom20_return
top20_excess_return
monthly_rank_ic_mean
monthly_rank_ic_ir
monthly_rank_ic_positive_rate
```

因此模型优先看 RankIC、top20 excess 和 top20-bottom20，而不是只看 MSE。

### 2. Live Production Training

用途：离线训练当前实盘模型权重。

运行：

```bash
python -c "from qlib_training_utils import run_transformer_live_production_training; print(run_transformer_live_production_training(force=True))"
```

如果最新因子日期是 2026 年，默认窗口为：

```text
2019-2024 train
2025 valid
```

输出：

```text
data/model_reports/transformer_model.pth
data/model_reports/transformer_live_feature_scaler.json
data/model_reports/transformer_live_training_summary.json
```

旧入口 `run_transformer_training(force=True)` 仍存在，但只是兼容 wrapper，会转调 `run_transformer_live_production_training()`，并在结果中标记：

```text
deprecated_alias = true
alias_for = run_transformer_live_production_training
```

### 3. Daily Inference

用途：每日实盘/日报。

运行：

```bash
python -c "from qlib_training_utils import run_transformer_inference; print(run_transformer_inference())"
```

每日推理只取每只股票最新一行因子：

```text
latest factor rows
-> live feature scaler
-> transformer_model.pth
-> transformer_predictions.csv
```

输出：

```text
data/model_reports/transformer_predictions.csv
data/model_reports/transformer_inference_summary.json
```

## Qlib 回测

`qlib_backtest_utils.py` 会把本地因子 CSV 转换为 Qlib provider 数据，并使用：

```text
TopkDropoutStrategy
SimulatorExecutor(time_per_step="day")
benchmark = SH000300
```

回测信号源：

```text
rule         = 因子 CSV 中的 cross_section_score；旧数据回退 signal_score
transformer  = walk-forward predictions
hybrid       = rule + walk-forward transformer score
transformer_live = live_predictions/YYYY-MM-DD.csv historical snapshots
hybrid_live       = rule + live-production transformer snapshots
agents      = data/agent_signals/YYYY-MM-DD.json final action + weight
```

默认 `run_qlib_backtest()` 会运行 comparison：

```text
rule
transformer
hybrid
transformer_live
hybrid_live
agents
```

同时会输出 rule 信号的调仓频率和成本敏感性比较：

```text
rebalance_frequency = daily / weekly / monthly
cost_profile        = low / current / high
```

每组报告包含收益、年化收益、Sharpe、最大回撤、turnover_rate、benchmark comparison、topk 和成本参数。

`agents` uses forward-only accumulated snapshots. The project saves only the current run's final TradingAgents action/weight snapshot and never backfills today's advice into past dates. If fewer than 2 agent snapshot dates exist, `signal_reports.agents` is reported as `skipped` with a reason.

报告字段会标记：

```text
prediction_mode
prediction_path
walk_forward_enabled
leakage_guard
fold_count
first_test_year
last_test_year
```

## TradingAgents 研究层

项目使用 `third_party/tradingagents`，本地适配在：

```text
agents.py
tradingagents_graph_runner.py
tradingagents_local_tools.py
```

研究层读取 `selected_candidates.csv`，为每只候选构造上下文，包括：

```text
因子快照
财务快照
Transformer prediction
Qlib comparison
agent_performance_feedback
市场摘要
Buffett skill 内容
```

输出结构：

```text
MacroAnalysis
StockRecommendation
RiskReview
```

`agent_performance_feedback` 来自 `data/agent_feedback/agent_performance_feedback.json`。它只作为历史表现上下文和风险纪律提示进入 Agents，不会覆盖 `strategy_validation` 或确定性风控规则。

`tradingagents_local_tools.py` 提供给图节点使用的本地工具：

```text
get_stock_data
get_indicators
get_verified_market_snapshot
get_fundamentals
get_balance_sheet
get_cashflow
get_income_statement
get_news
get_global_news
get_insider_transactions
```

## Agents 策略验证与降级/降仓

`tradingagents_graph_runner.py` 的 `build_strategy_validation()` 会读取 Qlib comparison 结果，对 agents 的建议进行二次验证。

Transformer 口径优先级：
- 优先使用 `hybrid_live` / `transformer_live`，检验每天实盘实际保存的 live-production 预测快照。
- 如果 live 快照不足或对应回测 `skipped`，自动回退 `hybrid` / `transformer` walk-forward 口径。
- validation 会输出 `validation_signal_scope=live_snapshot` 或 `walk_forward_fallback`。

逻辑：

```text
规则/混合回测支持 -> support
规则弱且 Transformer 不能抵消 -> downgrade
规则弱但 Transformer 独立达标且收益优势足够 -> cautious + reduce exposure
```

当 Transformer 抵消规则弱势时：

```text
agent_research_conclusion = cautious
exposure_adjustment = reduce
exposure_multiplier = 0.5
transformer_offsets_rule_weakness = true
```

映射结果：

- 真 downgrade：`buy -> hold`，仓位归零。
- reduce exposure：保持 `buy`，仓位乘以 0.5。

## Agents Qlib 回测与反馈闭环

Agents 最终输出会作为第四个 Qlib signal source 参与 comparison：

```text
rule vs transformer vs hybrid vs agents
```

实现口径：
- `run_tradingagents_research()` 会保存当天最终映射后的 `action + weight` 到 `data/agent_signals/YYYY-MM-DD.json`。
- `qlib_backtest_utils.py` 的 `signal_source="agents"` 读取历史快照；`buy` 且 `weight > 0` 使用 `weight` 作为 agent-adjusted score，其他情况使用低分排除。
- Agents 回测只使用真实前向积累快照，不把当天建议回填到过去；历史快照少于 2 个交易日时，日报显示 `agents_status=skipped`。
- `agent_feedback_utils.py` 根据 agents vs hybrid 的 Qlib 表现写 `data/agent_feedback/agent_performance_feedback.json`。
- 下一次 TradingAgents 研究会读取该反馈作为历史表现上下文，但最终交易仍受 `strategy_validation` 和 deterministic risk rules 约束。

## 风控与仓位

`portfolio_utils.py` 的 deterministic risk rules 负责最终裁剪：

```text
单票上限
宏观总仓位上限
live portfolio rules multiplier
止损百分比
卖出指令处理
```

默认值：

```text
单票上限：20%
默认止损：8%
正式回测口径由 Qlib comparison 提供。
```

`portfolio_rule_utils.py` 会读取：

```text
data/portfolio_rules/live_portfolio_rules.json
```

并限制 multiplier 范围，避免单日反馈过度影响仓位。

## 实盘归因反馈

掘金终端脚本每日收盘后写：

```text
data/gm_attribution/YYYY-MM-DD.json
```

下一次 `main.py` 启动时，`live_attribution_utils.py` 读取最新归因并更新：

```text
data/factor_reports/live_factor_feedback.json
data/portfolio_rules/live_portfolio_rules.json
```

反馈规则：

- 因子 feedback multiplier 单次温和调整，不直接反转方向。
- 组合规则根据组合相对基准表现调整总仓位、单票仓位倾向和止损偏好。

## 掘金终端执行

`main.py` 最后调用 `gm_executor.execute_final_orders()`，写入：

```text
data/gm_orders.json
```

`gm_strategy_runner.py` 需要复制到掘金终端策略编辑器中运行。它会：

```text
定时读取 gm_orders.json
打印日报摘要和 Qlib comparison
提交 order_target_percent
收盘后写 gm_attribution/YYYY-MM-DD.json
```

复制到掘金终端后，建议通过环境变量提供 `GM_TOKEN`，`GM_STRATEGY_ID` 可在脚本底部或环境变量中设置；缺少 token 时脚本只会打印缺参提示，不会进入 `init/on_bar/schedule` 事件循环。

路径默认：

```python
PROJECT_ROOT = D:\quantitative
ORDERS_PATH = D:\quantitative\data\gm_orders.json
```

## 报告内容

`report_utils.py` 生成：

```text
reports/YYYY-MM-DD/HHMMSS.json
reports/YYYY-MM-DD/HHMMSS.md
```

Markdown 日报包含：

```text
Market State
Stock Signals
Cross-Section Signal Candidates
Deterministic Risk Result
Final Orders
Qlib Professional Backtest
Transformer Research / Inference
Alpha Analysis
Live Attribution Feedback
Agents Backtest Feedback
LLM Explanation
TradingAgents 多角色研究过程
Raw Agent Outputs
```

## 数据契约

核心 Pydantic schema 在 `schemas.py`：

```text
MacroAnalysis
  market_environment
  suggested_position
  reason

StockRecommendation
  stock_code
  action
  weight
  reason

RiskOrder
  stock_code
  direction
  quantity_percent
  stop_loss_percent

RiskReview
  approved
  adjustments
  final_orders
```

代码会自动把 6 位 A 股代码标准化为 `sh600000` / `sz000001` 风格。

## 常用命令

每日完整流水线：

```bash
python main.py
```

正式 walk-forward 历史评估：

```bash
python -c "from qlib_training_utils import run_transformer_walk_forward_training; print(run_transformer_walk_forward_training(force=True))"
```

更新当前实盘 Transformer 模型：

```bash
python -c "from qlib_training_utils import run_transformer_live_production_training; print(run_transformer_live_production_training(force=True))"
```

只跑每日 Transformer 推理：

```bash
python -c "from qlib_training_utils import run_transformer_inference; print(run_transformer_inference())"
```

只跑 Qlib comparison：

```bash
python -c "from qlib_backtest_utils import run_qlib_backtest; print(run_qlib_backtest())"
```

只更新 Agents 回测反馈：

```bash
python -c "from agent_feedback_utils import run_agent_feedback_update; print(run_agent_feedback_update())"
```

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

## 配置说明

主要配置在 `config.py`。

| 配置组 | 关键字段 | 说明 |
|---|---|---|
| 存储 | `DATA_DIR`, `STORAGE_FORMAT` | 数据根目录和格式 |
| 股票池 | `UNIVERSE_INDEX_SYMBOL`, `UNIVERSE_SOURCE`, `UNIVERSE_SIZE_LIMIT` | 指数成分股来源和数量 |
| 策略类型 | `STRATEGY_TYPE`, `LONG_QUANTILE`, `SHORT_QUANTILE` | 默认 `equity_index_enhanced`；前20%做指数增强，后20%仅做 alpha long-short 指标 |
| 基准权重 | `BENCHMARK_SYMBOL`, `BENCHMARK_WEIGHT_SOURCE`, `BENCHMARK_WEIGHT_DIR` | 沪深300成分权重缓存；失败时回退等权 |
| 横截面标准化 | `FACTOR_WINSORIZE_LOWER`, `FACTOR_WINSORIZE_UPPER` | 因子 winsorize / z-score / rank / 中性化参数 |
| 数据窗口 | `LOOKBACK_DAYS`, `MIN_REQUIRED_DAYS` | 每日因子刷新窗口和最少数据量 |
| 风控 | `MAX_POSITION_PER_STOCK`, `MAX_TOTAL_POSITION`, `INDUSTRY_WEIGHT_CAP`, `BENCHMARK_ACTIVE_TILT` | 指数增强组合的单票、总仓位、行业和主动倾斜约束 |
| 过滤 | `FILTER_ST`, `FILTER_SUSPENDED` | ST 和停牌过滤 |
| Qlib | `QLIB_ENABLED`, `QLIB_ACCOUNT`, `QLIB_BENCHMARK`, `QLIB_BACKTEST_WINDOW` | Qlib 回测 |
| Transformer walk-forward | `TRANSFORMER_WALK_FORWARD_*` | 历史滚动训练和正式预测 |
| Transformer live | `TRANSFORMER_LIVE_TRAINING_ENABLED`, `TRANSFORMER_LIVE_TRAIN_YEARS`, `TRANSFORMER_LIVE_VALID_YEARS` | 当前生产模型训练 |
| Transformer inference | `TRANSFORMER_INFERENCE_ENABLED`, `TRANSFORMER_MODEL_PATH`, `TRANSFORMER_PREDICTION_PATH`, `TRANSFORMER_LIVE_PREDICTION_DIR`, `TRANSFORMER_LIVE_MIN_SNAPSHOT_DAYS` | 每日推理和 live 预测快照 |
| Transformer label/features | `TRANSFORMER_LABEL_MODE`, `TRANSFORMER_FEATURE_NORMALIZATION`, `TRANSFORMER_FEATURE_MISSING_MAX`, `TRANSFORMER_FEATURE_MIN_STD`, `TRANSFORMER_FEATURE_MIN_ABS_RANKIC` | 横截面 label、特征标准化和训练窗内特征筛选 |
| Alpha | `ALPHA_ANALYSIS_ENABLED`, `ALPHA_FACTOR_SELECTION_PATH` | 因子分析和权重文件 |
| Cross-section signal weights | `SIGNAL_CROSS_WEIGHTS` | Alpha/Transformer/风控流动性三路横截面分数融合权重 |
| Live feedback | `LIVE_ATTRIBUTION_ENABLED`, `GM_ATTRIBUTION_DIR`, `LIVE_FACTOR_FEEDBACK_PATH`, `LIVE_PORTFOLIO_RULES_PATH` | 实盘反馈 |
| TradingAgents | `TRADINGAGENTS_*` | 研究图开关、分析师、轮数、缓存 |
| Agents feedback | `AGENT_SIGNAL_DIR`, `AGENT_FEEDBACK_DIR`, `AGENT_PERFORMANCE_FEEDBACK_PATH` | Agents 快照、Qlib agents 回测反馈 |
| EastMoney | `EASTMONEY_LIGHTWEIGHT_SPOT_ENABLED`, `EASTMONEY_REQUEST_TIMEOUT` | 估值快照 fallback |
| GM | `GM_EXECUTION_ENABLED`, `GM_ACCOUNT_ID` | 掘金指令输出 |

`TRANSFORMER_RETRAIN_ON_DAILY_RUN` 仅保留兼容旧配置；当前 `main.py` 不会用它触发每日训练。

## 环境准备

```bash
conda create -n quant_env python=3.10
conda activate quant_env
pip install akshare pandas requests python-dotenv openai pyqlib torch
pip install ta alphalens
```

编辑 `environment.env`：

```text
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## 测试覆盖

项目测试位于 `tests/`：

```text
test_signal_fusion.py
  signal fusion 权重、Transformer 缺失容错、候选排序、live feedback multiplier

test_strategy_validation.py
  agents downgrade / cautious reduce exposure / support 逻辑

test_walk_forward_transformer.py
  Transformer 横截面 label/特征筛选、walk-forward 年度窗口、Qlib 预测文件优先级、daily main 不训练、live production wrapper
```

运行：

```bash
python -m unittest discover -s tests -v
```

## 注意事项

- `main.py` 是每日实盘/日报入口，不训练 Transformer。
- 正式历史回测使用 walk-forward predictions，避免未来信息泄漏。
- `transformer_predictions.csv` 是 daily inference 文件，只服务 signal fusion。
- 如需从 2010 年开始做 walk-forward，需要先让 `data/factors` 覆盖足够长历史。
- 部分股票上市晚于 2010，只能从上市日起参与 fold，这是正常现象。
- 掘金 runner 依赖本机路径，复制到终端前确认 `PROJECT_ROOT` 和 `ORDERS_PATH`。
- `third_party/tradingagents/` 是内置依赖代码，日常修改优先在本项目适配层完成。
