# polymarket_wallet_research

English overview:
- Start with [PROJECT_SUMMARY.md](/Users/davidgasper/Downloads/polymarket_wallet_research/PROJECT_SUMMARY.md) for a professor- or investor-facing summary.
- See [results/README.md](/Users/davidgasper/Downloads/polymarket_wallet_research/results/README.md) for the lightweight results bundle included in Git.

`polymarket_wallet_research` 是一个仅供研究用途的 Python MVP，用于评估公开 Polymarket 钱包是否表现出可用于研究的预测能力，以及更适合采用：

- `copy`：顺势跟随
- `fade`：反向均值回归
- `ignore`：样本或信号不足，不建议研究性跟踪

本项目严格限制在公开数据范围内：

- 不包含任何实盘交易、下单或执行逻辑
- 不使用任何需要身份验证的 Polymarket 端点
- 保留原始 API JSON 数据

## 功能概览

- 回填公开钱包交易历史
- 回填市场元数据与 token 元数据
- 回填相关 token 的公开历史价格
- 构建带来源标记和缺失标记的 `wallet_trades_enriched`
- 生成事件研究汇总表与交易级诊断表
- 生成钱包评分、排行榜、诊断 CSV 与简单图表
- 基于时间切分执行样本外（OOS）训练/测试验证
- 针对多种拆分方案执行多重 OOS 鲁棒性验证
- 基于现有 enriched 数据执行延迟、价差、滑点和费用条件下的净信号分析
- 将信号从“钱包层”拆解到“交易行为层”，识别更稳定的行为模式
- 通过公开市场交易扫描自动构建大规模钱包宇宙，并输出 strong / weak 钱包群体
- 提供 `--dry-run` 模式，便于在少量钱包上先审查抓取计划

## 安装

```bash
cd /Users/davidgasper/Downloads/polymarket_wallet_research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 配置

`.env.example`：

```env
POLY_GAMMA_BASE=https://gamma-api.polymarket.com
POLY_DATA_BASE=https://data-api.polymarket.com
POLY_CLOB_BASE=https://clob.polymarket.com
POLY_WS_MARKET=wss://ws-subscriptions-clob.polymarket.com/ws/market
DB_URL=sqlite:///./polymarket_wallets.db
REQUEST_TIMEOUT=20
MAX_CONCURRENCY=8
LOG_LEVEL=INFO
OOS_TRAIN_FRACTION=0.70
OOS_SELECT_TOP_N=10
OOS_TRAIN_MIN_TRADES=10
OOS_TRAIN_MIN_MARKETS=3
OOS_TRAIN_MAX_TOP_MARKET_FRACTION=0.70
OOS_RECENT_ACTIVITY_DAYS=90
OOS_TEST_MIN_TRADES=5
OOS_TEST_MIN_MARKETS=2
MULTI_OOS_N_SPLITS=10
MULTI_OOS_RANDOM_SEED=42
MULTI_OOS_MIN_OBSERVED_SPLITS=3
MULTI_OOS_MIN_SELECTED_SPLITS=2
MULTI_OOS_ROBUST_SELECTION_FREQUENCY=0.50
MULTI_OOS_ROBUST_POSITIVE_TEST_FREQUENCY=0.60
MULTI_OOS_MODE_CONSISTENCY_THRESHOLD=0.75
MULTI_OOS_INCONSISTENT_MODE_THRESHOLD=0.60
POLYMARKET_FEE_K=0.0625
FLAT_FEE_BPS=
COST_SCENARIO=base
EXTRA_COST_PENALTY=0.010
```

## wallets.txt 格式

项目根目录自带了一个示例 [`wallets.txt`](./wallets.txt)。

支持：

- 每行一个钱包
- 空行
- 以 `#` 开头的注释行
- 自动去重

示例：

```text
# Public wallets only
0x56687bf447db6ffa42ffe2204a05edaa20f55839

# Duplicate lines are fine; they will be deduped
0x56687bf447db6ffa42ffe2204a05edaa20f55839
```

## 运行方式

完整流程：

```bash
python scripts/run_backfill.py --wallets wallets.txt
python scripts/run_enrichment.py
python scripts/run_event_study.py
python scripts/run_reports.py
python scripts/run_oos_validation.py
python scripts/run_multi_oos_validation.py
python scripts/run_delay_analysis.py
python scripts/run_behavior_analysis.py
python scripts/run_wallet_universe_scan.py
```

空运行模式：

```bash
python scripts/run_backfill.py --wallets wallets.txt --dry-run
```

也可以直接传地址：

```bash
python scripts/run_backfill.py --wallets 0x56687bf447db6ffa42ffe2204a05edaa20f55839
```

## 样本外验证

`run_oos_validation.py` 使用已经构建好的 `wallet_trades_enriched` 作为研究数据源。

- 训练集与测试集严格按时间切分
- 支持显式日期切分，或按时间顺序做比例切分
- 训练集只用于评分和选钱包
- 测试集只用于评估泛化表现
- 当前期限比较仍基于现有 enriched 表中的 `1m / 5m / 30m` 分钟级窗口，不是自然月

示例：

```bash
python scripts/run_oos_validation.py --train-fraction 0.70
python scripts/run_oos_validation.py --split-date 2024-11-01 --top-n 5
```

多重拆分示例：

```bash
python scripts/run_multi_oos_validation.py --n-splits 10
python scripts/run_multi_oos_validation.py --n-splits 10 --include-random --random-splits 2
```

## 延迟与成本分析

`run_delay_analysis.py` 复用已存在的 `wallet_trades_enriched` 和 `price_history`，不会重新抓取公开 API。

- `0s` 作为基准信号
- 再模拟 `5s / 15s / 30s / 60s` 之后才入场
- 如果公开价格点不够密，使用“延迟时刻之后的第一个可用价格点”近似
- 同时输出毛 PnL、净 PnL、优势留存率和收支平衡成本

示例：

```bash
python scripts/run_delay_analysis.py
python scripts/run_delay_analysis.py --output-dir exports/delay_analysis
```

## 行为分析

`run_behavior_analysis.py` 复用已有的 `wallet_trades_enriched`、延迟/净 PnL 列，以及公开 `price_history` / `markets` 元数据。

它做三件事：

- 为每笔交易提取规则化特征并写入 `trade_features` 表
- 按特征分组做条件事件研究，而不是只按钱包汇总
- 用简单规则把交易归类为若干行为模式，例如 `early_positioning`、`late_fomo_chaser`

示例：

```bash
python scripts/run_behavior_analysis.py
python scripts/run_behavior_analysis.py --output-dir exports/behavior_analysis
```

如果净 PnL / 延迟列尚未填充，脚本会先基于本地已有的 `price_history` 自动补齐，不会重新抓取公开 API。

## 钱包宇宙扫描

`run_wallet_universe_scan.py` 不依赖手工钱包列表。它会：

- 分页扫描公开 Gamma `/markets`
- 对每个可交易市场分页扫描公开 Data API `/trades?market=...`
- 从 `proxyWallet` 提取并全局去重钱包地址
- 基于公开观察到的市场交易构建 master wallet universe
- 用公开 `/positions` 与 `/closed-positions` 估算 wallet-level realized PnL

示例：

```bash
python scripts/run_wallet_universe_scan.py --output-dir exports/wallet_universe
python scripts/run_wallet_universe_scan.py --max-markets 25 --output-dir exports/wallet_universe_smoke
```

导出：

- `master_wallet_universe.csv`
- `strong_wallets.csv`
- `strong_wallets.txt`
- `weak_wallets.csv`
- `weak_wallets.txt`
- `wallet_universe_assumptions.md`

## 原始信号 vs 净信号

- 原始信号：直接根据公开价格点计算的 `copy_pnl_*` / `fade_pnl_*`
- 净信号：从原始信号中扣除现实成本后的 `copy_pnl_net_*` / `fade_pnl_net_*`

成本情景：

- `optimistic`：`spread + slippage`
- `base`：`spread + slippage + fee`
- `conservative`：`spread + slippage + fee + extra_penalty`

费用近似：

- 默认使用 `fee = k * price * (1 - price)`
- 默认 `k = 0.0625`
- 若设置了 `FLAT_FEE_BPS`，则优先使用固定费率覆盖该曲线

## 为什么延迟会摧毁大多数信号

- 钱包成交被公开观察到时，市场可能已经吸收了同一信息
- 公开价格历史是离散点，不是逐笔成交流，短期边际优势会快速衰减
- 如果一个信号只能在 `0s` 或 `5s` 下为正，而在 `15s` 或 `30s` 下转负，它通常更像“研究上看见了”，而不是“现实里抓得住”

## 收支平衡成本的解读

- `break_even_cost_copy_5m` / `break_even_cost_fade_5m` 表示该钱包在 `5m` 窗口上的平均毛 PnL 最多能承受多少总成本
- 若真实总成本高于这个值，平均净 PnL 理论上会转为非正
- 这个指标越高，说明该钱包的边际优势越厚；越接近 `0`，说明信号很容易被执行摩擦吃掉

## 什么决定了一个信号是否可交易

当前研究标签：

- `tradable`：净 PnL 为正，并且在 `15s` 或 `30s` 延迟下仍保持为正，且跨拆分结果不明显翻转
- `borderline`：只在低延迟或乐观成本下勉强为正
- `not_tradable`：在现实成本近似下净 PnL 不再为正

这只是研究标签，不是交易建议；它依然受公开价格点粒度和缺失订单簿快照的限制。

## 为什么钱包层 Alpha 往往有限

- 单个钱包的样本量常常不够大
- 同一个钱包可能混合了多种完全不同的行为
- 钱包层评分容易把“偶然踩中某一种模式”误读成“全面能力”

因此，钱包层适合做入口筛查，但不适合直接当作最终信号单元。

## 为什么行为层 Alpha 可能更稳健

- 行为特征能把同一钱包里的好交易和坏交易拆开
- 不同钱包可能共享相同的可解释行为模式
- 如果同一种行为在多个钱包、多个市场里都重复出现，通常比“某一个钱包看起来很强”更值得研究

当前 MVP 的行为层输出仍是规则分桶，不是机器学习模型；目标是先拿到可解释、可审计的结论。

## 为什么要先构建钱包宇宙

- 如果先手工挑钱包，再做强/弱分组，很容易产生选择偏差
- 从公开市场交易中先自动发现钱包，再做 cohort 研究，更接近真实研究流程
- strong / weak 钱包群体只是研究入口，不是未来收益的承诺

## 预期写入磁盘的文件

数据库：

- `polymarket_wallets.db`

研究产物：

- `artifacts/event_study/wallet_event_study_summary.csv`
- `artifacts/event_study/wallet_trade_diagnostics.csv`
- `artifacts/reports/wallet_leaderboard.csv`
- `artifacts/plots/*.png`

检查型导出：

- `exports/endpoint_audit.csv`
- `exports/wallet_activity_summary.csv`
- `exports/wallet_diagnostics.csv`
- `exports/top_copy_wallets.csv`
- `exports/top_fade_wallets.csv`
- `exports/recommended_mode_counts.csv`
- `exports/dry_run_wallets.csv`
- `exports/dry_run_market_targets.csv`
- `exports/dry_run_token_targets.csv`
- `exports/oos_validation/wallet_train_metrics.csv`
- `exports/oos_validation/wallet_test_metrics.csv`
- `exports/oos_validation/wallet_oos_comparison.csv`
- `exports/oos_validation/train_selected_copy_wallets.csv`
- `exports/oos_validation/train_selected_fade_wallets.csv`
- `exports/oos_validation/selected_wallets_test_results.csv`
- `exports/oos_validation/test_portfolio_copy_summary.csv`
- `exports/oos_validation/test_portfolio_fade_summary.csv`
- `exports/oos_validation/plots/*.png`
- `exports/multi_oos/split_run_summary.csv`
- `exports/multi_oos/split_selected_wallets.csv`
- `exports/multi_oos/split_portfolio_performance.csv`
- `exports/multi_oos/wallet_robustness_summary.csv`
- `exports/multi_oos/portfolio_robustness_summary.csv`
- `exports/multi_oos/delay_robustness_summary.csv`
- `exports/multi_oos/portfolio_delay_performance.csv`
- `exports/multi_oos/splits/<split_id>/*`
- `exports/multi_oos/plots/*.png`
- `exports/delay_analysis/trade_delay_diagnostics.csv`
- `exports/delay_analysis/wallet_delay_event_study.csv`
- `exports/delay_analysis/wallet_delay_summary.csv`
- `exports/delay_analysis/portfolio_delay_performance.csv`
- `exports/delay_analysis/plots/*.png`
- `exports/behavior_analysis/feature_performance_summary.csv`
- `exports/behavior_analysis/trade_type_clusters.csv`
- `exports/behavior_analysis/wallet_behavior_breakdown.csv`
- `exports/wallet_universe/master_wallet_universe.csv`
- `exports/wallet_universe/strong_wallets.csv`
- `exports/wallet_universe/strong_wallets.txt`
- `exports/wallet_universe/weak_wallets.csv`
- `exports/wallet_universe/weak_wallets.txt`
- `exports/wallet_universe/wallet_universe_assumptions.md`

## 回退逻辑与缺失数据处理

`wallet_trades_enriched` 会保留研究透明度字段：

- `missing_price_history`
- `missing_market_metadata`
- `used_fallback_midpoint`
- `trade_price_source`
- `midpoint_source`
- `book_source`
- `enrichment_status`
- `missing_reason`

中点价格优先级：

1. 最近的公开历史价格点
2. 若历史价格缺失，则使用当前公开订单簿估算值
3. 若两者都不可用，则置为 `NULL` 并附带缺失标志

说明：

- `trade_price_source` 当前来自公开 trade feed
- `midpoint_source` 可能是 `price_history_exactish`、`price_history_nearest`、`live_book_approx` 或 `missing_prices`
- `book_source` 当前只表示 enrichment 时刻的公开订单簿可不可得
- 当前 MVP 中的订单簿字段是近似值，不是历史事件时点的真实快照
- 如果市场元数据缺失，交易仍然保留并继续参与后续流程
- 如果历史价格缺失，交易仍然保留，但相关 forward-return 字段会保持 `NULL`

## 为什么要做时间切分

- 如果先看较新的结果再回头筛钱包，研究会产生前视偏差
- 时间切分让训练集负责“发现信号”，测试集负责“验证信号”
- 如果钱包只在训练集好看、在测试集明显退化，就不应被当作可泛化信号

## 为什么单次 OOS 不够

- 单次切分很容易碰巧踩中某段特殊市场环境
- 某个钱包可能只在一个 cutoff 上看起来有效
- 多重拆分更接近真正的问题：信号是否在不同边界和训练长度下仍然成立

## 稳定性标签

- `stable`：训练集与测试集在主导模式上保持一致，且测试样本足够
- `unstable`：训练集与测试集方向不一致，或测试表现明显偏离
- `insufficient_data`：测试交易数或测试市场覆盖太少，无法严肃评估

## 多重拆分鲁棒性

`run_multi_oos_validation.py` 会复用现有单次 OOS 流程，并对多种 split 方案重复运行：

- 比例拆分：默认包含 `60/40`、`70/30`、`80/20`
- 滚动时间拆分：沿时间轴选择多个 cutoff
- 可选随机索引拆分：随机抽取时间排序后的边界索引，但 train/test 仍严格保持时间顺序

聚合后的关键指标：

- `selection_frequency`：钱包在多少比例的拆分中被训练集选中
- `positive_test_frequency`：钱包被选中后，在测试集对应模式下拿到正 `5m` PnL 的比例
- `mode_consistency`：钱包在不同拆分里的主导模式一致程度；越接近 `1` 越稳定
- `net_positive_test_frequency`：在现实成本近似下，测试集净 PnL 仍为正的比例

鲁棒性标签：

- `robust`：经常被选中，测试期多数为正，且模式很少翻转
- `fragile`：偶尔有效，但测试延续性不够
- `inconsistent`：`copy` / `fade` 模式在不同拆分里经常切换
- `insufficient_data`：被观察或被选中的拆分次数太少

## 延迟感知鲁棒性

多重拆分输出还会额外聚合：

- `avg_net_test_copy_pnl_5m`
- `avg_net_test_fade_pnl_5m`
- `delay_robustness_summary.csv`
- `portfolio_delay_performance.csv`

这部分用于回答两个更现实的问题：

- 这个信号在测试集里扣除成本后还剩多少？
- 这个优势是否只能在极低延迟下存在？

## 行为研究输出

行为研究的核心导出包括：

- `feature_performance_summary.csv`：按 `size_bucket`、`price_zone`、`pre_trade_trend_state`、`market_phase`、`liquidity_bucket`、`trade_type_cluster` 分组的条件表现
- `trade_type_clusters.csv`：逐笔交易的特征、行为标签和对应净 PnL
- `wallet_behavior_breakdown.csv`：每个钱包由哪些行为模式驱动收益，哪些模式拖累收益

首版规则型行为标签：

- `aggressive_momentum_chaser`
- `passive_accumulator`
- `late_fomo_chaser`
- `early_positioning`
- `other`

这些标签是研究分桶，不是对真实交易意图的确定性判定。

## 公开端点与假设

端点契约集中定义在 [`config/api_contracts.py`](./config/api_contracts.py)。

核心假设：

- `GET /trades` 使用 `user`, `limit`, `offset`
- `GET /trades?market=...` 在市场扫描时使用 `conditionId`，并从响应中的 `proxyWallet` 提取钱包
- `GET /markets` 使用 `condition_ids` 与 `clob_token_ids`；单值过滤比多值过滤更稳定，因此代码按单值逐个查
- `GET /markets/{market_id}` 使用 Gamma market id
- `GET /positions` 与 `GET /closed-positions` 用于 wallet-level realized PnL 估算
- `GET /book` 使用 `token_id`
- `GET /prices-history` 当前使用的 query key 是 `market`，但传入的是 token id

## Known Data Limitations

- 历史订单簿快照未在 MVP 中保存，因此订单簿相关字段只能视为 enrichment 时刻的近似值
- 一部分旧 token 会对公开 `prices-history` 或 `book` 返回 `400/404`
- 公开 profile trades 并不稳定提供显式 `trade_id`，因此项目使用确定性哈希派生主键
- 公开价格历史是离散时间点，不是逐笔成交流，因此 forward return 存在“最近点近似”
- 评分使用的是研究导向的 z-score 风格综合分数，不代表可执行 alpha
- 市场和 token 的公开接口字段偶尔会以 JSON 字符串而非数组形式返回
- OOS 验证虽然减少了训练内过拟合，但依然受限于公开价格点和近似的微观结构代理
- 因为当前没有历史订单簿快照，极短期结论特别容易被近似中点和离散时间点噪音污染
- 多重拆分改善的是“边界鲁棒性”，不是“真实可交易性”；它不能弥补公开数据粒度不足的问题
- 延迟分析不会假设毫秒级精度；`5s / 15s / 30s / 60s` 都是基于“延迟之后第一个可用公开价格点”的近似
- Polymarket 费用模型是研究近似，不是官方结算明细复刻
- 如果一个钱包的净优势只在 `0s` 存在，而在 `15s` 或 `30s` 明显消失，这更可能说明信号容易被他人复制或被市场快速消化
- 行为分析里的 `time_to_resolution_minutes` 和 `market_phase` 依赖公开 market JSON 中可解析的结束/结算时间；缺失时会退化为 `unknown` 或弱代理
- `size_to_liquidity_ratio` 是基于当前样本内可观察交易构建的流动性代理，不是完整市场深度
- 行为标签是规则分桶；它们适合做解释和筛查，不应被误读为完整的交易风格识别器
- 钱包宇宙扫描依赖公开 `/trades` 的可访问分页范围；极高成交量市场可能因公共 offset 限制而截断
- 钱包宇宙导出的 `realized_pnl_absolute` 和 `realized_pnl_percent` 明确是估算值，不是审计级精确值

## 测试

```bash
pytest
```

## 研究用途声明

这是研究工具，不是实盘交易系统，也不是投资建议生成器。

## 如何运行行为分析脚本

```bash
python scripts/run_behavior_analysis.py
python scripts/run_behavior_analysis.py --output-dir exports/behavior_analysis
```

## 已发现的典型行为模式示例

- 市场早期、价格仍接近 `0.5`、且没有明显趋势时的大中型建仓，常被归到 `early_positioning`
- 市场后期、价格已接近极值、且交易方向继续追随既有趋势时，常被归到 `late_fomo_chaser`
- 同一钱包在短时间内连续分批下单、但单笔不算大的交易，更容易落入 `passive_accumulator`

## 哪些行为模式看起来更值得研究

- 在低延迟与 `30s` 延迟下都还能保留净优势的 `trade_type_cluster`
- 在多个钱包里重复出现、而不是只由单一钱包贡献的行为分组
- 价格靠近 `0.5`、市场尚处早期、且净 PnL 不依赖极低延迟的行为
