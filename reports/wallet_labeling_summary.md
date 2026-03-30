# Wallet First-Pass Labeling Summary

## Scope
- Labeling is **time-aware as-of the end of the currently observed sample**, not a backfilled label for earlier dates.
- Current observed sample end: `2026-03-26 23:59:59 UTC`.
- Wallet rows in feature table: `934`
- Wallet rows in label table: `934`

## Data Sources Used
- `exports/current_market_wallet_scan_top1000/wallets_100plus_recent_20260312_20260326.csv`
- `exports/recent_wallet_trade_capture_top1000_cohort/recent_wallet_trade_summary_20260312_20260326.csv`
- `exports/recent_wallet_trade_capture_top1000_cohort/recent_wallet_trades_20260312_20260326.csv`
- `exports/recent_wallet_realized_pnl_only/recent_wallet_realized_closed_pnl_wallet_summary.csv`
- `exports/recent_wallet_realized_pnl_only/recent_wallet_realized_closed_pnl_trades.csv`
- `exports/copy_follow_wallet_exit_recent_closed_realized_sql/copy_follow_wallet_exit_recent_wallet_summary_5s_10s_15s_30s_20260312_20260326.csv`
- `exports/per_wallet_half_forward/wallet_half_forward_results_15s_30s_60s.csv`
- `exports/follow_wallet_repeat_oos/wallet_repeat_positive_summary.csv`

## Feature Set
- Wallet identity and scan-universe totals: `wallet_id`, total observed trades, distinct markets, first/last seen.
- Recent activity: active days, recent trades, per-minute trade intensity, burstiness, recent volume.
- Holding behavior: average / median holding time, and close-within-10s / 30s / 60s shares.
- Delay edge: average copy edge at `0s`, `5s`, `15s`, `30s`, plus `60s` proxy from the half-split report.
- Delay decay: absolute and retention-style decay from `0s -> 30s`, and `0s -> 60s proxy`.
- PnL / stability: realized PnL, realized win rate, max drawdown, profit concentration, rolling 3-day positivity share.
- Trade sizing: average size, median size, size standard deviation, size CV.
- Copyability friction: valid delayed slices and fast-exit share at `30s`.

Columns written to `data/wallet_features.csv`:
wallet_id, sample_name, recent_trades_window, active_days, avg_holding_seconds, median_holding_seconds, share_closed_within_10s, share_closed_within_30s, share_closed_within_60s, avg_copy_edge_net_0s, avg_copy_edge_net_5s, avg_copy_edge_net_15s, avg_copy_edge_net_30s, avg_copy_edge_net_60s_proxy, edge_retention_30s_from_0, edge_retention_60s_from_0_proxy, realized_pnl_abs, realized_pnl_pct_est, max_drawdown_abs, max_drawdown_pct_of_peak, pnl_concentration_top1_share, pnl_concentration_top5_share, pnl_concentration_top10_share, realized_win_rate, rolling_3d_positive_share, mean_trades_per_active_minute, trade_burstiness_cv, avg_position_size_usdc, position_size_cv, fast_exit_share_30s, repeat_oos_test_positive_windows

## Rule-Based Label Logic
- `positive_ev_copyable`
  - delayed net edge remains positive at `15s` and `30s`
  - delay retention does not collapse immediately
  - rolling realized consistency and win rate are acceptable
  - profits are not excessively concentrated in one trade
  - drawdown is not extreme
- `hft_latency_sensitive`
  - `0s` edge is positive, but it decays sharply by `30s` / `60s proxy`
  - holding times are very short
  - many positions close before a delayed follower can enter
  - per-minute activity is bursty
- `yolo_noise_unstable`
  - PnL is concentrated in a few trades
  - drawdown is large
  - sizing is erratic or active days are very sparse
  - delayed copy edge is weak or non-positive

These are **first-pass heuristic labels**, not ML predictions.

## Label Counts
- `positive_ev_copyable`: `30`
- `hft_latency_sensitive`: `5`
- `yolo_noise_unstable`: `899`

## Confidence Counts
- `high`: `126`
- `medium`: `76`
- `low`: `732`

## Most Likely Positive EV Wallets
| wallet_id                                  | sample_name            | label_confidence   |   positive_ev_score |   avg_copy_edge_net_15s |   avg_copy_edge_net_30s |   avg_copy_edge_net_60s_proxy |   edge_retention_30s_from_0 |   rolling_3d_positive_share |   repeat_oos_test_positive_windows | primary_reasons                                                                                                               |
|:-------------------------------------------|:-----------------------|:-------------------|--------------------:|------------------------:|------------------------:|------------------------------:|----------------------------:|----------------------------:|-----------------------------------:|:------------------------------------------------------------------------------------------------------------------------------|
| 0xcdc316f1fd3f5a4bd42f414582d7959119cea71f | Hilar                  | medium             |                  10 |              0.0753471  |              0.0753471  |                    0.0749611  |                    0.685385 |                        1    |                                  1 | 15s_edge_positive|30s_edge_positive|60s_proxy_edge_positive|delay_retention_30_ok|rolling_consistency_ok|win_rate_ok          |
| 0x30eead8be2dbf57303b78eb9e7404d37c9bca587 | laserjohnny            | medium             |                  10 |              0.0572098  |              0.0572098  |                    0.0395579  |                    0.782478 |                        0.75 |                                  0 | 15s_edge_positive|30s_edge_positive|60s_proxy_edge_positive|delay_retention_30_ok|rolling_consistency_ok|win_rate_ok          |
| 0x77fd7aec1952ea7d042a6eec83bc4782f67db6c8 | Everlasting-Gobstopper | medium             |                   9 |              0.108228   |              0.108228   |                   -0.00293937 |                    1.19217  |                        1    |                                nan | 15s_edge_positive|30s_edge_positive|delay_retention_30_ok|rolling_consistency_ok|win_rate_ok|pnl_not_single_trade_dominated   |
| 0x7cbefea2f57bf129ea3446d53397b9520747c4b7 | VGR22                  | medium             |                   9 |              0.0376982  |              0.0376982  |                    0.0722874  |                    1.60999  |                        0.5  |                                  1 | 15s_edge_positive|30s_edge_positive|60s_proxy_edge_positive|delay_retention_30_ok|win_rate_ok|pnl_not_single_trade_dominated  |
| 0x1e3f673cbd329df727d564e07e9f9dfcd40ae00b | pmic3                  | medium             |                   8 |              0.00364574 |              0.00364574 |                    0.00938367 |                   -0.241552 |                        1    |                                nan | 15s_edge_positive|30s_edge_positive|60s_proxy_edge_positive|rolling_consistency_ok|win_rate_ok|drawdown_not_extreme           |
| 0x187365dee1866e49c87fba10734375615d5d37b6 | RaphCrypto             | medium             |                   7 |              0.0700047  |              0.0700047  |                    0.214975   |                 -221.935    |                        0.75 |                                nan | 15s_edge_positive|30s_edge_positive|60s_proxy_edge_positive|rolling_consistency_ok|drawdown_not_extreme                       |
| 0x16ce1f954bfae560af862712fad9dd261f4ceba1 | 36oldtiger             | low                |                  10 |              0.0256145  |              0.0256145  |                    0.0256145  |                    0.151401 |                        1    |                                  4 | 15s_edge_positive|30s_edge_positive|60s_proxy_edge_positive|rolling_consistency_ok|win_rate_ok|pnl_not_single_trade_dominated |
| 0x53ecc53e7a69aad0e6dda60264cc2e363092df91 | 0x53eCc53E7            | low                |                   9 |              0.193032   |              0.193032   |                    0.392875   |                    2.25526  |                        1    |                                  0 | 15s_edge_positive|30s_edge_positive|60s_proxy_edge_positive|delay_retention_30_ok|rolling_consistency_ok|win_rate_ok          |


## Most Likely HFT / Delay-Sensitive Wallets
| wallet_id                                  | sample_name   | label_confidence   |   hft_score |   avg_copy_edge_net_0s |   avg_copy_edge_net_30s |   edge_retention_30s_from_0 |   median_holding_seconds |   share_closed_within_60s |   fast_exit_share_30s | primary_reasons                                                                                            |
|:-------------------------------------------|:--------------|:-------------------|------------:|-----------------------:|------------------------:|----------------------------:|-------------------------:|--------------------------:|----------------------:|:-----------------------------------------------------------------------------------------------------------|
| 0xcf37bbb2c1687b70ea553b6fe05f740251d7993f | rocky42004    | medium             |           6 |              0.0143761 |            -0.158141    |                 -11.0003    |                     9680 |                 0         |              0.5      | 0s_edge_positive|30s_edge_decays_fast|60s_proxy_edge_nearly_gone|wallet_often_exits_before_copy_can_enter  |
| 0x563ac35e74cb2c3901e2fa418552d76a0cf2f578 | pnmdpqomwc    | low                |           5 |             -0.0235644 |             0.000410625 |                  -0.0174257 |                     9099 |                 0         |              0.75     | 30s_edge_decays_fast|wallet_often_exits_before_copy_can_enter|bursty_per_minute_activity                   |
| 0x58ee3d58aca1cce95cd1516e3cde9c05e2e40bda | nan           | low                |           5 |             -0.120989  |            -0.00693447  |                   0.0573147 |                       39 |                 0.6875    |              0        | 30s_edge_decays_fast|60s_proxy_edge_nearly_gone|median_holding_very_short|many_positions_closed_within_60s |
| 0x7c9e0b03d7505dad7e87777cd282628f75b2db3d | 50cents       | high               |           6 |              0.0331262 |            -0.0174294   |                  -0.526153  |                     2152 |                 0.0154278 |              0.708275 | 0s_edge_positive|30s_edge_decays_fast|60s_proxy_edge_nearly_gone|wallet_often_exits_before_copy_can_enter  |
| 0xa58d4f278d7953cd38eeb929f7e242bfc7c0b9b8 | AiBird        | high               |           6 |              0.0241255 |            -0.0426838   |                  -1.76924   |                     9144 |                 0         |              0.60733  | 0s_edge_positive|30s_edge_decays_fast|60s_proxy_edge_nearly_gone|wallet_often_exits_before_copy_can_enter  |


## Most Likely YOLO / Noise / Unstable Wallets
| wallet_id                                  | sample_name   | label_confidence   |   yolo_score |   realized_pnl_abs |   max_drawdown_pct_of_peak |   pnl_concentration_top1_share |   pnl_concentration_top5_share |   position_size_cv |   rolling_3d_positive_share | primary_reasons                                                                                                                                                   |
|:-------------------------------------------|:--------------|:-------------------|-------------:|-------------------:|---------------------------:|-------------------------------:|-------------------------------:|-------------------:|----------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0x77fc9cdaea8b408a30e5b11c98d5efbf2d5a0eac | cluemster     | medium             |            9 |          -44.9179  |                  100.966   |                       0.93401  |                              1 |           2.06861  |                        0    | single_trade_dominates_profits|few_trades_dominate_profits|drawdown_extreme|position_size_highly_variable|rolling_consistency_weak|win_rate_weak                  |
| 0xc602e34792d9a5e129f3404daef2df43d1d37fc1 | Dr.PNL        | medium             |            9 |         -612.56    |                  121.016   |                       0.672881 |                              1 |           1.75218  |                        0    | single_trade_dominates_profits|few_trades_dominate_profits|drawdown_extreme|position_size_highly_variable|rolling_consistency_weak|win_rate_weak                  |
| 0xd399ba186f89721f79e0a72bfa6c9babd2e13f46 | Etanol        | medium             |            9 |          -27.868   |                   10.2776  |                       0.508255 |                              1 |           1.843    |                        0    | single_trade_dominates_profits|few_trades_dominate_profits|drawdown_extreme|position_size_highly_variable|rolling_consistency_weak|win_rate_weak                  |
| 0xfd433a536659570357b89369c8bb1f988597df4c | chosenoneeee  | medium             |            8 |           -3.20196 |                    1.61962 |                       1        |                              1 |           1.81805  |                        0.5  | single_trade_dominates_profits|few_trades_dominate_profits|drawdown_extreme|position_size_highly_variable|win_rate_weak|delayed_copy_edge_non_positive            |
| 0xe4cfbb89c05d3429280c69969aee2cfb7a4678c4 | shine         | medium             |            8 |           -4.19204 |                    2.54429 |                       0.932392 |                              1 |           1.44856  |                        0    | single_trade_dominates_profits|few_trades_dominate_profits|drawdown_extreme|rolling_consistency_weak|win_rate_weak|delayed_copy_edge_non_positive                 |
| 0x14f709b4833d42355e09b002ee9225a74b384ef7 | wow3          | medium             |            8 |           -1.26753 |                  448.89    |                       0.920214 |                              1 |           1.13621  |                        0    | single_trade_dominates_profits|few_trades_dominate_profits|drawdown_extreme|rolling_consistency_weak|win_rate_weak|delayed_copy_edge_non_positive                 |
| 0xeeb975706bba0fe52687a637eab8f95c05516f7a | elsak         | medium             |            8 |           -0.04098 |                    1.0081  |                       0.826073 |                              1 |           0.458052 |                        0.25 | single_trade_dominates_profits|few_trades_dominate_profits|drawdown_extreme|rolling_consistency_weak|win_rate_weak|delayed_copy_edge_non_positive                 |
| 0x79dcb362c47ba851d75cf5873735391b2c81ca77 | rocky42008    | medium             |            8 |           -0.93592 |                    3.38558 |                       0.662515 |                              1 |           1.57666  |                        0    | single_trade_dominates_profits|few_trades_dominate_profits|drawdown_extreme|position_size_highly_variable|rolling_consistency_weak|delayed_copy_edge_non_positive |


## Assumptions and Limitations
- `0s` edge is estimated from observed `BUY -> SELL` realized signal pairs and research-only cost assumptions.
- `5s / 15s / 30s` delay edges come from the existing delayed copy-follow wallet-exit backtest summary.
- `60s` edge is only a **proxy** reconstructed from the half-split report; cross-half closes are not preserved, so treat it as lower-fidelity.
- Unrealized PnL is not currently available from the public-data reconstruction used here; it is left null.
- Profit concentration is measured as the share of **positive realized PnL** contributed by the top 1 / 5 / 10 winning trades.
- Labels are computed using only information observed up to the sample end. They should not be reinterpreted as labels that would have been known earlier in the sample.
