# House Portfolio Performance Summary

This layer tracks the unified house portfolio after duplicate wallet signals and opposite-outcome market conflicts have already been consolidated away.

- Analysis cutoff: `2026-03-30T16:26:52.958755+00:00`
- Open house positions: `1249`
- Closed house positions: `890`
- Skipped / unpriceable ledger records: `0`
- Closed-position realized net PnL: `55945.78 USDC`
- Open-position MTM net PnL: `-9639.04 USDC`
- Combined house portfolio net PnL: `46306.74 USDC`
- Marked open positions: `331/1249`
- Average holding days for currently open positions: `172.65`
- Average realized holding days for closed positions: `16.69`

## Output Files
- `house_open_position_performance.csv`: `exports/manual_seed_paper_tracking/performance/house_open_position_performance.csv`
- `house_closed_position_performance.csv`: `exports/manual_seed_paper_tracking/performance/house_closed_position_performance.csv`
- `house_skipped_position_records.csv`: `exports/manual_seed_paper_tracking/performance/house_skipped_position_records.csv`
- `house_portfolio_daily_equity_curve.csv`: `exports/manual_seed_paper_tracking/performance/house_portfolio_daily_equity_curve.csv`
- `house_portfolio_daily_equity_curve.png`: `exports/manual_seed_paper_tracking/performance/house_portfolio_daily_equity_curve.png`

## Interpretation
- `realized net PnL` uses unified house exits when source wallets produce a consolidated sell signal.
- `open MTM net PnL` marks still-open house positions to the latest locally available public price history, with current Gamma terminal values only as a fallback.
- The daily equity curve combines cumulative realized PnL with day-end MTM on positions that were still open on each day.
