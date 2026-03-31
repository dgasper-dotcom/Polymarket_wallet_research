# House Portfolio Performance Summary

This layer tracks the unified house portfolio after duplicate wallet signals and opposite-outcome market conflicts have already been consolidated away.

- Analysis cutoff: `2026-03-31T14:57:05.803909+00:00`
- Open house positions: `111`
- Closed house positions: `60`
- Skipped / unpriceable ledger records: `219`
- Per-position notional cap: `100.00 USDC`
- Total concurrent house-notional cap: `50000.00 USDC`
- Closed-position realized net PnL: `709.91 USDC`
- Open-position MTM net PnL: `711.76 USDC`
- Combined house portfolio net PnL: `1421.67 USDC`
- Marked open positions: `107/111`
- Wallet contribution positive-share top 1: `50.8%`
- Wallet contribution positive-share top 5: `100.0%`
- Event contribution positive-share top 1: `19.7%`
- Event contribution positive-share top 5: `49.8%`
- Average holding days for currently open positions: `46.06`
- Average realized holding days for closed positions: `6.94`

## Output Files
- `house_open_position_performance.csv`: `/Users/davidgasper/Downloads/polymarket_wallet_research/exports/forward_paper_30d_core4/performance/house_open_position_performance.csv`
- `house_closed_position_performance.csv`: `/Users/davidgasper/Downloads/polymarket_wallet_research/exports/forward_paper_30d_core4/performance/house_closed_position_performance.csv`
- `house_skipped_position_records.csv`: `/Users/davidgasper/Downloads/polymarket_wallet_research/exports/forward_paper_30d_core4/performance/house_skipped_position_records.csv`
- `house_wallet_contribution.csv`: `/Users/davidgasper/Downloads/polymarket_wallet_research/exports/forward_paper_30d_core4/performance/house_wallet_contribution.csv`
- `house_event_contribution.csv`: `/Users/davidgasper/Downloads/polymarket_wallet_research/exports/forward_paper_30d_core4/performance/house_event_contribution.csv`
- `house_portfolio_daily_equity_curve.csv`: `/Users/davidgasper/Downloads/polymarket_wallet_research/exports/forward_paper_30d_core4/performance/house_portfolio_daily_equity_curve.csv`
- `house_portfolio_daily_equity_curve.png`: `/Users/davidgasper/Downloads/polymarket_wallet_research/exports/forward_paper_30d_core4/performance/house_portfolio_daily_equity_curve.png`

## Interpretation
- `realized net PnL` uses unified house exits when source wallets produce a consolidated sell signal.
- `open MTM net PnL` marks still-open house positions to the latest locally available public price history, with current Gamma terminal values only as a fallback.
- The daily equity curve combines cumulative realized PnL with day-end MTM on positions that were still open on each day.
