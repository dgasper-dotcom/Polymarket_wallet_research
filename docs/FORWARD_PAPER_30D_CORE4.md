# Forward Paper Book: Next 30 Days

This is the frozen forward paper setup starting `2026-03-31`.

## Copy Through Unified House Book

- `MikeMoore` `0x5d2f49295387e01a49f0a3e59449ceed791c4adb`
- `ELICHOU` `0x312bcca3bc77bdc1d37dc6db5b9c1493de61cafe`
- `aikko` `0x68d1b156197fc516c56fc95d325b3716322c3c4d`
- `sbinnala` `0xc483ee2ce773ae281131382ecc6285c968b88ac8`

## Monitor Only

- `0x53...` `0x53ecc53e7a69aad0e6dda60264cc2e363092df91`
- `SnowballHustle` `0xe36296a42555b95e95880412387e954d84b0bd00`
- `PetrGrepl` `0xe7590338d435112c032e3ea51ff3d08a27a1e7ca`
- `RobertoRubio` `0x3c4c03892f47d3166ee049a48a73d4743a17dd95`

## Portfolio Rules

- Unified house book only
- Max per position: `100 USDC`
- Max concurrent open notional: `50,000 USDC`
- Same-token duplicate signals are consolidated
- Opposite-outcome market conflicts are skipped
- Source-wallet sells close the unified house position

## Suggested Run Command

```bash
cd /Users/davidgasper/Downloads/polymarket_wallet_research
python3 scripts/run_forward_paper_cycle.py \
  --wallet-csv configs/forward_paper_30d_core4_wallets.csv \
  --mapped-trades-csv exports/house_book_total_cap_compare_post_jan1/manual_seed_pma_mapped_trades_from_2026_01_01.csv \
  --paper-output-dir exports/forward_paper_30d_core4 \
  --dashboard-output-dir exports/forward_paper_30d_core4/forward_tracker \
  --watchlist-csv configs/forward_paper_30d_watchlist.csv \
  --max-position-notional-usdc 100 \
  --max-total-open-notional-usdc 50000 \
  --skip-refresh
```

## Outputs To Watch

- `exports/forward_paper_30d_core4/current_house_positions.csv`
- `exports/forward_paper_30d_core4/closed_house_positions.csv`
- `exports/forward_paper_30d_core4/performance/house_portfolio_performance_summary.md`
- `exports/forward_paper_30d_core4/forward_tracker/forward_paper_dashboard.md`

Every consolidated signal is logged in:

- `exports/forward_paper_30d_core4/consolidated/house_signal_tape.csv`

Anything that does not become an active house position should appear in one of:

- `exports/forward_paper_30d_core4/skipped_market_conflicts.csv`
- `exports/forward_paper_30d_core4/skipped_position_cap_records.csv`
- `exports/forward_paper_30d_core4/performance/house_skipped_position_records.csv`
