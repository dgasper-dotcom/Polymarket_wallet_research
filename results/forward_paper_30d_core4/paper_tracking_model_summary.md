# Paper Tracking Model Summary

This is the operational paper-tracking layer for the unified house portfolio. It is not a historical backtest. It merges copy-ready wallet signals into one portfolio state so repeated same-thesis bets do not triple-size exposure.

- Wallet bucket tracked: `copy_ready`
- Raw mapped trades consumed: `1001`
- Consolidated house tape actions: `350`
- Current open house positions: `111`
- Closed house positions: `60`
- Reinforcement actions: `92`
- Market-conflict actions skipped: `4`
- Position-cap skip / partial-fill records: `219`
- Per-position notional cap: `100.00 USDC`
- Total concurrent house-notional cap: `50000.00 USDC`
- Average supporting wallets per open house position: `1.01`
- Approximate duplicate signal absorption count: `734`

## Interpretation
- `current_house_positions.csv` is the file to monitor going forward.
- `closed_house_positions.csv` shows where the unified house tape would have exited based on source wallet sells.
- `skipped_market_conflicts.csv` shows signals ignored because another wallet wanted the opposite outcome in a market where the house was already exposed.
- `skipped_position_cap_records.csv` shows signals that were fully blocked or partially clipped by the per-position cap.
- Reinforcements are recorded, but they do not automatically open a second independent position.
