"""Tests for resolved expiry-hold reporting."""

from __future__ import annotations

import pandas as pd

from research.resolved_expiry_report import compute_resolved_expiry_positions_from_frame


def test_compute_resolved_expiry_positions_marks_resolved_open_lot() -> None:
    """Open buy lots on resolved markets should count as observed hold-to-expiry."""

    recent_trades = pd.DataFrame(
        [
            {
                "trade_id": "buy-1",
                "wallet_address": "0x1111111111111111111111111111111111111111",
                "market_id": "m1",
                "token_id": "t1",
                "side": "BUY",
                "price": 0.40,
                "size": 10.0,
                "usdc_size": 4.0,
                "timestamp": "2026-03-10T00:00:00Z",
            }
        ]
    )
    terminal_lookup = {
        "t1": {
            "terminal_price": 1.0,
            "terminal_price_source": "gamma_outcome_prices_current",
            "resolution_ts": pd.Timestamp("2026-03-20T00:00:00Z"),
            "question": "Resolved market?",
        }
    }

    wallet_summary, resolved_trades, overview = compute_resolved_expiry_positions_from_frame(
        recent_trades,
        terminal_lookup,
        analysis_cutoff="2026-03-28T23:59:59Z",
    )

    assert int(overview.iloc[0]["held_to_expiry_observed_slices_total"]) == 1
    assert int(wallet_summary.iloc[0]["held_to_expiry_observed_slices"]) == 1
    assert float(wallet_summary.iloc[0]["hold_to_expiry_share_observed"]) == 1.0
    assert resolved_trades.iloc[0]["resolved_status"] == "held_to_expiry_observed"
