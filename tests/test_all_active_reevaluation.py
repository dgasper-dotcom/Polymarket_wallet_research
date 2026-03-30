"""Tests for the all-active reevaluation bucket logic."""

from __future__ import annotations

from contextlib import contextmanager

import pandas as pd

from research.all_active_reevaluation import classify_all_active_wallets, compute_open_position_evidence


def test_classify_all_active_wallets_uses_prior_style_plus_unresolved_open_bucket() -> None:
    """Style-compatible wallets with many unresolved opens should survive reevaluation."""

    frame = pd.DataFrame(
        [
            {
                "wallet_id": "0x1111111111111111111111111111111111111111",
                "sample_name": "long-holder",
                "manual_excluded": False,
                "avg_copy_edge_net_30s": 0.08,
                "fast_exit_share_30s": 0.0,
                "median_holding_seconds": 200000.0,
                "rolling_3d_positive_share": 1.0,
                "recent_trades_window": 80,
                "sell_share": 0.05,
                "open_share": 0.90,
                "held_to_expiry_observed_slices": 0,
                "unresolved_open_slices": 40,
                "long_window_trade_rows": 120,
            },
            {
                "wallet_id": "0x2222222222222222222222222222222222222222",
                "sample_name": "no-history",
                "manual_excluded": False,
                "avg_copy_edge_net_30s": None,
                "fast_exit_share_30s": None,
                "median_holding_seconds": None,
                "rolling_3d_positive_share": None,
                "recent_trades_window": None,
                "sell_share": None,
                "open_share": None,
                "held_to_expiry_observed_slices": 0,
                "unresolved_open_slices": 0,
                "long_window_trade_rows": 0,
            },
        ]
    )

    classified = classify_all_active_wallets(frame)
    row0 = classified.loc[classified["wallet_id"] == "0x1111111111111111111111111111111111111111"].iloc[0]
    row1 = classified.loc[classified["wallet_id"] == "0x2222222222222222222222222222222222222222"].iloc[0]

    assert bool(row0["meets_prior_style_filters"]) is True
    assert row0["reevaluation_bucket"] == "candidate_with_many_unresolved_open_positions"
    assert row1["reevaluation_bucket"] == "insufficient_public_long_window_data"


def test_compute_open_position_evidence_marks_to_market_and_forward(monkeypatch) -> None:
    """Unresolved open lots should get MTM and forward 7d/30d evidence."""

    recent_trades = pd.DataFrame(
        [
            {
                "trade_id": "buy-1",
                "wallet_address": "0xabc",
                "market_id": "market-1",
                "token_id": "token-1",
                "timestamp": "2026-01-01T00:00:00Z",
                "side": "BUY",
                "price": 0.40,
                "size": 10.0,
                "usdc_size": 4.0,
                "spread_at_trade": 0.01,
                "slippage_bps_assumed": 0.0,
                "liquidity_bucket": "large",
                "raw_json": "{}",
            }
        ]
    )

    price_history = pd.DataFrame(
        [
            {"token_id": "token-1", "ts": "2026-01-08T00:00:00Z", "price": 0.55},
            {"token_id": "token-1", "ts": "2026-01-31T00:00:00Z", "price": 0.70},
            {"token_id": "token-1", "ts": "2026-02-10T00:00:00Z", "price": 0.65},
        ]
    )

    @contextmanager
    def _fake_session():
        yield object()

    monkeypatch.setattr("research.all_active_reevaluation.get_session", _fake_session)
    monkeypatch.setattr(
        "research.all_active_reevaluation.load_price_history_frame",
        lambda session, token_ids=None, start_ts=None, end_ts=None: price_history,
    )

    summary, unresolved = compute_open_position_evidence(
        recent_trades,
        terminal_lookup={
            "token-1": {
                "terminal_price": 0.9,
                "resolution_ts": pd.Timestamp("2026-03-01T00:00:00Z"),
            }
        },
        analysis_cutoff="2026-02-15T00:00:00Z",
    )

    assert len(unresolved) == 1
    trade = unresolved.iloc[0]
    assert trade["unresolved_status"] == "open_unresolved"
    assert trade["mtm_price"] == 0.65
    assert trade["mtm_price_source"] == "price_history_latest_before_cutoff"
    assert trade["forward_price_7d"] == 0.55
    assert trade["forward_price_30d"] == 0.70
    assert trade["forward_price_source_7d"] == "price_history_exactish"
    assert trade["forward_price_source_30d"] == "price_history_exactish"
    assert float(trade["holding_days_open"]) == 45.0

    row = summary.iloc[0]
    assert row["wallet_id"] == "0xabc"
    assert int(row["unresolved_open_slices"]) == 1
    assert int(row["paired_wallet_sell_slices"]) == 0
    assert float(row["unresolved_open_share"]) == 1.0
    assert float(row["unresolved_open_mtm_total_net_usdc"]) > 0
    assert float(row["unresolved_open_avg_holding_days"]) == 45.0
    assert int(row["unresolved_open_valid_forward_7d"]) == 1
    assert int(row["unresolved_open_valid_forward_30d"]) == 1
    assert float(row["unresolved_open_avg_forward_7d_net"]) > 0
    assert float(row["unresolved_open_avg_forward_30d_net"]) > 0
