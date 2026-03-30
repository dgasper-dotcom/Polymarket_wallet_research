"""Tests for focused good-wallet analysis helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from research.good_wallet_analysis import (
    build_good_wallet_summary,
    build_mirror_wallet_candidates,
)


def test_build_good_wallet_summary_uses_trade_observations_and_feature_metrics() -> None:
    """Wallet summary should merge feature metrics with observed recent behavior."""

    wallet = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    features = pd.DataFrame(
        [
            {
                "wallet_id": wallet,
                "sample_name": "steady-wallet",
                "sample_pseudonym": "steady-wallet",
                "recent_trades_window": 3,
                "avg_position_size_usdc": 125.0,
                "avg_holding_seconds": 3600.0,
                "median_holding_seconds": 1800.0,
                "avg_copy_edge_net_15s": 0.03,
                "avg_copy_edge_net_30s": 0.02,
                "avg_copy_edge_net_60s_proxy": 0.01,
                "fast_exit_share_30s": 0.0,
                "rolling_3d_positive_share": 1.0,
                "repeat_oos_test_positive_windows": 2,
                "pending_open_copy_slices": 8,
                "paired_copy_slices": 2,
                "realized_pnl_abs": 50.0,
            }
        ]
    )
    recent_trades = pd.DataFrame(
        [
            {
                "wallet_address": wallet,
                "market_id": "m1",
                "event_slug": "sports-event",
                "title": "Sports Event?",
                "outcome": "Yes",
                "side": "BUY",
                "timestamp": datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc),
                "trade_date": datetime(2026, 3, 12, tzinfo=timezone.utc),
                "sample_name_observed": "steady-wallet",
                "sample_pseudonym_observed": "steady-wallet",
            },
            {
                "wallet_address": wallet,
                "market_id": "m2",
                "event_slug": "sports-event",
                "title": "Sports Event?",
                "outcome": "No",
                "side": "BUY",
                "timestamp": datetime(2026, 3, 12, 1, 0, tzinfo=timezone.utc),
                "trade_date": datetime(2026, 3, 12, tzinfo=timezone.utc),
                "sample_name_observed": "steady-wallet",
                "sample_pseudonym_observed": "steady-wallet",
            },
            {
                "wallet_address": wallet,
                "market_id": "m3",
                "event_slug": "macro-event",
                "title": "Macro Event?",
                "outcome": "Yes",
                "side": "SELL",
                "timestamp": datetime(2026, 3, 13, 1, 0, tzinfo=timezone.utc),
                "trade_date": datetime(2026, 3, 13, tzinfo=timezone.utc),
                "sample_name_observed": "steady-wallet",
                "sample_pseudonym_observed": "steady-wallet",
            },
        ]
    )

    summary = build_good_wallet_summary([wallet], features=features, recent_trades=recent_trades)
    row = summary.iloc[0]

    assert row["sample_name"] == "steady-wallet"
    assert int(row["observed_recent_trades"]) == 3
    assert int(row["active_days_observed"]) == 2
    assert float(row["sell_share_observed"]) == 1 / 3
    assert float(row["open_copy_slice_share"]) == 0.8
    assert row["top_event_slug"] == "sports-event"


def test_build_mirror_wallet_candidates_finds_near_synchronous_trade_copy() -> None:
    """Mirror detection should rank the closest near-synchronous wallet first."""

    anchor = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    mirror = "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    other = "0xcccccccccccccccccccccccccccccccccccccccc"
    ts = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)

    recent_trades = pd.DataFrame(
        [
            {
                "wallet_address": anchor,
                "token_id": "t1",
                "market_id": "m1",
                "event_slug": "sports-event",
                "side": "BUY",
                "price": 0.41,
                "usdc_size": 100.0,
                "timestamp": ts,
                "epoch_seconds": int(ts.timestamp()),
            },
            {
                "wallet_address": anchor,
                "token_id": "t2",
                "market_id": "m2",
                "event_slug": "sports-event",
                "side": "BUY",
                "price": 0.62,
                "usdc_size": 75.0,
                "timestamp": ts.replace(minute=1),
                "epoch_seconds": int(ts.replace(minute=1).timestamp()),
            },
            {
                "wallet_address": mirror,
                "token_id": "t1",
                "market_id": "m1",
                "event_slug": "sports-event",
                "side": "BUY",
                "price": 0.4105,
                "usdc_size": 101.0,
                "timestamp": ts.replace(second=3),
                "epoch_seconds": int(ts.replace(second=3).timestamp()),
            },
            {
                "wallet_address": mirror,
                "token_id": "t2",
                "market_id": "m2",
                "event_slug": "sports-event",
                "side": "BUY",
                "price": 0.6205,
                "usdc_size": 74.0,
                "timestamp": ts.replace(minute=1, second=4),
                "epoch_seconds": int(ts.replace(minute=1, second=4).timestamp()),
            },
            {
                "wallet_address": other,
                "token_id": "t1",
                "market_id": "m1",
                "event_slug": "sports-event",
                "side": "BUY",
                "price": 0.45,
                "usdc_size": 220.0,
                "timestamp": ts.replace(minute=10),
                "epoch_seconds": int(ts.replace(minute=10).timestamp()),
            },
        ]
    )
    features = pd.DataFrame(
        [
            {"wallet_id": mirror, "sample_name": "mirror-wallet", "recent_trades_window": 20, "recent_sell_trades": 1, "avg_copy_edge_net_30s": 0.03},
            {"wallet_id": other, "sample_name": "other-wallet", "recent_trades_window": 20, "recent_sell_trades": 10, "avg_copy_edge_net_30s": -0.01},
        ]
    )

    candidates = build_mirror_wallet_candidates(anchor, recent_trades=recent_trades, features=features)
    top = candidates.iloc[0]

    assert top["candidate_wallet"] == mirror
    assert int(top["exact_price_matches_15s"]) == 2
    assert int(top["same_size_matches_5s"]) == 2
