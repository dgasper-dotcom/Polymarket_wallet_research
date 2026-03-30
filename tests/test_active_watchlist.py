"""Tests for the active copyable-wallet watchlist builder."""

from __future__ import annotations

import pandas as pd

from research.active_watchlist import build_monitor_watchlist, build_strict_watchlist


def test_build_strict_watchlist_keeps_only_current_active_style_matches() -> None:
    """Strict watchlist should keep only current-active wallets that fit the style rules."""

    frame = pd.DataFrame(
        [
            {
                "wallet_id": "0x1111111111111111111111111111111111111111",
                "sample_name": "active-keeper",
                "is_current_active": True,
                "manual_excluded": False,
                "dominant_vertical": "sports",
                "dominant_vertical_share": 0.9,
                "recent_trades_window": 80,
                "active_days": 10,
                "sell_share": 0.05,
                "open_share": 0.9,
                "avg_copy_edge_net_30s": 0.08,
                "fast_exit_share_30s": 0.0,
                "median_holding_seconds": 200000.0,
                "holding_days": 200000.0 / 86400.0,
                "avg_trades_per_active_day_observed": 8.0,
                "rolling_3d_positive_share": 1.0,
            },
            {
                "wallet_id": "0x2222222222222222222222222222222222222222",
                "sample_name": "monitor-only",
                "is_current_active": False,
                "manual_excluded": False,
                "dominant_vertical": "politics",
                "dominant_vertical_share": 0.8,
                "recent_trades_window": 60,
                "active_days": 9,
                "sell_share": 0.10,
                "open_share": 0.85,
                "avg_copy_edge_net_30s": 0.06,
                "fast_exit_share_30s": 0.0,
                "median_holding_seconds": 250000.0,
                "holding_days": 250000.0 / 86400.0,
                "avg_trades_per_active_day_observed": 6.0,
                "rolling_3d_positive_share": 1.0,
            },
        ]
    )

    strict = build_strict_watchlist(frame)
    assert list(strict["wallet_id"]) == ["0x1111111111111111111111111111111111111111"]


def test_build_monitor_watchlist_keeps_style_matches_without_current_activity() -> None:
    """Monitor watchlist should keep only non-active wallets that still fit the style mask."""

    frame = pd.DataFrame(
        [
            {
                "wallet_id": "0x1111111111111111111111111111111111111111",
                "sample_name": "active-keeper",
                "is_current_active": True,
                "manual_excluded": False,
                "dominant_vertical": "sports",
                "dominant_vertical_share": 0.9,
                "recent_trades_window": 80,
                "active_days": 10,
                "sell_share": 0.05,
                "open_share": 0.9,
                "avg_copy_edge_net_30s": 0.08,
                "fast_exit_share_30s": 0.0,
                "median_holding_seconds": 200000.0,
                "holding_days": 200000.0 / 86400.0,
                "avg_trades_per_active_day_observed": 8.0,
                "rolling_3d_positive_share": 1.0,
            },
            {
                "wallet_id": "0x2222222222222222222222222222222222222222",
                "sample_name": "monitor-only",
                "is_current_active": False,
                "manual_excluded": False,
                "dominant_vertical": "politics",
                "dominant_vertical_share": 0.8,
                "recent_trades_window": 60,
                "active_days": 9,
                "sell_share": 0.10,
                "open_share": 0.85,
                "avg_copy_edge_net_30s": 0.06,
                "fast_exit_share_30s": 0.0,
                "median_holding_seconds": 250000.0,
                "holding_days": 250000.0 / 86400.0,
                "avg_trades_per_active_day_observed": 6.0,
                "rolling_3d_positive_share": 1.0,
            },
        ]
    )

    monitor = build_monitor_watchlist(frame)
    assert list(monitor["wallet_id"]) == ["0x2222222222222222222222222222222222222222"]
