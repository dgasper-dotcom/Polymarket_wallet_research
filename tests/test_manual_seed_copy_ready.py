"""Tests for the revised manual-seed copy-ready watchlist logic."""

from __future__ import annotations

from research.manual_seed_copy_ready import _classify_wallet


def test_classify_wallet_marks_positive_non_hft_wallet_copy_ready() -> None:
    decision = _classify_wallet(
        {
            "display_name": "TestWallet",
            "wallet_address": "0x1111111111111111111111111111111111111111",
            "notes": "",
        },
        {"pma_copy_combined_net_30s": "100.0"},
        {
            "combined_realized_plus_mtm_pnl_usdc_est": "50.0",
            "combined_realized_pnl_usdc_est": "20.0",
            "unresolved_open_mtm_net_pnl_usdc": "30.0",
            "unresolved_open_forward_30d_net_pnl_usdc": "10.0",
        },
        {
            "avg_copy_edge_net_30s": "0.02",
            "fast_exit_share_30s": "0.0",
            "median_holding_seconds": "86400",
            "recent_trades_window": "20",
            "recent_sell_trades": "5",
        },
    )
    assert decision.action_bucket == "copy_ready"
    assert decision.is_hft_like is False


def test_classify_wallet_marks_fast_exit_wallet_avoid() -> None:
    decision = _classify_wallet(
        {
            "display_name": "FastWallet",
            "wallet_address": "0x2222222222222222222222222222222222222222",
            "notes": "",
        },
        {"pma_copy_combined_net_30s": "50.0"},
        {"combined_realized_plus_mtm_pnl_usdc_est": "25.0"},
        {
            "avg_copy_edge_net_30s": "0.01",
            "fast_exit_share_30s": "0.50",
            "median_holding_seconds": "60",
            "recent_trades_window": "100",
            "recent_sell_trades": "80",
        },
    )
    assert decision.action_bucket == "avoid"
    assert decision.is_hft_like is True


def test_classify_wallet_keeps_crowded_wallet_on_monitor() -> None:
    decision = _classify_wallet(
        {
            "display_name": "Kickstand7",
            "wallet_address": "0xd1acd3925d895de9aec98ff95f3a30c5279d08d5",
            "notes": "manually approved but likely already crowded / copy-traded",
        },
        {"pma_copy_combined_net_30s": "999.0"},
        {"combined_realized_plus_mtm_pnl_usdc_est": "1000.0"},
        {
            "avg_copy_edge_net_30s": "0.2",
            "fast_exit_share_30s": "0.0",
            "median_holding_seconds": "100000",
            "recent_trades_window": "100",
            "recent_sell_trades": "10",
        },
    )
    assert decision.action_bucket == "monitor"
    assert decision.action_rationale == "crowded_reference_not_primary"
