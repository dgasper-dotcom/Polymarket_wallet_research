"""Tests for first-pass wallet labeling heuristics."""

from __future__ import annotations

import pandas as pd

from research.wallet_labeling import _merge_delay_features, assign_first_pass_labels


def test_merge_delay_features_tracks_expiry_exit_share() -> None:
    """Delay feature merge should retain expiry-hold counts and shares."""

    copy_summary = pd.DataFrame(
        [
            {
                "wallet_address": "0x1111111111111111111111111111111111111111",
                "paired_copy_slices": 20,
                "wallet_sell_exits_15s": 3,
                "wallet_sell_exits_30s": 2,
                "expiry_exits_15s": 7,
                "expiry_exits_30s": 8,
                "valid_copy_slices_15s": 10,
                "valid_copy_slices_30s": 9,
                "wallet_exit_before_entry_15s": 1,
                "wallet_exit_before_entry_30s": 1,
                "entry_after_or_at_delayed_exit_15s": 2,
                "entry_after_or_at_delayed_exit_30s": 1,
            }
        ]
    )

    merged = _merge_delay_features(copy_summary)
    row = merged.iloc[0]
    assert row["resolved_copy_exits_15s"] == 10
    assert row["resolved_copy_exits_30s"] == 10
    assert row["expiry_exit_share_15s"] == 0.7
    assert row["expiry_exit_share_30s"] == 0.8


def test_assign_first_pass_labels_marks_positive_ev_wallet() -> None:
    """Delayed-positive, consistent wallets should be marked copyable."""

    features = pd.DataFrame(
        [
            {
                "wallet_id": "0x1111111111111111111111111111111111111111",
                "sample_name": "steady-wallet",
                "sample_pseudonym": "steady-wallet",
                "recent_trades_window": 120,
                "active_days": 10,
                "realized_closed_trades": 40,
                "avg_copy_edge_net_0s": 0.020,
                "avg_copy_edge_net_15s": 0.012,
                "avg_copy_edge_net_30s": 0.010,
                "avg_copy_edge_net_60s_proxy": 0.006,
                "edge_retention_30s_from_0": 0.50,
                "edge_retention_60s_from_0_proxy": 0.30,
                "valid_copy_slices_15s": 25,
                "valid_copy_slices_30s": 21,
                "rolling_3d_positive_share": 0.75,
                "realized_win_rate": 0.65,
                "pnl_concentration_top1_share": 0.20,
                "max_drawdown_pct_of_peak": 0.25,
                "repeat_oos_test_positive_windows": 3,
                "median_holding_seconds": 3600.0,
                "share_closed_within_60s": 0.05,
                "fast_exit_share_30s": 0.02,
                "realized_pnl_abs": 1500.0,
                "realized_pnl_pct_est": 0.20,
                "pnl_concentration_top5_share": 0.45,
                "position_size_cv": 0.40,
            }
        ]
    )

    labels = assign_first_pass_labels(features)
    row = labels.iloc[0]
    assert row["primary_label"] == "positive_ev_copyable"
    assert row["label_confidence"] == "high"


def test_assign_first_pass_labels_marks_hft_wallet() -> None:
    """Short-hold, fast-decay wallets should be marked latency-sensitive."""

    features = pd.DataFrame(
        [
            {
                "wallet_id": "0x2222222222222222222222222222222222222222",
                "sample_name": "hft-wallet",
                "sample_pseudonym": "hft-wallet",
                "recent_trades_window": 400,
                "active_days": 7,
                "realized_closed_trades": 80,
                "avg_copy_edge_net_0s": 0.025,
                "avg_copy_edge_net_15s": 0.002,
                "avg_copy_edge_net_30s": -0.001,
                "avg_copy_edge_net_60s_proxy": -0.004,
                "edge_retention_30s_from_0": 0.04,
                "edge_retention_60s_from_0_proxy": -0.16,
                "valid_copy_slices_15s": 40,
                "valid_copy_slices_30s": 20,
                "rolling_3d_positive_share": 0.45,
                "realized_win_rate": 0.54,
                "pnl_concentration_top1_share": 0.22,
                "max_drawdown_pct_of_peak": 0.40,
                "repeat_oos_test_positive_windows": 0,
                "median_holding_seconds": 18.0,
                "share_closed_within_60s": 0.78,
                "fast_exit_share_30s": 0.66,
                "trade_burstiness_cv": 2.2,
                "realized_pnl_abs": 320.0,
                "realized_pnl_pct_est": 0.03,
                "pnl_concentration_top5_share": 0.52,
                "position_size_cv": 0.70,
            }
        ]
    )

    labels = assign_first_pass_labels(features)
    row = labels.iloc[0]
    assert row["primary_label"] == "hft_latency_sensitive"


def test_assign_first_pass_labels_marks_yolo_wallet() -> None:
    """Concentrated, unstable wallets should be marked YOLO / noise."""

    features = pd.DataFrame(
        [
            {
                "wallet_id": "0x3333333333333333333333333333333333333333",
                "sample_name": "yolo-wallet",
                "sample_pseudonym": "yolo-wallet",
                "recent_trades_window": 28,
                "active_days": 2,
                "realized_closed_trades": 8,
                "avg_copy_edge_net_0s": 0.004,
                "avg_copy_edge_net_15s": -0.003,
                "avg_copy_edge_net_30s": -0.006,
                "avg_copy_edge_net_60s_proxy": -0.008,
                "edge_retention_30s_from_0": -1.5,
                "edge_retention_60s_from_0_proxy": -2.0,
                "valid_copy_slices_15s": 4,
                "valid_copy_slices_30s": 1,
                "rolling_3d_positive_share": 0.20,
                "realized_win_rate": 0.25,
                "pnl_concentration_top1_share": 0.81,
                "max_drawdown_pct_of_peak": 0.92,
                "repeat_oos_test_positive_windows": 0,
                "median_holding_seconds": 2400.0,
                "share_closed_within_60s": 0.0,
                "fast_exit_share_30s": 0.05,
                "trade_burstiness_cv": 0.8,
                "realized_pnl_abs": -900.0,
                "realized_pnl_pct_est": -0.55,
                "pnl_concentration_top5_share": 0.96,
                "position_size_cv": 2.4,
            }
        ]
    )

    labels = assign_first_pass_labels(features)
    row = labels.iloc[0]
    assert row["primary_label"] == "yolo_noise_unstable"
