"""Tests for behavior-level feature extraction and rule-based summaries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from research.behavior_analysis import (
    build_wallet_behavior_breakdown,
    extract_trade_features_frame,
    summarize_feature_performance,
)


def test_extract_trade_features_classifies_early_and_late_behaviors() -> None:
    """Feature extraction should produce interpretable phase and behavior labels."""

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    enriched = pd.DataFrame(
        [
            {
                "trade_id": "early-trade",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "market_id": "m1",
                "token_id": "t1",
                "timestamp": start + timedelta(minutes=10),
                "side": "BUY",
                "price": 0.52,
                "size": 50.0,
                "mid_at_trade": 0.52,
                "liquidity_bucket": "medium",
                "copy_pnl_5m": 0.01,
                "fade_pnl_5m": -0.01,
                "copy_pnl_net_5m": 0.008,
                "fade_pnl_net_5m": -0.012,
                "copy_pnl_net_5m_delay_30s": 0.006,
                "fade_pnl_net_5m_delay_30s": -0.014,
            },
            {
                "trade_id": "late-trade",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "market_id": "m1",
                "token_id": "t1",
                "timestamp": start + timedelta(minutes=90),
                "side": "BUY",
                "price": 0.86,
                "size": 200.0,
                "mid_at_trade": 0.86,
                "liquidity_bucket": "small",
                "copy_pnl_5m": 0.03,
                "fade_pnl_5m": -0.03,
                "copy_pnl_net_5m": 0.02,
                "fade_pnl_net_5m": -0.04,
                "copy_pnl_net_5m_delay_30s": 0.01,
                "fade_pnl_net_5m_delay_30s": -0.05,
            },
            {
                "trade_id": "nearby-trade",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "market_id": "m1",
                "token_id": "t1",
                "timestamp": start + timedelta(minutes=91),
                "side": "BUY",
                "price": 0.87,
                "size": 20.0,
                "mid_at_trade": 0.87,
                "liquidity_bucket": "small",
                "copy_pnl_5m": 0.01,
                "fade_pnl_5m": -0.01,
                "copy_pnl_net_5m": 0.005,
                "fade_pnl_net_5m": -0.015,
                "copy_pnl_net_5m_delay_30s": 0.003,
                "fade_pnl_net_5m_delay_30s": -0.017,
            },
        ]
    )
    price_history = pd.DataFrame(
        [
            {"token_id": "t1", "ts": start + timedelta(minutes=5), "price": 0.52},
            {"token_id": "t1", "ts": start + timedelta(minutes=9), "price": 0.52},
            {"token_id": "t1", "ts": start + timedelta(minutes=85), "price": 0.70},
            {"token_id": "t1", "ts": start + timedelta(minutes=89), "price": 0.82},
            {"token_id": "t1", "ts": start + timedelta(minutes=90), "price": 0.86},
            {"token_id": "t1", "ts": start + timedelta(minutes=91), "price": 0.87},
        ]
    )
    markets = pd.DataFrame(
        [
            {
                "id": "m1",
                "condition_id": "cond-1",
                "created_at": start,
                "updated_at": start + timedelta(minutes=100),
                "closed": True,
                "raw_json": '{"endDate": "2025-01-01T01:40:00Z"}',
            }
        ]
    )

    features = extract_trade_features_frame(enriched, price_history, markets).set_index("trade_id")
    assert features.loc["early-trade", "market_phase"] == "early"
    assert features.loc["early-trade", "price_zone"] == "centered"
    assert features.loc["early-trade", "trade_type_cluster"] == "early_positioning"
    assert features.loc["late-trade", "market_phase"] == "late"
    assert features.loc["late-trade", "price_zone"] == "extreme"
    assert features.loc["late-trade", "pre_trade_trend_state"] == "uptrend"
    assert features.loc["late-trade", "trade_type_cluster"] == "late_fomo_chaser"


def test_feature_performance_summary_selects_behavior_portfolio_candidates() -> None:
    """Positive and delay-robust behavior clusters should be flagged as candidates."""

    trades = pd.DataFrame(
        [
            {
                "trade_id": f"trade-{index}",
                "wallet_address": f"0x{index:040x}",
                "market_id": f"m{index % 2}",
                "token_id": f"t{index}",
                "size_bucket": "large",
                "price_zone": "balanced",
                "pre_trade_trend_state": "uptrend",
                "market_phase": "mid",
                "liquidity_bucket": "small",
                "trade_type_cluster": "aggressive_momentum_chaser",
                "copy_pnl_5m": 0.04,
                "fade_pnl_5m": -0.04,
                "copy_pnl_net_5m": 0.03,
                "fade_pnl_net_5m": -0.05,
                "copy_pnl_net_5m_delay_30s": 0.015,
                "fade_pnl_net_5m_delay_30s": -0.06,
            }
            for index in range(6)
        ]
    )

    summary = summarize_feature_performance(trades)
    row = summary[
        (summary["feature_name"] == "trade_type_cluster")
        & (summary["feature_value"] == "aggressive_momentum_chaser")
    ].iloc[0]
    assert row["recommended_mode"] == "copy"
    assert bool(row["selected_for_behavior_portfolio"]) is True
    assert float(row["copy_edge_retention_30s"]) > 0


def test_wallet_behavior_breakdown_marks_edge_drivers_and_drags() -> None:
    """Wallet behavior breakdown should separate positive drivers from negative drags."""

    trades = pd.DataFrame(
        [
            {
                "trade_id": "driver-1",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "trade_type_cluster": "early_positioning",
                "copy_pnl_net_5m": 0.03,
                "fade_pnl_net_5m": -0.02,
            },
            {
                "trade_id": "driver-2",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "trade_type_cluster": "early_positioning",
                "copy_pnl_net_5m": 0.02,
                "fade_pnl_net_5m": -0.01,
            },
            {
                "trade_id": "drag-1",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "trade_type_cluster": "late_fomo_chaser",
                "copy_pnl_net_5m": -0.04,
                "fade_pnl_net_5m": 0.01,
            },
            {
                "trade_id": "drag-2",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "trade_type_cluster": "late_fomo_chaser",
                "copy_pnl_net_5m": -0.03,
                "fade_pnl_net_5m": 0.0,
            },
        ]
    )

    breakdown = build_wallet_behavior_breakdown(trades)
    driver = breakdown[breakdown["trade_type_cluster"] == "early_positioning"].iloc[0]
    drag = breakdown[breakdown["trade_type_cluster"] == "late_fomo_chaser"].iloc[0]

    assert driver["cluster_role"] == "copy_edge_driver"
    assert drag["cluster_role"] == "fade_edge_driver"
    assert float(driver["share_of_wallet_trades"]) == 0.5
