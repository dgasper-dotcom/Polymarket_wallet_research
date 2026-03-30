"""Tests for delay and realistic net-PnL analysis."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from config.settings import Settings
from research.delay_analysis import (
    classify_tradability,
    compute_delay_trade_metrics,
)


def test_delay_metrics_use_first_price_after_delay() -> None:
    """Delay entry should use the first public price at or after the delayed timestamp."""

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    enriched = pd.DataFrame(
        [
            {
                "trade_id": "trade-1",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "market_id": "m1",
                "token_id": "token-1",
                "timestamp": base,
                "side": "BUY",
                "price": 0.50,
                "ret_5m": 0.12,
                "spread_at_trade": 0.02,
                "liquidity_bucket": "medium",
                "slippage_bps_assumed": 8.0,
            }
        ]
    )
    price_history = pd.DataFrame(
        [
            {"token_id": "token-1", "ts": base + timedelta(seconds=6), "price": 0.52},
            {"token_id": "token-1", "ts": base + timedelta(seconds=16), "price": 0.54},
            {"token_id": "token-1", "ts": base + timedelta(seconds=31), "price": 0.55},
            {"token_id": "token-1", "ts": base + timedelta(seconds=61), "price": 0.56},
            {"token_id": "token-1", "ts": base + timedelta(seconds=306), "price": 0.60},
            {"token_id": "token-1", "ts": base + timedelta(seconds=316), "price": 0.62},
            {"token_id": "token-1", "ts": base + timedelta(seconds=331), "price": 0.63},
            {"token_id": "token-1", "ts": base + timedelta(seconds=361), "price": 0.64},
        ]
    )

    result = compute_delay_trade_metrics(enriched, price_history, settings=Settings(cost_scenario="base"))
    row = result.iloc[0]

    assert round(float(row["copy_pnl_5m_delay_5s"]), 6) == 0.08
    assert round(float(row["copy_pnl_5m_delay_15s"]), 6) == 0.08
    assert float(row["copy_pnl_net_5m"]) > 0
    assert float(row["copy_pnl_net_5m_delay_30s"]) > 0


def test_tradability_classification_logic() -> None:
    """Tradability labels should separate durable, marginal, and broken edges."""

    settings = Settings()
    assert classify_tradability(
        base_net_pnl=0.03,
        net_pnl_delay_15s=0.02,
        net_pnl_delay_30s=0.01,
        optimistic_net_pnl=0.03,
        mode_consistency=0.9,
        settings=settings,
    ) == "tradable"
    assert classify_tradability(
        base_net_pnl=0.01,
        net_pnl_delay_15s=-0.01,
        net_pnl_delay_30s=-0.02,
        optimistic_net_pnl=0.02,
        mode_consistency=0.9,
        settings=settings,
    ) == "borderline"
    assert classify_tradability(
        base_net_pnl=-0.01,
        net_pnl_delay_15s=-0.02,
        net_pnl_delay_30s=-0.03,
        optimistic_net_pnl=-0.005,
        mode_consistency=0.9,
        settings=settings,
    ) == "not_tradable"
