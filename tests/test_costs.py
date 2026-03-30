"""Tests for the simple trading cost model."""

from __future__ import annotations

import pandas as pd

from research.costs import (
    apply_copy_fade_costs,
    calculate_net_pnl,
    estimate_break_even_cost,
    estimate_polymarket_fee,
    estimate_trade_costs,
)


def test_estimate_trade_costs_and_copy_fade_outputs() -> None:
    """Round-trip cost and copy/fade PnL should reflect spread and slippage assumptions."""

    row = pd.Series(
        {
            "price": 0.50,
            "spread_at_trade": 0.02,
            "liquidity_bucket": "small",
            "ret_1m": 0.01,
            "ret_5m": 0.03,
            "ret_30m": -0.02,
        }
    )
    costs = estimate_trade_costs(row)
    adjusted = apply_copy_fade_costs(row)

    assert round(costs["round_trip_cost"], 6) == 0.0212
    assert round(adjusted["copy_pnl_5m"], 6) == 0.0088
    assert round(adjusted["fade_pnl_5m"], 6) == -0.0512


def test_polymarket_fee_curve_peaks_near_midpoint() -> None:
    """The quadratic fee proxy should be highest around price 0.5 and symmetric."""

    fee_low = estimate_polymarket_fee(0.2, fee_enabled=True, fee_k=0.0625, flat_fee_bps=None)
    fee_mid = estimate_polymarket_fee(0.5, fee_enabled=True, fee_k=0.0625, flat_fee_bps=None)
    fee_high = estimate_polymarket_fee(0.8, fee_enabled=True, fee_k=0.0625, flat_fee_bps=None)

    assert fee_mid > fee_low
    assert round(fee_low, 6) == round(fee_high, 6)


def test_net_pnl_and_break_even_cost_helpers() -> None:
    """Net-PnL subtraction and break-even cost should behave linearly."""

    assert round(calculate_net_pnl(0.05, 0.02), 6) == 0.03
    assert round(estimate_break_even_cost(pd.Series([0.10, 0.20, -0.10])), 6) == 0.066667
