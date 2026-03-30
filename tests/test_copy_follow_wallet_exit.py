"""Tests for delayed copy-follow analysis that exits with the wallet."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pandas as pd

from config.settings import Settings
from research.copy_follow_wallet_exit import (
    build_copy_exit_pairs,
    compute_copy_follow_wallet_exit_from_frame,
)


def _market_row(*, token_id: str, end_date: datetime, terminal_price: float) -> dict[str, object]:
    """Build one Gamma-like market row for expiry fallback tests."""

    return {
        "id": "market-1",
        "question": "Will this resolve?",
        "condition_id": "condition-1",
        "closed": True,
        "updated_at": end_date,
        "raw_json": json.dumps(
            {
                "clobTokenIds": [token_id],
                "outcomePrices": [terminal_price],
                "endDate": end_date.isoformat().replace("+00:00", "Z"),
            }
        ),
    }


def test_copy_follow_wallet_exit_uses_delayed_entry_and_delayed_exit() -> None:
    """Copied trades should enter and exit at the first public prices after both delays."""

    base = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    trades = pd.DataFrame(
        [
            {
                "trade_id": "buy-1",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "market_id": "condition-1",
                "token_id": "token-1",
                "timestamp": base,
                "side": "BUY",
                "price": 0.40,
                "size": 2.0,
                "usdc_size": 0.80,
                "spread_at_trade": 0.0,
                "slippage_bps_assumed": 0.0,
                "liquidity_bucket": "large",
            },
            {
                "trade_id": "sell-1",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "market_id": "condition-1",
                "token_id": "token-1",
                "timestamp": base + timedelta(seconds=60),
                "side": "SELL",
                "price": 0.60,
                "size": 2.0,
                "usdc_size": 1.20,
                "spread_at_trade": 0.0,
                "slippage_bps_assumed": 0.0,
                "liquidity_bucket": "large",
            },
        ]
    )
    price_history = pd.DataFrame(
        [
            {"token_id": "token-1", "ts": base + timedelta(seconds=6), "price": 0.42},
            {"token_id": "token-1", "ts": base + timedelta(seconds=11), "price": 0.43},
            {"token_id": "token-1", "ts": base + timedelta(seconds=16), "price": 0.44},
            {"token_id": "token-1", "ts": base + timedelta(seconds=31), "price": 0.45},
            {"token_id": "token-1", "ts": base + timedelta(seconds=66), "price": 0.58},
            {"token_id": "token-1", "ts": base + timedelta(seconds=71), "price": 0.57},
            {"token_id": "token-1", "ts": base + timedelta(seconds=76), "price": 0.56},
            {"token_id": "token-1", "ts": base + timedelta(seconds=91), "price": 0.55},
        ]
    )

    summary, diagnostics, overview = compute_copy_follow_wallet_exit_from_frame(
        trades,
        price_history,
        markets=pd.DataFrame(),
        delays=(5, 10, 15, 30),
        start_date="2026-03-20",
        end_date="2026-03-20",
        analysis_asof=base + timedelta(hours=1),
        settings=Settings(cost_scenario="optimistic"),
    )

    assert int(overview.iloc[0]["wallets_positive_net_total_usdc_5s"]) == 1
    assert diagnostics.iloc[0]["copy_status_5s"] == "ok"
    assert diagnostics.iloc[0]["copy_exit_type_5s"] == "wallet_sell"
    assert round(float(summary.iloc[0]["avg_copy_pnl_5s"]), 6) == 0.16
    assert round(float(summary.iloc[0]["total_copy_pnl_net_usdc_5s"]), 6) == 0.32
    assert round(float(summary.iloc[0]["avg_copy_pnl_30s"]), 6) == 0.10


def test_copy_follow_wallet_exit_handles_fifo_partial_close_and_pending_remainder() -> None:
    """FIFO matching should close earlier buys first and leave unmatched remainder open."""

    base = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    trades = pd.DataFrame(
        [
            {
                "trade_id": "buy-1",
                "wallet_address": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "market_id": "condition-1",
                "token_id": "token-1",
                "timestamp": base,
                "side": "BUY",
                "price": 0.40,
                "size": 3.0,
                "spread_at_trade": 0.0,
                "slippage_bps_assumed": 0.0,
                "liquidity_bucket": "large",
            },
            {
                "trade_id": "buy-2",
                "wallet_address": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "market_id": "condition-1",
                "token_id": "token-1",
                "timestamp": base + timedelta(seconds=10),
                "side": "BUY",
                "price": 0.45,
                "size": 2.0,
                "spread_at_trade": 0.0,
                "slippage_bps_assumed": 0.0,
                "liquidity_bucket": "large",
            },
            {
                "trade_id": "sell-1",
                "wallet_address": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "market_id": "condition-1",
                "token_id": "token-1",
                "timestamp": base + timedelta(seconds=40),
                "side": "SELL",
                "price": 0.60,
                "size": 4.0,
                "spread_at_trade": 0.0,
                "slippage_bps_assumed": 0.0,
                "liquidity_bucket": "large",
            },
        ]
    )

    pairs, open_positions = build_copy_exit_pairs(trades)

    assert len(pairs) == 2
    assert list(pairs["copied_size"]) == [3.0, 1.0]
    assert len(open_positions) == 1
    assert float(open_positions.iloc[0]["copied_size"]) == 1.0


def test_copy_follow_wallet_exit_skips_when_wallet_exits_before_delayed_entry() -> None:
    """A copied trade should be invalid when the wallet has already sold before we can enter."""

    base = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    trades = pd.DataFrame(
        [
            {
                "trade_id": "buy-1",
                "wallet_address": "0xcccccccccccccccccccccccccccccccccccccccc",
                "market_id": "condition-1",
                "token_id": "token-1",
                "timestamp": base,
                "side": "BUY",
                "price": 0.40,
                "size": 1.0,
                "spread_at_trade": 0.0,
                "slippage_bps_assumed": 0.0,
                "liquidity_bucket": "large",
            },
            {
                "trade_id": "sell-1",
                "wallet_address": "0xcccccccccccccccccccccccccccccccccccccccc",
                "market_id": "condition-1",
                "token_id": "token-1",
                "timestamp": base + timedelta(seconds=3),
                "side": "SELL",
                "price": 0.42,
                "size": 1.0,
                "spread_at_trade": 0.0,
                "slippage_bps_assumed": 0.0,
                "liquidity_bucket": "large",
            },
        ]
    )
    price_history = pd.DataFrame(
        [
            {"token_id": "token-1", "ts": base + timedelta(seconds=6), "price": 0.43},
            {"token_id": "token-1", "ts": base + timedelta(seconds=9), "price": 0.41},
        ]
    )

    summary, diagnostics, _ = compute_copy_follow_wallet_exit_from_frame(
        trades,
        price_history,
        markets=pd.DataFrame([_market_row(token_id="token-1", end_date=base + timedelta(days=1), terminal_price=1.0)]),
        delays=(5,),
        start_date="2026-03-20",
        end_date="2026-03-20",
        analysis_asof=base + timedelta(hours=1),
        settings=Settings(cost_scenario="optimistic"),
    )

    assert diagnostics.iloc[0]["copy_status_5s"] == "wallet_exit_before_delayed_entry"
    assert int(summary.iloc[0]["valid_copy_slices_5s"]) == 0
