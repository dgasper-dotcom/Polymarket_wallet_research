"""Tests for delayed copy analysis held to expiry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pandas as pd

from config.settings import Settings
from research.copy_follow_expiry import compute_copy_follow_expiry_from_frame


def _market_row(*, token_id: str, end_date: datetime, terminal_price: float) -> dict[str, object]:
    """Build one Gamma-market-like record for expiry-follow tests."""

    return {
        "id": "market-1",
        "question": "Will this test pass?",
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


def test_copy_follow_expiry_uses_delayed_entry_and_terminal_value() -> None:
    """Expiry-held copy should enter after the delay and exit at terminal value."""

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    enriched = pd.DataFrame(
        [
            {
                "trade_id": "trade-1",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "market_id": "condition-1",
                "token_id": "token-1",
                "timestamp": base,
                "side": "BUY",
                "price": 0.40,
                "spread_at_trade": 0.0,
                "slippage_bps_assumed": 0.0,
                "liquidity_bucket": "large",
            }
        ]
    )
    price_history = pd.DataFrame(
        [
            {"token_id": "token-1", "ts": base + timedelta(seconds=16), "price": 0.45},
            {"token_id": "token-1", "ts": base + timedelta(seconds=31), "price": 0.47},
        ]
    )
    markets = pd.DataFrame(
        [_market_row(token_id="token-1", end_date=base + timedelta(days=1), terminal_price=1.0)]
    )

    summary, diagnostics, overview = compute_copy_follow_expiry_from_frame(
        enriched,
        price_history,
        markets,
        start_date="2024-01-01",
        end_date="2024-01-02",
        settings=Settings(cost_scenario="optimistic"),
    )

    assert int(overview.iloc[0]["wallets_positive_net_15s"]) == 1
    assert round(float(summary.iloc[0]["avg_copy_pnl_expiry_15s"]), 6) == 0.55
    assert round(float(summary.iloc[0]["avg_copy_pnl_expiry_30s"]), 6) == 0.53
    assert round(float(summary.iloc[0]["avg_copy_pnl_net_expiry_15s"]), 6) == 0.55
    assert diagnostics.iloc[0]["copy_status_expiry_15s"] == "ok"


def test_copy_follow_expiry_rejects_entries_after_resolution() -> None:
    """If the delayed entry is at or after resolution, the copied trade should be invalid."""

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    enriched = pd.DataFrame(
        [
            {
                "trade_id": "trade-1",
                "wallet_address": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "market_id": "condition-1",
                "token_id": "token-1",
                "timestamp": base,
                "side": "BUY",
                "price": 0.40,
                "spread_at_trade": 0.0,
                "slippage_bps_assumed": 0.0,
                "liquidity_bucket": "large",
            }
        ]
    )
    price_history = pd.DataFrame(
        [
            {"token_id": "token-1", "ts": base + timedelta(seconds=16), "price": 0.45},
            {"token_id": "token-1", "ts": base + timedelta(seconds=31), "price": 0.47},
        ]
    )
    markets = pd.DataFrame(
        [_market_row(token_id="token-1", end_date=base + timedelta(seconds=20), terminal_price=1.0)]
    )

    summary, diagnostics, _ = compute_copy_follow_expiry_from_frame(
        enriched,
        price_history,
        markets,
        start_date="2024-01-01",
        end_date="2024-01-02",
        settings=Settings(cost_scenario="optimistic"),
    )

    assert diagnostics.iloc[0]["copy_status_expiry_30s"] == "delay_at_or_after_resolution"
    assert int(summary.iloc[0]["valid_trades_30s"]) == 0
    assert pd.isna(summary.iloc[0]["avg_copy_pnl_net_expiry_30s"])

