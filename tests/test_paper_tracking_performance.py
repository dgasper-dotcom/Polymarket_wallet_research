from __future__ import annotations

import pandas as pd

from research.paper_tracking_performance import (
    _build_contribution_tables,
    _build_house_position_ledger,
    _build_house_position_ledger_with_cap,
)


def test_build_house_position_ledger_weights_reinforcements() -> None:
    tape = pd.DataFrame.from_records(
        [
            {
                "cluster_id": "buy1",
                "action": "open_long",
                "side": "BUY",
                "token_id": "t1",
                "market_id": "m1",
                "event_title": "Event",
                "outcome": "Yes",
                "first_ts": "2026-01-01T00:00:00+00:00",
                "last_ts": "2026-01-01T00:00:00+00:00",
                "trade_count": 1,
                "unique_wallet_count": 1,
                "supporting_wallets": '["0xaaa"]',
                "total_notional_usdc": 100.0,
                "avg_signal_price": 0.50,
            },
            {
                "cluster_id": "buy2",
                "action": "reinforce_long",
                "side": "BUY",
                "token_id": "t1",
                "market_id": "m1",
                "event_title": "Event",
                "outcome": "Yes",
                "first_ts": "2026-01-02T00:00:00+00:00",
                "last_ts": "2026-01-02T00:00:00+00:00",
                "trade_count": 1,
                "unique_wallet_count": 1,
                "supporting_wallets": '["0xbbb"]',
                "total_notional_usdc": 60.0,
                "avg_signal_price": 0.75,
            },
            {
                "cluster_id": "sell1",
                "action": "close_long",
                "side": "SELL",
                "token_id": "t1",
                "market_id": "m1",
                "event_title": "Event",
                "outcome": "Yes",
                "first_ts": "2026-01-03T00:00:00+00:00",
                "last_ts": "2026-01-03T00:00:00+00:00",
                "trade_count": 1,
                "unique_wallet_count": 1,
                "supporting_wallets": '["0xaaa"]',
                "total_notional_usdc": 0.0,
                "avg_signal_price": 0.90,
            },
        ]
    )

    (open_positions, closed_positions), skipped = _build_house_position_ledger(tape)

    assert open_positions.empty
    assert skipped.empty
    assert len(closed_positions) == 1
    row = closed_positions.iloc[0]
    assert row["reinforcement_count"] == 1
    assert row["supporting_wallet_count"] == 2
    assert round(float(row["entry_contracts"]), 6) == 280.0
    assert round(float(row["weighted_avg_entry_price"]), 6) == round(160.0 / 280.0, 6)
    assert float(row["realized_pnl_raw_usdc"]) > 0


def test_build_house_position_ledger_respects_position_cap() -> None:
    tape = pd.DataFrame.from_records(
        [
            {
                "cluster_id": "buy1",
                "action": "open_long",
                "side": "BUY",
                "token_id": "t1",
                "market_id": "m1",
                "event_title": "Event",
                "outcome": "Yes",
                "first_ts": "2026-01-01T00:00:00+00:00",
                "last_ts": "2026-01-01T00:00:00+00:00",
                "trade_count": 1,
                "unique_wallet_count": 1,
                "supporting_wallets": '["0xaaa"]',
                "total_notional_usdc": 80.0,
                "avg_signal_price": 0.5,
            },
            {
                "cluster_id": "buy2",
                "action": "reinforce_long",
                "side": "BUY",
                "token_id": "t1",
                "market_id": "m1",
                "event_title": "Event",
                "outcome": "Yes",
                "first_ts": "2026-01-02T00:00:00+00:00",
                "last_ts": "2026-01-02T00:00:00+00:00",
                "trade_count": 1,
                "unique_wallet_count": 1,
                "supporting_wallets": '["0xbbb"]',
                "total_notional_usdc": 50.0,
                "avg_signal_price": 0.5,
            },
        ]
    )

    (open_positions, closed_positions), skipped = _build_house_position_ledger_with_cap(
        tape,
        max_position_notional_usdc=100.0,
    )

    assert closed_positions.empty
    assert len(open_positions) == 1
    row = open_positions.iloc[0]
    assert float(row["entry_notional_usdc"]) == 100.0
    assert float(row["signaled_notional_usdc"]) == 130.0
    assert float(row["suppressed_notional_usdc"]) == 30.0
    assert len(skipped) == 1
    assert skipped.iloc[0]["reason"] == "position_cap_partial_reinforce"


def test_build_house_position_ledger_respects_total_open_cap() -> None:
    tape = pd.DataFrame.from_records(
        [
            {
                "cluster_id": "buy1",
                "action": "open_long",
                "side": "BUY",
                "token_id": "t1",
                "market_id": "m1",
                "event_title": "Event 1",
                "outcome": "Yes",
                "first_ts": "2026-01-01T00:00:00+00:00",
                "last_ts": "2026-01-01T00:00:00+00:00",
                "trade_count": 1,
                "unique_wallet_count": 1,
                "supporting_wallets": '["0xaaa"]',
                "total_notional_usdc": 80.0,
                "avg_signal_price": 0.5,
            },
            {
                "cluster_id": "buy2",
                "action": "open_long",
                "side": "BUY",
                "token_id": "t2",
                "market_id": "m2",
                "event_title": "Event 2",
                "outcome": "No",
                "first_ts": "2026-01-02T00:00:00+00:00",
                "last_ts": "2026-01-02T00:00:00+00:00",
                "trade_count": 1,
                "unique_wallet_count": 1,
                "supporting_wallets": '["0xbbb"]',
                "total_notional_usdc": 50.0,
                "avg_signal_price": 0.5,
            },
        ]
    )

    (open_positions, closed_positions), skipped = _build_house_position_ledger_with_cap(
        tape,
        max_position_notional_usdc=None,
        max_total_open_notional_usdc=100.0,
    )

    assert closed_positions.empty
    assert len(open_positions) == 2
    assert float(open_positions.loc[open_positions["token_id"] == "t1", "entry_notional_usdc"].iloc[0]) == 80.0
    assert float(open_positions.loc[open_positions["token_id"] == "t2", "entry_notional_usdc"].iloc[0]) == 20.0
    assert len(skipped) == 1
    assert skipped.iloc[0]["reason"] == "book_cap_partial_open"
    assert float(skipped.iloc[0]["skipped_notional_usdc"]) == 30.0


def test_build_contribution_tables_treats_missing_mtm_as_zero() -> None:
    closed_positions = pd.DataFrame.from_records(
        [
            {
                "event_title": "Event A",
                "entry_notional_usdc": 100.0,
                "realized_pnl_net_usdc": 20.0,
                "wallet_notional_attribution": {"0xaaa": 100.0},
            }
        ]
    )
    open_positions = pd.DataFrame.from_records(
        [
            {
                "event_title": "Event B",
                "entry_notional_usdc": 50.0,
                "mtm_pnl_net_usdc": float("nan"),
                "wallet_notional_attribution": {"0xbbb": 50.0},
            }
        ]
    )

    wallet_contrib, event_contrib, wallet_metrics, event_metrics = _build_contribution_tables(
        closed_positions=closed_positions,
        open_positions=open_positions,
    )

    assert set(wallet_contrib["wallet"]) == {"0xaaa", "0xbbb"}
    assert float(wallet_contrib.loc[wallet_contrib["wallet"] == "0xaaa", "combined_net_pnl_usdc"].iloc[0]) == 20.0
    assert float(wallet_contrib.loc[wallet_contrib["wallet"] == "0xbbb", "combined_net_pnl_usdc"].iloc[0]) == 0.0
    assert wallet_metrics["top1_positive_share"] == 1.0
    assert event_metrics["top1_positive_share"] == 1.0
