from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.weekly_wallet_check import (
    build_wallet_delta_rows,
    build_watchlist_activity_rows,
    default_trade_start_utc,
    filter_pma_trades_from,
)


def test_default_trade_start_utc_uses_current_year() -> None:
    reference = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)
    assert default_trade_start_utc(reference) == datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_filter_pma_trades_from_keeps_only_rows_after_cutoff() -> None:
    frame = pd.DataFrame.from_records(
        [
            {"trade_dttm": "2025-12-31 23:59:59", "value": 1},
            {"trade_dttm": "2026-01-01 00:00:00", "value": 2},
            {"trade_dttm": "2026-01-05 12:00:00", "value": 3},
        ]
    )
    filtered = filter_pma_trades_from(frame, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert list(filtered["value"]) == [2, 3]
    assert list(frame["trade_dttm"]) == [
        "2025-12-31 23:59:59",
        "2026-01-01 00:00:00",
        "2026-01-05 12:00:00",
    ]


def test_build_wallet_delta_rows_and_activity_rows(tmp_path: Path) -> None:
    baseline = {
        "0xaaa": {"positions": 2.0, "combined_net_pnl_usdc": 10.0, "realized_net_pnl_usdc": 4.0, "open_mtm_net_pnl_usdc": 6.0},
        "0xbbb": {"positions": 1.0, "combined_net_pnl_usdc": 5.0, "realized_net_pnl_usdc": 2.0, "open_mtm_net_pnl_usdc": 3.0},
    }
    current = {
        "0xaaa": {"positions": 3.0, "combined_net_pnl_usdc": 12.5, "realized_net_pnl_usdc": 4.5, "open_mtm_net_pnl_usdc": 8.0},
        "0xbbb": {"positions": 1.0, "combined_net_pnl_usdc": 1.0, "realized_net_pnl_usdc": 2.0, "open_mtm_net_pnl_usdc": -1.0},
    }
    meta = {
        "0xaaa": {"display_name": "Wallet A", "bucket": "copy_ready"},
        "0xbbb": {"display_name": "Wallet B", "bucket": "monitor"},
    }

    delta_rows = build_wallet_delta_rows(baseline, current, meta)
    assert delta_rows[0]["display_name"] == "Wallet A"
    assert delta_rows[0]["delta_combined_net_pnl_usdc"] == 2.5
    assert delta_rows[1]["delta_combined_net_pnl_usdc"] == -4.0

    raw_csv = tmp_path / "raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trade_dttm", "trader_id", "side", "value"])
        writer.writeheader()
        writer.writerows(
            [
                {"trade_dttm": "2026-03-29 23:59:59", "trader_id": "0xaaa", "side": "buy", "value": "10"},
                {"trade_dttm": "2026-03-30 00:00:00", "trader_id": "0xaaa", "side": "buy", "value": "20"},
                {"trade_dttm": "2026-03-31 01:00:00", "trader_id": "0xbbb", "side": "sell", "value": "7.5"},
            ]
        )

    activity_rows = build_watchlist_activity_rows(
        raw_csv,
        meta,
        datetime(2026, 3, 30, tzinfo=timezone.utc),
    )
    activity_map = {row["wallet"]: row for row in activity_rows}
    assert activity_map["0xaaa"]["trades_since_start"] == 1
    assert activity_map["0xaaa"]["net_flow_usdc"] == 20.0
    assert activity_map["0xbbb"]["sell_value_usdc"] == 7.5
    assert activity_map["0xbbb"]["net_flow_usdc"] == -7.5
