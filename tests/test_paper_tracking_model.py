from __future__ import annotations

import csv
from pathlib import Path

from research.paper_tracking_model import run_paper_tracking_model


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_paper_tracking_model_builds_unified_open_and_conflict(tmp_path: Path) -> None:
    wallet_csv = tmp_path / "wallets.csv"
    trades_csv = tmp_path / "mapped_trades.csv"

    _write_csv(
        wallet_csv,
        ["display_name", "wallet_id", "action_bucket"],
        [
            {"display_name": "A", "wallet_id": "0xaaa", "action_bucket": "copy_ready"},
            {"display_name": "B", "wallet_id": "0xbbb", "action_bucket": "copy_ready"},
            {"display_name": "C", "wallet_id": "0xccc", "action_bucket": "monitor"},
        ],
    )
    _write_csv(
        trades_csv,
        [
            "trade_id",
            "wallet_address",
            "wallet_name",
            "timestamp",
            "side",
            "price",
            "size",
            "usdc_size",
            "event_id",
            "event_title",
            "market_subtitle",
            "outcome",
            "market_id",
            "token_id",
            "raw_json",
            "spread_at_trade",
            "slippage_bps_assumed",
            "liquidity_bucket",
        ],
        [
            {
                "trade_id": "1",
                "wallet_address": "0xaaa",
                "wallet_name": "A",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "side": "BUY",
                "price": "0.4",
                "size": "10",
                "usdc_size": "4",
                "event_id": "e1",
                "event_title": "Event 1",
                "market_subtitle": "",
                "outcome": "Yes",
                "market_id": "m1",
                "token_id": "t_yes",
                "raw_json": "{}",
                "spread_at_trade": "",
                "slippage_bps_assumed": "",
                "liquidity_bucket": "",
            },
            {
                "trade_id": "2",
                "wallet_address": "0xbbb",
                "wallet_name": "B",
                "timestamp": "2026-01-01T06:00:00+00:00",
                "side": "BUY",
                "price": "0.42",
                "size": "5",
                "usdc_size": "2.1",
                "event_id": "e1",
                "event_title": "Event 1",
                "market_subtitle": "",
                "outcome": "Yes",
                "market_id": "m1",
                "token_id": "t_yes",
                "raw_json": "{}",
                "spread_at_trade": "",
                "slippage_bps_assumed": "",
                "liquidity_bucket": "",
            },
            {
                "trade_id": "3",
                "wallet_address": "0xbbb",
                "wallet_name": "B",
                "timestamp": "2026-01-03T00:00:00+00:00",
                "side": "BUY",
                "price": "0.55",
                "size": "7",
                "usdc_size": "3.85",
                "event_id": "e1",
                "event_title": "Event 1",
                "market_subtitle": "",
                "outcome": "No",
                "market_id": "m1",
                "token_id": "t_no",
                "raw_json": "{}",
                "spread_at_trade": "",
                "slippage_bps_assumed": "",
                "liquidity_bucket": "",
            },
            {
                "trade_id": "4",
                "wallet_address": "0xaaa",
                "wallet_name": "A",
                "timestamp": "2026-01-04T00:00:00+00:00",
                "side": "SELL",
                "price": "0.7",
                "size": "4",
                "usdc_size": "2.8",
                "event_id": "e1",
                "event_title": "Event 1",
                "market_subtitle": "",
                "outcome": "Yes",
                "market_id": "m1",
                "token_id": "t_yes",
                "raw_json": "{}",
                "spread_at_trade": "",
                "slippage_bps_assumed": "",
                "liquidity_bucket": "",
            },
        ],
    )

    result = run_paper_tracking_model(
        wallet_csv=wallet_csv,
        mapped_trades_csv=trades_csv,
        output_dir=tmp_path / "out",
        cluster_window_hours=24,
        action_bucket="copy_ready",
    )

    assert len(result["open_rows"]) == 0
    assert len(result["closed_rows"]) == 1
    assert len(result["conflict_rows"]) == 1
    assert result["closed_rows"][0]["reinforcement_count"] == 0
    assert result["closed_rows"][0]["signal_cluster_count"] == 2
