"""Tests for out-of-sample validation behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import Settings
from research.oos_validation import (
    label_stability,
    run_oos_validation_from_frame,
    split_enriched_frame,
)


def _trade_row(
    wallet: str,
    trade_id: str,
    timestamp: datetime,
    market_id: str,
    *,
    copy_1m: float,
    copy_5m: float,
    copy_30m: float,
    fade_1m: float,
    fade_5m: float,
    fade_30m: float,
) -> dict[str, object]:
    """Build one enriched-trade-like row for frame-level OOS tests."""

    return {
        "trade_id": trade_id,
        "wallet_address": wallet,
        "market_id": market_id,
        "token_id": f"token-{trade_id}",
        "timestamp": timestamp,
        "side": "BUY",
        "price": 0.5,
        "ret_1m": copy_1m,
        "ret_5m": copy_5m,
        "ret_30m": copy_30m,
        "copy_pnl_1m": copy_1m,
        "copy_pnl_5m": copy_5m,
        "copy_pnl_30m": copy_30m,
        "fade_pnl_1m": fade_1m,
        "fade_pnl_5m": fade_5m,
        "fade_pnl_30m": fade_30m,
    }


def _wallet_rows(
    wallet: str,
    start: datetime,
    *,
    prefix: str,
    n_trades: int,
    copy_5m: float,
    fade_5m: float,
    copy_1m: float | None = None,
    copy_30m: float | None = None,
    fade_1m: float | None = None,
    fade_30m: float | None = None,
) -> list[dict[str, object]]:
    """Create a simple multi-market wallet path for testing."""

    rows: list[dict[str, object]] = []
    for index in range(n_trades):
        rows.append(
            _trade_row(
                wallet=wallet,
                trade_id=f"{prefix}-{index}",
                timestamp=start + timedelta(hours=index),
                market_id=f"{prefix}-market-{index % 3}",
                copy_1m=copy_1m if copy_1m is not None else copy_5m / 2,
                copy_5m=copy_5m,
                copy_30m=copy_30m if copy_30m is not None else copy_5m * 1.5,
                fade_1m=fade_1m if fade_1m is not None else fade_5m / 2,
                fade_5m=fade_5m,
                fade_30m=fade_30m if fade_30m is not None else fade_5m * 1.5,
            )
        )
    return rows


def test_split_enriched_frame_fraction_prevents_timestamp_leakage() -> None:
    """Fraction splits should keep boundary timestamps entirely in the test set."""

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        [
            _trade_row("0x1", "a", base, "m1", copy_1m=0.01, copy_5m=0.02, copy_30m=0.03, fade_1m=-0.01, fade_5m=-0.02, fade_30m=-0.03),
            _trade_row("0x1", "b", base + timedelta(days=1), "m1", copy_1m=0.01, copy_5m=0.02, copy_30m=0.03, fade_1m=-0.01, fade_5m=-0.02, fade_30m=-0.03),
            _trade_row("0x2", "c", base + timedelta(days=1), "m2", copy_1m=0.01, copy_5m=0.02, copy_30m=0.03, fade_1m=-0.01, fade_5m=-0.02, fade_30m=-0.03),
            _trade_row("0x2", "d", base + timedelta(days=2), "m2", copy_1m=0.01, copy_5m=0.02, copy_30m=0.03, fade_1m=-0.01, fade_5m=-0.02, fade_30m=-0.03),
        ]
    )

    split = split_enriched_frame(frame, train_fraction=0.5)
    assert split.train["timestamp"].max() < split.test["timestamp"].min()
    assert set(split.train["trade_id"]).isdisjoint(set(split.test["trade_id"]))


def test_split_enriched_frame_by_date_routes_rows_correctly() -> None:
    """Date-based splits should place older rows in train and newer rows in test."""

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        [
            _trade_row("0x1", "a", base, "m1", copy_1m=0.01, copy_5m=0.02, copy_30m=0.03, fade_1m=-0.01, fade_5m=-0.02, fade_30m=-0.03),
            _trade_row("0x1", "b", base + timedelta(days=10), "m2", copy_1m=0.01, copy_5m=0.02, copy_30m=0.03, fade_1m=-0.01, fade_5m=-0.02, fade_30m=-0.03),
        ]
    )

    split = split_enriched_frame(frame, split_date="2025-01-06")
    assert split.train["trade_id"].tolist() == ["a"]
    assert split.test["trade_id"].tolist() == ["b"]


def test_oos_wallet_selection_uses_train_data_only(tmp_path: Path) -> None:
    """Wallets that look strong only in test should not be selected from train."""

    frame = pd.DataFrame(
        _wallet_rows(
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            prefix="train-copy",
            n_trades=12,
            copy_5m=0.05,
            fade_5m=-0.03,
        )
        + _wallet_rows(
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            datetime(2025, 7, 1, tzinfo=timezone.utc),
            prefix="test-copy",
            n_trades=6,
            copy_5m=0.04,
            fade_5m=-0.02,
        )
        + _wallet_rows(
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            prefix="train-fade",
            n_trades=12,
            copy_5m=-0.02,
            fade_5m=0.04,
        )
        + _wallet_rows(
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            datetime(2025, 7, 1, tzinfo=timezone.utc),
            prefix="test-fade",
            n_trades=6,
            copy_5m=-0.01,
            fade_5m=0.03,
        )
        + _wallet_rows(
            "0xcccccccccccccccccccccccccccccccccccccccc",
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            prefix="train-ignore",
            n_trades=12,
            copy_5m=0.0,
            fade_5m=0.0,
        )
        + _wallet_rows(
            "0xcccccccccccccccccccccccccccccccccccccccc",
            datetime(2025, 7, 1, tzinfo=timezone.utc),
            prefix="test-late-copy",
            n_trades=6,
            copy_5m=0.08,
            fade_5m=-0.04,
        )
    )

    results = run_oos_validation_from_frame(
        frame,
        output_dir=tmp_path / "oos",
        split_date="2025-06-01",
        top_n=1,
        settings=Settings(),
    )

    selected_copy = results["selected_copy"]["wallet_address"].tolist()
    assert selected_copy == ["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
    assert "0xcccccccccccccccccccccccccccccccccccccccc" not in selected_copy


def test_label_stability_distinguishes_stable_and_unstable_modes() -> None:
    """Matching train/test modes should be stable, mismatches unstable."""

    settings = Settings()
    assert label_stability("copy", "copy", 12, 6, 2, settings) == "stable"
    assert label_stability("copy", "fade", 12, 6, 2, settings) == "unstable"


def test_oos_insufficient_data_marks_selected_wallet(tmp_path: Path) -> None:
    """Selected wallets with too little test data should be flagged as insufficient."""

    frame = pd.DataFrame(
        _wallet_rows(
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            prefix="train-copy",
            n_trades=12,
            copy_5m=0.05,
            fade_5m=-0.03,
        )
        + _wallet_rows(
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            prefix="train-fade",
            n_trades=12,
            copy_5m=-0.02,
            fade_5m=0.04,
        )
        + _wallet_rows(
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            datetime(2025, 7, 1, tzinfo=timezone.utc),
            prefix="test-copy",
            n_trades=1,
            copy_5m=0.04,
            fade_5m=-0.02,
        )
    )

    results = run_oos_validation_from_frame(
        frame,
        output_dir=tmp_path / "oos",
        split_date="2025-06-01",
        top_n=1,
        settings=Settings(),
    )

    selected = results["selected_test"].set_index("wallet_address")
    assert selected.loc["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "stability_label"] == "insufficient_data"
