"""Tests for multi-split OOS robustness validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import Settings
from research.multi_oos_validation import (
    classify_wallet_robustness,
    generate_multi_split_specs,
    run_multi_oos_validation_from_frame,
)


def _trade_row(
    wallet: str,
    trade_id: str,
    timestamp: datetime,
    market_id: str,
    *,
    copy_5m: float,
    fade_5m: float,
) -> dict[str, object]:
    """Build one enriched-trade-like row for multi-split testing."""

    return {
        "trade_id": trade_id,
        "wallet_address": wallet,
        "market_id": market_id,
        "token_id": f"token-{trade_id}",
        "timestamp": timestamp,
        "side": "BUY",
        "price": 0.5,
        "ret_1m": copy_5m / 2,
        "ret_5m": copy_5m,
        "ret_30m": copy_5m * 1.5,
        "copy_pnl_1m": copy_5m / 2,
        "copy_pnl_5m": copy_5m,
        "copy_pnl_30m": copy_5m * 1.5,
        "fade_pnl_1m": fade_5m / 2,
        "fade_pnl_5m": fade_5m,
        "fade_pnl_30m": fade_5m * 1.5,
    }


def _wallet_rows(
    wallet: str,
    *,
    start: datetime,
    prefix: str,
    n_trades: int,
    copy_5m_by_index,
    fade_5m_by_index,
) -> list[dict[str, object]]:
    """Create a simple wallet path across three markets."""

    rows: list[dict[str, object]] = []
    for index in range(n_trades):
        rows.append(
            _trade_row(
                wallet=wallet,
                trade_id=f"{prefix}-{index}",
                timestamp=start + timedelta(hours=index),
                market_id=f"{prefix}-market-{index % 3}",
                copy_5m=copy_5m_by_index(index),
                fade_5m=fade_5m_by_index(index),
            )
        )
    return rows


def _robust_frame() -> pd.DataFrame:
    """Build a frame with one stable copy wallet and one stable fade wallet."""

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    rows += _wallet_rows(
        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        start=start,
        prefix="copy",
        n_trades=30,
        copy_5m_by_index=lambda _: 0.05,
        fade_5m_by_index=lambda _: -0.03,
    )
    rows += _wallet_rows(
        "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        start=start,
        prefix="fade",
        n_trades=30,
        copy_5m_by_index=lambda _: -0.02,
        fade_5m_by_index=lambda _: 0.04,
    )
    return pd.DataFrame(rows)


def test_generate_multi_split_specs_respects_requested_count() -> None:
    """The split generator should produce the requested number of unique specs."""

    specs = generate_multi_split_specs(
        _robust_frame(),
        n_splits=5,
        ratio_splits=(0.6, 0.7),
        include_random=True,
        random_splits=1,
        random_seed=7,
    )
    assert len(specs) == 5
    assert any(spec.split_kind == "random_index" for spec in specs)


def test_multi_oos_validation_aggregates_wallet_and_portfolio_outputs(tmp_path: Path) -> None:
    """Stable wallets should be selected frequently and aggregate outputs should be written."""

    results = run_multi_oos_validation_from_frame(
        _robust_frame(),
        output_dir=tmp_path / "multi_oos",
        n_splits=4,
        ratio_splits=(0.6, 0.7),
        include_random=False,
        top_n=1,
        settings=Settings(),
    )

    robustness = results["wallet_robustness"].set_index("wallet_address")
    assert len(results["split_run_summary"]) == 4
    assert robustness.loc["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "selection_frequency"] == 1.0
    assert robustness.loc["0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "selection_frequency"] == 1.0
    assert robustness.loc["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "robustness_label"] == "robust"
    assert robustness.loc["0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "robustness_label"] == "robust"

    portfolio = results["portfolio_robustness"]
    copy_5m = portfolio[(portfolio["strategy_mode"] == "copy") & (portfolio["horizon"] == "5m")]
    assert int(copy_5m.iloc[0]["n_splits"]) == 4
    assert (tmp_path / "multi_oos" / "wallet_robustness_summary.csv").exists()
    assert (tmp_path / "multi_oos" / "portfolio_robustness_summary.csv").exists()


def test_classify_wallet_robustness_branches() -> None:
    """Robustness labels should follow the configured thresholds."""

    settings = Settings()
    assert classify_wallet_robustness(
        observed_splits=5,
        selected_splits=4,
        selection_frequency=0.8,
        positive_test_frequency=0.75,
        mode_consistency=1.0,
        settings=settings,
    ) == "robust"
    assert classify_wallet_robustness(
        observed_splits=5,
        selected_splits=4,
        selection_frequency=0.8,
        positive_test_frequency=0.4,
        mode_consistency=0.4,
        settings=settings,
    ) == "inconsistent"
    assert classify_wallet_robustness(
        observed_splits=5,
        selected_splits=4,
        selection_frequency=0.4,
        positive_test_frequency=0.5,
        mode_consistency=0.8,
        settings=settings,
    ) == "fragile"
    assert classify_wallet_robustness(
        observed_splits=2,
        selected_splits=1,
        selection_frequency=0.5,
        positive_test_frequency=1.0,
        mode_consistency=1.0,
        settings=settings,
    ) == "insufficient_data"
