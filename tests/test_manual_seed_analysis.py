"""Tests for the manual seed wallet analysis helpers."""

from __future__ import annotations

import pandas as pd

from research.manual_seed_analysis import build_manual_seed_wallet_overview, load_manual_seed_frame


def test_load_manual_seed_frame_keeps_only_resolved_wallets(tmp_path) -> None:
    """Only resolved wallets with normalized ids should be loaded."""

    csv_path = tmp_path / "manual_seed_wallets.csv"
    csv_path.write_text(
        "\n".join(
            [
                "display_name,wallet_address,status,priority_group,notes,address_source",
                "Alpha,0xABC,resolved,primary,ok,manual",
                "Beta,0xdef,unresolved,primary,skip,manual",
                "Gamma,0xAbC,resolved,review,duplicate,manual",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_manual_seed_frame(csv_path)

    assert frame["wallet_address"].tolist() == ["0xabc"]
    assert frame["display_name"].tolist() == ["Alpha"]


def test_build_manual_seed_wallet_overview_merges_delay_and_open_outputs() -> None:
    """The wallet overview should join seed metadata with subset outputs by wallet id."""

    seed_frame = pd.DataFrame(
        [
            {
                "display_name": "Alpha",
                "wallet_address": "0xabc",
                "status": "resolved",
                "priority_group": "primary",
                "notes": "seed",
                "address_source": "manual",
            }
        ]
    )
    raw_coverage = pd.DataFrame(
        [
            {
                "display_name": "Alpha",
                "wallet_id": "0xabc",
                "status": "resolved",
                "priority_group": "primary",
                "notes": "seed",
                "address_source": "manual",
                "raw_trades": 42,
            }
        ]
    )
    delay_summary = pd.DataFrame(
        [{"wallet_address": "0xabc", "avg_copy_pnl_net_5m_delay_30s": 0.12, "tradability_label": "tradable"}]
    )
    event_summary = pd.DataFrame([{"wallet_address": "0xabc", "n_trades": 42}])
    expiry_summary = pd.DataFrame([{"wallet_id": "0xabc", "held_to_expiry_observed_slices": 3}])
    open_summary = pd.DataFrame([{"wallet_id": "0xabc", "unresolved_open_slices": 5}])

    merged = build_manual_seed_wallet_overview(
        seed_frame,
        raw_coverage,
        delay_summary,
        event_summary,
        expiry_summary,
        open_summary,
    )

    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["wallet_id"] == "0xabc"
    assert row["raw_trades"] == 42
    assert row["avg_copy_pnl_net_5m_delay_30s"] == 0.12
    assert row["held_to_expiry_observed_slices"] == 3
    assert row["unresolved_open_slices"] == 5
