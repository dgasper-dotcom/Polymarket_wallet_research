"""Tests for vertical-specialist wallet screening."""

from __future__ import annotations

import pandas as pd

from research.vertical_specialists import (
    build_vertical_top_wallets,
    classify_vertical,
    rank_vertical_specialists,
)


def test_classify_vertical_maps_core_domains() -> None:
    """Keyword rules should classify obvious verticals correctly."""

    assert classify_vertical("which-teams-will-make-the-nba-playoffs", "Will the Knicks make the playoffs?") == "sports"
    assert classify_vertical("presidential-election-winner-2028", "Will Trump win the 2028 election?") == "politics"
    assert classify_vertical("oscar-best-picture", "Will Dune win Best Picture?") == "culture"
    assert classify_vertical("bitcoin-above-150k", "Will Bitcoin hit 150k?") == "crypto"
    assert classify_vertical("will-the-fed-cut-rates", "Will the Fed cut rates?") == "macro"


def test_rank_vertical_specialists_prefers_concentrated_positive_wallets() -> None:
    """Specialist ranking should keep only concentrated positive wallets."""

    summary = pd.DataFrame(
        [
            {
                "wallet_address": "0x1111111111111111111111111111111111111111",
                "sample_name": "sports-keeper",
                "observed_recent_trades": 50,
                "dominant_vertical": "sports",
                "dominant_vertical_share": 0.80,
                "avg_copy_edge_net_30s": 0.08,
                "sell_share_observed": 0.10,
                "fast_exit_share_30s": 0.05,
                "open_copy_slice_share": 0.90,
                "rolling_3d_positive_share": 0.90,
                "avg_trades_per_active_day_observed": 6.0,
            },
            {
                "wallet_address": "0x2222222222222222222222222222222222222222",
                "sample_name": "sports-noisy",
                "observed_recent_trades": 200,
                "dominant_vertical": "sports",
                "dominant_vertical_share": 0.52,
                "avg_copy_edge_net_30s": 0.02,
                "sell_share_observed": 0.20,
                "fast_exit_share_30s": 0.10,
                "open_copy_slice_share": 0.60,
                "rolling_3d_positive_share": 0.70,
                "avg_trades_per_active_day_observed": 20.0,
            },
            {
                "wallet_address": "0x3333333333333333333333333333333333333333",
                "sample_name": "politics-keeper",
                "observed_recent_trades": 40,
                "dominant_vertical": "politics",
                "dominant_vertical_share": 0.75,
                "avg_copy_edge_net_30s": 0.05,
                "sell_share_observed": 0.15,
                "fast_exit_share_30s": 0.05,
                "open_copy_slice_share": 0.85,
                "rolling_3d_positive_share": 0.80,
                "avg_trades_per_active_day_observed": 5.0,
            },
        ]
    )

    candidates = rank_vertical_specialists(summary)
    top_wallets = build_vertical_top_wallets(candidates, top_n=1)

    assert set(candidates["wallet_address"]) == {
        "0x1111111111111111111111111111111111111111",
        "0x3333333333333333333333333333333333333333",
    }
    assert set(top_wallets["dominant_vertical"]) == {"sports", "politics"}
