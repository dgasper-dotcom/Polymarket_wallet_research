"""Tests for unified house-book OOS summary generation."""

from __future__ import annotations

from pathlib import Path

from research.house_book_oos import run_house_book_oos


def test_house_book_oos_builds_train_test_outputs(tmp_path: Path) -> None:
    closed_csv = tmp_path / "closed.csv"
    closed_csv.write_text(
        "\n".join(
            [
                "house_position_id,opened_at,closed_at,entry_notional_usdc,exit_cost_total_usdc,realized_pnl_net_usdc",
                "p1,2026-01-01T00:00:00+00:00,2026-01-10T00:00:00+00:00,100,120,12",
                "p2,2026-02-01T00:00:00+00:00,2026-02-10T00:00:00+00:00,150,160,-5",
            ]
        ),
        encoding="utf-8",
    )
    open_csv = tmp_path / "open.csv"
    open_csv.write_text(
        "\n".join(
            [
                "house_position_id,opened_at,entry_notional_usdc,mtm_pnl_net_usdc,mark_price,analysis_cutoff",
                "p3,2026-03-01T00:00:00+00:00,200,8,0.55,2026-03-30T00:00:00+00:00",
                "p4,2026-03-15T00:00:00+00:00,50,0,,2026-03-30T00:00:00+00:00",
            ]
        ),
        encoding="utf-8",
    )

    results = run_house_book_oos(closed_csv=closed_csv, open_csv=open_csv, output_dir=tmp_path / "out")

    summary_rows = results["summary_rows"]
    assert summary_rows[0]["split"] == "full_history"
    assert any(row["split"] == "train_80_20" for row in summary_rows)
    assert any(row["split"] == "test_80_20" for row in summary_rows)
    assert (tmp_path / "out" / "house_book_oos_summary.csv").exists()
    assert (tmp_path / "out" / "house_book_oos_summary.md").exists()
