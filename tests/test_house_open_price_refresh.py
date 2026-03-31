from __future__ import annotations

from pathlib import Path

from research.house_open_price_refresh import build_house_open_refresh_specs


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_house_open_refresh_specs_without_performance_file_keeps_positions(tmp_path: Path) -> None:
    positions_csv = tmp_path / "current_house_positions.csv"
    _write(
        positions_csv,
        "\n".join(
            [
                "token_id,market_id,event_title,outcome,opened_at,last_signal_at,close_ts,status,opening_cluster_id,closing_cluster_id,signal_cluster_count,reinforcement_count,raw_trade_count,supporting_wallet_count,supporting_wallets,wallet_notional_attribution,total_signaled_notional_usdc,executed_notional_usdc,suppressed_notional_usdc,avg_open_signal_price",
                'tok1,m1,Event 1,Yes,2026-03-01T00:00:00+00:00,2026-03-01T00:00:00+00:00,,open,c1,,1,0,1,1,"[\\"0xabc\\"]","{\\"0xabc\\": 50.0}",50,50,0,0.5',
                'tok2,m2,Event 2,No,2026-03-02T00:00:00+00:00,2026-03-02T00:00:00+00:00,,open,c2,,1,0,1,1,"[\\"0xdef\\"]","{\\"0xdef\\": 75.0}",75,75,0,0.6',
            ]
        ),
    )

    specs = build_house_open_refresh_specs(
        positions_csv=positions_csv,
        performance_csv=tmp_path / "missing_performance.csv",
        only_missing_marks=True,
    )

    assert [spec.token_id for spec in specs] == ["tok1", "tok2"]


def test_build_house_open_refresh_specs_only_missing_marks_filters_existing_marks(tmp_path: Path) -> None:
    positions_csv = tmp_path / "current_house_positions.csv"
    performance_csv = tmp_path / "house_open_position_performance.csv"
    _write(
        positions_csv,
        "\n".join(
            [
                "token_id,market_id,event_title,outcome,opened_at,last_signal_at,close_ts,status,opening_cluster_id,closing_cluster_id,signal_cluster_count,reinforcement_count,raw_trade_count,supporting_wallet_count,supporting_wallets,wallet_notional_attribution,total_signaled_notional_usdc,executed_notional_usdc,suppressed_notional_usdc,avg_open_signal_price",
                'tok1,m1,Event 1,Yes,2026-03-01T00:00:00+00:00,2026-03-01T00:00:00+00:00,,open,c1,,1,0,1,1,"[\\"0xabc\\"]","{\\"0xabc\\": 50.0}",50,50,0,0.5',
                'tok2,m2,Event 2,No,2026-03-02T00:00:00+00:00,2026-03-02T00:00:00+00:00,,open,c2,,1,0,1,1,"[\\"0xdef\\"]","{\\"0xdef\\": 75.0}",75,75,0,0.6',
            ]
        ),
    )
    _write(
        performance_csv,
        "\n".join(
            [
                "token_id,entry_notional_usdc,mark_price,mark_price_age_seconds",
                "tok1,50,,100",
                "tok2,75,0.65,200",
            ]
        ),
    )

    specs = build_house_open_refresh_specs(
        positions_csv=positions_csv,
        performance_csv=performance_csv,
        only_missing_marks=True,
    )

    assert [spec.token_id for spec in specs] == ["tok1"]
