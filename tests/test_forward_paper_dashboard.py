from __future__ import annotations

from pathlib import Path

from research.forward_paper_dashboard import build_forward_paper_dashboard


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_forward_paper_dashboard(tmp_path: Path) -> None:
    base_dir = tmp_path / "paper"
    perf_dir = base_dir / "performance"
    consolidated_dir = base_dir / "consolidated"
    watchlist_csv = tmp_path / "watchlist.csv"

    _write(
        watchlist_csv,
        "\n".join(
            [
                "display_name,wallet_id,action_bucket,action_rationale,pma_copy_combined_net_30s,combined_realized_plus_mtm_pnl_usdc_est,notes",
                "WalletA,0xabc,copy_ready,copyable,10,12,good",
                "WalletB,0xdef,monitor,mixed,5,3,watch",
            ]
        ),
    )
    _write(
        base_dir / "current_house_positions.csv",
        "\n".join(
            [
                "token_id,market_id,event_title,outcome,opened_at,last_signal_at,close_ts,status,opening_cluster_id,closing_cluster_id,signal_cluster_count,reinforcement_count,raw_trade_count,supporting_wallet_count,supporting_wallets,total_signaled_notional_usdc,avg_open_signal_price",
                'tok1,m1,Event 1,Yes,2026-03-29T00:00:00+00:00,2026-03-29T00:00:00+00:00,,open,c1,,1,0,2,1,"[\\"0xabc\\"]",100,0.5',
            ]
        ),
    )
    _write(base_dir / "closed_house_positions.csv", "token_id\n")
    _write(base_dir / "skipped_market_conflicts.csv", "cluster_id\n")
    _write(base_dir / "skipped_position_cap_records.csv", "cluster_id,reason\n")
    _write(
        perf_dir / "house_portfolio_performance_summary.md",
        "\n".join(
            [
                "# House Portfolio Performance Summary",
                "",
                "- Analysis cutoff: `2026-03-30T00:00:00+00:00`",
                "- Open house positions: `1`",
                "- Closed house positions: `1`",
                "- Closed-position realized net PnL: `10.00 USDC`",
                "- Open-position MTM net PnL: `2.00 USDC`",
                "- Combined house portfolio net PnL: `12.00 USDC`",
            ]
        ),
    )
    _write(
        perf_dir / "house_open_position_performance.csv",
        "\n".join(
            [
                "house_position_id,token_id,market_id,event_title,outcome,opened_at,last_signal_at,closed_at,status,opening_cluster_id,closing_cluster_id,signal_cluster_count,reinforcement_count,raw_trade_count,supporting_wallet_count,supporting_wallets,entry_contracts,entry_notional_usdc,weighted_avg_entry_price,entry_cost_total_usdc,analysis_cutoff,mark_price,mark_price_source,mark_price_age_seconds,mtm_pnl_raw_usdc,mtm_pnl_net_usdc,holding_days_open",
                'hp1,tok1,m1,Event 1,Yes,2026-03-29T00:00:00+00:00,2026-03-29T00:00:00+00:00,,open,c1,,1,0,2,1,"[\\"0xabc\\"]",100,100,1.0,1,2026-03-30T00:00:00+00:00,0.6,public,60,10,2,1',
            ]
        ),
    )
    _write(
        perf_dir / "house_closed_position_performance.csv",
        "\n".join(
            [
                "house_position_id,token_id,market_id,event_title,outcome,opened_at,last_signal_at,closed_at,status,opening_cluster_id,closing_cluster_id,signal_cluster_count,reinforcement_count,raw_trade_count,supporting_wallet_count,supporting_wallets,entry_contracts,entry_notional_usdc,weighted_avg_entry_price,entry_cost_total_usdc,exit_price,exit_cost_total_usdc,realized_pnl_raw_usdc,realized_pnl_net_usdc",
                'hp2,tok2,m2,Event 2,No,2026-03-28T00:00:00+00:00,2026-03-28T00:00:00+00:00,2026-03-29T00:00:00+00:00,closed,c2,c3,2,0,1,1,"[\\"0xdef\\"]",50,25,0.5,1,0.7,1,12,10',
            ]
        ),
    )
    _write(perf_dir / "house_skipped_position_records.csv", "cluster_id,reason\n")
    _write(
        perf_dir / "house_portfolio_daily_equity_curve.csv",
        "\n".join(
            [
                "date,combined_equity_net_usdc",
                "2026-03-29,10",
                "2026-03-30,12",
            ]
        ),
    )
    _write(
        consolidated_dir / "house_signal_tape.csv",
        "\n".join(
            [
                "cluster_id,action,side,token_id,market_id,event_title,outcome,first_ts,last_ts,trade_count,unique_wallet_count,supporting_wallets,total_notional_usdc,avg_signal_price",
                'c1,open_long,BUY,tok1,m1,Event 1,Yes,2026-03-29T00:00:00+00:00,2026-03-29T00:00:00+00:00,1,1,"[\\"0xabc\\"]",100,0.5',
                'c2,close_long,SELL,tok2,m2,Event 2,No,2026-03-29T00:00:00+00:00,2026-03-29T00:00:00+00:00,1,1,"[\\"0xdef\\"]",25,0.7',
            ]
        ),
    )

    result = build_forward_paper_dashboard(
        base_dir=base_dir,
        watchlist_csv=watchlist_csv,
        output_dir=tmp_path / "dashboard",
    )

    dashboard_text = Path(result["dashboard_path"]).read_text(encoding="utf-8")
    assert "Forward Paper Tracking Dashboard" in dashboard_text
    assert "WalletA" in dashboard_text
    assert "Combined house portfolio net PnL" in dashboard_text
    assert Path(result["snapshot_path"]).exists()
    assert Path(result["top_open_path"]).exists()
    assert Path(result["history_root"]).exists()
