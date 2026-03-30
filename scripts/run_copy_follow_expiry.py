"""CLI for delayed copy analysis held to market expiry using public data only."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import func, select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logging
from config.settings import get_settings
from db.models import Market, Token
from db.session import get_session, init_db
from ingestion.markets import backfill_markets
from research.copy_follow_expiry import run_copy_follow_expiry_analysis


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="Inclusive UTC start date, e.g. 2022-12-12.")
    parser.add_argument("--end-date", required=True, help="Inclusive UTC end date, e.g. 2024-11-14.")
    parser.add_argument(
        "--output-dir",
        default="exports/copy_follow_expiry_report",
        help="Directory where expiry-held copy reports are written.",
    )
    parser.add_argument(
        "--active-last-days",
        type=int,
        default=30,
        help="Also export the subset active in the last N days of the analysis window.",
    )
    parser.add_argument(
        "--skip-market-backfill",
        action="store_true",
        help="Skip public market metadata backfill even if the local markets/tokens tables are empty.",
    )
    return parser.parse_args()


def _needs_market_backfill() -> bool:
    """Return whether persisted market metadata is currently missing."""

    with get_session() as session:
        market_count = session.scalar(select(func.count()).select_from(Market)) or 0
        token_count = session.scalar(select(func.count()).select_from(Token)) or 0
    return market_count == 0 or token_count == 0


def main() -> None:
    """Run delayed copy-to-expiry analysis on existing enriched trades."""

    args = parse_args()
    settings = get_settings()
    setup_logging()
    init_db()

    if not args.skip_market_backfill and _needs_market_backfill():
        with get_session() as session:
            asyncio.run(backfill_markets(session))

    with get_session() as session:
        results = run_copy_follow_expiry_analysis(
            session,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            active_last_days=args.active_last_days,
            settings=settings,
        )

    overview = results["overview"].iloc[0].to_dict() if not results["overview"].empty else {}
    print("Expiry-held copy report")
    for key in (
        "analysis_window_start",
        "analysis_window_end",
        "wallets_in_report",
        "active_wallets_in_last_30d_of_window",
        "total_trades_in_window",
        "valid_trades_15s",
        "valid_trades_30s",
        "wallets_positive_gross_15s",
        "wallets_positive_gross_30s",
        "wallets_positive_net_15s",
        "wallets_positive_net_30s",
        "wallets_positive_net_both_15s_30s",
    ):
        print(f"{key}: {overview.get(key)}")
    print(f"wallet_report: {results['wallet_path']}")
    print(f"active_wallet_report: {results['active_wallets_path']}")
    print(f"summary_report: {results['summary_path']}")
    print(f"trade_diagnostics: {results['diagnostics_path']}")
    print(f"assumptions: {results['assumptions_path']}")


if __name__ == "__main__":
    main()

