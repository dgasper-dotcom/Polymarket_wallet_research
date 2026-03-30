"""CLI to backfill wallets, raw trades, markets, and price history."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clients.profile_client import ProfileClient
from config.logging_config import setup_logging
from db.session import get_session, init_db
from ingestion.backfill import (
    backfill_wallet_trades,
    build_backfill_preview,
    fetch_wallet_trade_payloads,
)
from ingestion.markets import backfill_markets
from ingestion.prices import backfill_price_history
from ingestion.wallets import load_wallet_inputs, store_wallets
from reports.summary_exports import (
    export_backfill_dry_run_preview,
    export_endpoint_audit,
    export_research_summary,
    print_backfill_dry_run_preview,
    print_research_summary,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wallets",
        nargs="+",
        required=True,
        help="One or more wallet addresses, or a text file with one wallet per line.",
    )
    parser.add_argument(
        "--price-fidelity",
        type=int,
        default=1,
        help="Price history fidelity passed to the public CLOB endpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate wallets and inspect planned fetch targets without writing to the database.",
    )
    return parser.parse_args()


async def run_backfill_workflow(
    args: argparse.Namespace,
    profile_client: ProfileClient | None = None,
    init_db_fn=init_db,
    session_factory=get_session,
    exports_dir: str | Path = "exports",
) -> dict:
    """Execute backfill or dry-run mode."""

    wallet_load = load_wallet_inputs(args.wallets)
    LOGGER.info(
        "Loaded %s valid wallets from %s after ignoring %s blank/comment lines",
        wallet_load.valid_count,
        wallet_load.source_label,
        wallet_load.ignored_count,
    )
    if wallet_load.invalid_entries:
        raise ValueError(f"Invalid wallet entries: {', '.join(wallet_load.invalid_entries)}")

    export_endpoint_audit(output_dir=exports_dir)

    if args.dry_run:
        trade_payloads = await fetch_wallet_trade_payloads(wallets=wallet_load.wallets, client=profile_client)
        preview = build_backfill_preview(trade_payloads)
        export_backfill_dry_run_preview(preview, output_dir=exports_dir)
        print_backfill_dry_run_preview(preview)
        return preview

    init_db_fn()
    with session_factory() as session:
        stored_wallets = store_wallets(session, wallet_load.wallets, source="cli")
        trade_summary = await backfill_wallet_trades(session, stored_wallets, client=profile_client)
        market_summary = await backfill_markets(session)
        price_summary = await backfill_price_history(session, fidelity=args.price_fidelity)
        summary = export_research_summary(session, output_dir=exports_dir)
        print_research_summary(summary, title="Backfill Summary")
        LOGGER.info(
            "Backfill completed for %s wallets, %s market rows, %s token price history result rows",
            len(stored_wallets),
            market_summary.get("markets", 0),
            sum(price_summary.values()),
        )
        return {
            "trade_summary": trade_summary,
            "market_summary": market_summary,
            "price_summary": price_summary,
            "exports": summary,
        }


def main() -> None:
    """Run the backfill workflow."""

    args = parse_args()
    setup_logging()
    asyncio.run(run_backfill_workflow(args))


if __name__ == "__main__":
    main()
