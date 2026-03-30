"""Capture recent public trades for a wallet cohort using only public APIs."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logging
from db.session import get_session, init_db
from ingestion.wallets import load_wallet_inputs
from research.recent_wallet_trade_capture import (
    print_recent_wallet_trade_capture_summary,
    run_recent_wallet_trade_capture,
)


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
        "--recent-window-start",
        default="2026-03-12T00:00:00+00:00",
        help="Inclusive UTC lower bound for the recent-trades export.",
    )
    parser.add_argument(
        "--recent-window-end",
        default="2026-03-26T23:59:59+00:00",
        help="Inclusive UTC upper bound for the recent-trades export.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/recent_wallet_trade_capture",
        help="Directory where recent trade exports are written.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict:
    """Run the capture workflow."""

    wallet_load = load_wallet_inputs(args.wallets)
    if wallet_load.invalid_entries:
        raise ValueError(f"Invalid wallet entries: {', '.join(wallet_load.invalid_entries)}")

    init_db()
    with get_session() as session:
        return await run_recent_wallet_trade_capture(
            session,
            wallets=wallet_load.wallets,
            recent_window_start=args.recent_window_start,
            recent_window_end=args.recent_window_end,
            output_dir=args.output_dir,
        )


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    setup_logging()
    logging.getLogger("clients.profile_client").setLevel(logging.WARNING)
    logging.getLogger("ingestion.backfill").setLevel(logging.WARNING)
    results = asyncio.run(_run(args))
    print_recent_wallet_trade_capture_summary(results)

    recent_summary = results["recent_summary"]
    if recent_summary.empty:
        print("No recent trades found.")
        return

    print("Top wallets by recent trade count:")
    for row in recent_summary.head(25).itertuples(index=False):
        print(
            f"{row.wallet_address} | recent_trades={row.recent_trades} | "
            f"recent_markets={row.recent_distinct_markets} | most_recent={row.most_recent_trade_ts}"
        )
    print(f"recent_trades_csv: {results['paths']['recent_trades']}")
    print(f"recent_summary_csv: {results['paths']['recent_summary']}")
    print(f"tracked_wallets_txt: {results['paths']['tracked_wallets']}")


if __name__ == "__main__":
    main()
