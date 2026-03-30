"""CLI to scan public markets, discover wallets, and export strong/weak cohorts."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logging
from db.session import get_session, init_db
from research.wallet_universe import (
    DEFAULT_CLOSED_POSITIONS_PAGE_SIZE,
    DEFAULT_MARKET_PAGE_SIZE,
    DEFAULT_POSITIONS_PAGE_SIZE,
    DEFAULT_TRADE_PAGE_SIZE,
    print_wallet_universe_summary,
    run_wallet_universe_scan,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="exports/wallet_universe",
        help="Directory where master/strong/weak wallet exports are written.",
    )
    parser.add_argument(
        "--market-page-size",
        type=int,
        default=DEFAULT_MARKET_PAGE_SIZE,
        help="Gamma /markets page size.",
    )
    parser.add_argument(
        "--trade-page-size",
        type=int,
        default=DEFAULT_TRADE_PAGE_SIZE,
        help="Data API /trades page size for market scans.",
    )
    parser.add_argument(
        "--positions-page-size",
        type=int,
        default=DEFAULT_POSITIONS_PAGE_SIZE,
        help="Data API /positions page size used for realized-PnL estimates.",
    )
    parser.add_argument(
        "--closed-positions-page-size",
        type=int,
        default=DEFAULT_CLOSED_POSITIONS_PAGE_SIZE,
        help="Data API /closed-positions page size used for realized-PnL estimates.",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Optional smoke-test cap on how many markets to scan this run.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed progress rows and rescan markets from offset 0.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    """Run the wallet-universe workflow inside an async event loop."""

    init_db()
    with get_session() as session:
        results = await run_wallet_universe_scan(
            session,
            output_dir=args.output_dir,
            market_page_size=args.market_page_size,
            trade_page_size=args.trade_page_size,
            positions_page_size=args.positions_page_size,
            closed_positions_page_size=args.closed_positions_page_size,
            max_markets=args.max_markets,
            resume=not args.no_resume,
        )
        print_wallet_universe_summary(results)


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    setup_logging()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
