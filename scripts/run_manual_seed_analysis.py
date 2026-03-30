"""Run enrichment plus long-hold and delay review for the manual seed wallets."""

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
from research.manual_seed_analysis import (
    DEFAULT_MANUAL_SEED_CSV,
    DEFAULT_OUTPUT_DIR,
    print_manual_seed_analysis_summary,
    run_manual_seed_analysis,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-csv",
        default=DEFAULT_MANUAL_SEED_CSV,
        help="CSV containing resolved manual seed wallets and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the seed-only analysis exports are written.",
    )
    parser.add_argument(
        "--analysis-cutoff",
        default=None,
        help="Optional explicit UTC cutoff timestamp. Defaults to the latest raw seed trade timestamp.",
    )
    parser.add_argument(
        "--refresh-markets",
        action="store_true",
        help="Refresh Gamma market metadata for the seed subset before analysis. Off by default to avoid public 429 throttling.",
    )
    parser.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Refresh public token price history for the seed subset before analysis. Off by default to reuse cached prices.",
    )
    parser.add_argument(
        "--fetch-books",
        action="store_true",
        help="Fetch live order books during enrichment. Off by default to keep the seed pass fast and avoid large network fan-out.",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    init_db()
    with get_session() as session:
        results = await run_manual_seed_analysis(
            session,
            seed_csv=args.seed_csv,
            output_dir=args.output_dir,
            analysis_cutoff=args.analysis_cutoff,
            refresh_markets=args.refresh_markets,
            refresh_prices=args.refresh_prices,
            fetch_books=args.fetch_books,
        )
        print_manual_seed_analysis_summary(results)


def main() -> None:
    """Run the manual seed analysis workflow."""

    args = parse_args()
    setup_logging()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
