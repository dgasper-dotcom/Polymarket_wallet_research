"""Refresh price history for tokens currently held by the house portfolio."""

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
from research.house_open_price_refresh import (
    DEFAULT_HOUSE_OPEN_PERFORMANCE_CSV,
    DEFAULT_HOUSE_OPEN_POSITIONS_CSV,
    DEFAULT_OUTPUT_DIR,
    refresh_house_open_price_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positions-csv",
        default=DEFAULT_HOUSE_OPEN_POSITIONS_CSV,
        help="Current house positions CSV.",
    )
    parser.add_argument(
        "--performance-csv",
        default=DEFAULT_HOUSE_OPEN_PERFORMANCE_CSV,
        help="Optional open-position performance CSV used to focus on missing marks.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for refresh reports.",
    )
    parser.add_argument(
        "--all-open-tokens",
        action="store_true",
        help="Refresh all current open tokens, not only those missing marks.",
    )
    parser.add_argument(
        "--fidelity",
        type=int,
        default=1,
        help="Price history fidelity passed to the public CLOB history endpoint.",
    )
    parser.add_argument(
        "--insecure-tls",
        action="store_true",
        help="Disable TLS certificate verification for the public read client as a local-network fallback.",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    init_db()
    with get_session() as session:
        result = await refresh_house_open_price_history(
            session,
            positions_csv=args.positions_csv,
            performance_csv=args.performance_csv,
            only_missing_marks=not args.all_open_tokens,
            fidelity=args.fidelity,
            output_dir=args.output_dir,
            insecure_tls=args.insecure_tls,
        )
        print(result["summary_path"])
        print(result["detail_path"])


def main() -> None:
    args = parse_args()
    setup_logging()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
