"""Scan 1000 current Polymarket markets and export active high-trade wallets."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logging
from config.settings import get_settings
from research.current_market_wallet_scan import (
    MARKET_PAGE_SIZE_DEFAULT,
    TRADE_PAGE_SIZE_DEFAULT,
    print_current_market_wallet_scan_summary,
    run_current_market_wallet_scan,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="exports/current_market_wallet_scan")
    parser.add_argument("--max-markets", type=int, default=1000)
    parser.add_argument("--market-page-size", type=int, default=MARKET_PAGE_SIZE_DEFAULT)
    parser.add_argument("--trade-page-size", type=int, default=TRADE_PAGE_SIZE_DEFAULT)
    parser.add_argument(
        "--recent-window-start",
        default="2026-03-12T00:00:00+00:00",
        help="Inclusive UTC lower bound for the recent-activity filter.",
    )
    parser.add_argument(
        "--recent-window-end",
        default="2026-03-26T23:59:59+00:00",
        help="Inclusive UTC upper bound for the recent-activity filter.",
    )
    parser.add_argument(
        "--taker-only",
        action="store_true",
        help="Use takerOnly=true on /trades instead of the broader default takerOnly=false.",
    )
    return parser.parse_args()


def _parse_utc(value: str) -> datetime:
    """Parse an ISO datetime string into UTC."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


async def _run(args: argparse.Namespace) -> dict:
    """Run the market scan asynchronously."""

    return await run_current_market_wallet_scan(
        output_dir=args.output_dir,
        max_markets=args.max_markets,
        market_page_size=args.market_page_size,
        trade_page_size=args.trade_page_size,
        taker_only=args.taker_only,
        recent_window_start=_parse_utc(args.recent_window_start),
        recent_window_end=_parse_utc(args.recent_window_end),
        settings=get_settings(),
    )


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    setup_logging()
    results = asyncio.run(_run(args))
    print_current_market_wallet_scan_summary(results)

    eligible = results["eligible_wallets"]
    if eligible.empty:
        print("No eligible wallets found.")
        return

    print("Top eligible wallets:")
    for row in eligible.head(25).itertuples(index=False):
        label = row.sample_name or row.sample_pseudonym or ""
        print(
            f"{row.wallet_address} | trades={row.total_trades} | markets={row.distinct_markets} | "
            f"most_recent={row.most_recent_trade_ts} | name={label}"
        )


if __name__ == "__main__":
    main()

