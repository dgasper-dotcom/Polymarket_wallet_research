"""CLI to build the enriched wallet trades dataset."""

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
from research.enrich_trades import enrich_wallet_trades


def _load_wallets_from_file(path: str | None) -> list[str]:
    """Read wallet ids from a plain-text file."""

    if not path:
        return []
    wallet_path = Path(path)
    if not wallet_path.exists():
        raise FileNotFoundError(f"Wallet file not found: {wallet_path}")
    wallets: list[str] = []
    for raw_line in wallet_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        wallets.append(line)
    return wallets


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wallet",
        action="append",
        dest="wallets",
        default=[],
        help="Optional wallet address filter. May be supplied multiple times.",
    )
    parser.add_argument(
        "--wallet-file",
        default=None,
        help="Optional newline-delimited wallet file to restrict enrichment to a subset.",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    init_db()
    wallets = [*args.wallets, *_load_wallets_from_file(args.wallet_file)]
    with get_session() as session:
        await enrich_wallet_trades(session, wallets=wallets or None)


def main() -> None:
    """Run enrichment."""

    args = parse_args()
    setup_logging()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
