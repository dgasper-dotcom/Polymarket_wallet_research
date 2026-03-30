"""Run a past-week copy-follow backtest for the resolved manual seed wallets."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.modules.setdefault("pyarrow", None)

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logging
from config.settings import get_settings
from db.session import get_session
from ingestion.markets import backfill_markets_for_references
from ingestion.prices import backfill_price_history_for_token_bounds
from research.manual_seed_copy_backtest import (
    DEFAULT_DELAYS,
    DEFAULT_END_DATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_START_DATE,
    _collect_token_bounds,
    _load_wallet_file,
    run_manual_seed_copy_backtest,
)
from research.recent_wallet_trade_capture import load_recent_wallet_trades
from research.copy_follow_wallet_exit import build_copy_exit_pairs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wallets-file",
        default="data/manual_seed_wallets_resolved.txt",
        help="Newline-delimited wallet list to backtest.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Inclusive UTC analysis window start.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="Inclusive UTC analysis window end.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV reports are written.",
    )
    parser.add_argument(
        "--delays",
        default="5,15,30,60",
        help="Comma-separated delays in seconds.",
    )
    return parser.parse_args()


def _parse_delays(value: str) -> tuple[int, ...]:
    """Parse one comma-delimited delay set."""

    delays = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not delays:
        raise ValueError("At least one delay is required")
    return delays


def _to_utc(value: str, *, end_of_day: bool) -> datetime:
    """Parse an ISO timestamp/date into UTC."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    if end_of_day and len(value) <= 10:
        return parsed.replace(hour=23, minute=59, second=59)
    return parsed


async def _async_main(args: argparse.Namespace) -> None:
    settings = get_settings()
    wallets = _load_wallet_file(args.wallets_file)
    delays = _parse_delays(args.delays)
    start_dt = _to_utc(args.start_date, end_of_day=False)
    end_dt = _to_utc(args.end_date, end_of_day=True)

    with get_session() as session:
        recent_trades = load_recent_wallet_trades(
            session,
            wallets=wallets,
            recent_window_start=start_dt,
            recent_window_end=end_dt,
        )
        pairs, open_positions = build_copy_exit_pairs(recent_trades)
        condition_ids = [
            str(value)
            for value in recent_trades.get("market_id", [])
            if isinstance(value, str) and value.startswith("0x")
        ]
        token_ids = [
            str(value)
            for value in recent_trades.get("token_id", [])
            if isinstance(value, str) and value.strip()
        ]
        if condition_ids or token_ids:
            await backfill_markets_for_references(
                session,
                condition_ids=condition_ids,
                token_ids=token_ids,
            )
        token_bounds = _collect_token_bounds(
            pairs,
            open_positions,
            delays=delays,
            analysis_asof=pd.to_datetime(end_dt, utc=True),
        )
        if token_bounds:
            await backfill_price_history_for_token_bounds(
                session,
                token_bounds=token_bounds,
            )

        results = run_manual_seed_copy_backtest(
            session,
            wallets=wallets,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            delays=delays,
            analysis_asof=end_dt,
            settings=settings,
        )

    overview = results["overview"].iloc[0].to_dict() if not results["overview"].empty else {}
    print("Manual Seed Copy Backtest")
    print(f"wallets_requested: {len(wallets)}")
    print(f"raw_recent_trades_in_window: {overview.get('raw_recent_trades_in_window')}")
    print(f"copy_slices_total: {overview.get('copy_slices_total')}")
    for delay in delays:
        label = f"{int(delay)}s"
        print(f"realized_copy_slices_{label}: {overview.get(f'realized_copy_slices_{label}')}")
        print(f"open_copy_slices_{label}: {overview.get(f'open_copy_slices_{label}')}")
        print(f"entry_unfilled_slices_{label}: {overview.get(f'entry_unfilled_slices_{label}')}")
        print(f"combined_net_total_usdc_{label}: {overview.get(f'combined_net_total_usdc_{label}')}")
    print(f"wallet_report: {results['wallet_path']}")
    print(f"summary_report: {results['summary_path']}")
    print(f"trade_diagnostics: {results['diagnostics_path']}")
    print(f"assumptions: {results['assumptions_path']}")


def main() -> None:
    """Run the manual seed copy backtest."""

    args = parse_args()
    setup_logging()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
