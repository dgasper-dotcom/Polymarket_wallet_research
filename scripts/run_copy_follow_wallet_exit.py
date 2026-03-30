"""CLI for delayed copy analysis that exits when the tracked wallet exits."""

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
from db.session import get_session, init_db
from ingestion.prices import backfill_price_history_for_token_bounds
from research.copy_follow_wallet_exit import (
    ACTIVE_LAST_DAYS_DEFAULT,
    build_copy_exit_pairs,
    collect_price_backfill_targets,
    run_copy_follow_wallet_exit_analysis,
)
from research.recent_wallet_trade_capture import load_recent_wallet_trades


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wallets-file",
        default="exports/recent_wallet_trade_capture_top1000_cohort/tracked_wallets.txt",
        help="Newline-delimited wallet cohort file.",
    )
    parser.add_argument(
        "--start-date",
        default="2026-03-12T00:00:00Z",
        help="Inclusive UTC analysis window start.",
    )
    parser.add_argument(
        "--end-date",
        default="2026-03-26T23:59:59Z",
        help="Inclusive UTC analysis window end.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/copy_follow_wallet_exit_report_recent",
        help="Directory where delayed wallet-exit copy reports are written.",
    )
    parser.add_argument(
        "--delays",
        default="5,10,15,30",
        help="Comma-separated entry/exit delays in seconds.",
    )
    parser.add_argument(
        "--active-last-days",
        type=int,
        default=ACTIVE_LAST_DAYS_DEFAULT,
        help="Wallet is flagged active if it traded within the last N days of the window.",
    )
    parser.add_argument(
        "--skip-price-backfill",
        action="store_true",
        help="Use only locally cached public price history and do not fetch missing recent prices.",
    )
    return parser.parse_args()


def _load_wallets(path: str | Path) -> list[str]:
    """Load a newline-delimited wallet file."""

    values = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(line.lower())
    return sorted(set(values))


def _parse_delays(value: str) -> tuple[int, ...]:
    """Parse comma-delimited delay values."""

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


def main() -> None:
    """Run delayed copy-follow analysis on recent tracked-wallet trades."""

    args = parse_args()
    settings = get_settings()
    setup_logging()
    init_db()

    wallets = _load_wallets(args.wallets_file)
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
        token_bounds = collect_price_backfill_targets(pairs, open_positions.iloc[0:0], delays=delays)

        print(f"wallets_requested: {len(wallets)}", flush=True)
        print(f"recent_trades_loaded: {len(recent_trades)}", flush=True)
        print(
            f"wallets_with_recent_trades: "
            f"{recent_trades['wallet_address'].nunique() if not recent_trades.empty else 0}",
            flush=True,
        )
        print(f"paired_copy_slices: {len(pairs)}", flush=True)
        print(f"open_copy_slices: {len(open_positions)}", flush=True)
        print(f"price_backfill_token_ranges: {len(token_bounds)}", flush=True)

        if token_bounds and not args.skip_price_backfill:
            asyncio.run(
                backfill_price_history_for_token_bounds(
                    session,
                    token_bounds=token_bounds,
                )
            )

        results = run_copy_follow_wallet_exit_analysis(
            session,
            wallets=wallets,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            delays=delays,
            active_last_days=args.active_last_days,
            analysis_asof=end_dt,
            settings=settings,
        )

    overview = results["overview"].iloc[0].to_dict() if not results["overview"].empty else {}
    print("Copy Follow Wallet Exit Report")
    print(f"wallets_requested: {len(wallets)}")
    print(f"wallets_in_report: {overview.get('wallets_in_report')}")
    print(f"wallets_active_in_last_window_days: {overview.get('wallets_active_in_last_window_days')}")
    print(f"raw_recent_trades_in_window: {overview.get('raw_recent_trades_in_window')}")
    print(f"copy_slices_total: {overview.get('copy_slices_total')}")
    for delay in delays:
        label = f"{int(delay)}s"
        print(f"valid_copy_slices_{label}: {overview.get(f'valid_copy_slices_{label}')}")
        print(
            f"wallets_positive_net_total_usdc_{label}: "
            f"{overview.get(f'wallets_positive_net_total_usdc_{label}')}"
        )
    print(f"wallet_report: {results['wallet_path']}")
    print(f"summary_report: {results['summary_path']}")
    print(f"trade_diagnostics: {results['diagnostics_path']}")
    print(f"assumptions: {results['assumptions_path']}")


if __name__ == "__main__":
    main()
