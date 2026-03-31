"""Run the full forward paper-tracking refresh cycle."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.session import get_session
from research.forward_paper_dashboard import DEFAULT_OUTPUT_DIR as DEFAULT_DASHBOARD_DIR
from research.forward_paper_dashboard import build_forward_paper_dashboard
from research.house_open_price_refresh import refresh_house_open_price_history
from research.paper_tracking_model import run_paper_tracking_model
from research.paper_tracking_performance import run_paper_tracking_performance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wallet-csv",
        default=None,
        help="Optional explicit wallet list CSV. When omitted, the default action-bucket source is used.",
    )
    parser.add_argument(
        "--mapped-trades-csv",
        default=None,
        help="Optional mapped-trades CSV override.",
    )
    parser.add_argument(
        "--paper-output-dir",
        default="exports/manual_seed_paper_tracking",
        help="Base output directory for the unified paper-tracking model.",
    )
    parser.add_argument(
        "--dashboard-output-dir",
        default=DEFAULT_DASHBOARD_DIR,
        help="Directory for forward-paper dashboard outputs.",
    )
    parser.add_argument(
        "--watchlist-csv",
        default=None,
        help="Optional watchlist CSV used for the dashboard snapshot.",
    )
    parser.add_argument(
        "--action-bucket",
        default="copy_ready",
        help="Wallet bucket to track.",
    )
    parser.add_argument(
        "--cluster-window-hours",
        type=int,
        default=24,
        help="Hours to cluster near-simultaneous same-token signals.",
    )
    parser.add_argument(
        "--max-position-notional-usdc",
        type=float,
        default=None,
        help="Optional hard cap on cumulative house notional per token.",
    )
    parser.add_argument(
        "--max-event-notional-usdc",
        type=float,
        default=None,
        help="Optional hard cap on cumulative house notional per event title.",
    )
    parser.add_argument(
        "--max-wallet-open-notional-usdc",
        type=float,
        default=None,
        help="Optional hard cap on wallet-attributed open notional across the house book.",
    )
    parser.add_argument(
        "--max-total-open-notional-usdc",
        type=float,
        default=None,
        help="Optional hard cap on total concurrent open house notional across the whole portfolio.",
    )
    parser.add_argument(
        "--max-refresh-specs",
        type=int,
        default=250,
        help="Maximum number of prioritized open tokens to refresh during the MTM update step.",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip targeted open-token price refresh.",
    )
    parser.add_argument(
        "--refresh-all-open",
        action="store_true",
        help="Refresh all current open tokens instead of only missing marks.",
    )
    parser.add_argument(
        "--insecure-tls",
        action="store_true",
        help="Use insecure TLS fallback for the public price endpoint.",
    )
    args = parser.parse_args()

    paper_output = Path(args.paper_output_dir)
    performance_dir = paper_output / "performance"

    tracking_kwargs = {
        "output_dir": paper_output,
        "cluster_window_hours": args.cluster_window_hours,
        "action_bucket": args.action_bucket or None,
        "max_position_notional_usdc": args.max_position_notional_usdc,
        "max_event_notional_usdc": args.max_event_notional_usdc,
        "max_wallet_open_notional_usdc": args.max_wallet_open_notional_usdc,
        "max_total_open_notional_usdc": args.max_total_open_notional_usdc,
    }
    if args.wallet_csv:
        tracking_kwargs["wallet_csv"] = args.wallet_csv
    if args.mapped_trades_csv:
        tracking_kwargs["mapped_trades_csv"] = args.mapped_trades_csv
    tracking = run_paper_tracking_model(**tracking_kwargs)

    performance = run_paper_tracking_performance(
        consolidated_dir=paper_output / "consolidated",
        output_dir=performance_dir,
        max_position_notional_usdc=args.max_position_notional_usdc,
        max_event_notional_usdc=args.max_event_notional_usdc,
        max_wallet_open_notional_usdc=args.max_wallet_open_notional_usdc,
        max_total_open_notional_usdc=args.max_total_open_notional_usdc,
    )

    refresh_result = None
    if not args.skip_refresh:
        with get_session() as session:
            refresh_result = asyncio.run(
                refresh_house_open_price_history(
                    session,
                    positions_csv=tracking["open_path"],
                    performance_csv=performance_dir / "house_open_position_performance.csv",
                    only_missing_marks=not args.refresh_all_open,
                    output_dir=paper_output / "price_refresh",
                    insecure_tls=args.insecure_tls,
                    max_specs=args.max_refresh_specs,
                )
            )

    performance = run_paper_tracking_performance(
        consolidated_dir=paper_output / "consolidated",
        output_dir=performance_dir,
        max_position_notional_usdc=args.max_position_notional_usdc,
        max_event_notional_usdc=args.max_event_notional_usdc,
        max_wallet_open_notional_usdc=args.max_wallet_open_notional_usdc,
        max_total_open_notional_usdc=args.max_total_open_notional_usdc,
    )

    dashboard_kwargs = {
        "base_dir": paper_output,
        "output_dir": args.dashboard_output_dir,
    }
    if args.watchlist_csv:
        dashboard_kwargs["watchlist_csv"] = args.watchlist_csv
    dashboard = build_forward_paper_dashboard(**dashboard_kwargs)

    print(tracking["summary_path"])
    if refresh_result is not None:
        print(refresh_result["summary_path"])
    print(performance["summary_path"])
    print(dashboard["dashboard_path"])
    print(dashboard["snapshot_path"])


if __name__ == "__main__":
    main()
