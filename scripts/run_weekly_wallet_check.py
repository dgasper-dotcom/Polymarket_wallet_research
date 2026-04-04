"""Run the weekly PMA watchlist refresh and forward paper-book check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.weekly_wallet_check import run_weekly_wallet_check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/forward_paper_30d_core4.json",
        help="Frozen forward-paper JSON config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to exports/weekly_wallet_check_YYYYMMDD.",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="Optional baseline forward-paper directory to compare against. Defaults to config paper_output_dir.",
    )
    parser.add_argument(
        "--trade-start-utc",
        default=None,
        help="Override for the PMA trade-history filter start timestamp. Defaults to January 1 of the current UTC year.",
    )
    parser.add_argument(
        "--activity-start-utc",
        default=None,
        help="Explicit UTC timestamp for the activity report cutoff.",
    )
    parser.add_argument(
        "--activity-lookback-days",
        type=int,
        default=7,
        help="Lookback window for raw watchlist activity when activity-start-utc is omitted.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="PMA page size per wallet.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=25,
        help="Maximum PMA pages per wallet.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Also run targeted open-token price refresh before the final performance pass.",
    )
    parser.add_argument(
        "--insecure-tls",
        action="store_true",
        help="Use insecure TLS fallback for the public price endpoint during refresh.",
    )
    parser.add_argument(
        "--max-refresh-specs",
        type=int,
        default=250,
        help="Maximum number of targeted open tokens to refresh when --refresh is used.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_weekly_wallet_check(
        project_root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / args.config,
        output_dir=args.output_dir,
        baseline_dir=args.baseline_dir,
        trade_start_utc=args.trade_start_utc,
        activity_start_utc=args.activity_start_utc,
        activity_lookback_days=args.activity_lookback_days,
        page_size=args.page_size,
        max_pages=args.max_pages,
        refresh=args.refresh,
        insecure_tls=args.insecure_tls,
        max_refresh_specs=args.max_refresh_specs,
    )
    print(results["summary_path"])
    print(results["wallet_delta_path"])
    print(results["activity_path"])
    print(results["dashboard"]["dashboard_path"])


if __name__ == "__main__":
    main()
