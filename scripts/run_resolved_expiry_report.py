"""Build a resolved-expiry-hold report for the current watchlist wallets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.resolved_expiry_report import (
    DEFAULT_ANALYSIS_CUTOFF,
    DEFAULT_FEATURES_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RECENT_TRADES_CSV,
    load_watchlist_wallets,
    print_resolved_expiry_summary,
    run_resolved_expiry_watchlist_report,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wallet", action="append", dest="wallets", help="Wallet to include. Repeat for more.")
    parser.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--recent-trades-csv", default=DEFAULT_RECENT_TRADES_CSV)
    parser.add_argument("--analysis-cutoff", default=DEFAULT_ANALYSIS_CUTOFF)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the report."""

    args = parse_args()
    wallets = args.wallets or load_watchlist_wallets()
    results = run_resolved_expiry_watchlist_report(
        wallets=wallets,
        features_csv=args.features_csv,
        recent_trades_csv=args.recent_trades_csv,
        output_dir=args.output_dir,
        analysis_cutoff=args.analysis_cutoff,
    )
    print_resolved_expiry_summary(results)


if __name__ == "__main__":
    main()
