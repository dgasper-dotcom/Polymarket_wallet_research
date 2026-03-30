"""Build the current active copyable-wallet watchlist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.active_watchlist import (
    DEFAULT_CURRENT_ACTIVE_CSV,
    DEFAULT_FEATURES_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VERTICAL_SUMMARY_CSV,
    print_active_watchlist_summary,
    run_active_watchlist,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--current-active-csv", default=DEFAULT_CURRENT_ACTIVE_CSV)
    parser.add_argument("--vertical-summary-csv", default=DEFAULT_VERTICAL_SUMMARY_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the watchlist build."""

    args = parse_args()
    results = run_active_watchlist(
        features_csv=args.features_csv,
        current_active_csv=args.current_active_csv,
        vertical_summary_csv=args.vertical_summary_csv,
        output_dir=args.output_dir,
    )
    print_active_watchlist_summary(results)


if __name__ == "__main__":
    main()
