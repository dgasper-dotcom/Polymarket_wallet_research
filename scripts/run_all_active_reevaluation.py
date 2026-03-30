"""Re-evaluate the full active 1067-wallet universe with hold-to-expiry evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.all_active_reevaluation import (
    DEFAULT_ANALYSIS_CUTOFF,
    DEFAULT_CURRENT_ACTIVE_CSV,
    DEFAULT_FEATURES_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RECENT_TRADES_CSV,
    DEFAULT_VERTICAL_SUMMARY_CSV,
    print_all_active_reevaluation_summary,
    run_all_active_reevaluation,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-active-csv", default=DEFAULT_CURRENT_ACTIVE_CSV)
    parser.add_argument("--features-csv", default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--vertical-summary-csv", default=DEFAULT_VERTICAL_SUMMARY_CSV)
    parser.add_argument("--recent-trades-csv", default=DEFAULT_RECENT_TRADES_CSV)
    parser.add_argument("--analysis-cutoff", default=DEFAULT_ANALYSIS_CUTOFF)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the reevaluation."""

    args = parse_args()
    results = run_all_active_reevaluation(
        current_active_csv=args.current_active_csv,
        features_csv=args.features_csv,
        vertical_summary_csv=args.vertical_summary_csv,
        recent_trades_csv=args.recent_trades_csv,
        analysis_cutoff=args.analysis_cutoff,
        output_dir=args.output_dir,
    )
    print_all_active_reevaluation_summary(results)


if __name__ == "__main__":
    main()
