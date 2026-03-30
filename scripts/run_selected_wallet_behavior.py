"""Run a behavior deep-dive for the strict repeated-positive wallet shortlist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logging
from db.session import get_session, init_db
from research.selected_wallet_behavior import (
    DEFAULT_END_DATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SELECTION_CSV,
    DEFAULT_START_DATE,
    print_selected_wallet_behavior_summary,
    run_selected_wallet_behavior_analysis,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-csv",
        default=DEFAULT_SELECTION_CSV,
        help="CSV containing the strict repeated-positive shortlist with one preferred delay per wallet.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Inclusive UTC start timestamp for the recent-trade window.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="Inclusive UTC end timestamp for the recent-trade window.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the deep-dive CSV and Markdown report are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the strict-wallet behavior deep dive."""

    args = parse_args()
    setup_logging()
    init_db()
    with get_session() as session:
        results = run_selected_wallet_behavior_analysis(
            session,
            selection_csv=args.selection_csv,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print_selected_wallet_behavior_summary(results)


if __name__ == "__main__":
    main()
