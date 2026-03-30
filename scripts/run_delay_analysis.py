"""CLI to simulate delay, costs, and net-PnL using existing enriched data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logging
from config.settings import get_settings
from db.session import get_session, init_db
from research.delay_analysis import print_delay_summary, run_delay_analysis


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="exports/delay_analysis",
        help="Directory where delay-analysis CSVs and plots are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run delay and realistic-cost analysis on the existing enriched table."""

    args = parse_args()
    settings = get_settings()
    setup_logging()
    init_db()
    with get_session() as session:
        results = run_delay_analysis(
            session,
            output_dir=args.output_dir,
            settings=settings,
        )
        print_delay_summary(results)


if __name__ == "__main__":
    main()
