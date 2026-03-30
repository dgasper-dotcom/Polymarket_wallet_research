"""CLI to run behavior-level trade feature extraction and conditional signal analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logging
from db.session import get_session, init_db
from research.behavior_analysis import print_behavior_summary, run_behavior_analysis


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="exports/behavior_analysis",
        help="Directory where behavior-analysis CSVs are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run behavior-level research outputs from the existing enriched dataset."""

    args = parse_args()
    setup_logging()
    init_db()
    with get_session() as session:
        results = run_behavior_analysis(session, output_dir=args.output_dir)
        print_behavior_summary(results)


if __name__ == "__main__":
    main()
