"""CLI wrapper for unified house-book out-of-sample splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.house_book_oos import run_house_book_oos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--closed-csv",
        default="exports/manual_seed_paper_tracking/performance/house_closed_position_performance.csv",
        help="Closed house-position performance export.",
    )
    parser.add_argument(
        "--open-csv",
        default="exports/manual_seed_paper_tracking/performance/house_open_position_performance.csv",
        help="Open house-position performance export.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/manual_seed_paper_tracking/performance/oos_splits",
        help="Directory for OOS split summaries.",
    )
    args = parser.parse_args()

    results = run_house_book_oos(
        closed_csv=args.closed_csv,
        open_csv=args.open_csv,
        output_dir=args.output_dir,
    )
    print(f"Wrote {results['csv_path']}")
    print(f"Wrote {results['md_path']}")


if __name__ == "__main__":
    main()
