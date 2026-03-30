"""Run performance tracking for the unified house paper portfolio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.paper_tracking_performance import (
    DEFAULT_CONSOLIDATED_DIR,
    DEFAULT_OUTPUT_DIR,
    run_paper_tracking_performance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consolidated-dir",
        default=DEFAULT_CONSOLIDATED_DIR,
        help="Directory containing house_signal_tape.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for performance outputs.",
    )
    parser.add_argument(
        "--analysis-cutoff",
        default=None,
        help="Optional UTC timestamp cutoff for MTM evaluation.",
    )
    args = parser.parse_args()

    result = run_paper_tracking_performance(
        consolidated_dir=args.consolidated_dir,
        output_dir=args.output_dir,
        analysis_cutoff=args.analysis_cutoff,
    )
    print(result["summary_path"])
    print(result["open_path"])
    print(result["closed_path"])
    print(result["curve_path"])
    if result["plot_path"] is not None:
        print(result["plot_path"])


if __name__ == "__main__":
    main()
