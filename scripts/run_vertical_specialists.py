"""Run a vertical-specialist screen across the recent wallet universe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.vertical_specialists import (
    DEFAULT_OUTPUT_DIR,
    print_vertical_specialist_summary,
    run_vertical_specialist_analysis,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the vertical-specialist outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the vertical-specialist analysis."""

    args = parse_args()
    results = run_vertical_specialist_analysis(output_dir=args.output_dir)
    print_vertical_specialist_summary(results)


if __name__ == "__main__":
    main()
