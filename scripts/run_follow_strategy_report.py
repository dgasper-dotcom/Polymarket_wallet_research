"""Generate a breakdown CSV set and PDF report for the wallet half-forward analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.follow_strategy_report import generate_follow_strategy_report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="exports/per_wallet_half_forward",
        help="Directory containing the half-forward CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/per_wallet_half_forward",
        help="Directory where the report CSVs and PDF are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate report artifacts and print their locations."""

    args = parse_args()
    outputs = generate_follow_strategy_report(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print("Follow Strategy Report")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
