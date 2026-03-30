"""Build the revised manual-seed copy-ready watchlist and readiness report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from research.manual_seed_copy_ready import run_manual_seed_copy_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="exports/manual_seed_copy_ready",
        help="Directory where watchlist and readiness files will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_manual_seed_copy_ready(output_dir=args.output_dir)
    print(results["all_path"])
    print(results["summary_path"])
    print(f"investor_ready={results['investor_ready']}")


if __name__ == "__main__":
    main()
