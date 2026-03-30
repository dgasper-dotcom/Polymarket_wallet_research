"""CLI to run out-of-sample wallet validation from enriched trades."""

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
from research.oos_validation import print_oos_summary, run_oos_validation


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-date",
        default=None,
        help="UTC date boundary for train/test split, for example 2024-11-01.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help="Train fraction when using ratio-based splitting. Defaults to the configured setting.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Number of train-selected copy and fade wallets. Defaults to the configured setting.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/oos_validation",
        help="Directory where OOS validation CSVs and plots are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run OOS validation using the current enriched-trades table."""

    args = parse_args()
    settings = get_settings()
    setup_logging()
    init_db()
    with get_session() as session:
        results = run_oos_validation(
            session,
            output_dir=args.output_dir,
            split_date=args.split_date,
            train_fraction=args.train_fraction if args.train_fraction is not None else settings.oos_train_fraction,
            top_n=args.top_n if args.top_n is not None else settings.oos_select_top_n,
            settings=settings,
        )
        print_oos_summary(results)


if __name__ == "__main__":
    main()
