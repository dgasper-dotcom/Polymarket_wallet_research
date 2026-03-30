"""CLI to run multiple out-of-sample split schemes and aggregate robustness results."""

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
from research.multi_oos_validation import print_multi_oos_summary, run_multi_oos_validation


def _parse_ratio_splits(value: str) -> list[float]:
    """Parse a comma-separated ratio list like 0.6,0.7,0.8."""

    ratios: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        ratios.append(float(item))
    return ratios


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-splits",
        type=int,
        default=settings.multi_oos_n_splits,
        help="Total number of split schemes to execute.",
    )
    parser.add_argument(
        "--ratio-splits",
        default="0.60,0.70,0.80",
        help="Comma-separated ratio-based train fractions to include.",
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Include random index-boundary splits in addition to ratio and rolling splits.",
    )
    parser.add_argument(
        "--random-splits",
        type=int,
        default=2,
        help="How many random index-boundary splits to include when --include-random is enabled.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=settings.multi_oos_random_seed,
        help="Seed used for random index-boundary split generation.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=settings.oos_select_top_n,
        help="Top-N copy and fade wallets selected within each train split.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/multi_oos",
        help="Directory where split-level and aggregated outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run multi-split OOS validation."""

    args = parse_args()
    settings = get_settings()
    setup_logging()
    init_db()
    with get_session() as session:
        results = run_multi_oos_validation(
            session,
            output_dir=args.output_dir,
            n_splits=args.n_splits,
            ratio_splits=_parse_ratio_splits(args.ratio_splits),
            include_random=args.include_random,
            random_splits=args.random_splits,
            random_seed=args.random_seed,
            top_n=args.top_n,
            settings=settings,
        )
        print_multi_oos_summary(results)


if __name__ == "__main__":
    main()
