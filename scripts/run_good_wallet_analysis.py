"""Run a focused analysis for the currently preferred slower-hold wallets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.good_wallet_analysis import (
    DEFAULT_FEATURES_CSV,
    DEFAULT_GOOD_WALLETS,
    DEFAULT_MIRROR_ANCHOR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RECENT_TRADES_CSV,
    print_good_wallet_analysis_summary,
    run_good_wallet_analysis,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wallet",
        action="append",
        dest="wallets",
        help=(
            "Wallet to analyze. Repeat the flag for multiple wallets. "
            "Defaults to the current good-wallet pair."
        ),
    )
    parser.add_argument(
        "--mirror-anchor",
        default=DEFAULT_MIRROR_ANCHOR,
        help="Wallet to use as the mirror-detection anchor.",
    )
    parser.add_argument(
        "--features-csv",
        default=DEFAULT_FEATURES_CSV,
        help="Wallet feature CSV to read.",
    )
    parser.add_argument(
        "--recent-trades-csv",
        default=DEFAULT_RECENT_TRADES_CSV,
        help="Recent trade CSV to read.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the focused good-wallet analysis."""

    args = parse_args()
    wallets = args.wallets or list(DEFAULT_GOOD_WALLETS)
    results = run_good_wallet_analysis(
        wallets=wallets,
        mirror_anchor=args.mirror_anchor,
        features_csv=args.features_csv,
        recent_trades_csv=args.recent_trades_csv,
        output_dir=args.output_dir,
    )
    print_good_wallet_analysis_summary(results)


if __name__ == "__main__":
    main()
