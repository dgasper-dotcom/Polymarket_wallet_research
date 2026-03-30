"""CLI to export leaderboard CSVs and top-wallet plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logging
from db.session import get_session, init_db
from reports.plots import generate_top_wallet_plots
from reports.summary_exports import export_research_summary, print_research_summary
from reports.wallet_leaderboard import export_wallet_leaderboard


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leaderboard-path",
        default="artifacts/reports/wallet_leaderboard.csv",
        help="CSV file path for the leaderboard export.",
    )
    parser.add_argument(
        "--plots-dir",
        default="artifacts/plots",
        help="Directory where histogram plots are written.",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of wallets to plot.")
    return parser.parse_args()


def main() -> None:
    """Run leaderboard and plot generation."""

    args = parse_args()
    setup_logging()
    init_db()
    with get_session() as session:
        export_wallet_leaderboard(session, output_path=args.leaderboard_path)
        generate_top_wallet_plots(session, output_dir=args.plots_dir, top_n=args.top_n)
        summary = export_research_summary(session, output_dir="exports")
        print_research_summary(summary, title="Scoring Summary")


if __name__ == "__main__":
    main()
