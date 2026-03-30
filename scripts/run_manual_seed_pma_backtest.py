"""Run the full-history PMA-based copy backtest for manual seed wallets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from research.manual_seed_pma_backtest import run_manual_seed_pma_full_history_backtest


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-csv",
        default="data/manual_seed_wallets.csv",
        help="CSV containing the resolved manual seed wallet list.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/manual_seed_pma_full_history_backtest",
        help="Directory where CSV reports will be written.",
    )
    parser.add_argument(
        "--delays",
        default="0,5,15,30,60",
        help="Comma-separated delay list in seconds.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    delays = tuple(int(part.strip()) for part in args.delays.split(",") if part.strip())
    results = run_manual_seed_pma_full_history_backtest(
        seed_csv=args.seed_csv,
        output_dir=args.output_dir,
        delays=delays,
    )
    overview = results["overview"]
    wallet_summary = results["wallet_summary"]
    print(f"seed wallets: {len(results['seed_wallets'])}")
    print(f"raw PMA trades: {len(results['raw_trades'])}")
    print(f"mapped trades: {len(wallet_summary)} wallets with backtest output")
    if not overview.empty:
        row = overview.iloc[0]
        for delay in delays:
            label = f"{delay}s"
            print(f"combined net {label}: {row.get(f'combined_net_total_usdc_{label}')}")
    print(results["wallet_path"])


if __name__ == "__main__":
    main()
