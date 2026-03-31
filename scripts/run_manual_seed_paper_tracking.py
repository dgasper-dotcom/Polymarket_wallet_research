"""Run the unified paper-tracking model for copy-ready wallets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.paper_tracking_model import (
    DEFAULT_OUTPUT_DIR,
    run_paper_tracking_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for paper-tracking outputs.",
    )
    parser.add_argument(
        "--cluster-window-hours",
        type=int,
        default=24,
        help="Hours to cluster near-simultaneous same-token signals.",
    )
    parser.add_argument(
        "--action-bucket",
        default="copy_ready",
        help="Filter wallet csv rows by action_bucket. Use empty string for all rows.",
    )
    parser.add_argument(
        "--max-position-notional-usdc",
        type=float,
        default=None,
        help="Optional hard cap on cumulative house notional per token.",
    )
    parser.add_argument(
        "--max-event-notional-usdc",
        type=float,
        default=None,
        help="Optional hard cap on cumulative house notional per event title.",
    )
    parser.add_argument(
        "--max-wallet-open-notional-usdc",
        type=float,
        default=None,
        help="Optional hard cap on wallet-attributed open notional across the house book.",
    )
    parser.add_argument(
        "--max-total-open-notional-usdc",
        type=float,
        default=None,
        help="Optional hard cap on total concurrent open house notional across the whole portfolio.",
    )
    args = parser.parse_args()

    result = run_paper_tracking_model(
        output_dir=Path(args.output_dir),
        cluster_window_hours=args.cluster_window_hours,
        action_bucket=args.action_bucket or None,
        max_position_notional_usdc=args.max_position_notional_usdc,
        max_event_notional_usdc=args.max_event_notional_usdc,
        max_wallet_open_notional_usdc=args.max_wallet_open_notional_usdc,
        max_total_open_notional_usdc=args.max_total_open_notional_usdc,
    )

    print(result["summary_path"])
    print(result["open_path"])
    print(result["closed_path"])
    print(result["conflict_path"])
    print(result["cap_path"])


if __name__ == "__main__":
    main()
