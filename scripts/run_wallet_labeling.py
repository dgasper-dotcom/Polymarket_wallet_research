"""Generate first-pass wallet features and rule-based wallet archetype labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.wallet_labeling import run_wallet_labeling


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        default="data/wallet_features.csv",
        help="Output CSV path for wallet-level features, relative to project root.",
    )
    parser.add_argument(
        "--labels-path",
        default="data/wallet_first_pass_labels.csv",
        help="Output CSV path for first-pass labels, relative to project root.",
    )
    parser.add_argument(
        "--report-path",
        default="reports/wallet_labeling_summary.md",
        help="Output Markdown path for the labeling summary, relative to project root.",
    )
    return parser


def main() -> None:
    """Run the wallet labeling pipeline and print a concise summary."""

    args = build_parser().parse_args()
    results = run_wallet_labeling(
        PROJECT_ROOT,
        features_path=args.features_path,
        labels_path=args.labels_path,
        report_path=args.report_path,
    )
    labels = results["labels"]
    counts = labels["primary_label"].value_counts().to_dict()

    print("Wallet Labeling Summary")
    print(f"Wallet rows: {len(labels)}")
    print(f"Positive EV / copyable: {counts.get('positive_ev_copyable', 0)}")
    print(f"HFT / latency-sensitive: {counts.get('hft_latency_sensitive', 0)}")
    print(f"YOLO / noise / unstable: {counts.get('yolo_noise_unstable', 0)}")
    print(f"Features CSV: {results['paths']['features']}")
    print(f"Labels CSV: {results['paths']['labels']}")
    print(f"Summary report: {results['paths']['report']}")


if __name__ == "__main__":
    main()
