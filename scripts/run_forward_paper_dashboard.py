"""Build the forward paper-tracking dashboard and watchlist snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.forward_paper_dashboard import (
    DEFAULT_BASE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WATCHLIST_CSV,
    build_forward_paper_dashboard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help="Base paper-tracking directory.")
    parser.add_argument("--watchlist-csv", default=DEFAULT_WATCHLIST_CSV, help="Copy-ready watchlist CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Dashboard output directory.")
    args = parser.parse_args()

    result = build_forward_paper_dashboard(
        base_dir=args.base_dir,
        watchlist_csv=args.watchlist_csv,
        output_dir=args.output_dir,
    )
    print(result["dashboard_path"])
    print(result["snapshot_path"])
    print(result["top_open_path"])
    print(result["history_root"])


if __name__ == "__main__":
    main()

