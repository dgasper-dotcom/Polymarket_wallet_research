"""CLI to generate wallet event study tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.logging_config import setup_logging
from db.session import get_session, init_db
from reports.summary_exports import export_research_summary, print_research_summary
from research.event_study import build_event_study


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/event_study",
        help="Directory where CSV event study outputs are written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run wallet-level event study reporting."""

    args = parse_args()
    setup_logging()
    init_db()
    with get_session() as session:
        build_event_study(session, output_dir=args.output_dir)
        summary = export_research_summary(session, output_dir="exports")
        print_research_summary(summary, title="Event Study Summary")


if __name__ == "__main__":
    main()
