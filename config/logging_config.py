"""Central logging setup for scripts and modules."""

from __future__ import annotations

import logging

from config.settings import get_settings


def setup_logging(level: str | None = None) -> None:
    """Configure a simple process-wide logging format."""

    settings = get_settings()
    resolved_level = (level or settings.log_level).upper()
    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
