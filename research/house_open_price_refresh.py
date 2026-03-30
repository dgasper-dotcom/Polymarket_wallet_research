"""Refresh price history for tokens currently held by the house portfolio."""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from clients.clob_client import ClobClient
from ingestion.prices import backfill_price_history_for_token_bounds


DEFAULT_HOUSE_OPEN_POSITIONS_CSV = "exports/manual_seed_paper_tracking/current_house_positions.csv"
DEFAULT_HOUSE_OPEN_PERFORMANCE_CSV = (
    "exports/manual_seed_paper_tracking/performance/house_open_position_performance.csv"
)
DEFAULT_OUTPUT_DIR = "exports/manual_seed_paper_tracking/price_refresh"


@dataclass(frozen=True)
class RefreshSpec:
    token_id: str
    opened_at: datetime
    market_id: str
    event_title: str
    outcome: str


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_utc_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def build_house_open_refresh_specs(
    positions_csv: str | Path = DEFAULT_HOUSE_OPEN_POSITIONS_CSV,
    *,
    performance_csv: str | Path | None = DEFAULT_HOUSE_OPEN_PERFORMANCE_CSV,
    only_missing_marks: bool = True,
) -> list[RefreshSpec]:
    """Build per-token refresh bounds for current house positions."""

    position_rows = _read_rows(positions_csv)
    if not position_rows:
        return []

    missing_tokens: set[str] | None = None
    if only_missing_marks and performance_csv:
        perf_rows = _read_rows(performance_csv)
        missing_tokens = {
            str(row.get("token_id") or "")
            for row in perf_rows
            if str(row.get("token_id") or "").strip()
            and (row.get("mark_price") in ("", None))
        }

    specs: dict[str, RefreshSpec] = {}
    for row in position_rows:
        token_id = str(row.get("token_id") or "").strip()
        if not token_id:
            continue
        if missing_tokens is not None and token_id not in missing_tokens:
            continue
        opened_at = _to_utc_datetime(row.get("opened_at"))
        if opened_at is None:
            continue
        existing = specs.get(token_id)
        if existing is None or opened_at < existing.opened_at:
            specs[token_id] = RefreshSpec(
                token_id=token_id,
                opened_at=opened_at,
                market_id=str(row.get("market_id") or ""),
                event_title=str(row.get("event_title") or ""),
                outcome=str(row.get("outcome") or ""),
            )
    return sorted(specs.values(), key=lambda item: (item.opened_at, item.token_id))


async def refresh_house_open_price_history(
    session: Session,
    *,
    positions_csv: str | Path = DEFAULT_HOUSE_OPEN_POSITIONS_CSV,
    performance_csv: str | Path | None = DEFAULT_HOUSE_OPEN_PERFORMANCE_CSV,
    only_missing_marks: bool = True,
    fidelity: int = 1,
    margin_seconds: int = 3600,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    insecure_tls: bool = False,
) -> dict[str, Any]:
    specs = build_house_open_refresh_specs(
        positions_csv,
        performance_csv=performance_csv,
        only_missing_marks=only_missing_marks,
    )
    now = datetime.now(tz=timezone.utc)
    token_bounds = [(spec.token_id, spec.opened_at, now) for spec in specs]
    client = ClobClient(verify=False) if insecure_tls else ClobClient()
    try:
        refresh_counts = await backfill_price_history_for_token_bounds(
            session,
            token_bounds=token_bounds,
            fidelity=fidelity,
            margin_seconds=margin_seconds,
            client=client,
        )
    finally:
        await client.aclose()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    detail_rows = [
        {
            "token_id": spec.token_id,
            "opened_at": spec.opened_at.isoformat(),
            "market_id": spec.market_id,
            "event_title": spec.event_title,
            "outcome": spec.outcome,
            "refreshed_price_rows": int(refresh_counts.get(spec.token_id, 0)),
        }
        for spec in specs
    ]
    detail_path = output_root / "house_open_price_refresh_details.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()) if detail_rows else [
            "token_id",
            "opened_at",
            "market_id",
            "event_title",
            "outcome",
            "refreshed_price_rows",
        ])
        writer.writeheader()
        if detail_rows:
            writer.writerows(detail_rows)

    summary_lines = [
        "# House Open Price Refresh Summary\n",
        "\n",
        f"- Tokens targeted: `{len(specs)}`\n",
        f"- Only missing marks: `{only_missing_marks}`\n",
        f"- Insecure TLS fallback: `{insecure_tls}`\n",
        f"- Tokens with any new price rows: `{sum(1 for value in refresh_counts.values() if value > 0)}`\n",
        f"- Total new price rows written: `{sum(int(value) for value in refresh_counts.values())}`\n",
        f"- Detail CSV: `{detail_path}`\n",
    ]
    summary_path = output_root / "house_open_price_refresh_summary.md"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")

    return {
        "specs": specs,
        "refresh_counts": refresh_counts,
        "detail_path": detail_path,
        "summary_path": summary_path,
    }
