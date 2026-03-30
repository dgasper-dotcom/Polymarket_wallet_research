from __future__ import annotations

import csv
from pathlib import Path

from research.house_open_price_refresh import build_house_open_refresh_specs


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_house_open_refresh_specs_filters_to_missing_marks(tmp_path: Path) -> None:
    positions_csv = tmp_path / "positions.csv"
    performance_csv = tmp_path / "performance.csv"

    _write_csv(
        positions_csv,
        ["token_id", "opened_at", "market_id", "event_title", "outcome"],
        [
            {
                "token_id": "t1",
                "opened_at": "2026-01-01T00:00:00+00:00",
                "market_id": "m1",
                "event_title": "Event 1",
                "outcome": "Yes",
            },
            {
                "token_id": "t2",
                "opened_at": "2026-01-02T00:00:00+00:00",
                "market_id": "m2",
                "event_title": "Event 2",
                "outcome": "No",
            },
        ],
    )
    _write_csv(
        performance_csv,
        ["token_id", "mark_price"],
        [
            {"token_id": "t1", "mark_price": ""},
            {"token_id": "t2", "mark_price": "0.77"},
        ],
    )

    specs = build_house_open_refresh_specs(
        positions_csv,
        performance_csv=performance_csv,
        only_missing_marks=True,
    )

    assert [spec.token_id for spec in specs] == ["t1"]
