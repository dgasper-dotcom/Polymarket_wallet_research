"""Chronological out-of-sample splits for the unified house book."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


def _parse_ts(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _float_or_zero(value: str | None) -> float:
    if value is None:
        return 0.0
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return 0.0
    return float(raw)


@dataclass
class HousePosition:
    house_position_id: str
    opened_at: datetime
    closed_at: datetime | None
    entry_notional_usdc: float
    exit_cost_total_usdc: float
    realized_net_usdc: float
    mtm_net_usdc: float
    is_open: bool
    has_mark: bool


def _load_positions(
    closed_csv: Path,
    open_csv: Path,
) -> list[HousePosition]:
    positions: list[HousePosition] = []

    with closed_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            opened_at = _parse_ts(row.get("opened_at"))
            if opened_at is None:
                continue
            positions.append(
                HousePosition(
                    house_position_id=row["house_position_id"],
                    opened_at=opened_at,
                    closed_at=_parse_ts(row.get("closed_at")),
                    entry_notional_usdc=_float_or_zero(row.get("entry_notional_usdc")),
                    exit_cost_total_usdc=_float_or_zero(row.get("exit_cost_total_usdc")),
                    realized_net_usdc=_float_or_zero(row.get("realized_pnl_net_usdc")),
                    mtm_net_usdc=0.0,
                    is_open=False,
                    has_mark=False,
                )
            )

    with open_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            opened_at = _parse_ts(row.get("opened_at"))
            if opened_at is None:
                continue
            mark_price = row.get("mark_price")
            has_mark = bool(str(mark_price).strip()) and str(mark_price).strip().lower() != "nan"
            positions.append(
                HousePosition(
                    house_position_id=row["house_position_id"],
                    opened_at=opened_at,
                    closed_at=None,
                    entry_notional_usdc=_float_or_zero(row.get("entry_notional_usdc")),
                    exit_cost_total_usdc=0.0,
                    realized_net_usdc=0.0,
                    mtm_net_usdc=_float_or_zero(row.get("mtm_pnl_net_usdc")),
                    is_open=True,
                    has_mark=has_mark,
                )
            )

    return sorted(positions, key=lambda row: (row.opened_at, row.house_position_id))


def _peak_concurrent_notional(positions: Iterable[HousePosition], analysis_cutoff: datetime) -> float:
    events: list[tuple[datetime, float]] = []
    for row in positions:
        events.append((row.opened_at, row.entry_notional_usdc))
        close_ts = row.closed_at or analysis_cutoff
        events.append((close_ts, -row.entry_notional_usdc))

    current = 0.0
    peak = 0.0
    for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        current += delta
        peak = max(peak, current)
    return peak


def _summarize_positions(
    split_name: str,
    positions: list[HousePosition],
    cutoff_ts: datetime | None,
    analysis_cutoff: datetime,
) -> dict[str, object]:
    if not positions:
        return {
            "split": split_name,
            "cutoff_ts": cutoff_ts.isoformat() if cutoff_ts else "",
            "n_positions": 0,
            "n_open_positions": 0,
            "n_closed_positions": 0,
            "first_opened_at": "",
            "last_opened_at": "",
            "entry_volume_usdc": 0.0,
            "gross_turnover_usdc": 0.0,
            "peak_concurrent_notional_usdc": 0.0,
            "realized_net_pnl_usdc": 0.0,
            "unrealized_mtm_net_pnl_usdc": 0.0,
            "combined_net_pnl_usdc": 0.0,
            "marked_open_positions": 0,
            "mark_coverage_share": 0.0,
        }

    n_open = sum(1 for row in positions if row.is_open)
    n_closed = len(positions) - n_open
    marked_open = sum(1 for row in positions if row.is_open and row.has_mark)
    entry_volume = sum(row.entry_notional_usdc for row in positions)
    realized = sum(row.realized_net_usdc for row in positions)
    unrealized = sum(row.mtm_net_usdc for row in positions if row.is_open)
    gross_turnover = entry_volume + sum(row.exit_cost_total_usdc for row in positions if not row.is_open)
    peak = _peak_concurrent_notional(positions, analysis_cutoff)

    return {
        "split": split_name,
        "cutoff_ts": cutoff_ts.isoformat() if cutoff_ts else "",
        "n_positions": len(positions),
        "n_open_positions": n_open,
        "n_closed_positions": n_closed,
        "first_opened_at": positions[0].opened_at.isoformat(),
        "last_opened_at": positions[-1].opened_at.isoformat(),
        "entry_volume_usdc": entry_volume,
        "gross_turnover_usdc": gross_turnover,
        "peak_concurrent_notional_usdc": peak,
        "realized_net_pnl_usdc": realized,
        "unrealized_mtm_net_pnl_usdc": unrealized,
        "combined_net_pnl_usdc": realized + unrealized,
        "marked_open_positions": marked_open,
        "mark_coverage_share": (marked_open / n_open) if n_open else 0.0,
    }


def run_house_book_oos(
    *,
    closed_csv: str | Path = "exports/manual_seed_paper_tracking/performance/house_closed_position_performance.csv",
    open_csv: str | Path = "exports/manual_seed_paper_tracking/performance/house_open_position_performance.csv",
    output_dir: str | Path = "exports/manual_seed_paper_tracking/performance/oos_splits",
    ratios: tuple[float, ...] = (0.6, 0.7, 0.8),
) -> dict[str, object]:
    """Build chronological train/test splits for the unified house book."""

    closed_path = Path(closed_csv)
    open_path = Path(open_csv)
    export_root = Path(output_dir)
    export_root.mkdir(parents=True, exist_ok=True)

    positions = _load_positions(closed_path, open_path)
    if not positions:
        raise ValueError("No house positions available for OOS testing.")

    analysis_cutoff = max(
        [row.closed_at for row in positions if row.closed_at is not None]
        + [datetime.now(tz=positions[0].opened_at.tzinfo)]
    )
    for row in positions:
        if row.is_open:
            break
    open_rows = [row for row in positions if row.is_open]
    if open_rows:
        # Reuse the latest analysis timestamp encoded in the open-position export.
        with open_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
            if first and first.get("analysis_cutoff"):
                parsed = _parse_ts(first["analysis_cutoff"])
                if parsed is not None:
                    analysis_cutoff = parsed

    summary_rows: list[dict[str, object]] = []
    summary_rows.append(_summarize_positions("full_history", positions, None, analysis_cutoff))

    n_positions = len(positions)
    for ratio in ratios:
        cutoff_index = max(1, min(n_positions - 1, math.floor(n_positions * ratio)))
        cutoff_ts = positions[cutoff_index].opened_at
        train_rows = positions[:cutoff_index]
        test_rows = positions[cutoff_index:]
        ratio_label = int(ratio * 100)
        summary_rows.append(
            _summarize_positions(f"train_{ratio_label}_{100-ratio_label}", train_rows, cutoff_ts, analysis_cutoff)
        )
        summary_rows.append(
            _summarize_positions(f"test_{ratio_label}_{100-ratio_label}", test_rows, cutoff_ts, analysis_cutoff)
        )

    csv_path = export_root / "house_book_oos_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# House Book OOS Summary",
        "",
        f"- Total house positions: `{n_positions}`",
        f"- Analysis cutoff: `{analysis_cutoff.isoformat()}`",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"## {row['split']}",
                f"- cutoff: `{row['cutoff_ts'] or 'n/a'}`",
                f"- positions: `{row['n_positions']}`",
                f"- open positions: `{row['n_open_positions']}`",
                f"- closed positions: `{row['n_closed_positions']}`",
                f"- entry volume: `{float(row['entry_volume_usdc']):,.2f}` USDC",
                f"- gross turnover: `{float(row['gross_turnover_usdc']):,.2f}` USDC",
                f"- peak concurrent notional: `{float(row['peak_concurrent_notional_usdc']):,.2f}` USDC",
                f"- realized net PnL: `{float(row['realized_net_pnl_usdc']):,.2f}` USDC",
                f"- unrealized MTM net PnL: `{float(row['unrealized_mtm_net_pnl_usdc']):,.2f}` USDC",
                f"- combined net PnL: `{float(row['combined_net_pnl_usdc']):,.2f}` USDC",
                f"- open-mark coverage: `{int(row['marked_open_positions'])}/{int(row['n_open_positions'])}` ({float(row['mark_coverage_share'])*100:.1f}%)",
                "",
            ]
        )

    md_path = export_root / "house_book_oos_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "summary_rows": summary_rows,
        "analysis_cutoff": analysis_cutoff,
        "csv_path": csv_path,
        "md_path": md_path,
    }
