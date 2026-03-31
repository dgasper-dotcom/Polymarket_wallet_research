"""Build a forward paper-tracking dashboard and watchlist snapshot."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE_DIR = "exports/manual_seed_paper_tracking"
DEFAULT_WATCHLIST_CSV = "exports/manual_seed_copy_ready_v2/manual_seed_copy_ready_monitor_avoid.csv"
DEFAULT_OUTPUT_DIR = "exports/manual_seed_paper_tracking/forward_tracker"


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None) -> float | None:
    if value in ("", None, "None", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace(" ", "T").replace("Z", "+00:00"))


def _parse_wallets_field(raw: str | None) -> list[str]:
    if not raw:
        return []
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    cleaned = text.strip("[]")
    parts = [part.strip().strip('"').strip("'").replace("\\", "") for part in cleaned.split(",")]
    return [part for part in parts if part]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return path


def _extract_analysis_cutoff(summary_text: str) -> str | None:
    for line in summary_text.splitlines():
        prefix = "- Analysis cutoff: `"
        if line.startswith(prefix) and line.endswith("`"):
            return line[len(prefix):-1]
    return None


def _extract_metric(summary_text: str, label: str) -> str | None:
    prefix = f"- {label}: `"
    for line in summary_text.splitlines():
        if line.startswith(prefix) and line.endswith("`"):
            return line[len(prefix):-1]
    return None


def _wallet_position_rollup(current_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    rollup: dict[str, dict[str, float]] = defaultdict(lambda: {
        "open_positions": 0.0,
        "supporting_positions": 0.0,
        "signaled_notional_usdc": 0.0,
        "raw_trade_count": 0.0,
    })
    for row in current_rows:
        wallets = _parse_wallets_field(row.get("supporting_wallets"))
        signaled_notional = _to_float(row.get("total_signaled_notional_usdc")) or 0.0
        raw_trade_count = _to_float(row.get("raw_trade_count")) or 0.0
        for wallet in wallets:
            wallet_key = str(wallet).lower()
            rollup[wallet_key]["open_positions"] += 1.0
            rollup[wallet_key]["supporting_positions"] += 1.0
            rollup[wallet_key]["signaled_notional_usdc"] += signaled_notional
            rollup[wallet_key]["raw_trade_count"] += raw_trade_count
    return rollup


def build_forward_paper_dashboard(
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
    watchlist_csv: str | Path = DEFAULT_WATCHLIST_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    base_root = Path(base_dir)
    output_root = Path(output_dir)
    performance_root = base_root / "performance"
    consolidated_root = base_root / "consolidated"

    watchlist_rows = _read_rows(watchlist_csv)
    current_rows = _read_rows(base_root / "current_house_positions.csv")
    closed_rows = _read_rows(performance_root / "house_closed_position_performance.csv")
    open_perf_rows = _read_rows(performance_root / "house_open_position_performance.csv")
    skipped_rows = _read_rows(base_root / "skipped_market_conflicts.csv")
    cap_skipped_rows = _read_rows(base_root / "skipped_position_cap_records.csv")
    skipped_perf_rows = _read_rows(performance_root / "house_skipped_position_records.csv")
    signal_tape_rows = _read_rows(consolidated_root / "house_signal_tape.csv")
    wallet_contrib_rows = _read_rows(performance_root / "house_wallet_contribution.csv")
    event_contrib_rows = _read_rows(performance_root / "house_event_contribution.csv")

    perf_summary_path = performance_root / "house_portfolio_performance_summary.md"
    perf_summary_text = perf_summary_path.read_text(encoding="utf-8") if perf_summary_path.exists() else ""
    analysis_cutoff_raw = _extract_analysis_cutoff(perf_summary_text)
    analysis_cutoff = analysis_cutoff_raw or datetime.now(tz=timezone.utc).isoformat()

    watchlist_rollup = _wallet_position_rollup(current_rows)
    watchlist_snapshot: list[dict[str, Any]] = []
    for row in watchlist_rows:
        wallet_id = (row.get("wallet_id") or "").lower()
        live = watchlist_rollup.get(wallet_id, {})
        watchlist_snapshot.append(
            {
                "display_name": row.get("display_name") or wallet_id,
                "wallet_id": wallet_id,
                "action_bucket": row.get("action_bucket") or "",
                "action_rationale": row.get("action_rationale") or "",
                "pma_copy_combined_net_30s": row.get("pma_copy_combined_net_30s") or "",
                "combined_realized_plus_mtm_pnl_usdc_est": row.get("combined_realized_plus_mtm_pnl_usdc_est") or "",
                "open_house_positions_supported": int(live.get("open_positions", 0.0)),
                "supported_open_notional_usdc": live.get("signaled_notional_usdc", 0.0),
                "supported_open_raw_trades": int(live.get("raw_trade_count", 0.0)),
                "notes": row.get("notes") or "",
            }
        )
    watchlist_snapshot.sort(
        key=lambda item: (
            {"copy_ready": 0, "monitor": 1, "avoid": 2}.get(item["action_bucket"], 9),
            -item["supported_open_notional_usdc"],
            item["display_name"],
        )
    )

    open_positions_sorted = sorted(
        open_perf_rows,
        key=lambda row: _to_float(row.get("entry_notional_usdc")) or 0.0,
        reverse=True,
    )
    top_open_positions = [
        {
            "house_position_id": row.get("house_position_id") or "",
            "event_title": row.get("event_title") or "",
            "outcome": row.get("outcome") or "",
            "opened_at": row.get("opened_at") or "",
            "entry_notional_usdc": _to_float(row.get("entry_notional_usdc")) or 0.0,
            "mtm_pnl_net_usdc": _to_float(row.get("mtm_pnl_net_usdc")),
            "mark_price": _to_float(row.get("mark_price")),
            "mark_price_age_seconds": _to_float(row.get("mark_price_age_seconds")),
            "supporting_wallets": row.get("supporting_wallets") or "",
        }
        for row in open_positions_sorted[:25]
    ]

    recent_open_positions = sorted(
        current_rows,
        key=lambda row: _to_dt(row.get("opened_at")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )[:25]
    recent_closed_positions = sorted(
        closed_rows,
        key=lambda row: _to_dt(row.get("closed_at")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )[:25]

    signal_counts = Counter(row.get("action") or "" for row in signal_tape_rows)
    open_marked = sum(1 for row in open_perf_rows if row.get("mark_price") not in ("", None))

    dashboard_lines = [
        "# Forward Paper Tracking Dashboard\n",
        "\n",
        "This dashboard is the operational view for the unified house portfolio. It is designed for forward paper tracking, not historical re-selection.\n",
        "\n",
        f"- Analysis cutoff: `{analysis_cutoff}`\n",
        f"- Open house positions: `{_extract_metric(perf_summary_text, 'Open house positions') or len(current_rows)}`\n",
        f"- Closed house positions: `{_extract_metric(perf_summary_text, 'Closed house positions') or len(closed_rows)}`\n",
        f"- Closed-position realized net PnL: `{_extract_metric(perf_summary_text, 'Closed-position realized net PnL') or 'n/a'}`\n",
        f"- Open-position MTM net PnL: `{_extract_metric(perf_summary_text, 'Open-position MTM net PnL') or 'n/a'}`\n",
        f"- Combined house portfolio net PnL: `{_extract_metric(perf_summary_text, 'Combined house portfolio net PnL') or 'n/a'}`\n",
        f"- Open-position mark coverage: `{open_marked}/{len(open_perf_rows)}`\n",
        f"- Wallet contribution top-1 share: `{_extract_metric(perf_summary_text, 'Wallet contribution positive-share top 1') or 'n/a'}`\n",
        f"- Wallet contribution top-5 share: `{_extract_metric(perf_summary_text, 'Wallet contribution positive-share top 5') or 'n/a'}`\n",
        f"- Event contribution top-1 share: `{_extract_metric(perf_summary_text, 'Event contribution positive-share top 1') or 'n/a'}`\n",
        f"- Event contribution top-5 share: `{_extract_metric(perf_summary_text, 'Event contribution positive-share top 5') or 'n/a'}`\n",
        f"- Consolidated open signals: `{signal_counts.get('open_long', 0)}`\n",
        f"- Consolidated reinforce signals: `{signal_counts.get('reinforce_long', 0)}`\n",
        f"- Consolidated close signals: `{signal_counts.get('close_long', 0)}`\n",
        f"- Market-conflict skips: `{len(skipped_rows)}`\n",
        f"- Position-cap skips / clips: `{len(cap_skipped_rows)}`\n",
        f"- Ledger skips / unpriceable records: `{len(skipped_perf_rows)}`\n",
        "\n",
        "## Watchlist Snapshot\n",
    ]

    for row in watchlist_snapshot:
        if row["action_bucket"] not in {"copy_ready", "monitor"}:
            continue
        dashboard_lines.append(
            f"- `{row['display_name']}` `{row['wallet_id']}` | bucket `{row['action_bucket']}` | "
            f"supporting open positions `{row['open_house_positions_supported']}` | "
            f"supported open notional `{row['supported_open_notional_usdc']:.2f}`\n"
        )

    dashboard_lines.extend(
        [
            "\n",
            "## Top Open Positions By Entry Notional\n",
        ]
    )
    for row in top_open_positions[:10]:
        mtm_text = "unmarked" if row["mtm_pnl_net_usdc"] is None else f"{row['mtm_pnl_net_usdc']:.2f}"
        dashboard_lines.append(
            f"- `{row['event_title']}` `{row['outcome']}` | entry `{row['entry_notional_usdc']:.2f}` | MTM `{mtm_text}`\n"
        )

    dashboard_lines.extend(
        [
            "\n",
            "## Recent Opens\n",
        ]
    )
    for row in recent_open_positions[:10]:
        dashboard_lines.append(
            f"- `{row.get('opened_at')}` | `{row.get('event_title')}` `{row.get('outcome')}` | "
            f"signal notional `{_to_float(row.get('total_signaled_notional_usdc')) or 0.0:.2f}`\n"
        )

    dashboard_lines.extend(
        [
            "\n",
            "## Recent Closes\n",
        ]
    )
    for row in recent_closed_positions[:10]:
        dashboard_lines.append(
            f"- `{row.get('closed_at')}` | `{row.get('event_title')}` `{row.get('outcome')}` | "
            f"realized net `{(_to_float(row.get('realized_pnl_net_usdc')) or 0.0):.2f}`\n"
        )

    dashboard_lines.extend(
        [
            "\n",
            "## Top Wallet Contributors\n",
        ]
    )
    for row in wallet_contrib_rows[:10]:
        dashboard_lines.append(
            f"- `{row.get('wallet')}` | combined `{(_to_float(row.get('combined_net_pnl_usdc')) or 0.0):.2f}` | "
            f"positions `{int(_to_float(row.get('positions')) or 0)}`\n"
        )

    dashboard_lines.extend(
        [
            "\n",
            "## Top Event Contributors\n",
        ]
    )
    for row in event_contrib_rows[:10]:
        dashboard_lines.append(
            f"- `{row.get('event_title')}` | combined `{(_to_float(row.get('combined_net_pnl_usdc')) or 0.0):.2f}` | "
            f"positions `{int(_to_float(row.get('positions')) or 0)}`\n"
        )

    dashboard_lines.extend(
        [
            "\n",
            "## Audit Log Files\n",
            f"- `house_signal_tape.csv`: `{consolidated_root / 'house_signal_tape.csv'}`\n",
            f"- `current_house_positions.csv`: `{base_root / 'current_house_positions.csv'}`\n",
            f"- `closed_house_positions.csv`: `{base_root / 'closed_house_positions.csv'}`\n",
            f"- `skipped_market_conflicts.csv`: `{base_root / 'skipped_market_conflicts.csv'}`\n",
            f"- `skipped_position_cap_records.csv`: `{base_root / 'skipped_position_cap_records.csv'}`\n",
            f"- `house_skipped_position_records.csv`: `{performance_root / 'house_skipped_position_records.csv'}`\n",
            "\n",
            "Every consolidated wallet signal is recorded in the tape. Anything that does not turn into an active house position is expected to appear as a market conflict skip, a position-cap skip/clip, or a performance-layer skipped record.\n",
        ]
    )

    dashboard_path = output_root / "forward_paper_dashboard.md"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text("".join(dashboard_lines), encoding="utf-8")

    snapshot_path = _write_csv(
        output_root / "forward_watchlist_snapshot.csv",
        watchlist_snapshot,
        fieldnames=list(watchlist_snapshot[0].keys()) if watchlist_snapshot else [
            "display_name",
            "wallet_id",
            "action_bucket",
            "action_rationale",
            "pma_copy_combined_net_30s",
            "combined_realized_plus_mtm_pnl_usdc_est",
            "open_house_positions_supported",
            "supported_open_notional_usdc",
            "supported_open_raw_trades",
            "notes",
        ],
    )
    top_open_path = _write_csv(
        output_root / "forward_top_open_positions.csv",
        top_open_positions,
        fieldnames=list(top_open_positions[0].keys()) if top_open_positions else [
            "house_position_id",
            "event_title",
            "outcome",
            "opened_at",
            "entry_notional_usdc",
            "mtm_pnl_net_usdc",
            "mark_price",
            "mark_price_age_seconds",
            "supporting_wallets",
        ],
    )

    timestamp = analysis_cutoff.replace(":", "-")
    history_root = output_root / "history" / timestamp
    history_root.mkdir(parents=True, exist_ok=True)
    for path in (
        dashboard_path,
        snapshot_path,
        top_open_path,
        perf_summary_path,
        performance_root / "house_open_position_performance.csv",
        performance_root / "house_closed_position_performance.csv",
        performance_root / "house_portfolio_daily_equity_curve.csv",
        base_root / "current_house_positions.csv",
        base_root / "closed_house_positions.csv",
        base_root / "skipped_market_conflicts.csv",
        base_root / "skipped_position_cap_records.csv",
        consolidated_root / "house_signal_tape.csv",
    ):
        if path.exists():
            shutil.copy2(path, history_root / path.name)

    return {
        "dashboard_path": dashboard_path,
        "snapshot_path": snapshot_path,
        "top_open_path": top_open_path,
        "history_root": history_root,
    }
