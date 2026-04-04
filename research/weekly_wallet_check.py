"""Weekly PMA watchlist refresh plus unified house-book replay."""

from __future__ import annotations

import asyncio
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PANDAS_NO_IMPORT_PYARROW", "1")

import pandas as pd

from db.session import get_session
from research.forward_paper_dashboard import build_forward_paper_dashboard
from research.house_open_price_refresh import refresh_house_open_price_history
from research.manual_seed_pma_backtest import (
    DEFAULT_MAX_PAGES,
    DEFAULT_PAGE_SIZE,
    fetch_full_history_pma_trades,
    map_pma_trades_to_public_markets,
)
from research.paper_tracking_model import run_paper_tracking_model
from research.paper_tracking_performance import run_paper_tracking_performance


def _resolve_path(project_root: Path, raw: str | Path | None) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(raw)
    return path if path.is_absolute() else project_root / path


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def default_trade_start_utc(reference: datetime | None = None) -> datetime:
    now = reference or _utc_now()
    return datetime(now.year, 1, 1, tzinfo=timezone.utc)


def _parse_utc(value: str | datetime | None, *, fallback: datetime | None = None) -> datetime:
    if value is None:
        if fallback is None:
            raise ValueError("missing datetime value")
        return fallback
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    if "T" not in text and " " in text:
        text = text.replace(" ", "T", 1)
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _read_watchlist_meta(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row.get("wallet_id") or "").lower(): {
                "display_name": row.get("display_name") or (row.get("wallet_id") or ""),
                "bucket": row.get("action_bucket") or "",
            }
            for row in csv.DictReader(handle)
            if row.get("wallet_id")
        }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return path


def filter_pma_trades_from(raw_trades: pd.DataFrame, start_utc: datetime) -> pd.DataFrame:
    if raw_trades.empty:
        return raw_trades.copy()
    mask = pd.to_datetime(raw_trades["trade_dttm"], utc=True, errors="coerce") >= pd.Timestamp(start_utc)
    return raw_trades.loc[mask].copy()


def _extract_metric(summary_text: str, label: str) -> str | None:
    prefix = f"- {label}: `"
    for line in summary_text.splitlines():
        if line.startswith(prefix) and line.endswith("`"):
            return line[len(prefix):-1]
    return None


def _read_wallet_contribution(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            wallet = (row.get("wallet") or "").lower()
            if not wallet:
                continue
            rows[wallet] = {
                "positions": float(row.get("positions") or 0.0),
                "entry_notional_usdc": float(row.get("entry_notional_usdc") or 0.0),
                "realized_net_pnl_usdc": float(row.get("realized_net_pnl_usdc") or 0.0),
                "open_mtm_net_pnl_usdc": float(row.get("open_mtm_net_pnl_usdc") or 0.0),
                "combined_net_pnl_usdc": float(row.get("combined_net_pnl_usdc") or 0.0),
            }
    return rows


def build_wallet_delta_rows(
    baseline_rows: dict[str, dict[str, float]],
    current_rows: dict[str, dict[str, float]],
    watchlist_meta: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wallet, current in current_rows.items():
        baseline = baseline_rows.get(wallet, {})
        rows.append(
            {
                "wallet": wallet,
                "display_name": watchlist_meta.get(wallet, {}).get("display_name", wallet),
                "bucket": watchlist_meta.get(wallet, {}).get("bucket", ""),
                "old_combined_net_pnl_usdc": round(baseline.get("combined_net_pnl_usdc", 0.0), 6),
                "new_combined_net_pnl_usdc": round(current.get("combined_net_pnl_usdc", 0.0), 6),
                "delta_combined_net_pnl_usdc": round(
                    current.get("combined_net_pnl_usdc", 0.0) - baseline.get("combined_net_pnl_usdc", 0.0),
                    6,
                ),
                "old_positions": baseline.get("positions", 0.0),
                "new_positions": current.get("positions", 0.0),
                "delta_positions": current.get("positions", 0.0) - baseline.get("positions", 0.0),
                "new_realized_net_pnl_usdc": round(current.get("realized_net_pnl_usdc", 0.0), 6),
                "new_open_mtm_net_pnl_usdc": round(current.get("open_mtm_net_pnl_usdc", 0.0), 6),
            }
        )
    rows.sort(key=lambda row: (row["bucket"], -row["delta_combined_net_pnl_usdc"], row["display_name"]))
    return rows


def build_watchlist_activity_rows(
    raw_trades_csv: Path,
    watchlist_meta: dict[str, dict[str, str]],
    activity_start_utc: datetime,
) -> list[dict[str, Any]]:
    activity: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"trades": 0, "buy_value": 0.0, "sell_value": 0.0, "net_flow": 0.0, "last_trade": ""}
    )
    with raw_trades_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            trade_ts = _parse_utc((row.get("trade_dttm") or "").replace(" ", "T"))
            if trade_ts < activity_start_utc:
                continue
            wallet = (row.get("trader_id") or row.get("wallet_address") or "").lower()
            if wallet not in watchlist_meta:
                continue
            value = float(row.get("value") or 0.0)
            side = (row.get("side") or "").lower()
            rec = activity[wallet]
            rec["trades"] += 1
            if side == "buy":
                rec["buy_value"] += value
                rec["net_flow"] += value
            elif side == "sell":
                rec["sell_value"] += value
                rec["net_flow"] -= value
            if not rec["last_trade"] or trade_ts.isoformat() > rec["last_trade"]:
                rec["last_trade"] = trade_ts.isoformat()

    rows: list[dict[str, Any]] = []
    for wallet, meta in watchlist_meta.items():
        rec = activity[wallet]
        rows.append(
            {
                "wallet": wallet,
                "display_name": meta["display_name"],
                "bucket": meta["bucket"],
                "trades_since_start": rec["trades"],
                "buy_value_usdc": round(rec["buy_value"], 6),
                "sell_value_usdc": round(rec["sell_value"], 6),
                "net_flow_usdc": round(rec["net_flow"], 6),
                "last_trade_utc": rec["last_trade"],
            }
        )
    rows.sort(key=lambda row: (row["bucket"], -row["trades_since_start"], row["display_name"]))
    return rows


def _write_weekly_summary(
    *,
    summary_path: Path,
    baseline_summary_path: Path | None,
    current_summary_path: Path,
    wallet_delta_rows: list[dict[str, Any]],
    activity_rows: list[dict[str, Any]],
    activity_start_utc: datetime,
) -> Path:
    baseline_text = baseline_summary_path.read_text(encoding="utf-8") if baseline_summary_path and baseline_summary_path.exists() else ""
    current_text = current_summary_path.read_text(encoding="utf-8")

    old_combined = _extract_metric(baseline_text, "Combined house portfolio net PnL")
    old_realized = _extract_metric(baseline_text, "Closed-position realized net PnL")
    old_open_mtm = _extract_metric(baseline_text, "Open-position MTM net PnL")
    old_open_positions = _extract_metric(baseline_text, "Open house positions")
    old_closed_positions = _extract_metric(baseline_text, "Closed house positions")

    new_combined = _extract_metric(current_text, "Combined house portfolio net PnL") or "n/a"
    new_realized = _extract_metric(current_text, "Closed-position realized net PnL") or "n/a"
    new_open_mtm = _extract_metric(current_text, "Open-position MTM net PnL") or "n/a"
    new_open_positions = _extract_metric(current_text, "Open house positions") or "n/a"
    new_closed_positions = _extract_metric(current_text, "Closed house positions") or "n/a"
    current_cutoff = _extract_metric(current_text, "Analysis cutoff") or "n/a"

    lines = [
        "# Weekly Wallet Check\n",
        "\n",
        f"This report refreshes the forward paper-book through `{current_cutoff}`.\n",
        "\n",
        "## Unified House Book\n",
        "\n",
    ]

    if old_combined is not None:
        delta = round(float(new_combined.split(" USDC")[0]) - float(old_combined.split(" USDC")[0]), 2)
        lines.extend(
            [
                f"- Previous combined house net PnL: `{old_combined}`\n",
                f"- Current combined house net PnL: `{new_combined}`\n",
                f"- Change since prior snapshot: `{delta} USDC`\n",
                f"- Previous realized net PnL: `{old_realized}`\n",
                f"- Current realized net PnL: `{new_realized}`\n",
                f"- Previous open MTM net PnL: `{old_open_mtm}`\n",
                f"- Current open MTM net PnL: `{new_open_mtm}`\n",
                f"- Open positions: `{old_open_positions}` -> `{new_open_positions}`\n",
                f"- Closed positions: `{old_closed_positions}` -> `{new_closed_positions}`\n",
            ]
        )
    else:
        lines.extend(
            [
                f"- Current combined house net PnL: `{new_combined}`\n",
                f"- Current realized net PnL: `{new_realized}`\n",
                f"- Current open MTM net PnL: `{new_open_mtm}`\n",
                f"- Open positions: `{new_open_positions}`\n",
                f"- Closed positions: `{new_closed_positions}`\n",
            ]
        )

    lines.extend(["\n", "## Copy Wallet Delta Since Prior Snapshot\n", "\n"])
    for row in (item for item in wallet_delta_rows if item["bucket"] == "copy_ready"):
        lines.append(
            f"- `{row['display_name']}`: combined `{row['old_combined_net_pnl_usdc']:.2f}` -> "
            f"`{row['new_combined_net_pnl_usdc']:.2f}` | delta `{row['delta_combined_net_pnl_usdc']:+.2f}` | "
            f"positions `{int(row['old_positions'])}` -> `{int(row['new_positions'])}`\n"
        )

    lines.extend(
        [
            "\n",
            f"## Watchlist Activity Since {activity_start_utc.isoformat()}\n",
            "\n",
        ]
    )
    for row in activity_rows:
        last_trade = row["last_trade_utc"] or "n/a"
        lines.append(
            f"- `{row['display_name']}` `{row['bucket']}`: trades `{row['trades_since_start']}` | "
            f"buy `{row['buy_value_usdc']:.2f}` | sell `{row['sell_value_usdc']:.2f}` | "
            f"net flow `{row['net_flow_usdc']:+.2f}` | last trade `{last_trade}`\n"
        )

    summary_path.write_text("".join(lines), encoding="utf-8")
    return summary_path


def run_weekly_wallet_check(
    *,
    project_root: str | Path,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    baseline_dir: str | Path | None = None,
    trade_start_utc: str | datetime | None = None,
    activity_start_utc: str | datetime | None = None,
    activity_lookback_days: int = 7,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = DEFAULT_MAX_PAGES,
    refresh: bool = False,
    insecure_tls: bool = False,
    max_refresh_specs: int = 250,
) -> dict[str, Any]:
    project = Path(project_root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    now_utc = _utc_now()

    trade_start = _parse_utc(trade_start_utc, fallback=default_trade_start_utc(now_utc))
    activity_start = _parse_utc(
        activity_start_utc,
        fallback=now_utc - timedelta(days=int(activity_lookback_days)),
    )

    out_root = _resolve_path(project, output_dir) if output_dir else project / "exports" / f"weekly_wallet_check_{now_utc:%Y%m%d}"
    assert out_root is not None
    out_root.mkdir(parents=True, exist_ok=True)

    wallet_csv = _resolve_path(project, config.get("wallet_csv"))
    watchlist_csv = _resolve_path(project, config.get("watchlist_csv"))
    if wallet_csv is None or watchlist_csv is None:
        raise ValueError("config must include wallet_csv and watchlist_csv")

    watchlist_meta = _read_watchlist_meta(watchlist_csv)
    seed_wallets = pd.DataFrame.from_records(
        [
            {"display_name": meta["display_name"], "wallet_address": wallet}
            for wallet, meta in watchlist_meta.items()
        ]
    )

    raw_trades, coverage = fetch_full_history_pma_trades(
        seed_wallets,
        page_size=page_size,
        max_pages=max_pages,
    )
    raw_path = out_root / "pma_raw_trades_watchlist.csv"
    coverage_path = out_root / "pma_trade_coverage_watchlist.csv"
    raw_trades.to_csv(raw_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    filtered_trades = filter_pma_trades_from(raw_trades, trade_start)
    mapped_trades, mapping_audit, mapped_markets = map_pma_trades_to_public_markets(filtered_trades)
    mapped_path = out_root / "manual_seed_pma_mapped_trades_watchlist.csv"
    mapping_audit_path = out_root / "mapping_audit_watchlist.csv"
    mapped_markets_path = out_root / "mapped_markets_watchlist.csv"
    mapped_trades.to_csv(mapped_path, index=False)
    mapping_audit.to_csv(mapping_audit_path, index=False)
    mapped_markets.to_csv(mapped_markets_path, index=False)

    paper_root = out_root / "forward_paper"
    tracking = run_paper_tracking_model(
        wallet_csv=wallet_csv,
        mapped_trades_csv=mapped_path,
        output_dir=paper_root,
        cluster_window_hours=int(config.get("cluster_window_hours") or 24),
        action_bucket=None,
        max_position_notional_usdc=config.get("max_position_notional_usdc"),
        max_event_notional_usdc=config.get("max_event_notional_usdc"),
        max_wallet_open_notional_usdc=config.get("max_wallet_open_notional_usdc"),
        max_total_open_notional_usdc=config.get("max_total_open_notional_usdc"),
    )

    performance_dir = paper_root / "performance"
    performance = run_paper_tracking_performance(
        consolidated_dir=paper_root / "consolidated",
        output_dir=performance_dir,
        max_position_notional_usdc=config.get("max_position_notional_usdc"),
        max_event_notional_usdc=config.get("max_event_notional_usdc"),
        max_wallet_open_notional_usdc=config.get("max_wallet_open_notional_usdc"),
        max_total_open_notional_usdc=config.get("max_total_open_notional_usdc"),
    )

    refresh_result = None
    if refresh:
        with get_session() as session:
            refresh_result = asyncio.run(
                refresh_house_open_price_history(
                    session,
                    positions_csv=tracking["open_path"],
                    performance_csv=performance_dir / "house_open_position_performance.csv",
                    only_missing_marks=True,
                    output_dir=paper_root / "price_refresh",
                    insecure_tls=insecure_tls,
                    max_specs=max_refresh_specs,
                )
            )
        performance = run_paper_tracking_performance(
            consolidated_dir=paper_root / "consolidated",
            output_dir=performance_dir,
            max_position_notional_usdc=config.get("max_position_notional_usdc"),
            max_event_notional_usdc=config.get("max_event_notional_usdc"),
            max_wallet_open_notional_usdc=config.get("max_wallet_open_notional_usdc"),
            max_total_open_notional_usdc=config.get("max_total_open_notional_usdc"),
        )

    dashboard = build_forward_paper_dashboard(
        base_dir=paper_root,
        watchlist_csv=watchlist_csv,
        output_dir=paper_root / "forward_tracker",
    )

    baseline_root = _resolve_path(project, baseline_dir) if baseline_dir else _resolve_path(project, config.get("paper_output_dir"))
    baseline_summary_path = baseline_root / "performance" / "house_portfolio_performance_summary.md" if baseline_root else None
    baseline_wallet_path = baseline_root / "performance" / "house_wallet_contribution.csv" if baseline_root else None
    baseline_rows = _read_wallet_contribution(baseline_wallet_path) if baseline_wallet_path else {}
    current_rows = _read_wallet_contribution(performance_dir / "house_wallet_contribution.csv")
    wallet_delta_rows = build_wallet_delta_rows(baseline_rows, current_rows, watchlist_meta)

    wallet_delta_path = _write_csv(
        out_root / "weekly_copy_wallet_delta.csv",
        wallet_delta_rows,
        fieldnames=[
            "wallet",
            "display_name",
            "bucket",
            "old_combined_net_pnl_usdc",
            "new_combined_net_pnl_usdc",
            "delta_combined_net_pnl_usdc",
            "old_positions",
            "new_positions",
            "delta_positions",
            "new_realized_net_pnl_usdc",
            "new_open_mtm_net_pnl_usdc",
        ],
    )

    activity_rows = build_watchlist_activity_rows(raw_path, watchlist_meta, activity_start)
    activity_path = _write_csv(
        out_root / "watchlist_activity.csv",
        activity_rows,
        fieldnames=[
            "wallet",
            "display_name",
            "bucket",
            "trades_since_start",
            "buy_value_usdc",
            "sell_value_usdc",
            "net_flow_usdc",
            "last_trade_utc",
        ],
    )

    summary_path = _write_weekly_summary(
        summary_path=out_root / "weekly_wallet_check_summary.md",
        baseline_summary_path=baseline_summary_path,
        current_summary_path=performance["summary_path"],
        wallet_delta_rows=wallet_delta_rows,
        activity_rows=activity_rows,
        activity_start_utc=activity_start,
    )

    return {
        "output_dir": out_root,
        "raw_path": raw_path,
        "coverage_path": coverage_path,
        "mapped_path": mapped_path,
        "mapping_audit_path": mapping_audit_path,
        "mapped_markets_path": mapped_markets_path,
        "paper_root": paper_root,
        "tracking": tracking,
        "performance": performance,
        "dashboard": dashboard,
        "wallet_delta_path": wallet_delta_path,
        "activity_path": activity_path,
        "summary_path": summary_path,
        "refresh_result": refresh_result,
        "trade_start_utc": trade_start,
        "activity_start_utc": activity_start,
    }
