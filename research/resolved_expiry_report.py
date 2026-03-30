"""Report wallets' buy lots that were held through public market resolution."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd

from config.settings import Settings, get_settings
from db.session import get_session
from research.copy_follow_expiry import _build_terminal_lookup, load_markets_frame
from research.copy_follow_wallet_exit import build_copy_exit_pairs
from research.costs import calculate_net_pnl, estimate_entry_only_cost


DEFAULT_STRICT_WATCHLIST_CSV = "exports/active_copyable_watchlist/strict_active_watchlist.csv"
DEFAULT_MONITOR_WATCHLIST_CSV = "exports/active_copyable_watchlist/monitor_watchlist.csv"
DEFAULT_FEATURES_CSV = "data/wallet_features.csv"
DEFAULT_RECENT_TRADES_CSV = (
    "exports/recent_wallet_trade_capture_top2000_20240101_20260328/"
    "recent_wallet_trades_20240101_20260328.csv"
)
DEFAULT_OUTPUT_DIR = "exports/resolved_expiry_watchlist"
DEFAULT_ANALYSIS_CUTOFF = "2026-03-28T23:59:59Z"


def _read_csv(path: str | Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read one CSV file while tolerating missing paths."""

    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=usecols or [])
    return pd.read_csv(csv_path, usecols=usecols)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write one CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _normalize_wallets(values: list[str] | tuple[str, ...]) -> list[str]:
    """Normalize wallet ids."""

    seen: set[str] = set()
    ordered: list[str] = []
    for wallet in values:
        normalized = str(wallet or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def load_watchlist_wallets(
    *,
    strict_watchlist_csv: str | Path = DEFAULT_STRICT_WATCHLIST_CSV,
    monitor_watchlist_csv: str | Path = DEFAULT_MONITOR_WATCHLIST_CSV,
) -> list[str]:
    """Load the current strict + monitor watchlist wallets."""

    wallets: list[str] = []
    for csv_path in (strict_watchlist_csv, monitor_watchlist_csv):
        frame = _read_csv(csv_path, usecols=["wallet_id"])
        if frame.empty or "wallet_id" not in frame.columns:
            continue
        wallets.extend(frame["wallet_id"].astype(str).tolist())
    return _normalize_wallets(wallets)


def load_watchlist_features(path: str | Path = DEFAULT_FEATURES_CSV) -> pd.DataFrame:
    """Load wallet feature exports for later joins."""

    frame = _read_csv(path)
    if frame.empty:
        return frame
    if "wallet_id" in frame.columns:
        frame["wallet_id"] = frame["wallet_id"].astype(str).str.strip().str.lower()
    return frame


def load_long_window_recent_trades(
    *,
    recent_trades_csv: str | Path = DEFAULT_RECENT_TRADES_CSV,
    wallets: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Load the long-window public wallet-trade capture for selected wallets."""

    usecols = [
        "trade_id",
        "wallet_address",
        "market_id",
        "token_id",
        "side",
        "price",
        "size",
        "usdc_size",
        "timestamp",
    ]
    frame = _read_csv(recent_trades_csv, usecols=usecols)
    if frame.empty:
        return frame
    normalized_wallets = set(_normalize_wallets(list(wallets)))
    frame["wallet_address"] = frame["wallet_address"].astype(str).str.strip().str.lower()
    frame = frame.loc[frame["wallet_address"].isin(normalized_wallets)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
    frame["usdc_size"] = pd.to_numeric(frame["usdc_size"], errors="coerce")
    frame = frame.dropna(subset=["wallet_address", "timestamp", "token_id"]).reset_index(drop=True)
    return frame.sort_values(["wallet_address", "timestamp", "trade_id"]).reset_index(drop=True)


def compute_resolved_expiry_positions_from_frame(
    recent_trades: pd.DataFrame,
    terminal_lookup: dict[str, dict[str, Any]],
    *,
    analysis_cutoff: str | pd.Timestamp = DEFAULT_ANALYSIS_CUTOFF,
    settings: Settings | None = None,
    return_diagnostics: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute held-to-expiry observed slices from open wallet buy lots.

    Only open lots with public market resolution on or before the analysis
    cutoff are treated as observed held-to-expiry behavior.
    """

    cfg = settings or get_settings()
    if recent_trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cutoff = pd.to_datetime(analysis_cutoff, utc=True, errors="coerce")
    if cutoff is None or pd.isna(cutoff):
        cutoff = pd.to_datetime(recent_trades["timestamp"], utc=True, errors="coerce").max()

    pairs, open_positions = build_copy_exit_pairs(recent_trades)
    if open_positions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    open_frame = open_positions.copy()
    open_frame["wallet_address"] = open_frame["wallet_address"].astype(str).str.lower()
    open_frame["buy_timestamp"] = pd.to_datetime(open_frame["buy_timestamp"], utc=True, errors="coerce")
    open_frame["copied_size"] = pd.to_numeric(open_frame["copied_size"], errors="coerce")
    open_frame["buy_price_signal"] = pd.to_numeric(open_frame["buy_price_signal"], errors="coerce")

    records: list[dict[str, Any]] = []
    for row in open_frame.to_dict(orient="records"):
        token_id = str(row.get("token_id")) if row.get("token_id") is not None else None
        terminal_info = terminal_lookup.get(token_id or "") or {}
        resolution_ts = pd.to_datetime(terminal_info.get("resolution_ts"), utc=True, errors="coerce")
        terminal_price = terminal_info.get("terminal_price")
        trade_record = dict(row)
        trade_record["analysis_cutoff"] = cutoff
        trade_record["resolution_ts"] = resolution_ts
        trade_record["question"] = terminal_info.get("question")
        trade_record["terminal_price"] = terminal_price
        trade_record["terminal_price_source"] = terminal_info.get("terminal_price_source")
        if resolution_ts is None or pd.isna(resolution_ts) or resolution_ts > cutoff:
            trade_record["resolved_status"] = "unresolved_open"
            trade_record["holding_days_to_resolution"] = None
            trade_record["wallet_pnl_expiry_raw"] = None
            trade_record["wallet_pnl_expiry_net_est"] = None
            records.append(trade_record)
            continue
        if terminal_price is None:
            trade_record["resolved_status"] = "resolved_missing_terminal_price"
            trade_record["holding_days_to_resolution"] = None
            trade_record["wallet_pnl_expiry_raw"] = None
            trade_record["wallet_pnl_expiry_net_est"] = None
            records.append(trade_record)
            continue

        raw_pnl = float(terminal_price) - float(row.get("buy_price_signal") or 0.0)
        total_cost = float(
            estimate_entry_only_cost(
                pd.Series(row),
                entry_price=float(row.get("buy_price_signal") or 0.0),
                scenario=cfg.cost_scenario,
                settings=cfg,
            )["total_cost"]
        )
        trade_record["resolved_status"] = "held_to_expiry_observed"
        trade_record["holding_days_to_resolution"] = float(
            (resolution_ts - pd.to_datetime(row.get("buy_timestamp"), utc=True, errors="coerce")).total_seconds()
            / 86400.0
        )
        trade_record["wallet_pnl_expiry_raw"] = raw_pnl
        trade_record["wallet_pnl_expiry_net_est"] = calculate_net_pnl(raw_pnl, total_cost)
        trade_record["wallet_pnl_expiry_raw_usdc"] = raw_pnl * float(row.get("copied_size") or 0.0)
        trade_record["wallet_pnl_expiry_net_est_usdc"] = calculate_net_pnl(
            trade_record["wallet_pnl_expiry_raw_usdc"],
            total_cost * float(row.get("copied_size") or 0.0),
        )
        records.append(trade_record)

    diagnostics = pd.DataFrame.from_records(records).sort_values(
        ["wallet_address", "buy_timestamp", "signal_trade_id", "trade_id"]
    )
    resolved_trades = diagnostics.loc[diagnostics["resolved_status"] == "held_to_expiry_observed"].copy()

    paired_counts = (
        pairs.groupby("wallet_address", sort=False).size().astype(int).to_dict()
        if not pairs.empty
        else {}
    )
    unresolved_counts = (
        diagnostics.loc[diagnostics["resolved_status"] == "unresolved_open"]
        .groupby("wallet_address", sort=False)
        .size()
        .astype(int)
        .to_dict()
    )
    resolved_counts = (
        resolved_trades.groupby("wallet_address", sort=False).size().astype(int).to_dict()
        if not resolved_trades.empty
        else {}
    )

    wallets = sorted(set(open_frame["wallet_address"].astype(str).str.lower().tolist()))
    summary_rows: list[dict[str, Any]] = []
    for wallet in wallets:
        wallet_resolved = resolved_trades.loc[resolved_trades["wallet_address"] == wallet].copy()
        held_count = int(resolved_counts.get(wallet, 0))
        paired_count = int(paired_counts.get(wallet, 0))
        unresolved_count = int(unresolved_counts.get(wallet, 0))
        resolved_behavior = held_count + paired_count
        total_behavior = held_count + paired_count + unresolved_count
        summary_rows.append(
            {
                "wallet_id": wallet,
                "wallet_sell_closed_slices": paired_count,
                "held_to_expiry_observed_slices": held_count,
                "unresolved_open_slices": unresolved_count,
                "resolved_behavior_slices": resolved_behavior,
                "total_behavior_slices": total_behavior,
                "hold_to_expiry_share_observed": (
                    float(held_count / resolved_behavior) if resolved_behavior else None
                ),
                "hold_to_expiry_share_all_slices": (
                    float(held_count / total_behavior) if total_behavior else None
                ),
                "avg_wallet_pnl_expiry_raw": (
                    float(pd.to_numeric(wallet_resolved["wallet_pnl_expiry_raw"], errors="coerce").mean())
                    if not wallet_resolved.empty
                    else None
                ),
                "avg_wallet_pnl_expiry_net_est": (
                    float(pd.to_numeric(wallet_resolved["wallet_pnl_expiry_net_est"], errors="coerce").mean())
                    if not wallet_resolved.empty
                    else None
                ),
                "median_holding_days_to_resolution": (
                    float(pd.to_numeric(wallet_resolved["holding_days_to_resolution"], errors="coerce").median())
                    if not wallet_resolved.empty
                    else None
                ),
                "expiry_held_hit_rate": (
                    float((pd.to_numeric(wallet_resolved["wallet_pnl_expiry_net_est"], errors="coerce") > 0).mean())
                    if not wallet_resolved.empty
                    else None
                ),
            }
        )

    wallet_summary = pd.DataFrame.from_records(summary_rows).sort_values(
        ["held_to_expiry_observed_slices", "hold_to_expiry_share_observed", "wallet_id"],
        ascending=[False, False, True],
        na_position="last",
    )

    overview = pd.DataFrame(
        [
            {
                "analysis_cutoff": cutoff.isoformat(),
                "wallets_in_report": int(len(wallet_summary)),
                "wallets_with_observed_hold_to_expiry": int(
                    (pd.to_numeric(wallet_summary.get("held_to_expiry_observed_slices"), errors="coerce").fillna(0) > 0).sum()
                )
                if not wallet_summary.empty
                else 0,
                "held_to_expiry_observed_slices_total": int(
                    pd.to_numeric(wallet_summary.get("held_to_expiry_observed_slices"), errors="coerce").fillna(0).sum()
                )
                if not wallet_summary.empty
                else 0,
                "unresolved_open_slices_total": int(
                    pd.to_numeric(wallet_summary.get("unresolved_open_slices"), errors="coerce").fillna(0).sum()
                )
                if not wallet_summary.empty
                else 0,
            }
        ]
    )
    if return_diagnostics:
        return wallet_summary, resolved_trades, overview, diagnostics
    return wallet_summary, resolved_trades, overview


def build_report_markdown(wallet_summary: pd.DataFrame, overview: pd.DataFrame) -> str:
    """Render a concise Markdown report."""

    lines = [
        "# Resolved Expiry-Hold Report",
        "",
        "This report only counts buy lots that remained open and whose markets had",
        "publicly resolved by the analysis cutoff.",
        "",
    ]
    if not overview.empty:
        row = overview.iloc[0].to_dict()
        lines.extend(
            [
                "## Overview",
                "",
                f"- Wallets in report: `{row.get('wallets_in_report')}`",
                f"- Wallets with observed hold-to-expiry lots: `{row.get('wallets_with_observed_hold_to_expiry')}`",
                f"- Observed hold-to-expiry slices: `{row.get('held_to_expiry_observed_slices_total')}`",
                f"- Unresolved open slices: `{row.get('unresolved_open_slices_total')}`",
                "",
            ]
        )

    lines.extend(["## Wallets", ""])
    if wallet_summary.empty:
        lines.append("- No watchlist wallets were found in the selected trade window.")
    else:
        for row in wallet_summary.to_dict(orient="records"):
            lines.append(
                f"- `{row['wallet_id']}` | held_to_expiry `{row.get('held_to_expiry_observed_slices')}` | "
                f"unresolved `{row.get('unresolved_open_slices')}` | "
                f"share_observed `{row.get('hold_to_expiry_share_observed')}` | "
                f"avg_net_est `{row.get('avg_wallet_pnl_expiry_net_est')}`"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `held_to_expiry_observed_slices` only increments when the market has publicly resolved by the cutoff.",
            "- A value of zero can still be consistent with a long-horizon trader if most positions sit in `unresolved_open_slices`.",
            "- Net PnL here is an entry-cost estimate only; there is no pre-expiry exit cost because the observed behavior is hold-through-resolution.",
        ]
    )
    return "\n".join(lines)


def run_resolved_expiry_watchlist_report(
    *,
    wallets: list[str] | tuple[str, ...] | None = None,
    features_csv: str | Path = DEFAULT_FEATURES_CSV,
    recent_trades_csv: str | Path = DEFAULT_RECENT_TRADES_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    analysis_cutoff: str = DEFAULT_ANALYSIS_CUTOFF,
) -> dict[str, Any]:
    """Build the resolved expiry-hold report for watchlist wallets."""

    selected_wallets = _normalize_wallets(list(wallets or load_watchlist_wallets()))
    features = load_watchlist_features(features_csv)
    recent_trades = load_long_window_recent_trades(
        recent_trades_csv=recent_trades_csv,
        wallets=selected_wallets,
    )
    with get_session() as session:
        terminal_lookup = _build_terminal_lookup(load_markets_frame(session))

    wallet_summary, resolved_trades, overview = compute_resolved_expiry_positions_from_frame(
        recent_trades,
        terminal_lookup,
        analysis_cutoff=analysis_cutoff,
    )
    base_wallets = pd.DataFrame({"wallet_id": selected_wallets})
    wallet_summary = base_wallets.merge(wallet_summary, on="wallet_id", how="left")
    for column in (
        "wallet_sell_closed_slices",
        "held_to_expiry_observed_slices",
        "unresolved_open_slices",
        "resolved_behavior_slices",
        "total_behavior_slices",
    ):
        if column in wallet_summary.columns:
            wallet_summary[column] = pd.to_numeric(wallet_summary[column], errors="coerce").fillna(0).astype(int)
    if not wallet_summary.empty and not features.empty:
        wallet_summary = wallet_summary.merge(
            features[["wallet_id", "sample_name", "dominant_vertical" if "dominant_vertical" in features.columns else "wallet_id"]]
            if "dominant_vertical" in features.columns
            else features[["wallet_id", "sample_name"]],
            on="wallet_id",
            how="left",
        )
    overview = pd.DataFrame(
        [
            {
                "analysis_cutoff": pd.to_datetime(analysis_cutoff, utc=True, errors="coerce").isoformat(),
                "wallets_in_report": int(len(wallet_summary)),
                "wallets_with_observed_hold_to_expiry": int(
                    (pd.to_numeric(wallet_summary.get("held_to_expiry_observed_slices"), errors="coerce").fillna(0) > 0).sum()
                )
                if not wallet_summary.empty
                else 0,
                "held_to_expiry_observed_slices_total": int(
                    pd.to_numeric(wallet_summary.get("held_to_expiry_observed_slices"), errors="coerce").fillna(0).sum()
                )
                if not wallet_summary.empty
                else 0,
                "unresolved_open_slices_total": int(
                    pd.to_numeric(wallet_summary.get("unresolved_open_slices"), errors="coerce").fillna(0).sum()
                )
                if not wallet_summary.empty
                else 0,
            }
        ]
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    wallet_summary = wallet_summary.sort_values(
        ["held_to_expiry_observed_slices", "hold_to_expiry_share_observed", "wallet_id"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    wallet_path = _write_csv(wallet_summary, output_root / "watchlist_resolved_expiry_wallet_summary.csv")
    trades_path = _write_csv(resolved_trades, output_root / "watchlist_resolved_expiry_trades.csv")
    overview_path = _write_csv(overview, output_root / "watchlist_resolved_expiry_overview.csv")
    report_path = output_root / "watchlist_resolved_expiry_report.md"
    report_path.write_text(build_report_markdown(wallet_summary, overview), encoding="utf-8")

    return {
        "wallet_summary": wallet_summary,
        "resolved_trades": resolved_trades,
        "overview": overview,
        "paths": {
            "wallet_summary": wallet_path,
            "resolved_trades": trades_path,
            "overview": overview_path,
            "report": report_path,
        },
    }


def print_resolved_expiry_summary(results: dict[str, Any]) -> None:
    """Print a concise terminal summary."""

    overview = results["overview"].iloc[0].to_dict() if not results["overview"].empty else {}
    print("Resolved Expiry-Hold Report")
    print(f"Wallets in report: {overview.get('wallets_in_report')}")
    print(f"Wallets with observed hold-to-expiry: {overview.get('wallets_with_observed_hold_to_expiry')}")
    print(f"Observed hold-to-expiry slices: {overview.get('held_to_expiry_observed_slices_total')}")
    print(f"Unresolved open slices: {overview.get('unresolved_open_slices_total')}")
    if not results["wallet_summary"].empty:
        columns = [
            column
            for column in [
                "wallet_id",
                "sample_name",
                "held_to_expiry_observed_slices",
                "unresolved_open_slices",
                "hold_to_expiry_share_observed",
                "avg_wallet_pnl_expiry_net_est",
            ]
            if column in results["wallet_summary"].columns
        ]
        print(results["wallet_summary"][columns].to_string(index=False))
