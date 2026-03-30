"""Build an active copyable-wallet watchlist from existing research exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sys

sys.modules.setdefault("pyarrow", None)

import pandas as pd


DEFAULT_FEATURES_CSV = "data/wallet_features.csv"
DEFAULT_CURRENT_ACTIVE_CSV = (
    "exports/current_market_wallet_scan_top2000/"
    "wallets_100plus_recent_20260312_20260326.csv"
)
DEFAULT_VERTICAL_SUMMARY_CSV = "exports/vertical_specialists/vertical_wallet_summary.csv"
DEFAULT_OUTPUT_DIR = "exports/active_copyable_watchlist"

MANUAL_EXCLUDE_WALLETS = {
    "0x0a1cc6071c968086e5706a1a5cc6c9746bd6efaf",
    "0x74f1b8944c7b10f7239e3dda0de6b8ee42e816a8",
    "0xc02147dee42356b7a4edbb1c35ac4ffa95f61fa8",
    "0xb9cfb65e4ed4a953a72a6ea23d834a41daba75d1",
    "0xcf9555cef256c96a91dd4cd6c3a159c033867865",
    "0x30eead8be2dbf57303b78eb9e7404d37c9bca587",
    "0x1c266db0f8529b1f25b77123e9c0c918ac2f6e31",
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read one CSV or return an empty frame."""

    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write one CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_watchlist_frame(
    *,
    features_csv: str | Path = DEFAULT_FEATURES_CSV,
    current_active_csv: str | Path = DEFAULT_CURRENT_ACTIVE_CSV,
    vertical_summary_csv: str | Path = DEFAULT_VERTICAL_SUMMARY_CSV,
) -> pd.DataFrame:
    """Load and join the wallet-level inputs used by the watchlist rules."""

    features = _read_csv(features_csv)
    current = _read_csv(current_active_csv)
    vertical = _read_csv(vertical_summary_csv)

    if features.empty:
        return pd.DataFrame()

    features = features.copy()
    features["wallet_id"] = features["wallet_id"].astype(str).str.lower()
    if not current.empty:
        current = current.copy()
        current["wallet_address"] = current["wallet_address"].astype(str).str.lower()
    if not vertical.empty:
        vertical = vertical.copy()
        vertical["wallet_address"] = vertical["wallet_address"].astype(str).str.lower()

    frame = features.merge(
        current[["wallet_address", "total_trades", "distinct_markets", "most_recent_trade_ts"]],
        left_on="wallet_id",
        right_on="wallet_address",
        how="left",
    )
    frame = frame.merge(
        vertical[
            [
                "wallet_address",
                "dominant_vertical",
                "dominant_vertical_share",
                "avg_trades_per_active_day_observed",
            ]
        ],
        on="wallet_address",
        how="left",
    )
    frame["wallet_id"] = frame["wallet_id"].astype(str).str.lower()
    frame["is_current_active"] = frame["wallet_id"].isin(set(current.get("wallet_address", pd.Series(dtype=str))))
    frame["sell_share"] = (
        pd.to_numeric(frame["recent_sell_trades"], errors="coerce")
        / pd.to_numeric(frame["recent_trades_window"], errors="coerce")
    )
    pending = pd.to_numeric(frame["pending_open_copy_slices"], errors="coerce")
    paired = pd.to_numeric(frame["paired_copy_slices"], errors="coerce")
    frame["open_share"] = pending / (pending + paired)
    frame["holding_days"] = pd.to_numeric(frame["median_holding_seconds"], errors="coerce") / 86400.0
    frame["avg_trades_per_active_day_observed"] = pd.to_numeric(
        frame["avg_trades_per_active_day_observed"], errors="coerce"
    ).fillna(
        pd.to_numeric(frame["recent_trades_window"], errors="coerce")
        / pd.to_numeric(frame["active_days"], errors="coerce")
    )
    frame["dominant_vertical_share"] = pd.to_numeric(frame["dominant_vertical_share"], errors="coerce").fillna(0.0)
    frame["manual_excluded"] = frame["wallet_id"].isin(MANUAL_EXCLUDE_WALLETS)
    return frame


def _style_candidate_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the slow-copy style mask shared by strict and monitor lists."""

    return (
        (pd.to_numeric(frame["avg_copy_edge_net_30s"], errors="coerce") > 0.01)
        & (pd.to_numeric(frame["fast_exit_share_30s"], errors="coerce").fillna(1.0) <= 0.05)
        & (pd.to_numeric(frame["median_holding_seconds"], errors="coerce") > 3600.0)
        & (pd.to_numeric(frame["rolling_3d_positive_share"], errors="coerce").fillna(0.0) >= 0.40)
        & (pd.to_numeric(frame["recent_trades_window"], errors="coerce") >= 5)
        & (pd.to_numeric(frame["recent_trades_window"], errors="coerce") <= 250)
        & (pd.to_numeric(frame["sell_share"], errors="coerce").fillna(1.0) <= 0.40)
        & (pd.to_numeric(frame["open_share"], errors="coerce").fillna(0.0) >= 0.60)
        & (~frame["manual_excluded"])
    )


def score_watchlist(frame: pd.DataFrame) -> pd.DataFrame:
    """Add one interpretable ranking score."""

    scored = frame.copy()
    hold_to_expiry_share = pd.to_numeric(scored.get("hold_to_expiry_share_observed"), errors="coerce")
    if not isinstance(hold_to_expiry_share, pd.Series):
        hold_to_expiry_share = pd.Series(0.0, index=scored.index, dtype=float)
    hold_to_expiry_share = hold_to_expiry_share.fillna(0.0)
    held_to_expiry_count = pd.to_numeric(scored.get("held_to_expiry_observed_slices"), errors="coerce")
    if not isinstance(held_to_expiry_count, pd.Series):
        held_to_expiry_count = pd.Series(0.0, index=scored.index, dtype=float)
    held_to_expiry_count = held_to_expiry_count.fillna(0.0)
    if "expiry_exit_share_30s" in scored.columns:
        expiry_share_30s = pd.to_numeric(scored["expiry_exit_share_30s"], errors="coerce").fillna(0.0)
    else:
        expiry_share_30s = pd.Series(0.0, index=scored.index, dtype=float)
    scored["watchlist_score"] = (
        pd.to_numeric(scored["avg_copy_edge_net_30s"], errors="coerce").fillna(0.0) * 20.0
        + pd.to_numeric(scored["open_share"], errors="coerce").fillna(0.0) * 2.0
        + hold_to_expiry_share * 3.0
        + held_to_expiry_count.clip(upper=25.0) * 0.04
        + expiry_share_30s * 2.5
        + pd.to_numeric(scored["holding_days"], errors="coerce").fillna(0.0).clip(upper=14.0) * 0.15
        + pd.to_numeric(scored["dominant_vertical_share"], errors="coerce").fillna(0.0) * 1.5
        - pd.to_numeric(scored["sell_share"], errors="coerce").fillna(1.0) * 2.0
        - pd.to_numeric(scored["avg_trades_per_active_day_observed"], errors="coerce").fillna(999.0) * 0.03
    )
    return scored


def build_strict_watchlist(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the current-active strict watchlist."""

    if frame.empty:
        return pd.DataFrame()
    strict = score_watchlist(frame[_style_candidate_mask(frame) & frame["is_current_active"]].copy())
    return strict.sort_values(
        ["watchlist_score", "avg_copy_edge_net_30s", "wallet_id"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_monitor_watchlist(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the monitor list: style-compatible but not currently confirmed active."""

    if frame.empty:
        return pd.DataFrame()
    monitor = score_watchlist(frame[_style_candidate_mask(frame) & (~frame["is_current_active"])].copy())
    return monitor.sort_values(
        ["watchlist_score", "avg_copy_edge_net_30s", "wallet_id"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_summary_markdown(strict_watchlist: pd.DataFrame, monitor_watchlist: pd.DataFrame) -> str:
    """Render a concise watchlist summary."""

    lines = [
        "# Active Copyable Watchlist",
        "",
        "This watchlist targets currently active wallets that look followable under",
        "a slower-hold, positive-30s-edge style filter.",
        "",
        "## Rules",
        "",
        "- Current active confirmation comes from the expanded current-market scan (`top2000`).",
        "- `avg_copy_edge_net_30s > 0.01`",
        "- `fast_exit_share_30s <= 5%`",
        "- `median_holding_seconds > 1 hour`",
        "- `rolling_3d_positive_share >= 0.40`",
        "- `5 <= recent_trades_window <= 250`",
        "- `sell_share <= 40%`",
        "- `open_share >= 60%`",
        "- Tracked and score-weighted: `held_to_expiry_observed_slices` and `hold_to_expiry_share_observed`",
        "- Tracked and score-weighted: `expiry_exit_share_30s`",
        "- Manually excluded wallets stay excluded.",
        "",
        "## Strict Active Watchlist",
        "",
    ]

    if strict_watchlist.empty:
        lines.append("- No wallets passed the strict active screen.")
    else:
        for row in strict_watchlist.to_dict(orient="records"):
            lines.append(
                f"- `{row['wallet_id']}` ({row.get('sample_name') or 'no name'}) | "
                f"vertical `{row.get('dominant_vertical')}` | "
                f"edge30 `{row.get('avg_copy_edge_net_30s')}` | "
                f"sell `{row.get('sell_share')}` | "
                f"open `{row.get('open_share')}` | "
                f"hold_to_expiry `{row.get('held_to_expiry_observed_slices')}` | "
                f"unresolved `{row.get('unresolved_open_slices')}` | "
                f"hold_to_expiry_share `{row.get('hold_to_expiry_share_observed')}` | "
                f"expiry30 `{row.get('expiry_exit_share_30s')}` | "
                f"hold_days `{row.get('holding_days')}` | "
                f"score `{row.get('watchlist_score')}`"
            )

    lines.extend(["", "## Monitor List", ""])
    if monitor_watchlist.empty:
        lines.append("- No additional style-compatible wallets were found outside the current active screen.")
    else:
        for row in monitor_watchlist.head(10).to_dict(orient="records"):
            lines.append(
                f"- `{row['wallet_id']}` ({row.get('sample_name') or 'no name'}) | "
                f"vertical `{row.get('dominant_vertical')}` | "
                f"edge30 `{row.get('avg_copy_edge_net_30s')}` | "
                f"sell `{row.get('sell_share')}` | "
                f"open `{row.get('open_share')}` | "
                f"hold_to_expiry `{row.get('held_to_expiry_observed_slices')}` | "
                f"unresolved `{row.get('unresolved_open_slices')}` | "
                f"hold_to_expiry_share `{row.get('hold_to_expiry_share_observed')}` | "
                f"expiry30 `{row.get('expiry_exit_share_30s')}` | "
                f"hold_days `{row.get('holding_days')}` | "
                f"score `{row.get('watchlist_score')}`"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The strict list is the one to act on first for manual review.",
            "- The monitor list contains wallets whose style fits, but whose current activity was not confirmed inside the expanded active-market scan.",
            "- `held_to_expiry_observed_slices = 0` does not mean the wallet exits early; many current-event trades still sit in `unresolved_open_slices` and have not reached resolution by the sample cutoff.",
        ]
    )
    return "\n".join(lines)


def run_active_watchlist(
    *,
    features_csv: str | Path = DEFAULT_FEATURES_CSV,
    current_active_csv: str | Path = DEFAULT_CURRENT_ACTIVE_CSV,
    vertical_summary_csv: str | Path = DEFAULT_VERTICAL_SUMMARY_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build and export the active copyable watchlist."""

    frame = load_watchlist_frame(
        features_csv=features_csv,
        current_active_csv=current_active_csv,
        vertical_summary_csv=vertical_summary_csv,
    )
    strict_watchlist = build_strict_watchlist(frame)
    monitor_watchlist = build_monitor_watchlist(frame)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "watchlist_summary.md"
    summary_path.write_text(
        build_summary_markdown(strict_watchlist, monitor_watchlist),
        encoding="utf-8",
    )

    paths = {
        "strict_watchlist": _write_csv(strict_watchlist, output_root / "strict_active_watchlist.csv"),
        "monitor_watchlist": _write_csv(monitor_watchlist, output_root / "monitor_watchlist.csv"),
        "summary": summary_path,
    }
    return {
        "strict_watchlist": strict_watchlist,
        "monitor_watchlist": monitor_watchlist,
        "paths": paths,
    }


def print_active_watchlist_summary(results: dict[str, Any]) -> None:
    """Print a concise terminal summary."""

    strict_watchlist = results["strict_watchlist"]
    monitor_watchlist = results["monitor_watchlist"]
    print("Active Copyable Watchlist")
    print(f"Strict active wallets: {len(strict_watchlist)}")
    print(f"Monitor wallets: {len(monitor_watchlist)}")
    if not strict_watchlist.empty:
        columns = [
            "wallet_id",
            "sample_name",
            "dominant_vertical",
            "avg_copy_edge_net_30s",
            "sell_share",
            "open_share",
            "held_to_expiry_observed_slices",
            "unresolved_open_slices",
            "resolved_behavior_slices",
            "hold_to_expiry_share_observed",
            "expiry_exit_share_30s",
            "holding_days",
            "watchlist_score",
        ]
        columns = [column for column in columns if column in strict_watchlist.columns]
        print(
            strict_watchlist[columns].to_string(index=False)
        )
