"""First-pass wallet labeling built from existing research exports.

This module intentionally stays in the "research / explainable rules" layer.
It does not add trading or execution logic. Labels are produced as-of the end
of the currently observed sample and should be treated as descriptive heuristics,
not live-ready predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import numpy as np
import pandas as pd

from db.session import get_session
from research.copy_follow_expiry import _build_terminal_lookup, load_markets_frame
from research.copy_follow_wallet_exit import build_copy_exit_pairs
from research.costs import estimate_entry_only_cost


DEFAULT_UNIVERSE_CSV = (
    "exports/current_market_wallet_scan_top1000/wallets_100plus_recent_20260312_20260326.csv"
)
DEFAULT_RECENT_SUMMARY_CSV = (
    "exports/recent_wallet_trade_capture_top1000_cohort/recent_wallet_trade_summary_20260312_20260326.csv"
)
DEFAULT_RECENT_TRADES_CSV = (
    "exports/recent_wallet_trade_capture_top1000_cohort/recent_wallet_trades_20260312_20260326.csv"
)
DEFAULT_REALIZED_WALLET_SUMMARY_CSV = (
    "exports/recent_wallet_realized_pnl_only/recent_wallet_realized_closed_pnl_wallet_summary.csv"
)
DEFAULT_REALIZED_TRADES_CSV = (
    "exports/recent_wallet_realized_pnl_only/recent_wallet_realized_closed_pnl_trades.csv"
)
DEFAULT_COPY_EXIT_SUMMARY_CSV = (
    "exports/copy_follow_wallet_exit_recent_closed_realized_sql/"
    "copy_follow_wallet_exit_recent_wallet_summary_5s_10s_15s_30s_20260312_20260326.csv"
)
DEFAULT_HALF_FORWARD_CSV = (
    "exports/per_wallet_half_forward/wallet_half_forward_results_15s_30s_60s.csv"
)
DEFAULT_REPEAT_POSITIVE_CSV = (
    "exports/follow_wallet_repeat_oos/wallet_repeat_positive_summary.csv"
)
DEFAULT_FEATURES_CSV = "data/wallet_features.csv"
DEFAULT_LABELS_CSV = "data/wallet_first_pass_labels.csv"
DEFAULT_REPORT_MD = "reports/wallet_labeling_summary.md"


@dataclass(frozen=True)
class LabelThresholds:
    """Interpretable first-pass rule thresholds."""

    min_recent_trades_for_medium_confidence: int = 50
    min_realized_trades_for_medium_confidence: int = 10
    min_delay_slices_for_copyable: int = 10
    positive_edge_min: float = 0.0
    positive_consistency_min: float = 0.55
    positive_win_rate_min: float = 0.52
    positive_retention_30_min: float = 0.25
    hft_retention_30_max: float = 0.25
    hft_retention_60_max: float = 0.10
    hft_median_holding_seconds_max: float = 60.0
    hft_quick_close_60s_min: float = 0.50
    hft_fast_exit_share_30_min: float = 0.35
    hft_burstiness_min: float = 1.5
    yolo_top1_share_min: float = 0.50
    yolo_top5_share_min: float = 0.85
    yolo_drawdown_pct_min: float = 0.80
    yolo_position_cv_min: float = 1.50
    yolo_consistency_max: float = 0.35
    yolo_active_days_max: int = 3


def _read_csv(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read one CSV while tolerating missing files."""

    if not path.exists():
        return pd.DataFrame(columns=usecols or [])
    return pd.read_csv(path, usecols=usecols)


def _normalize_wallet_column(frame: pd.DataFrame, column: str = "wallet_address") -> pd.DataFrame:
    """Normalize wallet ids to lowercase hex strings."""

    if frame.empty or column not in frame.columns:
        return frame
    normalized = frame.copy()
    normalized[column] = normalized[column].astype(str).str.strip().str.lower()
    normalized = normalized.loc[normalized[column] != ""].copy()
    return normalized


def _safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """Return one ratio or None when it is not usable."""

    if numerator is None or denominator is None:
        return None
    denominator_float = float(denominator)
    if denominator_float == 0:
        return None
    return float(numerator) / denominator_float


def _safe_float(value: Any) -> float | None:
    """Convert one scalar into float or None."""

    series = pd.to_numeric(pd.Series([value]), errors="coerce")
    if series.empty or pd.isna(series.iloc[0]):
        return None
    return float(series.iloc[0])


def _nonnull_mean(series: pd.Series) -> float | None:
    """Return a mean only when values exist."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _nonnull_median(series: pd.Series) -> float | None:
    """Return a median only when values exist."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.median())


def _nonnull_std(series: pd.Series) -> float | None:
    """Return a standard deviation only when values exist."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.std(ddof=0))


def _pnl_concentration(group: pd.DataFrame, n: int) -> float | None:
    """Share of positive realized PnL contributed by the top-N winning trades."""

    profits = pd.to_numeric(group["realized_pnl_abs"], errors="coerce")
    positive = profits.loc[profits > 0].sort_values(ascending=False)
    if positive.empty:
        return None
    return float(positive.head(n).sum() / positive.sum())


def _compute_max_drawdown(group: pd.DataFrame) -> tuple[float | None, float | None]:
    """Compute absolute and relative max drawdown from realized trade PnL."""

    ordered = group.sort_values("sell_timestamp").copy()
    pnl = pd.to_numeric(ordered["realized_pnl_abs"], errors="coerce").fillna(0.0)
    if pnl.empty:
        return None, None

    cumulative = pnl.cumsum()
    running_peak = cumulative.cummax()
    drawdown = running_peak - cumulative
    max_drawdown_abs = float(drawdown.max()) if not drawdown.empty else None

    positive_peaks = running_peak.loc[running_peak > 0]
    if positive_peaks.empty or max_drawdown_abs is None:
        return max_drawdown_abs, None
    max_drawdown_pct = float(max_drawdown_abs / positive_peaks.max())
    return max_drawdown_abs, max_drawdown_pct


def _compute_rolling_consistency(group: pd.DataFrame) -> tuple[float | None, float | None]:
    """Chronological realized-PnL consistency without peeking past sample end.

    The metric is intentionally simple:
    - `positive_realized_day_share`: fraction of active realized days with positive PnL.
    - `rolling_3d_positive_share`: fraction of 3-day rolling sums above zero.
    """

    ordered = group.sort_values("sell_timestamp").copy()
    if ordered.empty:
        return None, None

    ordered["sell_date"] = pd.to_datetime(ordered["sell_timestamp"], utc=True, errors="coerce").dt.floor("D")
    daily = (
        ordered.dropna(subset=["sell_date"])
        .groupby("sell_date", as_index=False)["realized_pnl_abs"]
        .sum()
        .sort_values("sell_date")
    )
    if daily.empty:
        return None, None

    positive_day_share = float((daily["realized_pnl_abs"] > 0).mean())
    if len(daily) >= 3:
        rolling = daily["realized_pnl_abs"].rolling(3).sum().dropna()
        rolling_share = float((rolling > 0).mean()) if not rolling.empty else None
    else:
        rolling_share = positive_day_share
    return positive_day_share, rolling_share


def _compute_trade_activity_features(recent_trades: pd.DataFrame) -> pd.DataFrame:
    """Compute wallet-level activity, burstiness, and size features."""

    if recent_trades.empty:
        return pd.DataFrame(columns=["wallet_address"])

    frame = recent_trades.copy()
    frame["wallet_address"] = frame["wallet_address"].astype(str).str.lower()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["usdc_size"] = pd.to_numeric(frame["usdc_size"], errors="coerce").abs()
    frame = frame.dropna(subset=["wallet_address", "timestamp"])
    if frame.empty:
        return pd.DataFrame(columns=["wallet_address"])

    frame["trade_date"] = frame["timestamp"].dt.floor("D")
    frame["trade_minute"] = frame["timestamp"].dt.floor("min")

    records: list[dict[str, Any]] = []
    for wallet, group in frame.groupby("wallet_address", sort=False):
        minute_counts = group.groupby("trade_minute").size().astype(float)
        mean_trades_per_active_minute = float(minute_counts.mean()) if not minute_counts.empty else None
        max_trades_per_minute = float(minute_counts.max()) if not minute_counts.empty else None
        trades_per_minute_window = None
        if not group.empty:
            span_seconds = max(
                float((group["timestamp"].max() - group["timestamp"].min()).total_seconds()),
                60.0,
            )
            trades_per_minute_window = float(len(group) / (span_seconds / 60.0))

        position_sizes = pd.to_numeric(group["usdc_size"], errors="coerce").dropna()
        mean_position_size = float(position_sizes.mean()) if not position_sizes.empty else None
        std_position_size = float(position_sizes.std(ddof=0)) if not position_sizes.empty else None
        position_size_cv = (
            float(std_position_size / mean_position_size)
            if mean_position_size not in (None, 0.0) and std_position_size is not None
            else None
        )
        burstiness_cv = (
            float(minute_counts.std(ddof=0) / minute_counts.mean())
            if not minute_counts.empty and float(minute_counts.mean()) > 0
            else None
        )

        records.append(
            {
                "wallet_address": wallet,
                "recent_trades_window": int(len(group)),
                "active_days": int(group["trade_date"].nunique()),
                "mean_trades_per_active_minute": mean_trades_per_active_minute,
                "max_trades_per_minute": max_trades_per_minute,
                "trades_per_minute_window": trades_per_minute_window,
                "trade_burstiness_cv": burstiness_cv,
                "avg_position_size_usdc": mean_position_size,
                "median_position_size_usdc": float(position_sizes.median()) if not position_sizes.empty else None,
                "position_size_std_usdc": std_position_size,
                "position_size_cv": position_size_cv,
            }
        )

    return pd.DataFrame.from_records(records)


def _compute_zero_delay_features(realized_trades: pd.DataFrame) -> pd.DataFrame:
    """Compute 0-second copy-follow edge and realized-trade stability features."""

    if realized_trades.empty:
        return pd.DataFrame(columns=["wallet_address"])

    frame = realized_trades.copy()
    frame["wallet_address"] = frame["wallet_address"].astype(str).str.lower()
    frame["buy_timestamp"] = pd.to_datetime(frame["buy_timestamp"], utc=True, errors="coerce")
    frame["sell_timestamp"] = pd.to_datetime(frame["sell_timestamp"], utc=True, errors="coerce")
    frame["buy_price_signal"] = pd.to_numeric(frame["buy_price_signal"], errors="coerce")
    frame["sell_price_signal"] = pd.to_numeric(frame["sell_price_signal"], errors="coerce")
    frame["copied_size"] = pd.to_numeric(frame["copied_size"], errors="coerce")
    frame["realized_pnl_abs"] = pd.to_numeric(frame["realized_pnl_abs"], errors="coerce")
    frame["holding_seconds"] = pd.to_numeric(frame["holding_seconds"], errors="coerce")

    buy_cost_per_unit = frame.apply(
        lambda row: estimate_entry_only_cost(
            pd.Series(
                {
                    "spread_at_trade": row.get("spread_at_trade"),
                    "slippage_bps_assumed": row.get("slippage_bps_assumed"),
                    "liquidity_bucket": row.get("liquidity_bucket"),
                }
            ),
            entry_price=float(row["buy_price_signal"]) if pd.notna(row["buy_price_signal"]) else None,
        )["total_cost"],
        axis=1,
    )
    sell_cost_per_unit = frame.apply(
        lambda row: estimate_entry_only_cost(
            pd.Series(
                {
                    "spread_at_trade": row.get("spread_at_trade"),
                    "slippage_bps_assumed": row.get("slippage_bps_assumed"),
                    "liquidity_bucket": row.get("liquidity_bucket"),
                }
            ),
            entry_price=float(row["sell_price_signal"]) if pd.notna(row["sell_price_signal"]) else None,
        )["total_cost"],
        axis=1,
    )

    frame["copy_edge_raw_0s"] = frame["sell_price_signal"] - frame["buy_price_signal"]
    frame["copy_cost_0s_per_unit"] = buy_cost_per_unit + sell_cost_per_unit
    frame["copy_edge_net_0s"] = frame["copy_edge_raw_0s"] - frame["copy_cost_0s_per_unit"]
    frame["copy_edge_net_usdc_0s"] = frame["realized_pnl_abs"] - (
        frame["copy_cost_0s_per_unit"] * frame["copied_size"]
    )

    records: list[dict[str, Any]] = []
    for wallet, group in frame.groupby("wallet_address", sort=False):
        max_drawdown_abs, max_drawdown_pct = _compute_max_drawdown(group)
        positive_day_share, rolling_3d_positive_share = _compute_rolling_consistency(group)
        realized_win_rate = float((pd.to_numeric(group["realized_pnl_abs"], errors="coerce") > 0).mean())

        records.append(
            {
                "wallet_address": wallet,
                "realized_closed_trades": int(len(group)),
                "avg_holding_seconds": _nonnull_mean(group["holding_seconds"]),
                "median_holding_seconds": _nonnull_median(group["holding_seconds"]),
                "share_closed_within_10s": float((group["holding_seconds"] <= 10).mean()),
                "share_closed_within_30s": float((group["holding_seconds"] <= 30).mean()),
                "share_closed_within_60s": float((group["holding_seconds"] <= 60).mean()),
                "avg_copy_edge_raw_0s": _nonnull_mean(group["copy_edge_raw_0s"]),
                "avg_copy_edge_net_0s": _nonnull_mean(group["copy_edge_net_0s"]),
                "total_copy_edge_net_usdc_0s": float(
                    pd.to_numeric(group["copy_edge_net_usdc_0s"], errors="coerce").fillna(0.0).sum()
                ),
                "realized_win_rate": realized_win_rate,
                "max_drawdown_abs": max_drawdown_abs,
                "max_drawdown_pct_of_peak": max_drawdown_pct,
                "pnl_concentration_top1_share": _pnl_concentration(group, 1),
                "pnl_concentration_top5_share": _pnl_concentration(group, 5),
                "pnl_concentration_top10_share": _pnl_concentration(group, 10),
                "positive_realized_day_share": positive_day_share,
                "rolling_3d_positive_share": rolling_3d_positive_share,
            }
        )

    return pd.DataFrame.from_records(records)


def _compute_60s_proxy(half_forward: pd.DataFrame) -> pd.DataFrame:
    """Approximate 60s full-window edge from train/test halves.

    Assumption:
    - This is a lower-fidelity proxy because midpoint splits do not preserve
      positions that open in the first half and close in the second half.
    - It is still useful as a first-pass latency-sensitivity indicator.
    """

    if half_forward.empty:
        return pd.DataFrame(columns=["wallet_address"])

    frame = half_forward.copy()
    frame["wallet_address"] = frame["wallet_address"].astype(str).str.lower()
    numeric_columns = [
        "valid_copy_slices_60s_train",
        "valid_copy_slices_60s_test",
        "avg_copy_pnl_60s_train",
        "avg_copy_pnl_60s_test",
        "avg_copy_pnl_net_60s_train",
        "avg_copy_pnl_net_60s_test",
        "total_copy_pnl_net_usdc_60s_train",
        "total_copy_pnl_net_usdc_60s_test",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    total_valid = frame["valid_copy_slices_60s_train"] + frame["valid_copy_slices_60s_test"]
    weighted_raw = (
        frame["avg_copy_pnl_60s_train"] * frame["valid_copy_slices_60s_train"]
        + frame["avg_copy_pnl_60s_test"] * frame["valid_copy_slices_60s_test"]
    )
    weighted_net = (
        frame["avg_copy_pnl_net_60s_train"] * frame["valid_copy_slices_60s_train"]
        + frame["avg_copy_pnl_net_60s_test"] * frame["valid_copy_slices_60s_test"]
    )
    total_net_usdc = frame["total_copy_pnl_net_usdc_60s_train"] + frame["total_copy_pnl_net_usdc_60s_test"]

    proxy = pd.DataFrame(
        {
            "wallet_address": frame["wallet_address"],
            "valid_copy_slices_60s_proxy": total_valid.astype(int),
            "avg_copy_edge_raw_60s_proxy": np.where(total_valid > 0, weighted_raw / total_valid, np.nan),
            "avg_copy_edge_net_60s_proxy": np.where(total_valid > 0, weighted_net / total_valid, np.nan),
            "total_copy_edge_net_usdc_60s_proxy": total_net_usdc,
        }
    )
    return proxy


def _compute_repeat_oos_features(repeat_positive: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated-OOS summary into one row per wallet."""

    if repeat_positive.empty:
        return pd.DataFrame(columns=["wallet_address"])

    frame = repeat_positive.copy()
    frame["wallet_address"] = frame["wallet_address"].astype(str).str.lower()
    frame["selected_train_windows"] = pd.to_numeric(frame["selected_train_windows"], errors="coerce").fillna(0)
    frame["test_positive_windows"] = pd.to_numeric(frame["test_positive_windows"], errors="coerce").fillna(0)
    frame["test_valid_slices_total"] = pd.to_numeric(frame["test_valid_slices_total"], errors="coerce").fillna(0)
    frame["test_net_usdc_total"] = pd.to_numeric(frame["test_net_usdc_total"], errors="coerce").fillna(0.0)
    frame["test_positive_rate_when_selected"] = pd.to_numeric(
        frame["test_positive_rate_when_selected"], errors="coerce"
    )

    records: list[dict[str, Any]] = []
    for wallet, group in frame.groupby("wallet_address", sort=False):
        best = group.sort_values(
            ["test_positive_windows", "test_net_usdc_total", "selected_train_windows"],
            ascending=[False, False, False],
        ).iloc[0]
        records.append(
            {
                "wallet_address": wallet,
                "repeat_oos_best_delay": best.get("delay"),
                "repeat_oos_selected_train_windows": int(best["selected_train_windows"]),
                "repeat_oos_test_positive_windows": int(best["test_positive_windows"]),
                "repeat_oos_test_valid_slices_total": int(best["test_valid_slices_total"]),
                "repeat_oos_test_net_usdc_total": float(best["test_net_usdc_total"]),
                "repeat_oos_positive_rate": _safe_float(best.get("test_positive_rate_when_selected")),
                "repeat_oos_label": best.get("repeated_positive_label"),
            }
        )

    return pd.DataFrame.from_records(records)


def _merge_delay_features(copy_summary: pd.DataFrame) -> pd.DataFrame:
    """Rename existing delayed copy-follow wallet metrics into feature-friendly names."""

    if copy_summary.empty:
        return pd.DataFrame(columns=["wallet_address"])

    frame = copy_summary.copy()
    frame["wallet_address"] = frame["wallet_address"].astype(str).str.lower()
    keep_columns = {
        "valid_copy_slices_5s": "valid_copy_slices_5s",
        "valid_copy_slices_15s": "valid_copy_slices_15s",
        "valid_copy_slices_30s": "valid_copy_slices_30s",
        "avg_copy_pnl_5s": "avg_copy_edge_raw_5s",
        "avg_copy_pnl_net_5s": "avg_copy_edge_net_5s",
        "avg_copy_pnl_15s": "avg_copy_edge_raw_15s",
        "avg_copy_pnl_net_15s": "avg_copy_edge_net_15s",
        "avg_copy_pnl_30s": "avg_copy_edge_raw_30s",
        "avg_copy_pnl_net_30s": "avg_copy_edge_net_30s",
        "copy_hit_rate_5s": "copy_hit_rate_5s",
        "copy_net_hit_rate_5s": "copy_net_hit_rate_5s",
        "copy_hit_rate_15s": "copy_hit_rate_15s",
        "copy_net_hit_rate_15s": "copy_net_hit_rate_15s",
        "copy_hit_rate_30s": "copy_hit_rate_30s",
        "copy_net_hit_rate_30s": "copy_net_hit_rate_30s",
        "total_copy_pnl_net_usdc_5s": "total_copy_edge_net_usdc_5s",
        "total_copy_pnl_net_usdc_15s": "total_copy_edge_net_usdc_15s",
        "total_copy_pnl_net_usdc_30s": "total_copy_edge_net_usdc_30s",
        "paired_copy_slices": "paired_copy_slices",
        "pending_open_copy_slices": "pending_open_copy_slices",
        "wallet_sell_exits_15s": "wallet_sell_exits_15s",
        "wallet_sell_exits_30s": "wallet_sell_exits_30s",
        "expiry_exits_15s": "expiry_exits_15s",
        "expiry_exits_30s": "expiry_exits_30s",
        "pending_positions_15s": "pending_positions_15s",
        "pending_positions_30s": "pending_positions_30s",
        "wallet_exit_before_entry_15s": "wallet_exit_before_entry_15s",
        "wallet_exit_before_entry_30s": "wallet_exit_before_entry_30s",
        "entry_after_or_at_delayed_exit_15s": "entry_after_or_at_delayed_exit_15s",
        "entry_after_or_at_delayed_exit_30s": "entry_after_or_at_delayed_exit_30s",
        "missing_prices_15s": "missing_prices_15s",
        "missing_prices_30s": "missing_prices_30s",
    }
    available_columns = ["wallet_address", *[column for column in keep_columns if column in frame.columns]]
    trimmed = frame[available_columns].rename(columns=keep_columns)

    paired = pd.to_numeric(trimmed.get("paired_copy_slices"), errors="coerce")
    if paired is not None:
        trimmed["valid_copy_share_15s"] = trimmed.apply(
            lambda row: _safe_ratio(row.get("valid_copy_slices_15s"), row.get("paired_copy_slices")), axis=1
        )
        trimmed["valid_copy_share_30s"] = trimmed.apply(
            lambda row: _safe_ratio(row.get("valid_copy_slices_30s"), row.get("paired_copy_slices")), axis=1
        )
        trimmed["fast_exit_share_15s"] = trimmed.apply(
            lambda row: _safe_ratio(
                row.get("wallet_exit_before_entry_15s", 0) + row.get("entry_after_or_at_delayed_exit_15s", 0),
                row.get("paired_copy_slices"),
            ),
            axis=1,
        )
        trimmed["fast_exit_share_30s"] = trimmed.apply(
            lambda row: _safe_ratio(
                row.get("wallet_exit_before_entry_30s", 0) + row.get("entry_after_or_at_delayed_exit_30s", 0),
                row.get("paired_copy_slices"),
            ),
            axis=1,
        )
        trimmed["resolved_copy_exits_15s"] = trimmed.apply(
            lambda row: (row.get("wallet_sell_exits_15s") or 0) + (row.get("expiry_exits_15s") or 0),
            axis=1,
        )
        trimmed["resolved_copy_exits_30s"] = trimmed.apply(
            lambda row: (row.get("wallet_sell_exits_30s") or 0) + (row.get("expiry_exits_30s") or 0),
            axis=1,
        )
        trimmed["expiry_exit_share_15s"] = trimmed.apply(
            lambda row: _safe_ratio(row.get("expiry_exits_15s"), row.get("resolved_copy_exits_15s")),
            axis=1,
        )
        trimmed["expiry_exit_share_30s"] = trimmed.apply(
            lambda row: _safe_ratio(row.get("expiry_exits_30s"), row.get("resolved_copy_exits_30s")),
            axis=1,
        )
    return trimmed


def _compute_hold_to_expiry_features(recent_trades: pd.DataFrame) -> pd.DataFrame:
    """Estimate how often wallets hold buys through market resolution.

    This is a wallet-behavior metric, not a copy-PnL metric.

    Approximation:
    - Build FIFO buy lots and match later sells in the same wallet + token.
    - Remaining open lots are treated as "held to expiry" only when public
      market metadata shows the market resolved by the sample cutoff.
    - Open lots on unresolved markets stay in `unresolved_open_slices`.
    """

    if recent_trades.empty:
        return pd.DataFrame(columns=["wallet_address"])

    try:
        with get_session() as session:
            terminal_lookup = _build_terminal_lookup(load_markets_frame(session))
    except Exception:
        return pd.DataFrame(columns=["wallet_address"])

    pairs, open_positions = build_copy_exit_pairs(recent_trades)
    cutoff = pd.to_datetime(recent_trades["timestamp"], utc=True, errors="coerce").max()
    if cutoff is None or pd.isna(cutoff):
        return pd.DataFrame(columns=["wallet_address"])

    paired_by_wallet: dict[str, float] = {}
    if not pairs.empty:
        paired_by_wallet = (
            pairs.groupby("wallet_address", sort=False)["copied_size"].size().astype(float).to_dict()
        )

    open_enriched = open_positions.copy()
    if not open_enriched.empty:
        open_enriched["resolution_ts"] = open_enriched["token_id"].map(
            lambda token: (
                terminal_lookup.get(str(token), {}).get("resolution_ts")
                if token is not None
                else None
            )
        )
        open_enriched["resolution_ts"] = pd.to_datetime(open_enriched["resolution_ts"], utc=True, errors="coerce")
        open_enriched["held_to_expiry_observed"] = open_enriched["resolution_ts"].notna() & (
            open_enriched["resolution_ts"] <= cutoff
        )
    else:
        open_enriched["held_to_expiry_observed"] = pd.Series(dtype=bool)

    wallet_ids = sorted(
        set(paired_by_wallet)
        | set(open_enriched.get("wallet_address", pd.Series(dtype=str)).astype(str).str.lower().tolist())
    )
    records: list[dict[str, Any]] = []
    for wallet in wallet_ids:
        wallet_pairs = float(paired_by_wallet.get(wallet, 0.0))
        if open_enriched.empty:
            wallet_open = open_enriched
        else:
            wallet_open = open_enriched.loc[open_enriched["wallet_address"].astype(str).str.lower() == wallet].copy()
        held_count = int(wallet_open["held_to_expiry_observed"].sum()) if not wallet_open.empty else 0
        unresolved_count = int((~wallet_open["held_to_expiry_observed"]).sum()) if not wallet_open.empty else 0
        total_buy_slices = wallet_pairs + held_count + unresolved_count
        resolved_slices = wallet_pairs + held_count
        records.append(
            {
                "wallet_address": wallet,
                "wallet_sell_closed_slices": int(wallet_pairs),
                "held_to_expiry_observed_slices": held_count,
                "unresolved_open_slices": unresolved_count,
                "resolved_behavior_slices": int(resolved_slices),
                "total_behavior_slices": int(total_buy_slices),
                "hold_to_expiry_share_observed": _safe_ratio(held_count, resolved_slices),
                "hold_to_expiry_share_all_slices": _safe_ratio(held_count, total_buy_slices),
            }
        )

    return pd.DataFrame.from_records(records)


def build_wallet_features(project_root: str | Path) -> pd.DataFrame:
    """Build one row per wallet from current research exports."""

    root = Path(project_root)
    universe = _normalize_wallet_column(_read_csv(root / DEFAULT_UNIVERSE_CSV))
    recent_summary = _normalize_wallet_column(_read_csv(root / DEFAULT_RECENT_SUMMARY_CSV))
    recent_trades = _normalize_wallet_column(
        _read_csv(
            root / DEFAULT_RECENT_TRADES_CSV,
            usecols=[
                "trade_id",
                "wallet_address",
                "market_id",
                "token_id",
                "side",
                "price",
                "size",
                "usdc_size",
                "timestamp",
            ],
        )
    )
    realized_wallet_summary = _normalize_wallet_column(_read_csv(root / DEFAULT_REALIZED_WALLET_SUMMARY_CSV))
    realized_trades = _normalize_wallet_column(
        _read_csv(
            root / DEFAULT_REALIZED_TRADES_CSV,
            usecols=[
                "wallet_address",
                "buy_timestamp",
                "sell_timestamp",
                "buy_price_signal",
                "sell_price_signal",
                "copied_size",
                "spread_at_trade",
                "slippage_bps_assumed",
                "liquidity_bucket",
                "realized_pnl_abs",
                "realized_pnl_pct",
                "holding_seconds",
            ],
        )
    )
    copy_summary = _normalize_wallet_column(_read_csv(root / DEFAULT_COPY_EXIT_SUMMARY_CSV))
    half_forward = _normalize_wallet_column(_read_csv(root / DEFAULT_HALF_FORWARD_CSV))
    repeat_positive = _normalize_wallet_column(_read_csv(root / DEFAULT_REPEAT_POSITIVE_CSV))

    features = universe.rename(
        columns={
            "wallet_address": "wallet_id",
            "total_trades": "total_trades_scan_universe",
            "distinct_markets": "distinct_markets_scan_universe",
            "first_seen_trade_ts": "first_seen_trade_ts_scan_universe",
            "most_recent_trade_ts": "most_recent_trade_ts_scan_universe",
        }
    ).copy()
    features["wallet_id"] = features["wallet_id"].astype(str).str.lower()

    activity = _compute_trade_activity_features(recent_trades).rename(columns={"wallet_address": "wallet_id"})
    hold_to_expiry = _compute_hold_to_expiry_features(recent_trades).rename(columns={"wallet_address": "wallet_id"})
    zero_delay = _compute_zero_delay_features(realized_trades).rename(columns={"wallet_address": "wallet_id"})
    delay_features = _merge_delay_features(copy_summary).rename(columns={"wallet_address": "wallet_id"})
    proxy_60 = _compute_60s_proxy(half_forward).rename(columns={"wallet_address": "wallet_id"})
    repeat_features = _compute_repeat_oos_features(repeat_positive).rename(columns={"wallet_address": "wallet_id"})

    if not recent_summary.empty:
        recent_summary = recent_summary.rename(
            columns={
                "wallet_address": "wallet_id",
                "recent_trades": "recent_trades_summary",
                "recent_distinct_markets": "recent_distinct_markets",
                "recent_distinct_tokens": "recent_distinct_tokens",
                "first_recent_trade_ts": "first_recent_trade_ts",
                "most_recent_trade_ts": "most_recent_trade_ts",
                "recent_buy_trades": "recent_buy_trades",
                "recent_sell_trades": "recent_sell_trades",
                "recent_usdc_volume": "recent_usdc_volume",
            }
        )
    if not realized_wallet_summary.empty:
        realized_wallet_summary = realized_wallet_summary.rename(
            columns={
                "wallet_address": "wallet_id",
                "realized_closed_trades": "realized_closed_trades_summary",
                "realized_entry_notional_usdc": "realized_entry_notional_usdc",
                "realized_exit_notional_usdc": "realized_exit_notional_usdc",
                "realized_pnl_abs": "realized_pnl_abs",
                "realized_pnl_pct_est": "realized_pnl_pct_est",
            }
        )

    for frame in (
        recent_summary,
        activity,
        hold_to_expiry,
        realized_wallet_summary,
        zero_delay,
        delay_features,
        proxy_60,
        repeat_features,
    ):
        if frame.empty:
            continue
        features = features.merge(frame, on="wallet_id", how="left")

    features["recent_trades_window"] = features["recent_trades_window"].fillna(features.get("recent_trades_summary"))
    features["recent_trades_window"] = pd.to_numeric(features["recent_trades_window"], errors="coerce")
    features["active_days"] = pd.to_numeric(features["active_days"], errors="coerce")
    features["realized_closed_trades"] = pd.to_numeric(features["realized_closed_trades"], errors="coerce")
    features["total_trades_scan_universe"] = pd.to_numeric(features["total_trades_scan_universe"], errors="coerce")
    features["distinct_markets_scan_universe"] = pd.to_numeric(
        features["distinct_markets_scan_universe"], errors="coerce"
    )

    features["edge_decay_abs_0s_to_30s"] = (
        pd.to_numeric(features["avg_copy_edge_net_30s"], errors="coerce")
        - pd.to_numeric(features["avg_copy_edge_net_0s"], errors="coerce")
    )
    features["edge_retention_30s_from_0"] = features.apply(
        lambda row: _safe_ratio(row.get("avg_copy_edge_net_30s"), row.get("avg_copy_edge_net_0s")),
        axis=1,
    )
    features["edge_decay_abs_0s_to_60s_proxy"] = (
        pd.to_numeric(features["avg_copy_edge_net_60s_proxy"], errors="coerce")
        - pd.to_numeric(features["avg_copy_edge_net_0s"], errors="coerce")
    )
    features["edge_retention_60s_from_0_proxy"] = features.apply(
        lambda row: _safe_ratio(row.get("avg_copy_edge_net_60s_proxy"), row.get("avg_copy_edge_net_0s")),
        axis=1,
    )
    features["unrealized_pnl_abs_est"] = pd.NA
    features["unrealized_pnl_note"] = "not_available_from_current_public_close_only_reconstruction"

    sort_columns = [
        "repeat_oos_test_positive_windows",
        "avg_copy_edge_net_30s",
        "realized_pnl_abs",
        "wallet_id",
    ]
    ascending = [False, False, False, True]
    features = features.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)
    return features


def _label_scores(row: pd.Series, thresholds: LabelThresholds) -> tuple[int, int, int, list[str], list[str], list[str]]:
    """Compute explainable scores for each wallet archetype."""

    positive_score = 0
    hft_score = 0
    yolo_score = 0
    positive_reasons: list[str] = []
    hft_reasons: list[str] = []
    yolo_reasons: list[str] = []

    if _safe_float(row.get("avg_copy_edge_net_15s")) is not None and float(row["avg_copy_edge_net_15s"]) > thresholds.positive_edge_min:
        positive_score += 2
        positive_reasons.append("15s_edge_positive")
    if _safe_float(row.get("avg_copy_edge_net_30s")) is not None and float(row["avg_copy_edge_net_30s"]) > thresholds.positive_edge_min:
        positive_score += 2
        positive_reasons.append("30s_edge_positive")
    if _safe_float(row.get("avg_copy_edge_net_60s_proxy")) is not None and float(row["avg_copy_edge_net_60s_proxy"]) > thresholds.positive_edge_min:
        positive_score += 1
        positive_reasons.append("60s_proxy_edge_positive")
    if _safe_float(row.get("edge_retention_30s_from_0")) is not None and float(row["edge_retention_30s_from_0"]) >= thresholds.positive_retention_30_min:
        positive_score += 1
        positive_reasons.append("delay_retention_30_ok")
    if _safe_float(row.get("rolling_3d_positive_share")) is not None and float(row["rolling_3d_positive_share"]) >= thresholds.positive_consistency_min:
        positive_score += 1
        positive_reasons.append("rolling_consistency_ok")
    if _safe_float(row.get("realized_win_rate")) is not None and float(row["realized_win_rate"]) >= thresholds.positive_win_rate_min:
        positive_score += 1
        positive_reasons.append("win_rate_ok")
    if _safe_float(row.get("pnl_concentration_top1_share")) is not None and float(row["pnl_concentration_top1_share"]) < thresholds.yolo_top1_share_min:
        positive_score += 1
        positive_reasons.append("pnl_not_single_trade_dominated")
    if _safe_float(row.get("max_drawdown_pct_of_peak")) is not None and float(row["max_drawdown_pct_of_peak"]) < thresholds.yolo_drawdown_pct_min:
        positive_score += 1
        positive_reasons.append("drawdown_not_extreme")
    if (
        _safe_float(row.get("valid_copy_slices_15s")) is not None
        and _safe_float(row.get("valid_copy_slices_30s")) is not None
        and float(row["valid_copy_slices_15s"]) >= thresholds.min_delay_slices_for_copyable
        and float(row["valid_copy_slices_30s"]) >= thresholds.min_delay_slices_for_copyable
    ):
        positive_score += 1
        positive_reasons.append("enough_valid_delay_slices")
    if _safe_float(row.get("repeat_oos_test_positive_windows")) is not None and float(row["repeat_oos_test_positive_windows"]) >= 2:
        positive_score += 1
        positive_reasons.append("repeat_oos_positive")

    if _safe_float(row.get("avg_copy_edge_net_0s")) is not None and float(row["avg_copy_edge_net_0s"]) > 0:
        hft_score += 1
        hft_reasons.append("0s_edge_positive")
    if _safe_float(row.get("edge_retention_30s_from_0")) is not None and float(row["edge_retention_30s_from_0"]) <= thresholds.hft_retention_30_max:
        hft_score += 2
        hft_reasons.append("30s_edge_decays_fast")
    if _safe_float(row.get("edge_retention_60s_from_0_proxy")) is not None and float(row["edge_retention_60s_from_0_proxy"]) <= thresholds.hft_retention_60_max:
        hft_score += 1
        hft_reasons.append("60s_proxy_edge_nearly_gone")
    if _safe_float(row.get("median_holding_seconds")) is not None and float(row["median_holding_seconds"]) <= thresholds.hft_median_holding_seconds_max:
        hft_score += 1
        hft_reasons.append("median_holding_very_short")
    if _safe_float(row.get("share_closed_within_60s")) is not None and float(row["share_closed_within_60s"]) >= thresholds.hft_quick_close_60s_min:
        hft_score += 1
        hft_reasons.append("many_positions_closed_within_60s")
    if _safe_float(row.get("fast_exit_share_30s")) is not None and float(row["fast_exit_share_30s"]) >= thresholds.hft_fast_exit_share_30_min:
        hft_score += 2
        hft_reasons.append("wallet_often_exits_before_copy_can_enter")
    if _safe_float(row.get("trade_burstiness_cv")) is not None and float(row["trade_burstiness_cv"]) >= thresholds.hft_burstiness_min:
        hft_score += 1
        hft_reasons.append("bursty_per_minute_activity")

    if _safe_float(row.get("pnl_concentration_top1_share")) is not None and float(row["pnl_concentration_top1_share"]) >= thresholds.yolo_top1_share_min:
        yolo_score += 2
        yolo_reasons.append("single_trade_dominates_profits")
    if _safe_float(row.get("pnl_concentration_top5_share")) is not None and float(row["pnl_concentration_top5_share"]) >= thresholds.yolo_top5_share_min:
        yolo_score += 1
        yolo_reasons.append("few_trades_dominate_profits")
    if _safe_float(row.get("max_drawdown_pct_of_peak")) is not None and float(row["max_drawdown_pct_of_peak"]) >= thresholds.yolo_drawdown_pct_min:
        yolo_score += 2
        yolo_reasons.append("drawdown_extreme")
    if _safe_float(row.get("position_size_cv")) is not None and float(row["position_size_cv"]) >= thresholds.yolo_position_cv_min:
        yolo_score += 1
        yolo_reasons.append("position_size_highly_variable")
    if _safe_float(row.get("rolling_3d_positive_share")) is not None and float(row["rolling_3d_positive_share"]) <= thresholds.yolo_consistency_max:
        yolo_score += 1
        yolo_reasons.append("rolling_consistency_weak")
    if _safe_float(row.get("active_days")) is not None and float(row["active_days"]) <= thresholds.yolo_active_days_max:
        yolo_score += 1
        yolo_reasons.append("few_active_days")
    if _safe_float(row.get("realized_win_rate")) is not None and float(row["realized_win_rate"]) < thresholds.positive_win_rate_min:
        yolo_score += 1
        yolo_reasons.append("win_rate_weak")
    if (
        _safe_float(row.get("avg_copy_edge_net_15s")) is not None
        and _safe_float(row.get("avg_copy_edge_net_30s")) is not None
        and float(row["avg_copy_edge_net_15s"]) <= 0
        and float(row["avg_copy_edge_net_30s"]) <= 0
    ):
        yolo_score += 1
        yolo_reasons.append("delayed_copy_edge_non_positive")

    return positive_score, hft_score, yolo_score, positive_reasons, hft_reasons, yolo_reasons


def assign_first_pass_labels(
    features: pd.DataFrame,
    *,
    thresholds: LabelThresholds | None = None,
) -> pd.DataFrame:
    """Assign explainable first-pass wallet archetype labels."""

    cfg = thresholds or LabelThresholds()
    records: list[dict[str, Any]] = []
    for row in features.to_dict(orient="records"):
        series = pd.Series(row)
        positive_score, hft_score, yolo_score, positive_reasons, hft_reasons, yolo_reasons = _label_scores(
            series, cfg
        )
        recent_trades = _safe_float(series.get("recent_trades_window")) or 0.0
        realized_trades = _safe_float(series.get("realized_closed_trades")) or 0.0
        valid_30 = _safe_float(series.get("valid_copy_slices_30s")) or 0.0

        if (
            recent_trades >= cfg.min_recent_trades_for_medium_confidence
            and realized_trades >= cfg.min_realized_trades_for_medium_confidence
            and valid_30 >= cfg.min_delay_slices_for_copyable
        ):
            label_confidence = "high"
        elif recent_trades >= 20 and realized_trades >= 5 and valid_30 >= 3:
            label_confidence = "medium"
        else:
            label_confidence = "low"

        if positive_score >= 7 and positive_score >= hft_score + 2 and positive_score >= yolo_score + 1:
            primary_label = "positive_ev_copyable"
            primary_reasons = positive_reasons
        elif hft_score >= 5 and hft_score >= positive_score and hft_score >= yolo_score:
            primary_label = "hft_latency_sensitive"
            primary_reasons = hft_reasons
        else:
            primary_label = "yolo_noise_unstable"
            primary_reasons = yolo_reasons

        records.append(
            {
                "wallet_id": row["wallet_id"],
                "sample_name": row.get("sample_name"),
                "sample_pseudonym": row.get("sample_pseudonym"),
                "primary_label": primary_label,
                "label_confidence": label_confidence,
                "positive_ev_score": positive_score,
                "hft_score": hft_score,
                "yolo_score": yolo_score,
                "primary_reasons": "|".join(primary_reasons[:6]),
                "positive_ev_reasons": "|".join(positive_reasons[:8]),
                "hft_reasons": "|".join(hft_reasons[:8]),
                "yolo_reasons": "|".join(yolo_reasons[:8]),
                "recent_trades_window": row.get("recent_trades_window"),
                "active_days": row.get("active_days"),
                "realized_closed_trades": row.get("realized_closed_trades"),
                "avg_copy_edge_net_0s": row.get("avg_copy_edge_net_0s"),
                "avg_copy_edge_net_15s": row.get("avg_copy_edge_net_15s"),
                "avg_copy_edge_net_30s": row.get("avg_copy_edge_net_30s"),
                "avg_copy_edge_net_60s_proxy": row.get("avg_copy_edge_net_60s_proxy"),
                "edge_retention_30s_from_0": row.get("edge_retention_30s_from_0"),
                "edge_retention_60s_from_0_proxy": row.get("edge_retention_60s_from_0_proxy"),
                "median_holding_seconds": row.get("median_holding_seconds"),
                "share_closed_within_60s": row.get("share_closed_within_60s"),
                "fast_exit_share_30s": row.get("fast_exit_share_30s"),
                "realized_pnl_abs": row.get("realized_pnl_abs"),
                "realized_pnl_pct_est": row.get("realized_pnl_pct_est"),
                "max_drawdown_abs": row.get("max_drawdown_abs"),
                "max_drawdown_pct_of_peak": row.get("max_drawdown_pct_of_peak"),
                "pnl_concentration_top1_share": row.get("pnl_concentration_top1_share"),
                "pnl_concentration_top5_share": row.get("pnl_concentration_top5_share"),
                "rolling_3d_positive_share": row.get("rolling_3d_positive_share"),
                "trade_burstiness_cv": row.get("trade_burstiness_cv"),
                "position_size_cv": row.get("position_size_cv"),
                "repeat_oos_best_delay": row.get("repeat_oos_best_delay"),
                "repeat_oos_test_positive_windows": row.get("repeat_oos_test_positive_windows"),
            }
        )

    labels = pd.DataFrame.from_records(records).sort_values(
        ["primary_label", "label_confidence", "positive_ev_score", "hft_score", "yolo_score", "wallet_id"],
        ascending=[True, False, False, False, False, True],
    )
    return labels.reset_index(drop=True)


def _top_rows_for_label(labels: pd.DataFrame, label: str, *, limit: int = 8) -> pd.DataFrame:
    """Return the top rows for one label ordered by relevant scores."""

    subset = labels.loc[labels["primary_label"] == label].copy()
    if subset.empty:
        return subset

    if label == "positive_ev_copyable":
        sort_columns = [
            "label_confidence",
            "positive_ev_score",
            "avg_copy_edge_net_30s",
            "repeat_oos_test_positive_windows",
            "realized_pnl_abs",
        ]
        ascending = [False, False, False, False, False]
    elif label == "hft_latency_sensitive":
        sort_columns = [
            "label_confidence",
            "hft_score",
            "fast_exit_share_30s",
            "share_closed_within_60s",
            "avg_copy_edge_net_0s",
        ]
        ascending = [False, False, False, False, False]
    else:
        sort_columns = [
            "label_confidence",
            "yolo_score",
            "pnl_concentration_top1_share",
            "max_drawdown_pct_of_peak",
            "position_size_cv",
        ]
        ascending = [False, False, False, False, False]

    return subset.sort_values(sort_columns, ascending=ascending).head(limit)


def render_wallet_labeling_summary(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    output_path: str | Path,
) -> Path:
    """Render a concise Markdown summary for manual review."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    counts = labels["primary_label"].value_counts().to_dict()
    confidence_counts = labels["label_confidence"].value_counts().to_dict()

    def format_table(frame: pd.DataFrame, columns: list[str]) -> str:
        if frame.empty:
            return "_None_\n"
        return frame[columns].to_markdown(index=False) + "\n"

    positive_top = _top_rows_for_label(labels, "positive_ev_copyable")
    hft_top = _top_rows_for_label(labels, "hft_latency_sensitive")
    yolo_top = _top_rows_for_label(labels, "yolo_noise_unstable")

    feature_columns = [
        "wallet_id",
        "sample_name",
        "recent_trades_window",
        "active_days",
        "avg_holding_seconds",
        "median_holding_seconds",
        "share_closed_within_10s",
        "share_closed_within_30s",
        "share_closed_within_60s",
        "avg_copy_edge_net_0s",
        "avg_copy_edge_net_5s",
        "avg_copy_edge_net_15s",
        "avg_copy_edge_net_30s",
        "avg_copy_edge_net_60s_proxy",
        "edge_retention_30s_from_0",
        "edge_retention_60s_from_0_proxy",
        "realized_pnl_abs",
        "realized_pnl_pct_est",
        "max_drawdown_abs",
        "max_drawdown_pct_of_peak",
        "pnl_concentration_top1_share",
        "pnl_concentration_top5_share",
        "pnl_concentration_top10_share",
        "realized_win_rate",
        "rolling_3d_positive_share",
        "mean_trades_per_active_minute",
        "trade_burstiness_cv",
        "avg_position_size_usdc",
        "position_size_cv",
        "fast_exit_share_30s",
        "repeat_oos_test_positive_windows",
    ]

    report = f"""# Wallet First-Pass Labeling Summary

## Scope
- Labeling is **time-aware as-of the end of the currently observed sample**, not a backfilled label for earlier dates.
- Current observed sample end: `2026-03-26 23:59:59 UTC`.
- Wallet rows in feature table: `{len(features)}`
- Wallet rows in label table: `{len(labels)}`

## Data Sources Used
- `exports/current_market_wallet_scan_top1000/wallets_100plus_recent_20260312_20260326.csv`
- `exports/recent_wallet_trade_capture_top1000_cohort/recent_wallet_trade_summary_20260312_20260326.csv`
- `exports/recent_wallet_trade_capture_top1000_cohort/recent_wallet_trades_20260312_20260326.csv`
- `exports/recent_wallet_realized_pnl_only/recent_wallet_realized_closed_pnl_wallet_summary.csv`
- `exports/recent_wallet_realized_pnl_only/recent_wallet_realized_closed_pnl_trades.csv`
- `exports/copy_follow_wallet_exit_recent_closed_realized_sql/copy_follow_wallet_exit_recent_wallet_summary_5s_10s_15s_30s_20260312_20260326.csv`
- `exports/per_wallet_half_forward/wallet_half_forward_results_15s_30s_60s.csv`
- `exports/follow_wallet_repeat_oos/wallet_repeat_positive_summary.csv`

## Feature Set
- Wallet identity and scan-universe totals: `wallet_id`, total observed trades, distinct markets, first/last seen.
- Recent activity: active days, recent trades, per-minute trade intensity, burstiness, recent volume.
- Holding behavior: average / median holding time, and close-within-10s / 30s / 60s shares.
- Delay edge: average copy edge at `0s`, `5s`, `15s`, `30s`, plus `60s` proxy from the half-split report.
- Delay decay: absolute and retention-style decay from `0s -> 30s`, and `0s -> 60s proxy`.
- PnL / stability: realized PnL, realized win rate, max drawdown, profit concentration, rolling 3-day positivity share.
- Trade sizing: average size, median size, size standard deviation, size CV.
- Copyability friction: valid delayed slices and fast-exit share at `30s`.

Columns written to `data/wallet_features.csv`:
{", ".join(feature_columns)}

## Rule-Based Label Logic
- `positive_ev_copyable`
  - delayed net edge remains positive at `15s` and `30s`
  - delay retention does not collapse immediately
  - rolling realized consistency and win rate are acceptable
  - profits are not excessively concentrated in one trade
  - drawdown is not extreme
- `hft_latency_sensitive`
  - `0s` edge is positive, but it decays sharply by `30s` / `60s proxy`
  - holding times are very short
  - many positions close before a delayed follower can enter
  - per-minute activity is bursty
- `yolo_noise_unstable`
  - PnL is concentrated in a few trades
  - drawdown is large
  - sizing is erratic or active days are very sparse
  - delayed copy edge is weak or non-positive

These are **first-pass heuristic labels**, not ML predictions.

## Label Counts
- `positive_ev_copyable`: `{counts.get("positive_ev_copyable", 0)}`
- `hft_latency_sensitive`: `{counts.get("hft_latency_sensitive", 0)}`
- `yolo_noise_unstable`: `{counts.get("yolo_noise_unstable", 0)}`

## Confidence Counts
- `high`: `{confidence_counts.get("high", 0)}`
- `medium`: `{confidence_counts.get("medium", 0)}`
- `low`: `{confidence_counts.get("low", 0)}`

## Most Likely Positive EV Wallets
{format_table(
    positive_top,
    [
        "wallet_id",
        "sample_name",
        "label_confidence",
        "positive_ev_score",
        "avg_copy_edge_net_15s",
        "avg_copy_edge_net_30s",
        "avg_copy_edge_net_60s_proxy",
        "edge_retention_30s_from_0",
        "rolling_3d_positive_share",
        "repeat_oos_test_positive_windows",
        "primary_reasons",
    ],
)}

## Most Likely HFT / Delay-Sensitive Wallets
{format_table(
    hft_top,
    [
        "wallet_id",
        "sample_name",
        "label_confidence",
        "hft_score",
        "avg_copy_edge_net_0s",
        "avg_copy_edge_net_30s",
        "edge_retention_30s_from_0",
        "median_holding_seconds",
        "share_closed_within_60s",
        "fast_exit_share_30s",
        "primary_reasons",
    ],
)}

## Most Likely YOLO / Noise / Unstable Wallets
{format_table(
    yolo_top,
    [
        "wallet_id",
        "sample_name",
        "label_confidence",
        "yolo_score",
        "realized_pnl_abs",
        "max_drawdown_pct_of_peak",
        "pnl_concentration_top1_share",
        "pnl_concentration_top5_share",
        "position_size_cv",
        "rolling_3d_positive_share",
        "primary_reasons",
    ],
)}

## Assumptions and Limitations
- `0s` edge is estimated from observed `BUY -> SELL` realized signal pairs and research-only cost assumptions.
- `5s / 15s / 30s` delay edges come from the existing delayed copy-follow wallet-exit backtest summary.
- `60s` edge is only a **proxy** reconstructed from the half-split report; cross-half closes are not preserved, so treat it as lower-fidelity.
- Unrealized PnL is not currently available from the public-data reconstruction used here; it is left null.
- Profit concentration is measured as the share of **positive realized PnL** contributed by the top 1 / 5 / 10 winning trades.
- Labels are computed using only information observed up to the sample end. They should not be reinterpreted as labels that would have been known earlier in the sample.
"""

    output.write_text(report, encoding="utf-8")
    return output


def run_wallet_labeling(
    project_root: str | Path,
    *,
    features_path: str | Path = DEFAULT_FEATURES_CSV,
    labels_path: str | Path = DEFAULT_LABELS_CSV,
    report_path: str | Path = DEFAULT_REPORT_MD,
) -> dict[str, Any]:
    """Build wallet features, assign first-pass labels, and write outputs."""

    root = Path(project_root)
    features = build_wallet_features(root)
    labels = assign_first_pass_labels(features)

    features_out = root / features_path
    labels_out = root / labels_path
    report_out = root / report_path
    features_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(features_out, index=False)
    labels.to_csv(labels_out, index=False)
    render_wallet_labeling_summary(features, labels, output_path=report_out)

    return {
        "features": features,
        "labels": labels,
        "paths": {
            "features": features_out,
            "labels": labels_out,
            "report": report_out,
        },
    }
