"""Behavior deep-dive for a selected wallet shortlist using recent copy-follow outcomes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy.orm import Session

from research.behavior_analysis import (
    FEATURE_GROUP_COLUMNS,
    _build_market_lookup,
    _classify_market_phase,
    _classify_pre_trade_trend,
    _classify_price_zone,
    _classify_size_bucket,
    _load_market_frame,
    _lookup_price_at_or_before,
    _to_epoch_seconds,
    _trend_alignment,
    _window_prices,
    _compute_trade_cluster_density,
    classify_trade_behavior,
)
from research.copy_follow_wallet_exit import (
    _normalize_recent_trades,
    build_copy_exit_pairs,
)
from research.costs import calculate_net_pnl, estimate_entry_only_cost
from research.delay_analysis import lookup_forward_price
from research.enrich_trades import classify_liquidity_bucket
from research.recent_wallet_trade_capture import load_recent_wallet_trades


DEFAULT_SELECTION_CSV = (
    "exports/follow_wallet_repeat_oos/repeated_positive_wallets_strict_named.csv"
)
DEFAULT_OUTPUT_DIR = "exports/strict_wallet_behavior_deep_dive"
DEFAULT_START_DATE = "2026-03-12T00:00:00Z"
DEFAULT_END_DATE = "2026-03-26T23:59:59Z"
DEFAULT_DELAYS = (15, 30, 60)


@dataclass(frozen=True)
class WalletSelection:
    """One strict-wallet selection row plus its preferred delay."""

    wallet_address: str
    sample_name: str | None
    sample_pseudonym: str | None
    selected_delay: str


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _safe_float(value: Any) -> float | None:
    """Convert one scalar into float or None."""

    coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(coerced):
        return None
    return float(coerced)


def _safe_bool(value: Any) -> bool:
    """Normalize truthy CSV/database values."""

    return str(value).strip().lower() == "true"


def load_wallet_selection(path: str | Path = DEFAULT_SELECTION_CSV) -> list[WalletSelection]:
    """Load the strict repeated-positive shortlist and deduplicate wallets."""

    selection_path = Path(path)
    with selection_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    selections: dict[str, WalletSelection] = {}
    for row in rows:
        wallet = str(row.get("wallet_address") or "").strip().lower()
        if not wallet:
            continue
        selections[wallet] = WalletSelection(
            wallet_address=wallet,
            sample_name=(row.get("sample_name") or "").strip() or None,
            sample_pseudonym=(row.get("sample_pseudonym") or "").strip() or None,
            selected_delay=str(row.get("delay") or "15s"),
        )
    return list(selections.values())


def _build_price_index_chunked(
    session: Session,
    *,
    token_ids: list[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    chunk_size: int = 250,
) -> dict[str, tuple[list[int], list[float]]]:
    """Build a price index directly from SQLite in chunks.

    This avoids materializing a multi-million-row pandas frame for one small
    wallet subset while still reusing the existing public `price_history` cache.
    """

    index: dict[str, tuple[list[int], list[float]]] = {}
    if not token_ids:
        return index

    connection: sqlite3.Connection = session.bind.raw_connection()  # type: ignore[assignment]
    try:
        for start in range(0, len(token_ids), chunk_size):
            chunk = token_ids[start:start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "select token_id, cast(strftime('%s', ts) as integer) as epoch_ts, price "
                "from price_history "
                f"where token_id in ({placeholders}) and ts >= ? and ts <= ? "
                "order by token_id, ts"
            )
            params: list[Any] = [*chunk, start_ts.isoformat(), end_ts.isoformat()]
            for token_id, epoch_ts, price in connection.execute(query, params):
                if token_id is None or epoch_ts is None or price is None:
                    continue
                times, prices = index.setdefault(str(token_id), ([], []))
                times.append(int(epoch_ts))
                prices.append(float(price))
    finally:
        connection.close()
    return index


def _extract_trade_features_from_price_index(
    recent_trades: pd.DataFrame,
    *,
    price_index: dict[str, tuple[list[int], list[float]]],
    markets: pd.DataFrame,
) -> pd.DataFrame:
    """Extract buy-signal behavior features without loading price history into pandas."""

    frame = recent_trades.copy()
    if frame.empty:
        return frame

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["trade_id", "wallet_address", "timestamp", "price"]).reset_index(drop=True)
    if frame.empty:
        return frame

    frame["trade_notional"] = pd.to_numeric(frame["size"], errors="coerce") * pd.to_numeric(frame["price"], errors="coerce")
    logged_notional = pd.to_numeric(frame["trade_notional"], errors="coerce").map(
        lambda value: pd.NA if pd.isna(value) or value <= 0 else math.log1p(float(value))
    )
    logged_valid = pd.to_numeric(logged_notional, errors="coerce")
    std = logged_valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        frame["standardized_size"] = 0.0
    else:
        frame["standardized_size"] = (logged_valid - logged_valid.mean()) / std

    market_medians = frame.groupby("market_id", dropna=True)["trade_notional"].median()
    token_medians = frame.groupby("token_id", dropna=True)["trade_notional"].median()

    def _relative_size(row: pd.Series) -> float | None:
        median = market_medians.get(row.get("market_id"))
        if median is None or pd.isna(median) or median <= 0:
            median = token_medians.get(row.get("token_id"))
        trade_notional = pd.to_numeric(pd.Series([row.get("trade_notional")]), errors="coerce").iloc[0]
        if pd.isna(trade_notional) or median is None or pd.isna(median) or median <= 0:
            return None
        return float(trade_notional / median)

    frame["size_to_liquidity_ratio"] = frame.apply(_relative_size, axis=1)
    frame["price_distance_from_mid"] = (pd.to_numeric(frame["price"], errors="coerce") - 0.5).abs()
    frame["size_bucket"] = frame["standardized_size"].apply(_classify_size_bucket)
    frame["price_zone"] = frame["price_distance_from_mid"].apply(_classify_price_zone)

    market_lookup = _build_market_lookup(markets)
    recent_momentum_1m: list[float | None] = []
    recent_momentum_5m: list[float | None] = []
    short_term_volatility_5m: list[float | None] = []
    market_phases: list[str] = []
    time_to_resolution_minutes: list[float | None] = []
    trend_states: list[str] = []
    trend_alignments: list[bool | None] = []

    for row in frame.itertuples(index=False):
        trade_ts = _to_epoch_seconds(row.timestamp)
        reference_price = float(row.price)
        price_1m_ago = _lookup_price_at_or_before(price_index, row.token_id, trade_ts - 60)
        price_5m_ago = _lookup_price_at_or_before(price_index, row.token_id, trade_ts - 5 * 60)
        prices_5m_window = _window_prices(price_index, row.token_id, trade_ts - 5 * 60, trade_ts)

        momentum_1m = None if price_1m_ago is None else reference_price - price_1m_ago
        momentum_5m = None if price_5m_ago is None else reference_price - price_5m_ago
        volatility_5m = (
            float(pd.Series(prices_5m_window, dtype=float).std(ddof=0))
            if len(prices_5m_window) >= 2
            else None
        )

        market_record = market_lookup.get(str(row.market_id)) if row.market_id is not None else None
        created_at = None if market_record is None else market_record.get("created_at")
        resolution_ts = None if market_record is None else market_record.get("resolution_ts")
        market_phase, minutes_to_resolution = _classify_market_phase(row.timestamp, created_at, resolution_ts)
        trend_state = _classify_pre_trade_trend(momentum_1m, momentum_5m, volatility_5m)
        alignment = _trend_alignment(row.side, trend_state)

        recent_momentum_1m.append(momentum_1m)
        recent_momentum_5m.append(momentum_5m)
        short_term_volatility_5m.append(volatility_5m)
        market_phases.append(market_phase)
        time_to_resolution_minutes.append(minutes_to_resolution)
        trend_states.append(trend_state)
        trend_alignments.append(alignment)

    frame["recent_momentum_1m"] = recent_momentum_1m
    frame["recent_momentum_5m"] = recent_momentum_5m
    frame["short_term_volatility_5m"] = short_term_volatility_5m
    frame["market_phase"] = market_phases
    frame["time_to_resolution_minutes"] = time_to_resolution_minutes
    frame["pre_trade_trend_state"] = trend_states
    frame["trend_alignment"] = trend_alignments
    frame["trade_cluster_density_5m"] = _compute_trade_cluster_density(frame)
    frame["trade_type_cluster"] = frame.apply(lambda row: classify_trade_behavior(pd.Series(row)), axis=1)
    return frame


def _compute_copy_exit_diagnostics_from_price_index(
    recent_trades: pd.DataFrame,
    *,
    price_index: dict[str, tuple[list[int], list[float]]],
    delays: tuple[int, ...],
) -> pd.DataFrame:
    """Compute delayed wallet-exit copy diagnostics using one shared price index."""

    normalized = _normalize_recent_trades(recent_trades)
    pairs, open_positions = build_copy_exit_pairs(normalized)
    rows: list[dict[str, Any]] = []
    source_rows = []
    if not pairs.empty:
        source_rows.extend(pairs.to_dict(orient="records"))
    if not open_positions.empty:
        source_rows.extend(open_positions.to_dict(orient="records"))

    for record in source_rows:
        token_id = str(record.get("token_id")) if record.get("token_id") is not None else None
        buy_ts = pd.to_datetime(record.get("buy_timestamp"), utc=True, errors="coerce")
        sell_ts = pd.to_datetime(record.get("sell_timestamp"), utc=True, errors="coerce")
        buy_epoch = _to_epoch_seconds(buy_ts)
        sell_epoch = _to_epoch_seconds(sell_ts) if pd.notna(sell_ts) else None
        copied_size = float(record.get("copied_size") or 0.0)
        diagnostic = dict(record)
        for delay in delays:
            label = f"{int(delay)}s"
            status_col = f"copy_status_{label}"
            raw_col = f"copy_pnl_usdc_{label}"
            net_col = f"copy_pnl_net_usdc_{label}"
            cost_col = f"copy_cost_{label}"

            diagnostic[status_col] = None
            diagnostic[raw_col] = None
            diagnostic[net_col] = None
            diagnostic[cost_col] = None

            entry_target_epoch = buy_epoch + int(delay)
            entry_forward = lookup_forward_price(price_index, token_id, entry_target_epoch)
            diagnostic[f"copy_entry_price_{label}"] = entry_forward.price
            diagnostic[f"copy_entry_source_{label}"] = entry_forward.source
            diagnostic[f"copy_entry_lag_seconds_{label}"] = entry_forward.delta_seconds
            if entry_forward.price is None:
                diagnostic[status_col] = "missing_entry_price_after_delay"
                continue

            entry_epoch = entry_target_epoch + int(entry_forward.delta_seconds or 0)
            if sell_epoch is None:
                diagnostic[status_col] = "open_without_observed_exit"
                continue
            if sell_epoch <= entry_target_epoch:
                diagnostic[status_col] = "wallet_exit_before_delayed_entry"
                continue

            exit_target_epoch = sell_epoch + int(delay)
            if entry_epoch >= exit_target_epoch:
                diagnostic[status_col] = "entry_after_or_at_delayed_exit"
                continue

            exit_forward = lookup_forward_price(price_index, token_id, exit_target_epoch)
            diagnostic[f"copy_exit_price_{label}"] = exit_forward.price
            diagnostic[f"copy_exit_source_{label}"] = exit_forward.source
            diagnostic[f"copy_exit_lag_seconds_{label}"] = exit_forward.delta_seconds
            diagnostic["copy_exit_type_signal"] = "wallet_sell" if sell_epoch is not None else "open"
            if exit_forward.price is None:
                diagnostic[status_col] = "missing_exit_price_after_delay"
                continue

            raw_unit = float(exit_forward.price) - float(entry_forward.price)
            entry_cost = estimate_entry_only_cost(
                pd.Series(record),
                entry_price=float(entry_forward.price),
            )["total_cost"]
            exit_cost = estimate_entry_only_cost(
                pd.Series(record),
                entry_price=float(exit_forward.price),
            )["total_cost"]
            total_cost = float(entry_cost) + float(exit_cost)
            raw_usdc = raw_unit * copied_size
            net_usdc = calculate_net_pnl(raw_usdc, total_cost * copied_size)
            diagnostic[raw_col] = raw_usdc
            diagnostic[net_col] = net_usdc
            diagnostic[cost_col] = total_cost * copied_size
            diagnostic[status_col] = "ok"
        rows.append(diagnostic)

    return pd.DataFrame.from_records(rows)


def _aggregate_signal_outcomes(trade_diagnostics: pd.DataFrame, delays: tuple[int, ...]) -> pd.DataFrame:
    """Aggregate wallet-exit diagnostics from slice-level rows back to buy-signal rows."""

    if trade_diagnostics.empty:
        return pd.DataFrame(columns=["trade_id"])

    working = trade_diagnostics.copy()
    records: list[dict[str, Any]] = []
    for trade_id, group in working.groupby("signal_trade_id", dropna=False):
        record: dict[str, Any] = {"trade_id": trade_id}
        for delay in delays:
            label = f"{int(delay)}s"
            status_col = f"copy_status_{label}"
            net_col = f"copy_pnl_net_usdc_{label}"
            raw_col = f"copy_pnl_usdc_{label}"
            cost_col = f"copy_cost_{label}"

            valid_mask = group[status_col].eq("ok") if status_col in group.columns else pd.Series(False, index=group.index)
            valid_net = pd.to_numeric(group.loc[valid_mask, net_col], errors="coerce").dropna() if net_col in group.columns else pd.Series(dtype=float)
            valid_raw = pd.to_numeric(group.loc[valid_mask, raw_col], errors="coerce").dropna() if raw_col in group.columns else pd.Series(dtype=float)
            valid_cost = pd.to_numeric(group.loc[valid_mask, cost_col], errors="coerce").dropna() if cost_col in group.columns else pd.Series(dtype=float)

            record[f"signal_valid_slices_{label}"] = int(valid_mask.sum())
            record[f"signal_total_copy_pnl_usdc_{label}"] = float(valid_raw.sum()) if not valid_raw.empty else None
            record[f"signal_total_copy_pnl_net_usdc_{label}"] = float(valid_net.sum()) if not valid_net.empty else None
            record[f"signal_total_copy_cost_usdc_{label}"] = float(valid_cost.sum()) if not valid_cost.empty else None
            record[f"signal_positive_{label}"] = bool(not valid_net.empty and float(valid_net.sum()) > 0)
            record[f"signal_open_slices_{label}"] = int(group[status_col].eq("open_without_observed_exit").sum()) if status_col in group.columns else 0
            record[f"signal_missing_price_slices_{label}"] = int(
                group[status_col].isin(
                    ["missing_entry_price_after_delay", "missing_exit_price_after_delay"]
                ).sum()
            ) if status_col in group.columns else 0
            record[f"signal_too_fast_slices_{label}"] = int(
                group[status_col].isin(
                    ["wallet_exit_before_delayed_entry", "entry_after_or_at_delayed_exit"]
                ).sum()
            ) if status_col in group.columns else 0
        records.append(record)
    return pd.DataFrame.from_records(records)


def _pick_selected_delay_value(row: pd.Series, metric_prefix: str) -> Any:
    """Return one per-signal metric from the wallet's chosen delay."""

    delay = str(row.get("selected_delay") or "15s")
    return row.get(f"{metric_prefix}_{delay}")


def _build_selected_delay_frame(
    buy_features: pd.DataFrame,
    signal_outcomes: pd.DataFrame,
    selections: list[WalletSelection],
) -> pd.DataFrame:
    """Merge behavior features with per-signal copy-follow outcomes."""

    if buy_features.empty:
        return buy_features

    selection_frame = pd.DataFrame.from_records(
        [
            {
                "wallet_address": selection.wallet_address,
                "sample_name": selection.sample_name,
                "sample_pseudonym": selection.sample_pseudonym,
                "selected_delay": selection.selected_delay,
            }
            for selection in selections
        ]
    )

    merged = buy_features.merge(signal_outcomes, on="trade_id", how="left")
    merged = merged.merge(selection_frame, on="wallet_address", how="left")
    merged["selected_signal_valid_slices"] = merged.apply(
        lambda row: _pick_selected_delay_value(row, "signal_valid_slices"),
        axis=1,
    )
    merged["selected_signal_total_copy_pnl_usdc"] = merged.apply(
        lambda row: _pick_selected_delay_value(row, "signal_total_copy_pnl_usdc"),
        axis=1,
    )
    merged["selected_signal_total_copy_pnl_net_usdc"] = merged.apply(
        lambda row: _pick_selected_delay_value(row, "signal_total_copy_pnl_net_usdc"),
        axis=1,
    )
    merged["selected_signal_total_copy_cost_usdc"] = merged.apply(
        lambda row: _pick_selected_delay_value(row, "signal_total_copy_cost_usdc"),
        axis=1,
    )
    merged["selected_signal_positive"] = merged.apply(
        lambda row: _pick_selected_delay_value(row, "signal_positive"),
        axis=1,
    )
    merged["selected_signal_open_slices"] = merged.apply(
        lambda row: _pick_selected_delay_value(row, "signal_open_slices"),
        axis=1,
    )
    merged["selected_signal_missing_price_slices"] = merged.apply(
        lambda row: _pick_selected_delay_value(row, "signal_missing_price_slices"),
        axis=1,
    )
    merged["selected_signal_too_fast_slices"] = merged.apply(
        lambda row: _pick_selected_delay_value(row, "signal_too_fast_slices"),
        axis=1,
    )
    return merged


def summarize_selected_delay_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-delay performance by feature bucket."""

    if frame.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for feature_name in FEATURE_GROUP_COLUMNS:
        if feature_name not in frame.columns:
            continue
        grouped = frame.dropna(subset=[feature_name]).groupby(feature_name, dropna=False)
        for feature_value, group in grouped:
            valid = pd.to_numeric(group["selected_signal_total_copy_pnl_net_usdc"], errors="coerce").dropna()
            raw_valid = pd.to_numeric(group["selected_signal_total_copy_pnl_usdc"], errors="coerce").dropna()
            cost_valid = pd.to_numeric(group["selected_signal_total_copy_cost_usdc"], errors="coerce").dropna()
            records.append(
                {
                    "feature_name": feature_name,
                    "feature_value": feature_value,
                    "n_signals": int(len(group)),
                    "n_wallets": int(group["wallet_address"].nunique(dropna=True)),
                    "valid_signals": int(valid.shape[0]),
                    "positive_signals": int((valid > 0).sum()) if not valid.empty else 0,
                    "hit_rate": float((valid > 0).mean()) if not valid.empty else None,
                    "avg_copy_pnl_net_selected_delay_usdc": float(valid.mean()) if not valid.empty else None,
                    "median_copy_pnl_net_selected_delay_usdc": float(valid.median()) if not valid.empty else None,
                    "total_copy_pnl_net_selected_delay_usdc": float(valid.sum()) if not valid.empty else None,
                    "avg_copy_pnl_selected_delay_usdc": float(raw_valid.mean()) if not raw_valid.empty else None,
                    "avg_copy_cost_selected_delay_usdc": float(cost_valid.mean()) if not cost_valid.empty else None,
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    return result.sort_values(
        ["feature_name", "avg_copy_pnl_net_selected_delay_usdc", "n_signals", "feature_value"],
        ascending=[True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_wallet_behavior_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    """Break each selected wallet into behavior clusters using selected-delay net PnL."""

    if frame.empty:
        return pd.DataFrame()

    total_signals = frame.groupby("wallet_address").size()
    wallet_totals = pd.to_numeric(frame["selected_signal_total_copy_pnl_net_usdc"], errors="coerce").groupby(frame["wallet_address"]).sum(min_count=1)

    records: list[dict[str, Any]] = []
    for (wallet_address, cluster), group in frame.groupby(["wallet_address", "trade_type_cluster"], dropna=False):
        valid = pd.to_numeric(group["selected_signal_total_copy_pnl_net_usdc"], errors="coerce").dropna()
        wallet_total = wallet_totals.get(wallet_address)
        total = float(valid.sum()) if not valid.empty else None
        role = "neutral"
        if total is not None and total > 0:
            role = "copy_edge_driver"
        elif total is not None and total < 0:
            role = "copy_drag"

        records.append(
            {
                "wallet_address": wallet_address,
                "sample_name": group["sample_name"].dropna().iloc[0] if group["sample_name"].notna().any() else None,
                "selected_delay": group["selected_delay"].dropna().iloc[0] if group["selected_delay"].notna().any() else None,
                "trade_type_cluster": cluster,
                "n_signals": int(len(group)),
                "valid_signals": int(valid.shape[0]),
                "share_of_wallet_signals": float(len(group) / total_signals[wallet_address]),
                "avg_copy_pnl_net_selected_delay_usdc": float(valid.mean()) if not valid.empty else None,
                "median_copy_pnl_net_selected_delay_usdc": float(valid.median()) if not valid.empty else None,
                "total_copy_pnl_net_selected_delay_usdc": total,
                "copy_hit_rate_selected_delay": float((valid > 0).mean()) if not valid.empty else None,
                "contribution_share_of_wallet_net": (
                    None
                    if wallet_total is None or pd.isna(wallet_total) or wallet_total == 0 or total is None
                    else float(total / wallet_total)
                ),
                "cluster_role": role,
            }
        )

    result = pd.DataFrame.from_records(records)
    return result.sort_values(
        ["wallet_address", "total_copy_pnl_net_selected_delay_usdc", "n_signals", "trade_type_cluster"],
        ascending=[True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_wallet_summary(frame: pd.DataFrame, wallet_behavior: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-delay behavior outcomes at the wallet level."""

    if frame.empty:
        return pd.DataFrame()

    drivers = (
        wallet_behavior[wallet_behavior["cluster_role"] == "copy_edge_driver"]
        .sort_values(["wallet_address", "total_copy_pnl_net_selected_delay_usdc"], ascending=[True, False])
        .drop_duplicates(subset=["wallet_address"])
        .rename(columns={"trade_type_cluster": "top_positive_cluster"})
    )
    drags = (
        wallet_behavior[wallet_behavior["cluster_role"] == "copy_drag"]
        .sort_values(["wallet_address", "total_copy_pnl_net_selected_delay_usdc"], ascending=[True, True])
        .drop_duplicates(subset=["wallet_address"])
        .rename(columns={"trade_type_cluster": "top_negative_cluster"})
    )

    records: list[dict[str, Any]] = []
    for wallet_address, group in frame.groupby("wallet_address"):
        valid = pd.to_numeric(group["selected_signal_total_copy_pnl_net_usdc"], errors="coerce").dropna()
        record = {
            "wallet_address": wallet_address,
            "sample_name": group["sample_name"].dropna().iloc[0] if group["sample_name"].notna().any() else None,
            "sample_pseudonym": group["sample_pseudonym"].dropna().iloc[0] if group["sample_pseudonym"].notna().any() else None,
            "selected_delay": group["selected_delay"].dropna().iloc[0] if group["selected_delay"].notna().any() else None,
            "n_buy_signals": int(len(group)),
            "valid_buy_signals": int(valid.shape[0]),
            "positive_buy_signals": int((valid > 0).sum()) if not valid.empty else 0,
            "copy_hit_rate_selected_delay": float((valid > 0).mean()) if not valid.empty else None,
            "avg_copy_pnl_net_selected_delay_usdc": float(valid.mean()) if not valid.empty else None,
            "median_copy_pnl_net_selected_delay_usdc": float(valid.median()) if not valid.empty else None,
            "total_copy_pnl_net_selected_delay_usdc": float(valid.sum()) if not valid.empty else None,
            "first_signal_ts": pd.to_datetime(group["timestamp"], utc=True, errors="coerce").min(),
            "most_recent_signal_ts": pd.to_datetime(group["timestamp"], utc=True, errors="coerce").max(),
        }
        records.append(record)

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    result = result.merge(
        drivers[["wallet_address", "top_positive_cluster"]],
        on="wallet_address",
        how="left",
    )
    result = result.merge(
        drags[["wallet_address", "top_negative_cluster"]],
        on="wallet_address",
        how="left",
    )
    result["first_signal_ts"] = pd.to_datetime(result["first_signal_ts"], utc=True, errors="coerce").apply(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    result["most_recent_signal_ts"] = pd.to_datetime(result["most_recent_signal_ts"], utc=True, errors="coerce").apply(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    return result.sort_values(
        ["total_copy_pnl_net_selected_delay_usdc", "wallet_address"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_markdown_report(
    *,
    wallet_summary: pd.DataFrame,
    wallet_behavior: pd.DataFrame,
    feature_summary: pd.DataFrame,
) -> str:
    """Create a short Markdown report for the selected-wallet deep dive."""

    lines = [
        "# Strict Wallet Behavior Deep Dive",
        "",
        "This report analyzes the strict repeated-positive shortlist at each wallet's preferred delay.",
        "",
    ]

    if wallet_summary.empty:
        lines.append("No wallet signals were available for analysis.")
        return "\n".join(lines)

    lines.extend(
        [
            "## Wallet Summary",
            "",
            f"- Wallets analyzed: {len(wallet_summary)}",
            f"- Total buy signals: {int(wallet_summary['n_buy_signals'].sum())}",
            f"- Total valid buy signals: {int(wallet_summary['valid_buy_signals'].sum())}",
            "",
        ]
    )

    lines.append("## Top Wallets")
    lines.append("")
    for row in wallet_summary.head(7).to_dict(orient="records"):
        lines.append(
            f"- `{row['wallet_address']}` ({row.get('sample_name') or 'no name'}) | "
            f"delay `{row.get('selected_delay')}` | "
            f"valid `{row.get('valid_buy_signals')}` | "
            f"hit rate `{row.get('copy_hit_rate_selected_delay')}` | "
            f"net `{row.get('total_copy_pnl_net_selected_delay_usdc')}` | "
            f"driver `{row.get('top_positive_cluster')}` | "
            f"drag `{row.get('top_negative_cluster')}`"
        )

    lines.extend(["", "## Best Behavior Clusters", ""])
    cluster_rows = feature_summary[feature_summary["feature_name"] == "trade_type_cluster"].head(10)
    for row in cluster_rows.to_dict(orient="records"):
        lines.append(
            f"- `{row['feature_value']}` | signals `{row['n_signals']}` | "
            f"wallets `{row['n_wallets']}` | valid `{row['valid_signals']}` | "
            f"avg net `{row['avg_copy_pnl_net_selected_delay_usdc']}` | "
            f"hit rate `{row['hit_rate']}`"
        )

    lines.extend(["", "## Wallet Drivers", ""])
    driver_rows = wallet_behavior[wallet_behavior["cluster_role"] == "copy_edge_driver"].head(20)
    for row in driver_rows.to_dict(orient="records"):
        lines.append(
            f"- `{row['wallet_address']}` | `{row['trade_type_cluster']}` | "
            f"signals `{row['n_signals']}` | total net `{row['total_copy_pnl_net_selected_delay_usdc']}`"
        )

    return "\n".join(lines)


def run_selected_wallet_behavior_analysis(
    session: Session,
    *,
    selection_csv: str | Path = DEFAULT_SELECTION_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    delays: tuple[int, ...] = DEFAULT_DELAYS,
) -> dict[str, Any]:
    """Run behavior decomposition for the strict repeated-positive shortlist."""

    selections = load_wallet_selection(selection_csv)
    wallets = [selection.wallet_address for selection in selections]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    recent_trades = load_recent_wallet_trades(
        session,
        wallets=wallets,
        recent_window_start=pd.to_datetime(start_date, utc=True).to_pydatetime(),
        recent_window_end=pd.to_datetime(end_date, utc=True).to_pydatetime(),
    )
    normalized = _normalize_recent_trades(recent_trades)
    if normalized.empty:
        empty = pd.DataFrame()
        paths = {
            "wallet_summary": _write_csv(empty, output_root / "selected_wallet_summary.csv"),
            "wallet_behavior_breakdown": _write_csv(empty, output_root / "selected_wallet_behavior_breakdown.csv"),
            "feature_performance_summary": _write_csv(empty, output_root / "selected_wallet_feature_performance.csv"),
            "signal_features": _write_csv(empty, output_root / "selected_wallet_signal_features.csv"),
        }
        return {
            "recent_trades": empty,
            "signal_features": empty,
            "wallet_summary": empty,
            "wallet_behavior_breakdown": empty,
            "feature_performance_summary": empty,
            "paths": paths,
        }

    normalized["liquidity_bucket"] = normalized["size"].apply(classify_liquidity_bucket)
    token_ids = sorted({str(token_id) for token_id in normalized["token_id"].dropna().tolist()})
    price_index = _build_price_index_chunked(
        session,
        token_ids=token_ids,
        start_ts=pd.to_datetime(start_date, utc=True) - pd.Timedelta(minutes=10),
        end_ts=pd.to_datetime(end_date, utc=True) + pd.Timedelta(minutes=10),
    )
    markets = _load_market_frame(session)
    trade_features = _extract_trade_features_from_price_index(
        normalized,
        price_index=price_index,
        markets=markets,
    )
    buy_features = trade_features[trade_features["side"].astype(str).str.upper() == "BUY"].copy()

    wallet_exit_diagnostics = _compute_copy_exit_diagnostics_from_price_index(
        recent_trades,
        price_index=price_index,
        delays=delays,
    )
    signal_outcomes = _aggregate_signal_outcomes(wallet_exit_diagnostics, delays)
    selected_delay_frame = _build_selected_delay_frame(buy_features, signal_outcomes, selections)
    copy_follow_dir = output_root / "copy_follow_wallet_exit_subset"
    copy_follow_dir.mkdir(parents=True, exist_ok=True)
    wallet_exit_overview = pd.DataFrame(
        [
            {
                "analysis_window_start": start_date,
                "analysis_window_end": end_date,
                "wallets_in_report": len(wallets),
                "buy_signal_rows": int(len(selected_delay_frame)),
                **{
                    f"valid_signals_{int(delay)}s": int(
                        pd.to_numeric(signal_outcomes.get(f"signal_valid_slices_{int(delay)}s"), errors="coerce")
                        .fillna(0)
                        .gt(0)
                        .sum()
                    )
                    if not signal_outcomes.empty
                    else 0
                    for delay in delays
                },
            }
        ]
    )
    copy_wallet_path = _write_csv(
        selected_delay_frame,
        copy_follow_dir / "copy_follow_wallet_exit_subset_wallet_summary.csv",
    )
    copy_diagnostics_path = _write_csv(
        wallet_exit_diagnostics,
        copy_follow_dir / "copy_follow_wallet_exit_subset_trade_diagnostics.csv",
    )
    copy_overview_path = _write_csv(
        wallet_exit_overview,
        copy_follow_dir / "copy_follow_wallet_exit_subset_overview.csv",
    )
    feature_summary = summarize_selected_delay_features(selected_delay_frame)
    wallet_behavior = build_wallet_behavior_breakdown(selected_delay_frame)
    wallet_summary = build_wallet_summary(selected_delay_frame, wallet_behavior)

    report_path = output_root / "selected_wallet_behavior_report.md"
    report_path.write_text(
        build_markdown_report(
            wallet_summary=wallet_summary,
            wallet_behavior=wallet_behavior,
            feature_summary=feature_summary,
        ),
        encoding="utf-8",
    )

    paths = {
        "wallet_summary": _write_csv(wallet_summary, output_root / "selected_wallet_summary.csv"),
        "wallet_behavior_breakdown": _write_csv(
            wallet_behavior,
            output_root / "selected_wallet_behavior_breakdown.csv",
        ),
        "feature_performance_summary": _write_csv(
            feature_summary,
            output_root / "selected_wallet_feature_performance.csv",
        ),
        "signal_features": _write_csv(
            selected_delay_frame,
            output_root / "selected_wallet_signal_features.csv",
        ),
        "report": report_path,
        "copy_follow_wallet_summary": copy_wallet_path,
        "copy_follow_trade_diagnostics": copy_diagnostics_path,
        "copy_follow_overview": copy_overview_path,
    }
    return {
        "selections": selections,
        "recent_trades": normalized,
        "signal_features": selected_delay_frame,
        "wallet_summary": wallet_summary,
        "wallet_behavior_breakdown": wallet_behavior,
        "feature_performance_summary": feature_summary,
        "paths": paths,
    }


def print_selected_wallet_behavior_summary(results: dict[str, Any]) -> None:
    """Print a concise console summary for the strict-wallet behavior deep dive."""

    wallet_summary = results["wallet_summary"]
    behavior = results["wallet_behavior_breakdown"]
    feature_summary = results["feature_performance_summary"]

    print("Strict Wallet Behavior Deep Dive")
    if wallet_summary.empty:
        print("No recent wallet trades were available for the selected shortlist.")
        return

    print(f"Wallets analyzed: {len(wallet_summary)}")
    print(f"Total buy signals: {int(wallet_summary['n_buy_signals'].sum())}")
    print(f"Total valid buy signals: {int(wallet_summary['valid_buy_signals'].sum())}")
    print("Top Wallets")
    print(
        wallet_summary[
            [
                "wallet_address",
                "sample_name",
                "selected_delay",
                "valid_buy_signals",
                "copy_hit_rate_selected_delay",
                "total_copy_pnl_net_selected_delay_usdc",
                "top_positive_cluster",
                "top_negative_cluster",
            ]
        ].to_string(index=False)
    )

    if not feature_summary.empty:
        print("Top Trade Type Clusters")
        trade_clusters = feature_summary[feature_summary["feature_name"] == "trade_type_cluster"].head(10)
        print(
            trade_clusters[
                [
                    "feature_value",
                    "n_signals",
                    "n_wallets",
                    "valid_signals",
                    "avg_copy_pnl_net_selected_delay_usdc",
                    "hit_rate",
                ]
            ].to_string(index=False)
        )

    if not behavior.empty:
        drivers = behavior[behavior["cluster_role"] == "copy_edge_driver"].head(10)
        if not drivers.empty:
            print("Top Wallet Behavior Drivers")
            print(
                drivers[
                    [
                        "wallet_address",
                        "sample_name",
                        "trade_type_cluster",
                        "n_signals",
                        "total_copy_pnl_net_selected_delay_usdc",
                    ]
                ].to_string(index=False)
            )
