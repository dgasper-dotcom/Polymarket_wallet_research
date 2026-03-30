"""Research-only delayed copy analysis that exits when the wallet exits."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from research.copy_follow_expiry import (
    _build_terminal_lookup,
    _filter_window,
    load_markets_frame,
)
from research.costs import calculate_net_pnl, estimate_entry_only_cost
from research.delay_analysis import _build_price_index, _to_epoch_seconds, load_price_history_frame, lookup_forward_price
from research.recent_wallet_trade_capture import load_recent_wallet_trades


DEFAULT_DELAYS = (5, 10, 15, 30)
ACTIVE_LAST_DAYS_DEFAULT = 14
SIZE_EPSILON = 1e-9


@dataclass
class _OpenLot:
    """One open copied buy lot for FIFO matching."""

    trade_id: str
    wallet_address: str
    market_id: str | None
    token_id: str
    timestamp: pd.Timestamp
    price: float
    size_remaining: float
    usdc_size: float | None
    spread_at_trade: float | None
    slippage_bps_assumed: float | None
    liquidity_bucket: str | None
    raw_json: Any


def _normalize_side(value: Any) -> str | None:
    """Normalize public trade side values."""

    normalized = str(value or "").upper().strip()
    if normalized in {"BUY", "SELL"}:
        return normalized
    return None


def _coerce_size(row: pd.Series) -> float | None:
    """Return a usable trade size, falling back to usdc_size / price when needed."""

    size = pd.to_numeric(pd.Series([row.get("size")]), errors="coerce").iloc[0]
    if pd.notna(size) and float(size) > SIZE_EPSILON:
        return float(size)

    usdc_size = pd.to_numeric(pd.Series([row.get("usdc_size")]), errors="coerce").iloc[0]
    price = pd.to_numeric(pd.Series([row.get("price")]), errors="coerce").iloc[0]
    if pd.notna(usdc_size) and pd.notna(price) and float(price) > SIZE_EPSILON:
        derived = float(usdc_size) / float(price)
        if derived > SIZE_EPSILON:
            return derived
    return None


def _normalize_recent_trades(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter recent raw trades down to usable chronological buy/sell records."""

    if frame.empty:
        return frame.copy()

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized["side"] = normalized["side"].map(_normalize_side)
    normalized["size"] = normalized.apply(_coerce_size, axis=1)
    normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
    normalized["usdc_size"] = pd.to_numeric(normalized.get("usdc_size"), errors="coerce")
    normalized = normalized.dropna(subset=["wallet_address", "token_id", "timestamp", "side", "price", "size"])
    normalized = normalized.loc[normalized["size"] > SIZE_EPSILON].copy()
    return normalized.sort_values(["wallet_address", "token_id", "timestamp", "trade_id"]).reset_index(drop=True)


def build_copy_exit_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build FIFO buy lots paired against later wallet sells in the same token."""

    normalized = _normalize_recent_trades(frame)
    if normalized.empty:
        return pd.DataFrame(), pd.DataFrame()

    paired_rows: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []

    for (wallet, token_id), group in normalized.groupby(["wallet_address", "token_id"], sort=False):
        open_lots: deque[_OpenLot] = deque()
        for row in group.to_dict(orient="records"):
            side = _normalize_side(row.get("side"))
            size = float(row.get("size") or 0.0)
            if side == "BUY":
                open_lots.append(
                    _OpenLot(
                        trade_id=str(row.get("trade_id")),
                        wallet_address=str(wallet),
                        market_id=str(row.get("market_id")) if row.get("market_id") is not None else None,
                        token_id=str(token_id),
                        timestamp=pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce"),
                        price=float(row.get("price")),
                        size_remaining=size,
                        usdc_size=float(row["usdc_size"]) if pd.notna(row.get("usdc_size")) else None,
                        spread_at_trade=(
                            float(row["spread_at_trade"]) if pd.notna(row.get("spread_at_trade")) else None
                        ),
                        slippage_bps_assumed=(
                            float(row["slippage_bps_assumed"])
                            if pd.notna(row.get("slippage_bps_assumed"))
                            else None
                        ),
                        liquidity_bucket=str(row.get("liquidity_bucket"))
                        if row.get("liquidity_bucket") is not None
                        else None,
                        raw_json=row.get("raw_json"),
                    )
                )
                continue

            if side != "SELL":
                continue

            remaining_sell = size
            while remaining_sell > SIZE_EPSILON and open_lots:
                lot = open_lots[0]
                matched_size = min(lot.size_remaining, remaining_sell)
                paired_rows.append(
                    {
                        "trade_id": f"{lot.trade_id}::sell::{row.get('trade_id')}::{len(paired_rows)}",
                        "signal_trade_id": lot.trade_id,
                        "exit_trade_id": str(row.get("trade_id")),
                        "wallet_address": wallet,
                        "market_id": lot.market_id,
                        "token_id": token_id,
                        "buy_timestamp": lot.timestamp,
                        "buy_price_signal": lot.price,
                        "sell_timestamp": pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce"),
                        "sell_price_signal": float(row.get("price")),
                        "copied_size": matched_size,
                        "entry_notional_usdc_signal": lot.price * matched_size,
                        "exit_notional_usdc_signal": float(row.get("price")) * matched_size,
                        "exit_type_signal": "wallet_sell",
                        "spread_at_trade": lot.spread_at_trade,
                        "slippage_bps_assumed": lot.slippage_bps_assumed,
                        "liquidity_bucket": lot.liquidity_bucket,
                        "raw_json": lot.raw_json,
                    }
                )
                lot.size_remaining -= matched_size
                remaining_sell -= matched_size
                if lot.size_remaining <= SIZE_EPSILON:
                    open_lots.popleft()

        while open_lots:
            lot = open_lots.popleft()
            open_rows.append(
                {
                    "trade_id": f"{lot.trade_id}::open::{len(open_rows)}",
                    "signal_trade_id": lot.trade_id,
                    "exit_trade_id": None,
                    "wallet_address": lot.wallet_address,
                    "market_id": lot.market_id,
                    "token_id": lot.token_id,
                    "buy_timestamp": lot.timestamp,
                    "buy_price_signal": lot.price,
                    "sell_timestamp": None,
                    "sell_price_signal": None,
                    "copied_size": lot.size_remaining,
                    "entry_notional_usdc_signal": lot.price * lot.size_remaining,
                    "exit_notional_usdc_signal": None,
                    "exit_type_signal": "open",
                    "spread_at_trade": lot.spread_at_trade,
                    "slippage_bps_assumed": lot.slippage_bps_assumed,
                    "liquidity_bucket": lot.liquidity_bucket,
                    "raw_json": lot.raw_json,
                }
            )

    paired = pd.DataFrame.from_records(paired_rows)
    open_positions = pd.DataFrame.from_records(open_rows)
    return paired, open_positions


def collect_price_backfill_targets(
    pairs: pd.DataFrame,
    open_positions: pd.DataFrame,
    *,
    delays: Sequence[int] = DEFAULT_DELAYS,
) -> list[tuple[str, datetime, datetime]]:
    """Collect minimal token time bounds needed for delayed entry/exit lookups."""

    max_delay = max(int(value) for value in delays) if delays else 0
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    def update(token_id: str | None, start: pd.Timestamp | None, end: pd.Timestamp | None) -> None:
        if token_id is None or start is None or end is None or pd.isna(start) or pd.isna(end):
            return
        existing = bounds.get(str(token_id))
        if existing is None:
            bounds[str(token_id)] = (start, end)
            return
        bounds[str(token_id)] = (min(existing[0], start), max(existing[1], end))

    for row in pairs.to_dict(orient="records"):
        buy_ts = pd.to_datetime(row.get("buy_timestamp"), utc=True, errors="coerce")
        sell_ts = pd.to_datetime(row.get("sell_timestamp"), utc=True, errors="coerce")
        if pd.isna(buy_ts) or pd.isna(sell_ts):
            continue
        update(
            str(row.get("token_id")),
            buy_ts,
            sell_ts + pd.Timedelta(seconds=max_delay),
        )

    for row in open_positions.to_dict(orient="records"):
        buy_ts = pd.to_datetime(row.get("buy_timestamp"), utc=True, errors="coerce")
        if pd.isna(buy_ts):
            continue
        update(
            str(row.get("token_id")),
            buy_ts,
            buy_ts + pd.Timedelta(seconds=max_delay),
        )

    return [
        (token_id, start.to_pydatetime(), end.to_pydatetime())
        for token_id, (start, end) in bounds.items()
    ]


def _status_summary(group: pd.DataFrame, column: str, value: str) -> int:
    """Count rows with one status value in one diagnostics group."""

    if column not in group.columns:
        return 0
    return int((group[column] == value).sum())


def compute_copy_follow_wallet_exit_from_frame(
    recent_trades: pd.DataFrame,
    price_history: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    delays: Sequence[int] = DEFAULT_DELAYS,
    start_date: str | None = None,
    end_date: str | None = None,
    active_last_days: int = ACTIVE_LAST_DAYS_DEFAULT,
    analysis_asof: str | datetime | None = None,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute delayed copy-follow PnL that exits with the wallet when observable.

    Assumptions:
    - Only BUY trades are treated as copy-entry signals.
    - Later wallet SELL trades in the same token close prior buys FIFO.
    - If the wallet has not sold by the analysis cutoff, the copied trade remains
      open unless public market metadata shows resolution before the cutoff.
    - Delayed entry and delayed exit use the first public price point at or after
      the target timestamps.
    """

    cfg = settings or get_settings()
    filtered, start_ts, end_ts = _filter_window(recent_trades, start_date=start_date, end_date=end_date)
    normalized = _normalize_recent_trades(filtered)
    diagnostics_source = normalized.copy()
    if diagnostics_source.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    analysis_cutoff = pd.to_datetime(analysis_asof, utc=True, errors="coerce") if analysis_asof is not None else None
    if analysis_cutoff is None or pd.isna(analysis_cutoff):
        analysis_cutoff = end_ts or diagnostics_source["timestamp"].max()

    pairs, open_positions = build_copy_exit_pairs(diagnostics_source)
    base_records = []
    if not pairs.empty:
        base_records.extend(pairs.to_dict(orient="records"))
    if not open_positions.empty:
        base_records.extend(open_positions.to_dict(orient="records"))
    if not base_records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    price_index = _build_price_index(price_history)
    terminal_lookup = _build_terminal_lookup(markets)
    recent_cutoff = analysis_cutoff - pd.Timedelta(days=active_last_days)

    diagnostic_rows: list[dict[str, Any]] = []
    for record in base_records:
        token_id = str(record.get("token_id")) if record.get("token_id") is not None else None
        buy_ts = pd.to_datetime(record.get("buy_timestamp"), utc=True, errors="coerce")
        sell_ts = pd.to_datetime(record.get("sell_timestamp"), utc=True, errors="coerce")
        terminal_info = terminal_lookup.get(token_id or "")
        resolution_ts = terminal_info.get("resolution_ts") if terminal_info else None
        resolution_epoch = _to_epoch_seconds(resolution_ts) if resolution_ts is not None else None
        terminal_price = terminal_info.get("terminal_price") if terminal_info else None

        trade_record = dict(record)
        trade_record["analysis_window_start"] = start_ts
        trade_record["analysis_window_end"] = analysis_cutoff
        trade_record["active_in_last_window_days"] = bool(pd.notna(buy_ts) and buy_ts >= recent_cutoff)
        trade_record["terminal_price"] = terminal_price
        trade_record["terminal_price_source"] = (
            terminal_info.get("terminal_price_source") if terminal_info else "missing_market_terminal"
        )
        trade_record["resolution_ts"] = resolution_ts
        trade_record["resolution_source"] = (
            terminal_info.get("resolution_source") if terminal_info else "missing_market_terminal"
        )

        buy_epoch = _to_epoch_seconds(buy_ts)
        sell_epoch = _to_epoch_seconds(sell_ts) if pd.notna(sell_ts) else None
        copied_size = float(record.get("copied_size") or 0.0)

        for delay_seconds in delays:
            label = f"{int(delay_seconds)}s"
            entry_price_column = f"copy_entry_price_{label}"
            entry_source_column = f"copy_entry_source_{label}"
            entry_lag_column = f"copy_entry_lag_seconds_{label}"
            exit_price_column = f"copy_exit_price_{label}"
            exit_source_column = f"copy_exit_source_{label}"
            exit_lag_column = f"copy_exit_lag_seconds_{label}"
            raw_unit_column = f"copy_pnl_{label}"
            net_unit_column = f"copy_pnl_net_{label}"
            raw_usdc_column = f"copy_pnl_usdc_{label}"
            net_usdc_column = f"copy_pnl_net_usdc_{label}"
            cost_column = f"copy_cost_{label}"
            status_column = f"copy_status_{label}"
            exit_type_column = f"copy_exit_type_{label}"

            trade_record[entry_price_column] = None
            trade_record[entry_source_column] = "missing_prices"
            trade_record[entry_lag_column] = None
            trade_record[exit_price_column] = None
            trade_record[exit_source_column] = None
            trade_record[exit_lag_column] = None
            trade_record[raw_unit_column] = None
            trade_record[net_unit_column] = None
            trade_record[raw_usdc_column] = None
            trade_record[net_usdc_column] = None
            trade_record[cost_column] = None
            trade_record[exit_type_column] = None

            if sell_epoch is None:
                if terminal_info is None:
                    trade_record[status_column] = "open_without_market_metadata"
                    continue
                if terminal_price is None:
                    trade_record[status_column] = "open_without_terminal_price"
                    continue
                if resolution_epoch is None or resolution_ts is None or resolution_ts > analysis_cutoff:
                    trade_record[status_column] = "open_without_observed_exit"
                    continue

            entry_target_epoch = buy_epoch + int(delay_seconds)
            entry_forward = lookup_forward_price(price_index, token_id, entry_target_epoch)
            trade_record[entry_price_column] = entry_forward.price
            trade_record[entry_source_column] = entry_forward.source
            trade_record[entry_lag_column] = entry_forward.delta_seconds

            if entry_forward.price is None:
                trade_record[status_column] = "missing_entry_price_after_delay"
                continue

            entry_epoch = entry_target_epoch + int(entry_forward.delta_seconds or 0)

            if sell_epoch is not None:
                if sell_epoch <= entry_target_epoch:
                    trade_record[status_column] = "wallet_exit_before_delayed_entry"
                    trade_record[entry_price_column] = None
                    trade_record[entry_source_column] = "entry_after_wallet_exit"
                    continue

                exit_target_epoch = sell_epoch + int(delay_seconds)
                if entry_epoch >= exit_target_epoch:
                    trade_record[status_column] = "entry_after_or_at_delayed_exit"
                    trade_record[entry_price_column] = None
                    trade_record[entry_source_column] = "entry_after_wallet_exit"
                    continue

                exit_forward = lookup_forward_price(price_index, token_id, exit_target_epoch)
                trade_record[exit_price_column] = exit_forward.price
                trade_record[exit_source_column] = exit_forward.source
                trade_record[exit_lag_column] = exit_forward.delta_seconds
                if exit_forward.price is None:
                    trade_record[status_column] = "missing_exit_price_after_delay"
                    continue

                exit_epoch = exit_target_epoch + int(exit_forward.delta_seconds or 0)
                if exit_epoch <= entry_epoch:
                    trade_record[status_column] = "exit_price_not_after_entry"
                    trade_record[exit_price_column] = None
                    trade_record[exit_source_column] = "exit_not_after_entry"
                    continue

                raw_unit_pnl = float(exit_forward.price) - float(entry_forward.price)
                entry_cost = estimate_entry_only_cost(
                    pd.Series(record),
                    entry_price=float(entry_forward.price),
                    scenario=cfg.cost_scenario,
                    settings=cfg,
                )["total_cost"]
                exit_cost = estimate_entry_only_cost(
                    pd.Series(record),
                    entry_price=float(exit_forward.price),
                    scenario=cfg.cost_scenario,
                    settings=cfg,
                )["total_cost"]
                total_cost = float(entry_cost) + float(exit_cost)
                trade_record[exit_type_column] = "wallet_sell"
            else:
                if entry_epoch >= resolution_epoch:
                    trade_record[status_column] = "entry_at_or_after_resolution"
                    trade_record[entry_price_column] = None
                    trade_record[entry_source_column] = "entry_after_resolution"
                    continue

                raw_unit_pnl = float(terminal_price) - float(entry_forward.price)
                total_cost = float(
                    estimate_entry_only_cost(
                        pd.Series(record),
                        entry_price=float(entry_forward.price),
                        scenario=cfg.cost_scenario,
                        settings=cfg,
                    )["total_cost"]
                )
                trade_record[exit_price_column] = terminal_price
                trade_record[exit_source_column] = "gamma_terminal_price"
                trade_record[exit_lag_column] = None
                trade_record[exit_type_column] = "expiry"

            raw_usdc = raw_unit_pnl * copied_size
            net_unit = calculate_net_pnl(raw_unit_pnl, total_cost)
            net_usdc = calculate_net_pnl(raw_usdc, total_cost * copied_size)
            trade_record[raw_unit_column] = raw_unit_pnl
            trade_record[net_unit_column] = net_unit
            trade_record[raw_usdc_column] = raw_usdc
            trade_record[net_usdc_column] = net_usdc
            trade_record[cost_column] = total_cost * copied_size
            trade_record[status_column] = "ok"

        diagnostic_rows.append(trade_record)

    trade_diagnostics = pd.DataFrame.from_records(diagnostic_rows).sort_values(
        ["wallet_address", "buy_timestamp", "signal_trade_id", "trade_id"]
    )

    summary_rows: list[dict[str, Any]] = []
    grouped_source = diagnostics_source.groupby("wallet_address")
    for wallet, group in trade_diagnostics.groupby("wallet_address"):
        source_group = grouped_source.get_group(wallet) if wallet in grouped_source.groups else pd.DataFrame()
        record: dict[str, Any] = {
            "wallet_address": wallet,
            "n_recent_trades": int(len(source_group)),
            "n_buy_signals": int((source_group["side"] == "BUY").sum()) if not source_group.empty else 0,
            "n_markets": int(source_group["market_id"].nunique(dropna=True)) if not source_group.empty else 0,
            "first_trade_ts": source_group["timestamp"].min() if not source_group.empty else None,
            "most_recent_trade_ts": source_group["timestamp"].max() if not source_group.empty else None,
            "active_in_last_window_days": bool(
                not source_group.empty and (source_group["timestamp"] >= recent_cutoff).any()
            ),
            "copy_slices_total": int(len(group)),
            "copy_size_total": float(pd.to_numeric(group["copied_size"], errors="coerce").fillna(0.0).sum()),
            "wallet_sell_signal_slices": int((group["exit_type_signal"] == "wallet_sell").sum()),
            "open_signal_slices": int((group["exit_type_signal"] == "open").sum()),
        }
        for delay_seconds in delays:
            label = f"{int(delay_seconds)}s"
            raw_unit_column = f"copy_pnl_{label}"
            net_unit_column = f"copy_pnl_net_{label}"
            net_usdc_column = f"copy_pnl_net_usdc_{label}"
            status_column = f"copy_status_{label}"
            exit_type_column = f"copy_exit_type_{label}"

            gross_unit = pd.to_numeric(group[raw_unit_column], errors="coerce").dropna()
            valid_unit = pd.to_numeric(group[net_unit_column], errors="coerce").dropna()
            valid_usdc = pd.to_numeric(group[net_usdc_column], errors="coerce").dropna()
            record[f"valid_copy_slices_{label}"] = int(valid_unit.shape[0])
            record[f"avg_copy_pnl_{label}"] = float(gross_unit.mean()) if not gross_unit.empty else None
            record[f"median_copy_pnl_{label}"] = float(gross_unit.median()) if not gross_unit.empty else None
            record[f"copy_hit_rate_{label}"] = float((gross_unit > 0).mean()) if not gross_unit.empty else None
            record[f"avg_copy_pnl_net_{label}"] = float(valid_unit.mean()) if not valid_unit.empty else None
            record[f"median_copy_pnl_net_{label}"] = float(valid_unit.median()) if not valid_unit.empty else None
            record[f"copy_net_hit_rate_{label}"] = float((valid_unit > 0).mean()) if not valid_unit.empty else None
            record[f"total_copy_pnl_net_usdc_{label}"] = float(valid_usdc.sum()) if not valid_usdc.empty else None
            record[f"positive_wallet_net_avg_{label}"] = bool(not valid_unit.empty and float(valid_unit.mean()) > 0)
            record[f"positive_wallet_net_total_usdc_{label}"] = bool(
                not valid_usdc.empty and float(valid_usdc.sum()) > 0
            )
            record[f"wallet_sell_exits_{label}"] = _status_summary(group, exit_type_column, "wallet_sell")
            record[f"expiry_exits_{label}"] = _status_summary(group, exit_type_column, "expiry")
            record[f"pending_positions_{label}"] = _status_summary(group, status_column, "open_without_observed_exit")
            record[f"missing_prices_{label}"] = _status_summary(group, status_column, "missing_entry_price_after_delay") + _status_summary(group, status_column, "missing_exit_price_after_delay")
            record[f"wallet_exit_before_entry_{label}"] = _status_summary(
                group, status_column, "wallet_exit_before_delayed_entry"
            )
        summary_rows.append(record)

    wallet_summary = pd.DataFrame.from_records(summary_rows)
    sort_columns = [
        column
        for column in (
            f"total_copy_pnl_net_usdc_{int(delay)}s" for delay in sorted(delays, reverse=True)
        )
        if column in wallet_summary.columns
    ]
    sort_columns.append("wallet_address")
    wallet_summary = wallet_summary.sort_values(
        by=sort_columns,
        ascending=[False] * (len(sort_columns) - 1) + [True],
        na_position="last",
    )

    overview = pd.DataFrame(
        [
            {
                "analysis_window_start": start_ts.isoformat() if start_ts is not None else None,
                "analysis_window_end": analysis_cutoff.isoformat() if analysis_cutoff is not None else None,
                "wallets_in_report": int(wallet_summary["wallet_address"].nunique()) if not wallet_summary.empty else 0,
                "wallets_active_in_last_window_days": int(wallet_summary["active_in_last_window_days"].sum())
                if not wallet_summary.empty
                else 0,
                "raw_recent_trades_in_window": int(len(diagnostics_source)),
                "copy_slices_total": int(len(trade_diagnostics)),
                "wallet_sell_signal_slices": int((trade_diagnostics["exit_type_signal"] == "wallet_sell").sum()),
                "open_signal_slices": int((trade_diagnostics["exit_type_signal"] == "open").sum()),
                **{
                    f"valid_copy_slices_{int(delay)}s": int(wallet_summary.get(f"valid_copy_slices_{int(delay)}s", pd.Series(dtype=float)).sum())
                    if not wallet_summary.empty
                    else 0
                    for delay in delays
                },
                **{
                    f"wallets_positive_net_total_usdc_{int(delay)}s": int(
                        wallet_summary.get(f"positive_wallet_net_total_usdc_{int(delay)}s", pd.Series(dtype=bool)).sum()
                    )
                    if not wallet_summary.empty
                    else 0
                    for delay in delays
                },
            }
        ]
    )

    return wallet_summary, trade_diagnostics, overview


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Persist one DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_assumptions(path: Path) -> Path:
    """Persist the public-data assumptions behind wallet-exit copy analysis."""

    text = """# Copy Follow With Wallet Exit Assumptions

1. Signal definition:
   - Only observed public BUY trades are treated as copy-entry signals.
   - Observed public SELL trades are treated as exit signals for prior buys in
     the same wallet and token.

2. Position matching:
   - Buy-to-sell matching uses FIFO within the same wallet and token.
   - Partial sells can close part of one copied buy lot and leave the remainder open.

3. Delayed entry and exit:
   - Entry uses the first public `prices-history` point at or after
     `buy_timestamp + delay`.
   - If the wallet later sells, copied exit uses the first public `prices-history`
     point at or after `sell_timestamp + delay`.
   - If the wallet has already sold before the delayed copy entry would happen,
     the copied trade is marked invalid for that delay.

4. Open positions:
   - If the wallet has not publicly sold the copied lot by the analysis cutoff,
     the position is left open unless public market metadata shows resolution by
     that cutoff.
   - Open positions without an observed sell or public terminal value are kept in
     the diagnostics as pending, not forced closed.

5. Cost model:
   - Wallet-sell exits use one-way entry cost plus one-way exit cost.
   - Expiry exits use entry-only cost.
   - Costs are still research approximations, not historical fee ledger exports.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_copy_follow_wallet_exit_analysis(
    session: Session,
    *,
    wallets: Sequence[str],
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    delays: Sequence[int] = DEFAULT_DELAYS,
    active_last_days: int = ACTIVE_LAST_DAYS_DEFAULT,
    analysis_asof: str | datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run delayed copy-follow analysis using observed wallet exits when available."""

    start_dt = pd.to_datetime(start_date, utc=True).to_pydatetime()
    end_dt = pd.to_datetime(end_date, utc=True).to_pydatetime()
    recent_trades = load_recent_wallet_trades(
        session,
        wallets=list(wallets),
        recent_window_start=start_dt,
        recent_window_end=end_dt,
    )
    normalized = _normalize_recent_trades(recent_trades)
    pairs, open_positions = build_copy_exit_pairs(normalized)
    token_bounds = collect_price_backfill_targets(pairs, open_positions.iloc[0:0], delays=delays)
    token_ids = sorted({token_id for token_id, _, _ in token_bounds})
    max_delay = max(int(value) for value in delays) if delays else 0
    price_history = load_price_history_frame(
        session,
        token_ids=token_ids,
        start_ts=start_dt - pd.Timedelta(hours=1),
        end_ts=end_dt + pd.Timedelta(seconds=max_delay + 300),
    )
    markets = load_markets_frame(session)

    wallet_summary, trade_diagnostics, overview = compute_copy_follow_wallet_exit_from_frame(
        recent_trades,
        price_history,
        markets,
        delays=delays,
        start_date=start_date,
        end_date=end_date,
        active_last_days=active_last_days,
        analysis_asof=analysis_asof or end_date,
        settings=settings,
    )

    output_path = Path(output_dir)
    delay_label = "_".join(f"{int(delay)}s" for delay in delays)
    window_label = f"{start_date[:10].replace('-', '')}_{end_date[:10].replace('-', '')}"
    wallet_path = _write_csv(
        wallet_summary,
        output_path / f"copy_follow_wallet_exit_{delay_label}_{window_label}.csv",
    )
    summary_path = _write_csv(
        overview,
        output_path / "copy_follow_wallet_exit_summary.csv",
    )
    diagnostics_path = _write_csv(
        trade_diagnostics,
        output_path / f"copy_follow_wallet_exit_trade_diagnostics_{delay_label}_{window_label}.csv",
    )
    assumptions_path = _write_assumptions(output_path / "copy_follow_wallet_exit_assumptions.md")

    return {
        "wallet_summary": wallet_summary,
        "trade_diagnostics": trade_diagnostics,
        "overview": overview,
        "wallet_path": wallet_path,
        "summary_path": summary_path,
        "diagnostics_path": diagnostics_path,
        "assumptions_path": assumptions_path,
        "token_bounds": token_bounds,
        "paired_rows": pairs,
        "open_rows": open_positions,
    }
