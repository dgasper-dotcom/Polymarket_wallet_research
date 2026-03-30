"""Research-only copy-follow backtest for the manual seed wallet cohort.

This module extends the existing wallet-exit copy logic in two ways:
1. It keeps unresolved copied positions open and marks them to current public
   prices instead of dropping them.
2. When the delayed target timestamp has no "exactish" public print nearby, it
   uses a conservative maker-order approximation rather than assuming a taker
   fill was always possible.

The maker approximation is intentionally simple because historical order-book
    snapshots are not available from public data:
- if the first public price at or after the delayed target is within
  ``MAKER_REQUIRED_LAG_SECONDS`` we treat it as an executable taker fill;
- otherwise we post a passive limit at the latest public price at or before the
  target time and only fill if later public prints cross that limit before the
  position must be closed;
- maker fills pay only the modeled event fee by default, not spread/slippage.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Sequence

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from research.all_active_reevaluation import _lookup_latest_price_at_or_before
from research.copy_follow_expiry import _build_terminal_lookup, load_markets_frame
from research.copy_follow_wallet_exit import build_copy_exit_pairs
from research.costs import calculate_net_pnl, estimate_entry_only_cost, estimate_polymarket_fee
from research.delay_analysis import _build_price_index, _to_epoch_seconds, lookup_forward_price, load_price_history_frame
from research.recent_wallet_trade_capture import load_recent_wallet_trades


DEFAULT_DELAYS = (5, 15, 30, 60)
DEFAULT_OUTPUT_DIR = "exports/manual_seed_copy_backtest_week"
DEFAULT_START_DATE = "2026-03-22T00:00:00Z"
DEFAULT_END_DATE = "2026-03-29T23:59:59Z"
MAKER_REQUIRED_LAG_SECONDS = 60


@dataclass
class FillAttempt:
    """One modeled copy fill."""

    price: float | None
    source: str
    delta_seconds: int | None
    fill_epoch: int | None
    fill_mode: str


def _normalize_wallets(values: Sequence[str]) -> list[str]:
    """Normalize wallet ids while preserving order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        wallet = str(value or "").strip().lower()
        if not wallet or wallet in seen:
            continue
        seen.add(wallet)
        ordered.append(wallet)
    return ordered


def _load_wallet_file(path: str | Path) -> list[str]:
    """Load newline-delimited wallet ids."""

    return _normalize_wallets(Path(path).read_text(encoding="utf-8").splitlines())


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Persist one frame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_assumptions(path: Path) -> Path:
    """Write the backtest assumptions to disk."""

    content = """# Manual Seed Copy Backtest Assumptions

- Window is fixed to the requested recent week; dates are recorded in the CSV.
- Only public Polymarket data are used.
- Wallet BUY trades are copy-entry signals.
- Wallet SELL trades close copied positions FIFO.
- If a wallet has not sold, the copied position remains open unless the market
  publicly resolved by the analysis cutoff.
- Delayed fills use the first public price at or after the delayed target when
  a print appears within 60 seconds.
- If no such near-immediate print exists, the model uses a conservative passive
  maker approximation:
  - BUY maker order posts at the latest public price at or before the delayed
    entry time and only fills if later public prices trade at or below that
    level before the wallet exit or analysis cutoff.
  - SELL maker order posts at the latest public price at or before the delayed
    exit time and only fills if later public prices trade at or above that
    level before the analysis cutoff.
- Maker fills are charged modeled fees only; taker fills are charged the
  existing entry-only cost model on each executed leg.
- Unrealized PnL is marked to the latest public price at or before the analysis
  cutoff; if missing, Gamma terminal outcome prices are used only when a public
  terminal value is already available.
- Because historical order-book snapshots are not publicly available, maker vs
  taker classification is a research approximation rather than ground truth.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _collect_token_bounds(
    pairs: pd.DataFrame,
    open_positions: pd.DataFrame,
    *,
    delays: Sequence[int],
    analysis_asof: pd.Timestamp,
) -> list[tuple[str, datetime, datetime]]:
    """Collect price-history bounds for realized and open copied positions."""

    max_delay = max(int(value) for value in delays) if delays else 0
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    def update(token_id: str | None, start: pd.Timestamp | None, end: pd.Timestamp | None) -> None:
        if token_id is None or start is None or end is None or pd.isna(start) or pd.isna(end):
            return
        token = str(token_id)
        existing = bounds.get(token)
        if existing is None:
            bounds[token] = (start, end)
            return
        bounds[token] = (min(existing[0], start), max(existing[1], end))

    for row in pairs.to_dict(orient="records"):
        buy_ts = pd.to_datetime(row.get("buy_timestamp"), utc=True, errors="coerce")
        sell_ts = pd.to_datetime(row.get("sell_timestamp"), utc=True, errors="coerce")
        if pd.isna(buy_ts):
            continue
        end_ts = sell_ts + pd.Timedelta(seconds=max_delay) if pd.notna(sell_ts) else analysis_asof
        update(row.get("token_id"), buy_ts, end_ts)

    for row in open_positions.to_dict(orient="records"):
        buy_ts = pd.to_datetime(row.get("buy_timestamp"), utc=True, errors="coerce")
        if pd.isna(buy_ts):
            continue
        update(row.get("token_id"), buy_ts, analysis_asof)

    return [(token_id, start.to_pydatetime(), end.to_pydatetime()) for token_id, (start, end) in bounds.items()]


def _maker_one_way_cost(price: float, *, settings: Settings | None = None) -> float:
    """Return a conservative maker-order one-way cost estimate in price units."""

    cfg = settings or get_settings()
    scenario = cfg.cost_scenario
    fee_enabled = scenario in {"base", "conservative"}
    fee = estimate_polymarket_fee(price, fee_enabled=fee_enabled, settings=cfg)
    extra_penalty = float(cfg.extra_cost_penalty) if scenario == "conservative" else 0.0
    return float(fee) + extra_penalty


def _scan_passive_fill(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    *,
    target_epoch: int,
    limit_price: float,
    is_buy: bool,
    until_epoch: int,
) -> FillAttempt:
    """Scan later public prints for a conservative maker fill."""

    if token_id is None or token_id not in index:
        return FillAttempt(None, "missing_prices", None, None, "unfilled_maker")

    times, prices = index[str(token_id)]
    position = bisect_left(times, target_epoch)
    while position < len(times) and times[position] <= until_epoch:
        observed_price = float(prices[position])
        crossed = observed_price <= limit_price if is_buy else observed_price >= limit_price
        if crossed:
            return FillAttempt(
                price=float(limit_price),
                source="maker_limit_from_last_price",
                delta_seconds=int(times[position] - target_epoch),
                fill_epoch=int(times[position]),
                fill_mode="maker",
            )
        position += 1

    return FillAttempt(None, "maker_unfilled", None, None, "unfilled_maker")


def _entry_fill(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    *,
    target_epoch: int,
    until_epoch: int,
) -> FillAttempt:
    """Return one modeled copied entry fill."""

    forward = lookup_forward_price(index, token_id, target_epoch)
    if forward.price is not None and forward.delta_seconds is not None and forward.delta_seconds <= MAKER_REQUIRED_LAG_SECONDS:
        return FillAttempt(
            price=float(forward.price),
            source=forward.source,
            delta_seconds=int(forward.delta_seconds),
            fill_epoch=target_epoch + int(forward.delta_seconds),
            fill_mode="taker",
        )

    last_price, _, _age = _lookup_latest_price_at_or_before(index, token_id, target_epoch)
    if last_price is not None:
        maker = _scan_passive_fill(
            index,
            token_id,
            target_epoch=target_epoch,
            limit_price=float(last_price),
            is_buy=True,
            until_epoch=until_epoch,
        )
        if maker.price is not None:
            return maker
        return maker

    if forward.price is not None and forward.delta_seconds is not None:
        return FillAttempt(
            price=float(forward.price),
            source=forward.source,
            delta_seconds=int(forward.delta_seconds),
            fill_epoch=target_epoch + int(forward.delta_seconds),
            fill_mode="taker_late_no_prior_price",
        )
    return FillAttempt(None, "missing_entry_price_after_delay", None, None, "unfilled")


def _exit_fill(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    *,
    target_epoch: int,
    until_epoch: int,
) -> FillAttempt:
    """Return one modeled copied exit fill."""

    forward = lookup_forward_price(index, token_id, target_epoch)
    if forward.price is not None and forward.delta_seconds is not None and forward.delta_seconds <= MAKER_REQUIRED_LAG_SECONDS:
        return FillAttempt(
            price=float(forward.price),
            source=forward.source,
            delta_seconds=int(forward.delta_seconds),
            fill_epoch=target_epoch + int(forward.delta_seconds),
            fill_mode="taker",
        )

    last_price, _, _age = _lookup_latest_price_at_or_before(index, token_id, target_epoch)
    if last_price is not None:
        maker = _scan_passive_fill(
            index,
            token_id,
            target_epoch=target_epoch,
            limit_price=float(last_price),
            is_buy=False,
            until_epoch=until_epoch,
        )
        if maker.price is not None:
            return maker
        return maker

    if forward.price is not None and forward.delta_seconds is not None:
        return FillAttempt(
            price=float(forward.price),
            source=forward.source,
            delta_seconds=int(forward.delta_seconds),
            fill_epoch=target_epoch + int(forward.delta_seconds),
            fill_mode="taker_late_no_prior_price",
        )
    return FillAttempt(None, "missing_exit_price_after_delay", None, None, "unfilled")


def _mark_price(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    *,
    cutoff_epoch: int,
    terminal_info: dict[str, Any] | None,
) -> tuple[float | None, str, int | None]:
    """Return the latest public mark at or before the cutoff."""

    price, source, age_seconds = _lookup_latest_price_at_or_before(index, token_id, cutoff_epoch)
    if price is not None:
        return float(price), source, age_seconds
    if terminal_info and terminal_info.get("terminal_price") is not None:
        return float(terminal_info["terminal_price"]), "gamma_terminal_current_fallback", None
    return None, "missing_mark_price", None


def _sum_or_none(series: pd.Series) -> float | None:
    """Sum one numeric series if any values exist."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.sum())


def compute_manual_seed_copy_backtest_from_frame(
    recent_trades: pd.DataFrame,
    price_history: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    delays: Sequence[int] = DEFAULT_DELAYS,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    analysis_asof: str | datetime | None = None,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest delayed copy-follow for the selected recent wallet cohort."""

    cfg = settings or get_settings()
    if recent_trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    normalized = recent_trades.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["wallet_address", "token_id", "timestamp"]).reset_index(drop=True)
    normalized["wallet_address"] = normalized["wallet_address"].astype(str).str.lower()

    analysis_cutoff = pd.to_datetime(analysis_asof, utc=True, errors="coerce")
    if analysis_cutoff is None or pd.isna(analysis_cutoff):
        analysis_cutoff = pd.to_datetime(end_date, utc=True, errors="coerce")
    cutoff_epoch = _to_epoch_seconds(analysis_cutoff)

    pairs, open_positions = build_copy_exit_pairs(normalized)
    base_records: list[dict[str, Any]] = []
    if not pairs.empty:
        base_records.extend(pairs.to_dict(orient="records"))
    if not open_positions.empty:
        base_records.extend(open_positions.to_dict(orient="records"))
    if not base_records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    price_index = _build_price_index(price_history)
    terminal_lookup = _build_terminal_lookup(markets)

    diagnostic_rows: list[dict[str, Any]] = []
    for record in base_records:
        wallet_id = str(record.get("wallet_address") or "")
        token_id = str(record.get("token_id") or "")
        copied_size = float(record.get("copied_size") or 0.0)
        buy_ts = pd.to_datetime(record.get("buy_timestamp"), utc=True, errors="coerce")
        sell_ts = pd.to_datetime(record.get("sell_timestamp"), utc=True, errors="coerce")
        buy_epoch = _to_epoch_seconds(buy_ts)
        sell_epoch = _to_epoch_seconds(sell_ts) if pd.notna(sell_ts) else None
        terminal_info = terminal_lookup.get(token_id) or {}
        resolution_ts = pd.to_datetime(terminal_info.get("resolution_ts"), utc=True, errors="coerce")
        resolution_epoch = _to_epoch_seconds(resolution_ts) if pd.notna(resolution_ts) else None
        terminal_price = terminal_info.get("terminal_price")

        trade_record: dict[str, Any] = {
            "wallet_address": wallet_id,
            "token_id": token_id,
            "market_id": record.get("market_id"),
            "trade_id": record.get("trade_id"),
            "signal_trade_id": record.get("signal_trade_id"),
            "exit_trade_id": record.get("exit_trade_id"),
            "buy_timestamp": buy_ts,
            "sell_timestamp": sell_ts,
            "buy_price_signal": record.get("buy_price_signal"),
            "sell_price_signal": record.get("sell_price_signal"),
            "copied_size": copied_size,
            "exit_type_signal": record.get("exit_type_signal"),
            "analysis_asof": analysis_cutoff,
            "resolution_ts": resolution_ts,
            "terminal_price": terminal_price,
            "terminal_price_source": terminal_info.get("terminal_price_source"),
        }

        for delay_seconds in delays:
            label = f"{int(delay_seconds)}s"
            entry_target_epoch = buy_epoch + int(delay_seconds)
            entry_until_epoch = sell_epoch + int(delay_seconds) if sell_epoch is not None else cutoff_epoch
            entry = _entry_fill(price_index, token_id, target_epoch=entry_target_epoch, until_epoch=entry_until_epoch)

            trade_record[f"copy_entry_price_{label}"] = entry.price
            trade_record[f"copy_entry_source_{label}"] = entry.source
            trade_record[f"copy_entry_lag_seconds_{label}"] = entry.delta_seconds
            trade_record[f"copy_entry_fill_mode_{label}"] = entry.fill_mode

            trade_record[f"copy_exit_price_{label}"] = None
            trade_record[f"copy_exit_source_{label}"] = None
            trade_record[f"copy_exit_lag_seconds_{label}"] = None
            trade_record[f"copy_exit_fill_mode_{label}"] = None
            trade_record[f"copy_status_{label}"] = None
            trade_record[f"copy_exit_type_{label}"] = None
            trade_record[f"copy_pnl_net_usdc_{label}"] = None
            trade_record[f"copy_unrealized_mtm_net_usdc_{label}"] = None
            trade_record[f"copy_combined_net_usdc_{label}"] = None

            if entry.price is None or entry.fill_epoch is None:
                trade_record[f"copy_status_{label}"] = "entry_unfilled"
                continue

            entry_cost_unit = (
                _maker_one_way_cost(float(entry.price), settings=cfg)
                if entry.fill_mode == "maker"
                else float(
                    estimate_entry_only_cost(
                        pd.Series(record),
                        entry_price=float(entry.price),
                        scenario=cfg.cost_scenario,
                        settings=cfg,
                    )["total_cost"]
                )
            )

            if sell_epoch is not None:
                exit_target_epoch = sell_epoch + int(delay_seconds)
                if entry.fill_epoch >= exit_target_epoch:
                    trade_record[f"copy_status_{label}"] = "entry_after_or_at_targeted_exit"
                    continue

                exit_fill = _exit_fill(
                    price_index,
                    token_id,
                    target_epoch=exit_target_epoch,
                    until_epoch=cutoff_epoch,
                )
                trade_record[f"copy_exit_price_{label}"] = exit_fill.price
                trade_record[f"copy_exit_source_{label}"] = exit_fill.source
                trade_record[f"copy_exit_lag_seconds_{label}"] = exit_fill.delta_seconds
                trade_record[f"copy_exit_fill_mode_{label}"] = exit_fill.fill_mode

                if exit_fill.price is not None and exit_fill.fill_epoch is not None and exit_fill.fill_epoch > entry.fill_epoch:
                    exit_cost_unit = (
                        _maker_one_way_cost(float(exit_fill.price), settings=cfg)
                        if exit_fill.fill_mode == "maker"
                        else float(
                            estimate_entry_only_cost(
                                pd.Series(record),
                                entry_price=float(exit_fill.price),
                                scenario=cfg.cost_scenario,
                                settings=cfg,
                            )["total_cost"]
                        )
                    )
                    raw_unit = float(exit_fill.price) - float(entry.price)
                    total_cost_unit = float(entry_cost_unit) + float(exit_cost_unit)
                    raw_usdc = raw_unit * copied_size
                    net_usdc = calculate_net_pnl(raw_usdc, total_cost_unit * copied_size)
                    trade_record[f"copy_exit_type_{label}"] = "wallet_sell"
                    trade_record[f"copy_status_{label}"] = "realized"
                    trade_record[f"copy_pnl_net_usdc_{label}"] = net_usdc
                    trade_record[f"copy_combined_net_usdc_{label}"] = net_usdc
                    continue

                mark_price, mark_source, mark_age = _mark_price(
                    price_index,
                    token_id,
                    cutoff_epoch=cutoff_epoch,
                    terminal_info=terminal_info,
                )
                trade_record[f"copy_exit_price_{label}"] = mark_price
                trade_record[f"copy_exit_source_{label}"] = mark_source
                trade_record[f"copy_exit_lag_seconds_{label}"] = mark_age
                trade_record[f"copy_exit_fill_mode_{label}"] = "mtm_after_unfilled_exit"
                if mark_price is None:
                    trade_record[f"copy_status_{label}"] = "open_without_mark_price"
                    continue
                raw_unit = float(mark_price) - float(entry.price)
                raw_usdc = raw_unit * copied_size
                mtm_net_usdc = calculate_net_pnl(raw_usdc, entry_cost_unit * copied_size)
                trade_record[f"copy_exit_type_{label}"] = "mark_to_market_after_unfilled_exit"
                trade_record[f"copy_status_{label}"] = "unrealized_after_unfilled_exit"
                trade_record[f"copy_unrealized_mtm_net_usdc_{label}"] = mtm_net_usdc
                trade_record[f"copy_combined_net_usdc_{label}"] = mtm_net_usdc
                continue

            # Wallet has not sold yet.
            if resolution_epoch is not None and resolution_epoch <= cutoff_epoch and terminal_price is not None:
                if entry.fill_epoch >= resolution_epoch:
                    trade_record[f"copy_status_{label}"] = "entry_at_or_after_resolution"
                    continue
                raw_unit = float(terminal_price) - float(entry.price)
                raw_usdc = raw_unit * copied_size
                net_usdc = calculate_net_pnl(raw_usdc, entry_cost_unit * copied_size)
                trade_record[f"copy_exit_price_{label}"] = float(terminal_price)
                trade_record[f"copy_exit_source_{label}"] = str(terminal_info.get("terminal_price_source") or "gamma_terminal")
                trade_record[f"copy_exit_fill_mode_{label}"] = "expiry"
                trade_record[f"copy_exit_type_{label}"] = "expiry"
                trade_record[f"copy_status_{label}"] = "realized_expiry"
                trade_record[f"copy_pnl_net_usdc_{label}"] = net_usdc
                trade_record[f"copy_combined_net_usdc_{label}"] = net_usdc
                continue

            mark_price, mark_source, mark_age = _mark_price(
                price_index,
                token_id,
                cutoff_epoch=cutoff_epoch,
                terminal_info=terminal_info,
            )
            trade_record[f"copy_exit_price_{label}"] = mark_price
            trade_record[f"copy_exit_source_{label}"] = mark_source
            trade_record[f"copy_exit_lag_seconds_{label}"] = mark_age
            trade_record[f"copy_exit_fill_mode_{label}"] = "mark_to_market"
            if mark_price is None:
                trade_record[f"copy_status_{label}"] = "open_without_mark_price"
                continue
            raw_unit = float(mark_price) - float(entry.price)
            raw_usdc = raw_unit * copied_size
            mtm_net_usdc = calculate_net_pnl(raw_usdc, entry_cost_unit * copied_size)
            trade_record[f"copy_exit_type_{label}"] = "mark_to_market"
            trade_record[f"copy_status_{label}"] = "unrealized_open"
            trade_record[f"copy_unrealized_mtm_net_usdc_{label}"] = mtm_net_usdc
            trade_record[f"copy_combined_net_usdc_{label}"] = mtm_net_usdc
        diagnostic_rows.append(trade_record)

    trade_diagnostics = pd.DataFrame.from_records(diagnostic_rows).sort_values(
        ["wallet_address", "buy_timestamp", "signal_trade_id", "trade_id"]
    )

    summary_rows: list[dict[str, Any]] = []
    grouped_source = normalized.groupby("wallet_address")
    for wallet_id, group in trade_diagnostics.groupby("wallet_address", sort=False):
        source_group = grouped_source.get_group(wallet_id) if wallet_id in grouped_source.groups else pd.DataFrame()
        row: dict[str, Any] = {
            "wallet_address": wallet_id,
            "n_recent_trades": int(len(source_group)),
            "n_buy_signals": int((source_group["side"].astype(str).str.upper() == "BUY").sum()) if not source_group.empty else 0,
            "n_markets": int(source_group["market_id"].nunique(dropna=True)) if not source_group.empty else 0,
            "first_trade_ts": source_group["timestamp"].min() if not source_group.empty else None,
            "most_recent_trade_ts": source_group["timestamp"].max() if not source_group.empty else None,
            "copy_slices_total": int(len(group)),
        }
        for delay_seconds in delays:
            label = f"{int(delay_seconds)}s"
            realized_col = f"copy_pnl_net_usdc_{label}"
            mtm_col = f"copy_unrealized_mtm_net_usdc_{label}"
            combined_col = f"copy_combined_net_usdc_{label}"
            status_col = f"copy_status_{label}"
            entry_mode_col = f"copy_entry_fill_mode_{label}"
            exit_mode_col = f"copy_exit_fill_mode_{label}"

            row[f"realized_copy_slices_{label}"] = int(group[status_col].isin(["realized", "realized_expiry"]).sum())
            row[f"open_copy_slices_{label}"] = int(group[status_col].isin(["unrealized_open", "unrealized_after_unfilled_exit"]).sum())
            row[f"entry_unfilled_slices_{label}"] = int((group[status_col] == "entry_unfilled").sum())
            row[f"realized_net_total_usdc_{label}"] = _sum_or_none(group[realized_col])
            row[f"unrealized_mtm_net_total_usdc_{label}"] = _sum_or_none(group[mtm_col])
            row[f"combined_net_total_usdc_{label}"] = _sum_or_none(group[combined_col])
            row[f"wallets_positive_combined_{label}"] = None
            row[f"maker_entry_fills_{label}"] = int((group[entry_mode_col] == "maker").sum())
            row[f"maker_exit_fills_{label}"] = int((group[exit_mode_col] == "maker").sum())
            row[f"maker_unfilled_entry_slices_{label}"] = int((group[entry_mode_col] == "unfilled_maker").sum())
            row[f"mtm_after_unfilled_exit_slices_{label}"] = int(
                (group[status_col] == "unrealized_after_unfilled_exit").sum()
            )
        summary_rows.append(row)

    wallet_summary = pd.DataFrame.from_records(summary_rows).sort_values(
        ["combined_net_total_usdc_30s", "wallet_address"],
        ascending=[False, True],
        na_position="last",
    )

    overview_row: dict[str, Any] = {
        "wallets_requested": int(normalized["wallet_address"].nunique()),
        "raw_recent_trades_in_window": int(len(normalized)),
        "copy_slices_total": int(len(trade_diagnostics)),
        "analysis_window_start": start_date,
        "analysis_window_end": end_date,
        "analysis_asof": analysis_cutoff.isoformat(),
    }
    for delay_seconds in delays:
        label = f"{int(delay_seconds)}s"
        overview_row[f"realized_copy_slices_{label}"] = int(
            trade_diagnostics[f"copy_status_{label}"].isin(["realized", "realized_expiry"]).sum()
        )
        overview_row[f"open_copy_slices_{label}"] = int(
            trade_diagnostics[f"copy_status_{label}"].isin(["unrealized_open", "unrealized_after_unfilled_exit"]).sum()
        )
        overview_row[f"entry_unfilled_slices_{label}"] = int(
            (trade_diagnostics[f"copy_status_{label}"] == "entry_unfilled").sum()
        )
        overview_row[f"combined_net_total_usdc_{label}"] = _sum_or_none(
            trade_diagnostics[f"copy_combined_net_usdc_{label}"]
        )
    overview = pd.DataFrame([overview_row])
    return wallet_summary.reset_index(drop=True), trade_diagnostics.reset_index(drop=True), overview


def run_manual_seed_copy_backtest(
    session: Session,
    *,
    wallets: Sequence[str],
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    delays: Sequence[int] = DEFAULT_DELAYS,
    analysis_asof: str | datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the manual-seed copy backtest and persist reports."""

    cfg = settings or get_settings()
    normalized_wallets = _normalize_wallets(wallets)
    start_dt = pd.to_datetime(start_date, utc=True).to_pydatetime()
    end_dt = pd.to_datetime(end_date, utc=True).to_pydatetime()
    analysis_cutoff = pd.to_datetime(analysis_asof, utc=True, errors="coerce")
    if analysis_cutoff is None or pd.isna(analysis_cutoff):
        analysis_cutoff = pd.to_datetime(end_date, utc=True, errors="coerce")

    recent_trades = load_recent_wallet_trades(
        session,
        wallets=normalized_wallets,
        recent_window_start=start_dt,
        recent_window_end=end_dt,
    )
    normalized_trades = recent_trades.copy()
    if not normalized_trades.empty:
        normalized_trades["timestamp"] = pd.to_datetime(normalized_trades["timestamp"], utc=True, errors="coerce")
        normalized_trades["wallet_address"] = normalized_trades["wallet_address"].astype(str).str.lower()

    pairs, open_positions = build_copy_exit_pairs(normalized_trades)
    token_bounds = _collect_token_bounds(
        pairs,
        open_positions,
        delays=delays,
        analysis_asof=analysis_cutoff,
    )
    token_ids = sorted({str(token_id) for token_id, _, _ in token_bounds})
    price_history = load_price_history_frame(
        session,
        token_ids=token_ids,
        start_ts=start_dt - pd.Timedelta(hours=1),
        end_ts=analysis_cutoff.to_pydatetime() + pd.Timedelta(hours=1),
    )
    markets = load_markets_frame(session)

    wallet_summary, trade_diagnostics, overview = compute_manual_seed_copy_backtest_from_frame(
        normalized_trades,
        price_history,
        markets,
        delays=delays,
        start_date=start_date,
        end_date=end_date,
        analysis_asof=analysis_cutoff,
        settings=cfg,
    )

    output_path = Path(output_dir)
    delay_label = "_".join(f"{int(delay)}s" for delay in delays)
    window_label = f"{start_date[:10].replace('-', '')}_{end_date[:10].replace('-', '')}"
    wallet_path = _write_csv(
        wallet_summary,
        output_path / f"manual_seed_copy_backtest_wallet_summary_{delay_label}_{window_label}.csv",
    )
    diagnostics_path = _write_csv(
        trade_diagnostics,
        output_path / f"manual_seed_copy_backtest_trade_diagnostics_{delay_label}_{window_label}.csv",
    )
    summary_path = _write_csv(
        overview,
        output_path / "manual_seed_copy_backtest_summary.csv",
    )
    assumptions_path = _write_assumptions(output_path / "manual_seed_copy_backtest_assumptions.md")

    return {
        "wallet_summary": wallet_summary,
        "trade_diagnostics": trade_diagnostics,
        "overview": overview,
        "wallet_path": wallet_path,
        "diagnostics_path": diagnostics_path,
        "summary_path": summary_path,
        "assumptions_path": assumptions_path,
        "token_bounds": token_bounds,
        "pairs": pairs,
        "open_positions": open_positions,
    }
