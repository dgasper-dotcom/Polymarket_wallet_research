"""Performance tracking for the unified house paper portfolio."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any
import sys

sys.modules.setdefault("pyarrow", None)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from db.session import get_session
from ingestion.markets import backfill_markets_for_references
from research.all_active_reevaluation import (
    _load_price_history_chunked,
    _lookup_latest_price_at_or_before,
)
from research.copy_follow_expiry import _build_terminal_lookup, load_markets_frame
from research.costs import calculate_net_pnl, estimate_entry_only_cost
from research.delay_analysis import _build_price_index, _to_epoch_seconds
from research.house_portfolio_rules import (
    aggregate_wallet_contributions,
    allowed_house_notional,
    apply_wallet_open_notional_delta,
    positive_contribution_shares,
    release_wallet_open_notional,
    _safe_float,
)


DEFAULT_CONSOLIDATED_DIR = "exports/manual_seed_paper_tracking/consolidated"
DEFAULT_OUTPUT_DIR = "exports/manual_seed_paper_tracking/performance"


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _build_house_position_ledger(
    tape: pd.DataFrame,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Reconstruct weighted-cost house positions from the consolidated signal tape."""
    return _build_house_position_ledger_with_cap(
        tape,
        max_position_notional_usdc=None,
    )


def _build_house_position_ledger_with_cap(
    tape: pd.DataFrame,
    *,
    max_position_notional_usdc: float | None,
    max_event_notional_usdc: float | None = None,
    max_wallet_open_notional_usdc: float | None = None,
    max_total_open_notional_usdc: float | None = None,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Reconstruct weighted-cost house positions from the consolidated signal tape."""

    if tape.empty:
        return pd.DataFrame(), pd.DataFrame()

    normalized = tape.copy()
    normalized["first_ts"] = pd.to_datetime(normalized["first_ts"], utc=True, errors="coerce")
    normalized["last_ts"] = pd.to_datetime(normalized["last_ts"], utc=True, errors="coerce")
    normalized["trade_count"] = pd.to_numeric(normalized["trade_count"], errors="coerce").fillna(0).astype(int)
    normalized["unique_wallet_count"] = (
        pd.to_numeric(normalized["unique_wallet_count"], errors="coerce").fillna(0).astype(int)
    )
    normalized["total_notional_usdc"] = pd.to_numeric(
        normalized["total_notional_usdc"], errors="coerce"
    ).fillna(0.0)
    normalized["avg_signal_price"] = pd.to_numeric(normalized["avg_signal_price"], errors="coerce")
    normalized = normalized.sort_values(["first_ts", "cluster_id"]).reset_index(drop=True)

    open_positions: dict[str, dict[str, Any]] = {}
    closed_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    event_open_notional: dict[str, float] = {}
    wallet_open_notional: dict[str, float] = {}
    total_open_notional = 0.0

    for row in normalized.to_dict(orient="records"):
        action = str(row.get("action") or "")
        token_id = str(row.get("token_id") or "")
        price = row.get("avg_signal_price")
        signaled_notional = float(row.get("total_notional_usdc") or 0.0)
        contracts = None
        if price is not None and pd.notna(price) and float(price) > 0:
            contracts = signaled_notional / float(price)
        support_wallets = set(json.loads(row.get("supporting_wallets") or "[]"))
        event_key = str(row.get("event_title") or "")

        if action == "open_long":
            executed_notional, skipped_notional, binding_caps = allowed_house_notional(
                signaled_notional=signaled_notional,
                current_position_notional=0.0,
                current_event_notional=event_open_notional.get(event_key, 0.0),
                current_total_open_notional=total_open_notional,
                supporting_wallets=support_wallets,
                wallet_open_notional=wallet_open_notional,
                max_position_notional_usdc=max_position_notional_usdc,
                max_event_notional_usdc=max_event_notional_usdc,
                max_wallet_open_notional_usdc=max_wallet_open_notional_usdc,
                max_total_open_notional_usdc=max_total_open_notional_usdc,
            )
            executed_contracts = (
                None
                if contracts is None or signaled_notional <= 0
                else contracts * (executed_notional / signaled_notional)
            )
            if executed_contracts is None or executed_contracts <= 0:
                reason = "missing_open_price" if contracts is None else "position_cap_reached"
                if contracts is not None and "event_cap" in binding_caps:
                    reason = "event_cap_reached"
                elif contracts is not None and "wallet_cap" in binding_caps:
                    reason = "wallet_cap_reached"
                elif contracts is not None and "book_cap" in binding_caps:
                    reason = "book_cap_reached"
                skipped_rows.append(
                    {
                        "cluster_id": row.get("cluster_id"),
                        "action": action,
                        "reason": reason,
                        "binding_caps": ",".join(binding_caps),
                        "token_id": token_id,
                        "market_id": row.get("market_id"),
                        "event_title": row.get("event_title"),
                        "outcome": row.get("outcome"),
                        "signaled_notional_usdc": signaled_notional,
                        "executed_notional_usdc": executed_notional,
                        "skipped_notional_usdc": skipped_notional,
                    }
                )
                continue
            entry_cost_unit = float(
                estimate_entry_only_cost(pd.Series(row), entry_price=float(price))["total_cost"]
            )
            wallet_attribution = apply_wallet_open_notional_delta(wallet_open_notional, support_wallets, executed_notional)
            event_open_notional[event_key] = float(event_open_notional.get(event_key, 0.0)) + executed_notional
            total_open_notional += executed_notional
            open_positions[token_id] = {
                "house_position_id": str(row.get("cluster_id")),
                "token_id": token_id,
                "market_id": str(row.get("market_id") or ""),
                "event_title": row.get("event_title") or "",
                "outcome": row.get("outcome") or "",
                "opened_at": row["first_ts"],
                "last_signal_at": row["last_ts"],
                "closed_at": pd.NaT,
                "status": "open",
                "opening_cluster_id": str(row.get("cluster_id") or ""),
                "closing_cluster_id": "",
                "signal_cluster_count": 1,
                "reinforcement_count": 0,
                "raw_trade_count": int(row.get("trade_count") or 0),
                "supporting_wallet_count": len(support_wallets),
                "supporting_wallets": support_wallets,
                "wallet_notional_attribution": wallet_attribution,
                "entry_contracts": float(executed_contracts),
                "entry_notional_usdc": executed_notional,
                "signaled_notional_usdc": signaled_notional,
                "suppressed_notional_usdc": skipped_notional,
                "weighted_avg_entry_price": float(price),
                "entry_cost_total_usdc": entry_cost_unit * float(executed_contracts),
            }
            if skipped_notional > 0:
                reason = "position_cap_partial_open"
                if "event_cap" in binding_caps:
                    reason = "event_cap_partial_open"
                elif "wallet_cap" in binding_caps:
                    reason = "wallet_cap_partial_open"
                elif "book_cap" in binding_caps:
                    reason = "book_cap_partial_open"
                skipped_rows.append(
                    {
                        "cluster_id": row.get("cluster_id"),
                        "action": action,
                        "reason": reason,
                        "binding_caps": ",".join(binding_caps),
                        "token_id": token_id,
                        "market_id": row.get("market_id"),
                        "event_title": row.get("event_title"),
                        "outcome": row.get("outcome"),
                        "signaled_notional_usdc": signaled_notional,
                        "executed_notional_usdc": executed_notional,
                        "skipped_notional_usdc": skipped_notional,
                    }
                )
        elif action == "reinforce_long":
            current = open_positions.get(token_id)
            executed_notional, skipped_notional, binding_caps = allowed_house_notional(
                signaled_notional=signaled_notional,
                current_position_notional=float(current.get("entry_notional_usdc") or 0.0) if current is not None else 0.0,
                current_event_notional=event_open_notional.get(event_key, 0.0),
                current_total_open_notional=total_open_notional,
                supporting_wallets=support_wallets,
                wallet_open_notional=wallet_open_notional,
                max_position_notional_usdc=max_position_notional_usdc,
                max_event_notional_usdc=max_event_notional_usdc,
                max_wallet_open_notional_usdc=max_wallet_open_notional_usdc,
                max_total_open_notional_usdc=max_total_open_notional_usdc,
            )
            executed_contracts = (
                None
                if contracts is None or signaled_notional <= 0
                else contracts * (executed_notional / signaled_notional)
            )
            if current is None or executed_contracts is None or executed_contracts <= 0:
                reason = (
                    "missing_parent_or_price"
                    if current is None or contracts is None
                    else "position_cap_reached"
                )
                if current is not None and contracts is not None and "event_cap" in binding_caps:
                    reason = "event_cap_reached"
                elif current is not None and contracts is not None and "wallet_cap" in binding_caps:
                    reason = "wallet_cap_reached"
                elif current is not None and contracts is not None and "book_cap" in binding_caps:
                    reason = "book_cap_reached"
                skipped_rows.append(
                    {
                        "cluster_id": row.get("cluster_id"),
                        "action": action,
                        "reason": reason,
                        "binding_caps": ",".join(binding_caps),
                        "token_id": token_id,
                        "market_id": row.get("market_id"),
                        "event_title": row.get("event_title"),
                        "outcome": row.get("outcome"),
                        "signaled_notional_usdc": signaled_notional,
                        "executed_notional_usdc": executed_notional,
                        "skipped_notional_usdc": skipped_notional,
                    }
                )
                continue
            entry_cost_unit = float(
                estimate_entry_only_cost(pd.Series(row), entry_price=float(price))["total_cost"]
            )
            total_contracts = float(current["entry_contracts"]) + float(executed_contracts)
            total_notional = float(current["entry_notional_usdc"]) + executed_notional
            current["entry_contracts"] = total_contracts
            current["entry_notional_usdc"] = total_notional
            current["signaled_notional_usdc"] = float(current.get("signaled_notional_usdc") or 0.0) + signaled_notional
            current["suppressed_notional_usdc"] = float(current.get("suppressed_notional_usdc") or 0.0) + skipped_notional
            current["weighted_avg_entry_price"] = total_notional / total_contracts if total_contracts > 0 else None
            current["entry_cost_total_usdc"] = float(current["entry_cost_total_usdc"]) + (
                entry_cost_unit * float(executed_contracts)
            )
            current["last_signal_at"] = row["last_ts"]
            current["signal_cluster_count"] = int(current["signal_cluster_count"]) + 1
            current["reinforcement_count"] = int(current["reinforcement_count"]) + 1
            current["raw_trade_count"] = int(current["raw_trade_count"]) + int(row.get("trade_count") or 0)
            current["supporting_wallets"].update(support_wallets)
            current["supporting_wallet_count"] = len(current["supporting_wallets"])
            current_wallet_attr = dict(current.get("wallet_notional_attribution") or {})
            delta_wallet_attr = apply_wallet_open_notional_delta(wallet_open_notional, support_wallets, executed_notional)
            for wallet, value in delta_wallet_attr.items():
                current_wallet_attr[wallet] = float(current_wallet_attr.get(wallet, 0.0)) + value
            current["wallet_notional_attribution"] = current_wallet_attr
            event_open_notional[event_key] = float(event_open_notional.get(event_key, 0.0)) + executed_notional
            total_open_notional += executed_notional
            if skipped_notional > 0:
                reason = "position_cap_partial_reinforce"
                if "event_cap" in binding_caps:
                    reason = "event_cap_partial_reinforce"
                elif "wallet_cap" in binding_caps:
                    reason = "wallet_cap_partial_reinforce"
                elif "book_cap" in binding_caps:
                    reason = "book_cap_partial_reinforce"
                skipped_rows.append(
                    {
                        "cluster_id": row.get("cluster_id"),
                        "action": action,
                        "reason": reason,
                        "binding_caps": ",".join(binding_caps),
                        "token_id": token_id,
                        "market_id": row.get("market_id"),
                        "event_title": row.get("event_title"),
                        "outcome": row.get("outcome"),
                        "signaled_notional_usdc": signaled_notional,
                        "executed_notional_usdc": executed_notional,
                        "skipped_notional_usdc": skipped_notional,
                    }
                )
        elif action == "close_long":
            current = open_positions.get(token_id)
            if current is None or price is None or pd.isna(price):
                skipped_rows.append(
                    {
                        "cluster_id": row.get("cluster_id"),
                        "action": action,
                        "reason": "missing_parent_or_close_price",
                        "token_id": token_id,
                        "market_id": row.get("market_id"),
                        "event_title": row.get("event_title"),
                        "outcome": row.get("outcome"),
                    }
                )
                continue
            exit_price = float(price)
            contracts_total = float(current["entry_contracts"])
            entry_price = float(current["weighted_avg_entry_price"])
            exit_cost_unit = float(
                estimate_entry_only_cost(pd.Series(row), entry_price=exit_price)["total_cost"]
            )
            realized_raw_usdc = (exit_price - entry_price) * contracts_total
            total_cost_usdc = float(current["entry_cost_total_usdc"]) + exit_cost_unit * contracts_total
            realized_net_usdc = calculate_net_pnl(realized_raw_usdc, total_cost_usdc)

            current["closed_at"] = row["first_ts"]
            current["last_signal_at"] = row["last_ts"]
            current["status"] = "closed"
            current["closing_cluster_id"] = str(row.get("cluster_id") or "")
            current["signal_cluster_count"] = int(current["signal_cluster_count"]) + 1
            current["raw_trade_count"] = int(current["raw_trade_count"]) + int(row.get("trade_count") or 0)
            current["supporting_wallets"].update(support_wallets)
            current["supporting_wallet_count"] = len(current["supporting_wallets"])
            event_open_notional[event_key] = max(
                0.0,
                float(event_open_notional.get(event_key, 0.0)) - float(current.get("entry_notional_usdc") or 0.0),
            )
            if event_open_notional[event_key] <= 1e-9:
                event_open_notional.pop(event_key, None)
            release_wallet_open_notional(wallet_open_notional, current.get("wallet_notional_attribution"))
            total_open_notional = max(
                0.0,
                total_open_notional - float(current.get("entry_notional_usdc") or 0.0),
            )
            current["exit_price"] = exit_price
            current["exit_cost_total_usdc"] = exit_cost_unit * contracts_total
            current["realized_pnl_raw_usdc"] = realized_raw_usdc
            current["realized_pnl_net_usdc"] = realized_net_usdc
            closed_rows.append(current.copy())
            del open_positions[token_id]

    open_rows: list[dict[str, Any]] = []
    for position in open_positions.values():
        position["supporting_wallets"] = json.dumps(sorted(position["supporting_wallets"]))
        position["wallet_notional_attribution"] = json.dumps(
            {key: round(float(value), 10) for key, value in sorted((position.get("wallet_notional_attribution") or {}).items())}
        )
        open_rows.append(position)

    normalized_closed: list[dict[str, Any]] = []
    for position in closed_rows:
        position["supporting_wallets"] = json.dumps(sorted(position["supporting_wallets"]))
        position["wallet_notional_attribution"] = json.dumps(
            {key: round(float(value), 10) for key, value in sorted((position.get("wallet_notional_attribution") or {}).items())}
        )
        normalized_closed.append(position)

    if skipped_rows:
        skipped = pd.DataFrame.from_records(skipped_rows)
        skipped = skipped.sort_values(["cluster_id"]).reset_index(drop=True)
    else:
        skipped = pd.DataFrame(
            columns=[
                "cluster_id",
                "action",
                "reason",
                "token_id",
                "market_id",
                "event_title",
                "outcome",
                "binding_caps",
                "signaled_notional_usdc",
                "executed_notional_usdc",
                "skipped_notional_usdc",
            ]
        )

    open_frame = pd.DataFrame.from_records(open_rows)
    if not open_frame.empty:
        open_frame = open_frame.sort_values(["opened_at", "token_id"]).reset_index(drop=True)
    closed_frame = pd.DataFrame.from_records(normalized_closed)
    if not closed_frame.empty:
        closed_frame = closed_frame.sort_values(["opened_at", "token_id"]).reset_index(drop=True)

    return (open_frame, closed_frame), skipped


def _mark_open_positions(
    open_positions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    terminal_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if open_positions.empty:
        return open_positions.copy()

    token_ids = sorted(open_positions["token_id"].dropna().astype(str).unique().tolist())
    price_history = _load_price_history_chunked(token_ids, end_ts=cutoff)
    price_index = _build_price_index(price_history)
    cutoff_epoch = _to_epoch_seconds(cutoff)

    marked_rows: list[dict[str, Any]] = []
    for row in open_positions.to_dict(orient="records"):
        token_id = str(row["token_id"])
        terminal_info = terminal_lookup.get(token_id) or {}
        mark_price, mark_source, mark_age_seconds = _lookup_latest_price_at_or_before(price_index, token_id, cutoff_epoch)
        if mark_price is None and terminal_info.get("terminal_price") is not None:
            mark_price = float(terminal_info["terminal_price"])
            mark_source = "gamma_outcome_prices_current_fallback"
            mark_age_seconds = None

        contracts_total = float(row.get("entry_contracts") or 0.0)
        entry_price = float(row.get("weighted_avg_entry_price") or 0.0)
        raw_mtm_usdc = None if mark_price is None else (float(mark_price) - entry_price) * contracts_total
        net_mtm_usdc = (
            calculate_net_pnl(raw_mtm_usdc, float(row.get("entry_cost_total_usdc") or 0.0))
            if raw_mtm_usdc is not None
            else None
        )
        marked = dict(row)
        marked["analysis_cutoff"] = cutoff
        marked["mark_price"] = mark_price
        marked["mark_price_source"] = mark_source
        marked["mark_price_age_seconds"] = mark_age_seconds
        marked["mtm_pnl_raw_usdc"] = raw_mtm_usdc
        marked["mtm_pnl_net_usdc"] = net_mtm_usdc
        marked["holding_days_open"] = (
            (cutoff - pd.to_datetime(row["opened_at"], utc=True)).total_seconds() / 86400.0
        )
        try:
            marked["wallet_notional_attribution"] = json.loads(row.get("wallet_notional_attribution") or "{}")
        except json.JSONDecodeError:
            marked["wallet_notional_attribution"] = {}
        marked_rows.append(marked)

    return pd.DataFrame.from_records(marked_rows).sort_values(
        ["mtm_pnl_net_usdc", "opened_at"], ascending=[False, True], na_position="last"
    )


def _annotate_closed_wallet_attribution(closed_positions: pd.DataFrame) -> pd.DataFrame:
    if closed_positions.empty:
        return closed_positions.copy()
    annotated = closed_positions.copy()
    def _parse(raw: object) -> dict[str, float]:
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
    annotated["wallet_notional_attribution"] = annotated["wallet_notional_attribution"].apply(_parse)
    return annotated


def _build_contribution_tables(
    closed_positions: pd.DataFrame,
    open_positions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], dict[str, float]]:
    closed_records = closed_positions.to_dict(orient="records") if not closed_positions.empty else []
    open_records = open_positions.to_dict(orient="records") if not open_positions.empty else []

    wallet_rows = aggregate_wallet_contributions(
        [
            {
                "wallet_notional_attribution": row.get("wallet_notional_attribution") or {},
                "entry_notional_usdc": row.get("entry_notional_usdc") or 0.0,
                "realized_net_pnl_usdc": row.get("realized_pnl_net_usdc") or 0.0,
                "mtm_net_pnl_usdc": 0.0,
            }
            for row in closed_records
        ]
        + [
            {
                "wallet_notional_attribution": row.get("wallet_notional_attribution") or {},
                "entry_notional_usdc": row.get("entry_notional_usdc") or 0.0,
                "realized_net_pnl_usdc": 0.0,
                "mtm_net_pnl_usdc": row.get("mtm_pnl_net_usdc") or 0.0,
            }
            for row in open_records
        ]
    )
    event_rollup: dict[str, dict[str, float]] = {}
    for row in closed_records:
        key = str(row.get("event_title") or "")
        agg = event_rollup.setdefault(
            key,
            {"event_title": key, "positions": 0.0, "realized_net_pnl_usdc": 0.0, "open_mtm_net_pnl_usdc": 0.0, "combined_net_pnl_usdc": 0.0},
        )
        agg["positions"] += 1.0
        realized = _safe_float(row.get("realized_pnl_net_usdc"))
        agg["realized_net_pnl_usdc"] += realized
        agg["combined_net_pnl_usdc"] += realized
    for row in open_records:
        key = str(row.get("event_title") or "")
        agg = event_rollup.setdefault(
            key,
            {"event_title": key, "positions": 0.0, "realized_net_pnl_usdc": 0.0, "open_mtm_net_pnl_usdc": 0.0, "combined_net_pnl_usdc": 0.0},
        )
        mtm = _safe_float(row.get("mtm_pnl_net_usdc"))
        agg["positions"] += 1.0
        agg["open_mtm_net_pnl_usdc"] += mtm
        agg["combined_net_pnl_usdc"] += mtm
    event_rows = sorted(event_rollup.values(), key=lambda item: item["combined_net_pnl_usdc"], reverse=True)

    wallet_metrics = positive_contribution_shares(wallet_rows)
    event_metrics = positive_contribution_shares(event_rows)
    return (
        pd.DataFrame.from_records(wallet_rows),
        pd.DataFrame.from_records(event_rows),
        wallet_metrics,
        event_metrics,
    )


def _build_daily_equity_curve(
    closed_positions: pd.DataFrame,
    open_positions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    terminal_lookup: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not closed_positions.empty:
        frames.append(closed_positions[["opened_at", "closed_at"]].copy())
    if not open_positions.empty:
        frames.append(open_positions[["opened_at"]].copy())
    if not frames:
        return pd.DataFrame()

    anchors = pd.concat(frames, ignore_index=True)
    min_opened = pd.to_datetime(anchors["opened_at"], utc=True, errors="coerce").min()
    if pd.isna(min_opened):
        return pd.DataFrame()

    daily_index = pd.date_range(min_opened.floor("D"), cutoff.floor("D"), freq="D", tz="UTC")

    token_ids = sorted(
        pd.concat(
            [
                closed_positions["token_id"] if "token_id" in closed_positions.columns else pd.Series(dtype=str),
                open_positions["token_id"] if "token_id" in open_positions.columns else pd.Series(dtype=str),
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    price_history = _load_price_history_chunked(token_ids, end_ts=cutoff)
    price_index = _build_price_index(price_history)

    curve_rows: list[dict[str, Any]] = []
    closed_records = closed_positions.to_dict(orient="records")
    open_records = open_positions.to_dict(orient="records")

    for day in daily_index:
        day_end = day + pd.Timedelta(hours=23, minutes=59, seconds=59)
        day_epoch = _to_epoch_seconds(day_end)
        realized_net = 0.0
        open_mtm_net = 0.0
        open_count = 0
        marked_count = 0

        for row in closed_records:
            close_ts = pd.to_datetime(row.get("closed_at"), utc=True, errors="coerce")
            if close_ts is not None and not pd.isna(close_ts) and close_ts <= day_end:
                realized_net += float(row.get("realized_pnl_net_usdc") or 0.0)
                continue

            open_ts = pd.to_datetime(row.get("opened_at"), utc=True, errors="coerce")
            if open_ts is None or pd.isna(open_ts) or open_ts > day_end:
                continue
            open_count += 1
            token_id = str(row.get("token_id") or "")
            mark_price, _, _ = _lookup_latest_price_at_or_before(price_index, token_id, day_epoch)
            if mark_price is None:
                terminal_price = (terminal_lookup.get(token_id) or {}).get("terminal_price")
                if terminal_price is not None:
                    mark_price = float(terminal_price)
            if mark_price is None:
                continue
            marked_count += 1
            entry_price = float(row.get("weighted_avg_entry_price") or 0.0)
            contracts = float(row.get("entry_contracts") or 0.0)
            raw_mtm_usdc = (float(mark_price) - entry_price) * contracts
            open_mtm_net += float(
                calculate_net_pnl(raw_mtm_usdc, float(row.get("entry_cost_total_usdc") or 0.0)) or 0.0
            )

        for row in open_records:
            open_ts = pd.to_datetime(row.get("opened_at"), utc=True, errors="coerce")
            if open_ts is None or pd.isna(open_ts) or open_ts > day_end:
                continue
            open_count += 1
            token_id = str(row.get("token_id") or "")
            mark_price, _, _ = _lookup_latest_price_at_or_before(price_index, token_id, day_epoch)
            if mark_price is None:
                terminal_price = (terminal_lookup.get(token_id) or {}).get("terminal_price")
                if terminal_price is not None:
                    mark_price = float(terminal_price)
            if mark_price is None:
                continue
            marked_count += 1
            entry_price = float(row.get("weighted_avg_entry_price") or 0.0)
            contracts = float(row.get("entry_contracts") or 0.0)
            raw_mtm_usdc = (float(mark_price) - entry_price) * contracts
            open_mtm_net += float(
                calculate_net_pnl(raw_mtm_usdc, float(row.get("entry_cost_total_usdc") or 0.0)) or 0.0
            )

        curve_rows.append(
            {
                "date": day.date().isoformat(),
                "day_end_ts": day_end.isoformat(),
                "cumulative_realized_net_usdc": realized_net,
                "open_mtm_net_usdc": open_mtm_net,
                "combined_equity_net_usdc": realized_net + open_mtm_net,
                "open_positions_count": open_count,
                "marked_open_positions_count": marked_count,
                "mark_coverage_share": (marked_count / open_count) if open_count else None,
            }
        )

    return pd.DataFrame.from_records(curve_rows)


def _plot_equity_curve(curve: pd.DataFrame, path: Path) -> Path | None:
    if curve.empty:
        return None
    x_positions = list(range(len(curve)))
    labels = curve["date"].astype(str).tolist()

    plt.style.use("default")
    figure, axis = plt.subplots(figsize=(16, 9))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#EAEAF2")
    axis.plot(
        x_positions,
        curve["combined_equity_net_usdc"],
        label="Combined Equity",
        linewidth=3.0,
        color="#1f77b4",
    )
    axis.plot(
        x_positions,
        curve["cumulative_realized_net_usdc"],
        label="Realized Only",
        linewidth=2.2,
        color="#ff7f0e",
    )
    axis.set_title("Unified House Portfolio Equity Curve", fontsize=24, pad=12)
    axis.set_ylabel("Net PnL (USDC)", fontsize=20)
    axis.set_xlabel("Date", fontsize=20)
    axis.tick_params(axis="y", labelsize=16)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.legend(loc="upper left", fontsize=20, framealpha=0.85)
    axis.grid(axis="y", alpha=0.35, linewidth=1.0)
    axis.grid(axis="x", alpha=0.8, linewidth=0.8, color="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def run_paper_tracking_performance(
    consolidated_dir: str | Path = DEFAULT_CONSOLIDATED_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    analysis_cutoff: str | None = None,
    max_position_notional_usdc: float | None = None,
    max_event_notional_usdc: float | None = None,
    max_wallet_open_notional_usdc: float | None = None,
    max_total_open_notional_usdc: float | None = None,
) -> dict[str, Any]:
    consolidated_root = Path(consolidated_dir)
    output_root = Path(output_dir)

    tape = _read_csv(consolidated_root / "house_signal_tape.csv")
    if tape.empty:
        raise ValueError("house_signal_tape.csv is required to build performance tracking")

    (open_positions, closed_positions), skipped = _build_house_position_ledger_with_cap(
        tape,
        max_position_notional_usdc=max_position_notional_usdc,
        max_event_notional_usdc=max_event_notional_usdc,
        max_wallet_open_notional_usdc=max_wallet_open_notional_usdc,
        max_total_open_notional_usdc=max_total_open_notional_usdc,
    )
    cutoff = pd.to_datetime(analysis_cutoff, utc=True, errors="coerce")
    if cutoff is None or pd.isna(cutoff):
        cutoff = pd.Timestamp.now(tz="UTC")

    with get_session() as session:
        markets = load_markets_frame(session)
        terminal_lookup = _build_terminal_lookup(markets)
        missing_terminal_tokens = sorted(
            {
                str(token_id)
                for token_id in open_positions.get("token_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
                if str(token_id) not in terminal_lookup
            }
        )
        if missing_terminal_tokens:
            asyncio.run(backfill_markets_for_references(session, token_ids=missing_terminal_tokens))
            session.commit()
            markets = load_markets_frame(session)
            terminal_lookup = _build_terminal_lookup(markets)

    marked_open_positions = _mark_open_positions(open_positions, cutoff=cutoff, terminal_lookup=terminal_lookup)
    closed_positions = _annotate_closed_wallet_attribution(closed_positions)
    curve = _build_daily_equity_curve(
        closed_positions=closed_positions,
        open_positions=open_positions,
        cutoff=cutoff,
        terminal_lookup=terminal_lookup,
    )

    open_path = _write_csv(marked_open_positions, output_root / "house_open_position_performance.csv")
    closed_path = _write_csv(closed_positions, output_root / "house_closed_position_performance.csv")
    skipped_path = _write_csv(skipped, output_root / "house_skipped_position_records.csv")
    curve_path = _write_csv(curve, output_root / "house_portfolio_daily_equity_curve.csv")
    plot_path = _plot_equity_curve(curve, output_root / "house_portfolio_daily_equity_curve.png")
    wallet_contrib, event_contrib, wallet_metrics, event_metrics = _build_contribution_tables(
        closed_positions=closed_positions,
        open_positions=marked_open_positions,
    )
    wallet_contrib_path = _write_csv(wallet_contrib, output_root / "house_wallet_contribution.csv")
    event_contrib_path = _write_csv(event_contrib, output_root / "house_event_contribution.csv")

    realized_total = float(closed_positions.get("realized_pnl_net_usdc", pd.Series(dtype=float)).fillna(0).sum())
    open_mtm_total = float(marked_open_positions.get("mtm_pnl_net_usdc", pd.Series(dtype=float)).fillna(0).sum())
    combined_total = realized_total + open_mtm_total
    markable_open = int(marked_open_positions.get("mtm_pnl_net_usdc", pd.Series(dtype=float)).notna().sum())
    total_open = int(len(marked_open_positions))
    total_closed = int(len(closed_positions))
    skipped_count = int(len(skipped))
    avg_holding_days_open = (
        float(marked_open_positions["holding_days_open"].mean())
        if not marked_open_positions.empty and "holding_days_open" in marked_open_positions
        else None
    )
    avg_realized_holding_days = (
        float(
            (
                pd.to_datetime(closed_positions["closed_at"], utc=True, errors="coerce")
                - pd.to_datetime(closed_positions["opened_at"], utc=True, errors="coerce")
            )
            .dt.total_seconds()
            .div(86400.0)
            .mean()
        )
        if not closed_positions.empty
        else None
    )

    summary_lines = [
        "# House Portfolio Performance Summary\n",
        "\n",
        "This layer tracks the unified house portfolio after duplicate wallet signals and opposite-outcome market conflicts have already been consolidated away.\n",
        "\n",
        f"- Analysis cutoff: `{cutoff.isoformat()}`\n",
        f"- Open house positions: `{total_open}`\n",
        f"- Closed house positions: `{total_closed}`\n",
        f"- Skipped / unpriceable ledger records: `{skipped_count}`\n",
        (
            f"- Per-position notional cap: `{max_position_notional_usdc:.2f} USDC`\n"
            if max_position_notional_usdc is not None
            else ""
        ),
        (
            f"- Per-event notional cap: `{max_event_notional_usdc:.2f} USDC`\n"
            if max_event_notional_usdc is not None
            else ""
        ),
        (
            f"- Per-wallet open-notional cap: `{max_wallet_open_notional_usdc:.2f} USDC`\n"
            if max_wallet_open_notional_usdc is not None
            else ""
        ),
        (
            f"- Total concurrent house-notional cap: `{max_total_open_notional_usdc:.2f} USDC`\n"
            if max_total_open_notional_usdc is not None
            else ""
        ),
        f"- Closed-position realized net PnL: `{realized_total:.2f} USDC`\n",
        f"- Open-position MTM net PnL: `{open_mtm_total:.2f} USDC`\n",
        f"- Combined house portfolio net PnL: `{combined_total:.2f} USDC`\n",
        f"- Marked open positions: `{markable_open}/{total_open}`\n",
        f"- Wallet contribution positive-share top 1: `{wallet_metrics['top1_positive_share']*100:.1f}%`\n",
        f"- Wallet contribution positive-share top 5: `{wallet_metrics['top5_positive_share']*100:.1f}%`\n",
        f"- Event contribution positive-share top 1: `{event_metrics['top1_positive_share']*100:.1f}%`\n",
        f"- Event contribution positive-share top 5: `{event_metrics['top5_positive_share']*100:.1f}%`\n",
        (
            f"- Average holding days for currently open positions: `{avg_holding_days_open:.2f}`\n"
            if avg_holding_days_open is not None
            else ""
        ),
        (
            f"- Average realized holding days for closed positions: `{avg_realized_holding_days:.2f}`\n"
            if avg_realized_holding_days is not None
            else ""
        ),
        "\n",
        "## Output Files\n",
        f"- `house_open_position_performance.csv`: `{open_path}`\n",
        f"- `house_closed_position_performance.csv`: `{closed_path}`\n",
        f"- `house_skipped_position_records.csv`: `{skipped_path}`\n",
        f"- `house_wallet_contribution.csv`: `{wallet_contrib_path}`\n",
        f"- `house_event_contribution.csv`: `{event_contrib_path}`\n",
        f"- `house_portfolio_daily_equity_curve.csv`: `{curve_path}`\n",
    ]
    if plot_path is not None:
        summary_lines.append(
            f"- `house_portfolio_daily_equity_curve.png`: `{plot_path}`\n"
        )

    summary_lines.extend(
        [
            "\n",
            "## Interpretation\n",
            "- `realized net PnL` uses unified house exits when source wallets produce a consolidated sell signal.\n",
            "- `open MTM net PnL` marks still-open house positions to the latest locally available public price history, with current Gamma terminal values only as a fallback.\n",
            "- The daily equity curve combines cumulative realized PnL with day-end MTM on positions that were still open on each day.\n",
        ]
    )
    summary_path = output_root / "house_portfolio_performance_summary.md"
    summary_path.write_text("".join(part for part in summary_lines if part), encoding="utf-8")

    return {
        "open_positions": marked_open_positions,
        "closed_positions": closed_positions,
        "skipped_positions": skipped,
        "wallet_contribution": wallet_contrib,
        "event_contribution": event_contrib,
        "curve": curve,
        "summary_path": summary_path,
        "open_path": open_path,
        "closed_path": closed_path,
        "skipped_path": skipped_path,
        "wallet_contrib_path": wallet_contrib_path,
        "event_contrib_path": event_contrib_path,
        "curve_path": curve_path,
        "plot_path": plot_path,
    }
