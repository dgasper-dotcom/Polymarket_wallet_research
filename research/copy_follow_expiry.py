"""Research-only delayed copy analysis that holds copied trades to market expiry."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from db.models import Market
from research.costs import calculate_net_pnl, estimate_entry_only_cost
from research.delay_analysis import (
    _build_price_index,
    _to_epoch_seconds,
    load_price_history_frame,
    lookup_forward_price,
)
from research.event_study import load_enriched_trades


DEFAULT_DELAYS = (15, 30)


def _parse_json_list(value: Any) -> list[Any]:
    """Parse Polymarket fields that sometimes arrive as JSON-encoded strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    """Normalize a timestamp-like value to UTC."""

    if value in (None, ""):
        return None
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(timestamp) else timestamp


def _parse_market_payload(raw_json: Any) -> dict[str, Any]:
    """Decode stored Gamma market raw JSON into a dict."""

    if isinstance(raw_json, dict):
        return raw_json
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_market_resolution_ts(
    raw_json: Any,
    *,
    updated_at: Any,
    closed: Any,
) -> tuple[pd.Timestamp | None, str]:
    """Best-effort public-data resolution timestamp inference.

    Assumptions:
    - Prefer explicit market-end or resolution-like keys when present.
    - If the market is marked closed but no explicit end is present, fall back to
      `updated_at` as a weak late-stage proxy instead of fabricating a timestamp.
    """

    payload = _parse_market_payload(raw_json)
    for key in (
        "endDate",
        "end_date",
        "resolutionDate",
        "resolution_date",
        "resolveBy",
        "resolve_by",
        "closeTime",
        "close_time",
        "closedTime",
        "closed_time",
    ):
        timestamp = _to_utc_timestamp(payload.get(key))
        if timestamp is not None:
            return timestamp, key

    if bool(closed):
        timestamp = _to_utc_timestamp(updated_at)
        if timestamp is not None:
            return timestamp, "updated_at_closed_fallback"
    return None, "missing_resolution_ts"


def load_markets_frame(session: Session) -> pd.DataFrame:
    """Load persisted Gamma market metadata."""

    return pd.read_sql(select(Market), session.bind)


def _build_terminal_lookup(markets: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Map each token id to its best available terminal market information."""

    if markets.empty:
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    frame = markets.copy()
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True, errors="coerce")
    for row in frame.to_dict(orient="records"):
        payload = _parse_market_payload(row.get("raw_json"))
        token_ids = _parse_json_list(payload.get("clobTokenIds"))
        outcome_prices = _parse_json_list(payload.get("outcomePrices"))
        resolution_ts, resolution_source = _parse_market_resolution_ts(
            row.get("raw_json"),
            updated_at=row.get("updated_at"),
            closed=row.get("closed"),
        )
        for index, token_id in enumerate(token_ids):
            token_key = str(token_id)
            terminal_price: float | None = None
            if index < len(outcome_prices):
                try:
                    terminal_price = float(outcome_prices[index])
                except (TypeError, ValueError):
                    terminal_price = None
            candidate = {
                "market_id": str(row.get("id")) if row.get("id") is not None else None,
                "condition_id": str(row.get("condition_id")) if row.get("condition_id") is not None else None,
                "question": row.get("question"),
                "closed": bool(row.get("closed")),
                "terminal_price": terminal_price,
                "terminal_price_source": (
                    "gamma_outcome_prices_current"
                    if terminal_price is not None
                    else "missing_terminal_price"
                ),
                "resolution_ts": resolution_ts,
                "resolution_source": resolution_source,
            }
            existing = lookup.get(token_key)
            if existing is None:
                lookup[token_key] = candidate
                continue

            existing_score = int(bool(existing.get("closed"))) + int(existing.get("terminal_price") is not None)
            candidate_score = int(bool(candidate.get("closed"))) + int(candidate.get("terminal_price") is not None)
            if candidate_score > existing_score:
                lookup[token_key] = candidate
    return lookup


def _normalize_side(value: Any) -> str | None:
    """Normalize trade side to BUY / SELL."""

    normalized = str(value or "").upper().strip()
    if normalized in {"BUY", "SELL"}:
        return normalized
    return None


def _signed_copy_pnl(side: str | None, entry_price: float | None, terminal_price: float | None) -> float | None:
    """Return signed copy PnL for a held-to-expiry copied trade."""

    if side not in {"BUY", "SELL"} or entry_price is None or terminal_price is None:
        return None
    direction = 1.0 if side == "BUY" else -1.0
    return direction * (float(terminal_price) - float(entry_price))


def _filter_window(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    """Restrict the analysis frame to the requested inclusive UTC date window."""

    if frame.empty:
        return frame.copy(), None, None

    filtered = frame.copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], utc=True, errors="coerce")
    filtered = filtered.dropna(subset=["timestamp"]).sort_values(["wallet_address", "timestamp", "trade_id"])

    start_ts = _to_utc_timestamp(start_date)
    end_ts = _to_utc_timestamp(end_date)
    if end_ts is not None and len(str(end_date or "")) <= 10:
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    if start_ts is not None:
        filtered = filtered.loc[filtered["timestamp"] >= start_ts]
    if end_ts is not None:
        filtered = filtered.loc[filtered["timestamp"] <= end_ts]
    return filtered.reset_index(drop=True), start_ts, end_ts


def compute_copy_follow_expiry_from_frame(
    enriched: pd.DataFrame,
    price_history: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    delays: tuple[int, ...] = DEFAULT_DELAYS,
    start_date: str | None = None,
    end_date: str | None = None,
    active_last_days: int = 30,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute delayed-entry copy PnL through market expiry.

    Assumptions:
    - Delayed entry uses the first public price point at or after the delayed
      timestamp.
    - Exit uses the current public Gamma `outcomePrices` value for the same
      outcome token as the best public proxy for expiry value.
    - Net PnL uses entry-only costs because this path models holding through
      expiry instead of exiting through the order book before resolution.
    """

    cfg = settings or get_settings()
    filtered, start_ts, end_ts = _filter_window(enriched, start_date=start_date, end_date=end_date)
    diagnostics = filtered.copy()
    if diagnostics.empty:
        return pd.DataFrame(), diagnostics, pd.DataFrame()

    price_index = _build_price_index(price_history)
    terminal_lookup = _build_terminal_lookup(markets)
    window_end = end_ts or diagnostics["timestamp"].max()
    recent_cutoff = window_end - pd.Timedelta(days=active_last_days)

    diagnostic_rows: list[dict[str, Any]] = []
    for row in diagnostics.to_dict(orient="records"):
        trade_ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
        side = _normalize_side(row.get("side"))
        token_id = str(row.get("token_id")) if row.get("token_id") is not None else None
        terminal_info = terminal_lookup.get(token_id or "")
        trade_record = dict(row)
        trade_record["timestamp"] = trade_ts
        trade_record["analysis_window_start"] = start_ts
        trade_record["analysis_window_end"] = window_end
        trade_record["active_in_last_30d_of_window"] = bool(pd.notna(trade_ts) and trade_ts >= recent_cutoff)
        trade_record["terminal_price"] = None
        trade_record["terminal_price_source"] = "missing_market_terminal"
        trade_record["resolution_ts"] = None
        trade_record["resolution_source"] = "missing_market_terminal"

        resolution_epoch: int | None = None
        if terminal_info:
            trade_record["terminal_price"] = terminal_info.get("terminal_price")
            trade_record["terminal_price_source"] = terminal_info.get("terminal_price_source")
            trade_record["resolution_ts"] = terminal_info.get("resolution_ts")
            trade_record["resolution_source"] = terminal_info.get("resolution_source")
            if terminal_info.get("resolution_ts") is not None:
                resolution_epoch = _to_epoch_seconds(terminal_info["resolution_ts"])

        trade_epoch = _to_epoch_seconds(trade_ts)
        for delay_seconds in delays:
            label = f"{delay_seconds}s"
            entry_price_column = f"copy_entry_price_{label}"
            entry_source_column = f"copy_entry_source_{label}"
            entry_lag_column = f"copy_entry_lag_seconds_{label}"
            raw_pnl_column = f"copy_pnl_expiry_{label}"
            net_pnl_column = f"copy_pnl_net_expiry_{label}"
            status_column = f"copy_status_expiry_{label}"
            cost_column = f"copy_cost_expiry_{label}"

            trade_record[entry_price_column] = None
            trade_record[entry_source_column] = "missing_prices"
            trade_record[entry_lag_column] = None
            trade_record[raw_pnl_column] = None
            trade_record[net_pnl_column] = None
            trade_record[cost_column] = None

            if side is None:
                trade_record[status_column] = "missing_side"
                continue
            if terminal_info is None:
                trade_record[status_column] = "missing_market_terminal"
                continue
            if terminal_info.get("terminal_price") is None:
                trade_record[status_column] = "missing_terminal_price"
                continue

            target_epoch = trade_epoch + int(delay_seconds)
            if resolution_epoch is not None and target_epoch >= resolution_epoch:
                trade_record[status_column] = "delay_at_or_after_resolution"
                continue

            forward = lookup_forward_price(price_index, token_id, target_epoch)
            trade_record[entry_price_column] = forward.price
            trade_record[entry_source_column] = forward.source
            trade_record[entry_lag_column] = forward.delta_seconds
            if forward.price is None:
                trade_record[status_column] = "missing_entry_price_after_delay"
                continue

            matched_epoch = target_epoch + int(forward.delta_seconds or 0)
            if resolution_epoch is not None and matched_epoch > resolution_epoch:
                trade_record[status_column] = "entry_price_after_resolution"
                trade_record[entry_price_column] = None
                trade_record[entry_source_column] = "price_after_resolution"
                trade_record[entry_lag_column] = forward.delta_seconds
                continue

            raw_pnl = _signed_copy_pnl(side, forward.price, terminal_info.get("terminal_price"))
            cost_details = estimate_entry_only_cost(
                pd.Series(row),
                entry_price=forward.price,
                scenario=cfg.cost_scenario,
                settings=cfg,
            )
            trade_record[raw_pnl_column] = raw_pnl
            trade_record[cost_column] = cost_details["total_cost"]
            trade_record[net_pnl_column] = calculate_net_pnl(raw_pnl, cost_details["total_cost"])
            trade_record[status_column] = "ok"

        diagnostic_rows.append(trade_record)

    trade_diagnostics = pd.DataFrame.from_records(diagnostic_rows).sort_values(
        ["wallet_address", "timestamp", "trade_id"]
    )
    if trade_diagnostics.empty:
        return pd.DataFrame(), trade_diagnostics, pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for wallet, group in trade_diagnostics.groupby("wallet_address"):
        record: dict[str, Any] = {
            "wallet_address": wallet,
            "n_trades_total": int(len(group)),
            "n_markets": int(group["market_id"].nunique(dropna=True)),
            "first_trade_ts": group["timestamp"].min(),
            "most_recent_trade_ts": group["timestamp"].max(),
            "active_in_last_30d_of_window": bool((group["timestamp"] >= recent_cutoff).any()),
        }
        for delay_seconds in delays:
            label = f"{delay_seconds}s"
            raw_column = f"copy_pnl_expiry_{label}"
            net_column = f"copy_pnl_net_expiry_{label}"
            status_column = f"copy_status_expiry_{label}"
            valid = pd.to_numeric(group[net_column], errors="coerce").dropna()
            gross_valid = pd.to_numeric(group[raw_column], errors="coerce").dropna()
            record[f"valid_trades_{label}"] = int(valid.shape[0])
            record[f"avg_copy_pnl_expiry_{label}"] = float(gross_valid.mean()) if not gross_valid.empty else None
            record[f"median_copy_pnl_expiry_{label}"] = float(gross_valid.median()) if not gross_valid.empty else None
            record[f"copy_hit_rate_expiry_{label}"] = (
                float((gross_valid > 0).mean()) if not gross_valid.empty else None
            )
            record[f"avg_copy_pnl_net_expiry_{label}"] = float(valid.mean()) if not valid.empty else None
            record[f"median_copy_pnl_net_expiry_{label}"] = float(valid.median()) if not valid.empty else None
            record[f"copy_net_hit_rate_expiry_{label}"] = (
                float((valid > 0).mean()) if not valid.empty else None
            )
            record[f"positive_wallet_net_{label}"] = bool(not valid.empty and float(valid.mean()) > 0)
            record[f"positive_wallet_gross_{label}"] = bool(
                not gross_valid.empty and float(gross_valid.mean()) > 0
            )
            record[f"missing_trades_{label}"] = int((group[status_column] != "ok").sum())
        summary_rows.append(record)

    wallet_summary = pd.DataFrame.from_records(summary_rows).sort_values(
        by=["avg_copy_pnl_net_expiry_30s", "avg_copy_pnl_net_expiry_15s", "wallet_address"],
        ascending=[False, False, True],
        na_position="last",
    )
    wallet_summary["positive_wallet_net_both_15s_30s"] = (
        wallet_summary["positive_wallet_net_15s"] & wallet_summary["positive_wallet_net_30s"]
    )

    overview = pd.DataFrame(
        [
            {
                "analysis_window_start": start_ts.isoformat() if start_ts is not None else None,
                "analysis_window_end": window_end.isoformat() if window_end is not None else None,
                "wallets_in_report": int(wallet_summary["wallet_address"].nunique()),
                "active_wallets_in_last_30d_of_window": int(
                    wallet_summary["active_in_last_30d_of_window"].sum()
                ),
                "total_trades_in_window": int(len(trade_diagnostics)),
                "valid_trades_15s": int(wallet_summary["valid_trades_15s"].sum()),
                "valid_trades_30s": int(wallet_summary["valid_trades_30s"].sum()),
                "wallets_positive_gross_15s": int(wallet_summary["positive_wallet_gross_15s"].sum()),
                "wallets_positive_gross_30s": int(wallet_summary["positive_wallet_gross_30s"].sum()),
                "wallets_positive_net_15s": int(wallet_summary["positive_wallet_net_15s"].sum()),
                "wallets_positive_net_30s": int(wallet_summary["positive_wallet_net_30s"].sum()),
                "wallets_positive_net_both_15s_30s": int(
                    wallet_summary["positive_wallet_net_both_15s_30s"].sum()
                ),
            }
        ]
    )
    return wallet_summary, trade_diagnostics, overview


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Persist one DataFrame as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_assumptions(path: Path) -> Path:
    """Persist the public-data assumptions behind expiry-held copy analysis."""

    text = """# Copy Follow To Expiry Assumptions

1. Entry timing:
   - For each observed wallet trade, delayed copy entry uses the first public
     `prices-history` point at or after `trade_timestamp + delay`.
   - If the first available public price point is already after resolution, the
     copied trade is marked invalid for that delay.

2. Exit timing:
   - This report does not exit after 5 minutes.
   - It holds the copied position through expiry / resolution.

3. Terminal value source:
   - Terminal token value is approximated using public Gamma market metadata
     `outcomePrices` for the matching `clobTokenIds` entry.
   - This is a public-data approximation, not a settlement ledger export.

4. Net cost model:
   - Because the copied position is held to expiry, net PnL uses entry-only
     costs rather than a round-trip order-book exit cost.
   - Entry-only cost = half-spread + entry slippage + optional fee
     (+ extra penalty in conservative mode).

5. Side handling:
   - BUY uses `terminal_price - delayed_entry_price`.
   - SELL uses the same signed-return convention as the rest of the project:
     `-(terminal_price - delayed_entry_price)`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_copy_follow_expiry_analysis(
    session: Session,
    *,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    active_last_days: int = 30,
    delays: tuple[int, ...] = DEFAULT_DELAYS,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run delayed-entry copy analysis held through market expiry and export CSVs."""

    cfg = settings or get_settings()
    enriched = load_enriched_trades(session)
    token_ids = sorted({str(token_id) for token_id in enriched["token_id"].dropna().unique()})
    price_history = load_price_history_frame(session, token_ids=token_ids)
    markets = load_markets_frame(session)

    wallet_summary, trade_diagnostics, overview = compute_copy_follow_expiry_from_frame(
        enriched,
        price_history,
        markets,
        delays=delays,
        start_date=start_date,
        end_date=end_date,
        active_last_days=active_last_days,
        settings=cfg,
    )

    output_path = Path(output_dir)
    window_label = f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"
    wallet_path = _write_csv(
        wallet_summary,
        output_path / f"copy_follow_expiry_15s_30s_full_sample_{window_label}.csv",
    )
    summary_path = _write_csv(
        overview,
        output_path / "copy_follow_expiry_15s_30s_full_sample_summary.csv",
    )
    diagnostics_path = _write_csv(
        trade_diagnostics,
        output_path / f"copy_follow_expiry_15s_30s_trade_diagnostics_{window_label}.csv",
    )
    active_wallets = wallet_summary.loc[wallet_summary["active_in_last_30d_of_window"]].copy()
    active_wallets_path = _write_csv(
        active_wallets,
        output_path / f"copy_follow_expiry_15s_30s_active_last{active_last_days}d_{window_label}.csv",
    )
    assumptions_path = _write_assumptions(output_path / "copy_follow_expiry_assumptions.md")

    return {
        "wallet_summary": wallet_summary,
        "trade_diagnostics": trade_diagnostics,
        "overview": overview,
        "wallet_path": wallet_path,
        "summary_path": summary_path,
        "diagnostics_path": diagnostics_path,
        "active_wallets_path": active_wallets_path,
        "assumptions_path": assumptions_path,
    }
