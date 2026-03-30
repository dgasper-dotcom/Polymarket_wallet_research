"""Full-history copy backtest for manual seed wallets using PMA trade data.

This module uses Polymarket Analytics trader activity as the wallet trade
source, then maps those trades back to public Gamma/CLOB markets so delayed
copy fills and current MTM can still be grounded in public pricing.

Important limitations:
- PMA trade timestamps appear timezone-naive; this implementation treats them as
  UTC to stay aligned with the rest of the research pipeline.
- Public historical order-book snapshots are still unavailable, so maker fills
  remain a conservative research approximation rather than observed ground
  truth.
- Event-to-market mapping is heuristic for grouped multi-market events.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sys
from typing import Any

import certifi
import httpx

sys.modules.setdefault("pyarrow", None)

import pandas as pd

from clients.clob_client import ClobClient
from clients.polymarket_analytics_client import PolymarketAnalyticsClient
from config.settings import Settings, get_settings
from ingestion.prices import _history_rows
from research.copy_follow_expiry import _build_terminal_lookup
from research.copy_follow_wallet_exit import build_copy_exit_pairs
from research.manual_seed_copy_backtest import (
    DEFAULT_DELAYS,
    FillAttempt,
    _lookup_latest_price_at_or_before,
    _mark_price,
    _normalize_wallets,
    _sum_or_none,
    _write_assumptions,
    _write_csv,
    MAKER_REQUIRED_LAG_SECONDS,
    _maker_one_way_cost,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "exports/manual_seed_pma_full_history_backtest"
DEFAULT_SEED_CSV = "data/manual_seed_wallets.csv"
DEFAULT_PAGE_SIZE = 500
DEFAULT_MAX_PAGES = 200
MAX_PRICE_HISTORY_SPAN_DAYS = 14
PRICE_FETCH_BATCH_SIZE = 80
SPARSE_LOOKBACK_SECONDS = 3600


def _normalize_text(value: Any) -> str:
    """Lowercase and strip non-alphanumeric characters for fuzzy matching."""

    raw = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", raw)


def _parse_json_list(value: Any) -> list[Any]:
    """Parse Gamma fields that may arrive as JSON-encoded strings."""

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


def load_manual_seed_wallets(path: str | Path = DEFAULT_SEED_CSV) -> pd.DataFrame:
    """Load the curated seed-wallet table."""

    frame = pd.read_csv(path)
    frame["wallet_address"] = frame["wallet_address"].astype(str).str.lower().str.strip()
    frame = frame.loc[frame["wallet_address"].str.match(r"^0x[a-f0-9]{40}$", na=False)].copy()
    frame["display_name"] = frame.get("display_name", pd.Series(index=frame.index, dtype="object")).fillna(
        frame.get("sample_name", "")
    )
    return frame.reset_index(drop=True)


def _fetch_gamma_event(event_id: str, client: httpx.Client) -> dict[str, Any] | None:
    """Fetch one Gamma event payload."""

    response = client.get(f"https://gamma-api.polymarket.com/events/{event_id}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else None


def _select_gamma_market(event_payload: dict[str, Any], trade_row: dict[str, Any]) -> dict[str, Any] | None:
    """Map one PMA trade row back to one Gamma market within the event."""

    markets = event_payload.get("markets", [])
    if not isinstance(markets, list) or not markets:
        return None
    if len(markets) == 1:
        return markets[0] if isinstance(markets[0], dict) else None

    subtitle = _normalize_text(trade_row.get("market_subtitle"))
    market_title = _normalize_text(trade_row.get("market_title"))

    candidates = [market for market in markets if isinstance(market, dict)]
    if subtitle:
        direct_group = [
            market
            for market in candidates
            if _normalize_text(market.get("groupItemTitle")) == subtitle
        ]
        if direct_group:
            candidates = direct_group
        else:
            fuzzy_group = [
                market
                for market in candidates
                if subtitle in _normalize_text(market.get("groupItemTitle"))
                or subtitle in _normalize_text(market.get("question"))
                or _normalize_text(market.get("groupItemTitle")) in subtitle
            ]
            if fuzzy_group:
                candidates = fuzzy_group

    if len(candidates) == 1:
        return candidates[0]

    exact_question = [
        market for market in candidates if _normalize_text(market.get("question")) == market_title
    ]
    if len(exact_question) == 1:
        return exact_question[0]

    if subtitle:
        slug_match = [
            market
            for market in candidates
            if subtitle in _normalize_text(market.get("slug"))
            or subtitle in _normalize_text(market.get("question"))
        ]
        if len(slug_match) == 1:
            return slug_match[0]

    # Final fallback: if all remaining candidates share the same title pattern,
    # keep the first deterministic market rather than silently failing.
    if candidates:
        return sorted(candidates, key=lambda market: str(market.get("id") or ""))[0]
    return None


def _map_trade_to_gamma(
    trade_row: dict[str, Any],
    event_cache: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Attach Gamma market/token identifiers to one PMA trade."""

    event_id = str(trade_row.get("event_id") or "")
    event_payload = event_cache.get(event_id)
    if not event_payload:
        return None

    market = _select_gamma_market(event_payload, trade_row)
    if not market:
        return None

    outcomes = _parse_json_list(market.get("outcomes"))
    token_ids = _parse_json_list(market.get("clobTokenIds"))
    outcome_label = str(trade_row.get("outcome") or "").strip().lower()
    outcome_index = next(
        (index for index, value in enumerate(outcomes) if str(value).strip().lower() == outcome_label),
        None,
    )
    if outcome_index is None or outcome_index >= len(token_ids):
        return None

    timestamp = pd.to_datetime(trade_row.get("trade_dttm"), utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None

    wallet = str(trade_row.get("trader_id") or "").strip().lower()
    side = str(trade_row.get("side") or "").strip().upper()
    price = pd.to_numeric(pd.Series([trade_row.get("price")]), errors="coerce").iloc[0]
    amount = pd.to_numeric(pd.Series([trade_row.get("amount")]), errors="coerce").iloc[0]
    value = pd.to_numeric(pd.Series([trade_row.get("value")]), errors="coerce").iloc[0]
    if pd.isna(price) or pd.isna(amount) or side not in {"BUY", "SELL"}:
        return None

    trade_hash = hashlib.sha256(
        "|".join(
            [
                wallet,
                str(timestamp.isoformat()),
                event_id,
                str(market.get("id") or ""),
                str(token_ids[outcome_index]),
                side,
                f"{float(amount):.12f}",
                f"{float(price):.12f}",
            ]
        ).encode("utf-8")
    ).hexdigest()

    return {
        "trade_id": trade_hash,
        "wallet_address": wallet,
        "wallet_name": trade_row.get("trader_name"),
        "timestamp": timestamp,
        "side": side,
        "price": float(price),
        "size": float(amount),
        "usdc_size": None if pd.isna(value) else float(value),
        "event_id": event_id,
        "event_title": trade_row.get("market_title"),
        "market_subtitle": trade_row.get("market_subtitle"),
        "outcome": trade_row.get("outcome"),
        "market_id": str(market.get("id")) if market.get("id") is not None else None,
        "token_id": str(token_ids[outcome_index]),
        "raw_json": json.dumps(trade_row, sort_keys=True),
        "spread_at_trade": None,
        "slippage_bps_assumed": None,
        "liquidity_bucket": None,
    }


def _build_markets_frame(event_cache: dict[str, dict[str, Any] | None]) -> pd.DataFrame:
    """Convert cached Gamma events into the market frame expected by the backtest."""

    rows: list[dict[str, Any]] = []
    for event_id, event_payload in event_cache.items():
        if not event_payload:
            continue
        for market in event_payload.get("markets", []):
            if not isinstance(market, dict):
                continue
            rows.append(
                {
                    "id": str(market.get("id")) if market.get("id") is not None else None,
                    "question": market.get("question"),
                    "slug": market.get("slug"),
                    "condition_id": market.get("conditionId"),
                    "closed": bool(market.get("closed")),
                    "updated_at": market.get("updatedAt"),
                    "raw_json": json.dumps(market, sort_keys=True),
                    "event_id": event_id,
                    "event_title": event_payload.get("title"),
                }
            )
    return pd.DataFrame.from_records(rows)


def fetch_full_history_pma_trades(
    seed_wallets: pd.DataFrame,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int = DEFAULT_MAX_PAGES,
    bearer_token: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch full trade history for every seed wallet from PMA."""

    coverage_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    with PolymarketAnalyticsClient(bearer_token=bearer_token) as client:
        for wallet_row in seed_wallets.to_dict(orient="records"):
            wallet = str(wallet_row.get("wallet_address") or "").lower()
            display_name = wallet_row.get("display_name") or wallet
            print(f"[pma] fetching trades for {display_name} ({wallet})", flush=True)
            trades = client.get_all_activity_trades(wallet, page_size=page_size, max_pages=max_pages)
            timestamps = [row.get("trade_dttm") for row in trades if row.get("trade_dttm")]
            print(
                f"[pma] fetched {len(trades)} rows for {display_name}; "
                f"first={min(timestamps) if timestamps else 'n/a'} last={max(timestamps) if timestamps else 'n/a'}",
                flush=True,
            )
            coverage_rows.append(
                {
                    "wallet_address": wallet,
                    "display_name": display_name,
                    "trade_rows": len(trades),
                    "possible_page_cap_truncation": len(trades) >= page_size * max_pages,
                    "first_trade_ts": min(timestamps) if timestamps else None,
                    "most_recent_trade_ts": max(timestamps) if timestamps else None,
                    "unique_events": len({row.get("event_id") for row in trades if row.get("event_id")}),
                }
            )
            for row in trades:
                enriched = dict(row)
                enriched["wallet_address"] = wallet
                enriched["display_name"] = display_name
                raw_rows.append(enriched)

    raw_frame = pd.DataFrame.from_records(raw_rows)
    coverage = pd.DataFrame.from_records(coverage_rows).sort_values(
        ["trade_rows", "wallet_address"],
        ascending=[False, True],
    )
    return raw_frame, coverage


def map_pma_trades_to_public_markets(
    raw_trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map PMA trade rows to Gamma event markets and token ids."""

    if raw_trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    unique_event_ids = sorted({str(value) for value in raw_trades["event_id"].dropna().unique()})
    event_cache: dict[str, dict[str, Any] | None] = {}
    settings = get_settings()

    def fetch_one_event(event_id: str) -> tuple[str, dict[str, Any] | None]:
        with httpx.Client(
            verify=certifi.where(),
            timeout=max(settings.request_timeout, 30),
            headers={"User-Agent": "polymarket-wallet-research/0.1"},
            follow_redirects=True,
        ) as gamma_client:
            try:
                return event_id, _fetch_gamma_event(event_id, gamma_client)
            except httpx.HTTPError as exc:
                LOGGER.warning("Failed to fetch Gamma event %s: %s", event_id, exc)
                return event_id, None

    max_workers = min(16, max(4, settings.max_concurrency))
    print(f"[gamma] fetching {len(unique_event_ids)} events with {max_workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fetch_one_event, event_id): event_id for event_id in unique_event_ids}
        completed = 0
        total = len(future_map)
        for future in as_completed(future_map):
            event_id, payload = future.result()
            event_cache[event_id] = payload
            completed += 1
            if completed % 250 == 0 or completed == total:
                print(f"[gamma] fetched {completed}/{total} events", flush=True)

    mapped_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in raw_trades.to_dict(orient="records"):
        mapped = _map_trade_to_gamma(row, event_cache)
        audit = {
            "wallet_address": row.get("wallet_address"),
            "display_name": row.get("display_name"),
            "trade_dttm": row.get("trade_dttm"),
            "event_id": row.get("event_id"),
            "market_title": row.get("market_title"),
            "market_subtitle": row.get("market_subtitle"),
            "outcome": row.get("outcome"),
            "side": row.get("side"),
            "price": row.get("price"),
            "amount": row.get("amount"),
            "mapped": mapped is not None,
            "market_id": mapped.get("market_id") if mapped else None,
            "token_id": mapped.get("token_id") if mapped else None,
        }
        audit_rows.append(audit)
        if mapped is not None:
            mapped_rows.append(mapped)

    mapped_frame = pd.DataFrame.from_records(mapped_rows).sort_values(
        ["wallet_address", "timestamp", "trade_id"]
    )
    audit_frame = pd.DataFrame.from_records(audit_rows)
    markets_frame = _build_markets_frame(event_cache)
    return mapped_frame.reset_index(drop=True), audit_frame.reset_index(drop=True), markets_frame


async def _fetch_price_payloads(
    token_bounds: list[tuple[str, datetime, datetime]],
    *,
    settings: Settings,
) -> pd.DataFrame:
    """Fetch public price history into one in-memory frame."""

    if not token_bounds:
        return pd.DataFrame(columns=["token_id", "ts", "price"])

    request_specs: list[tuple[str, int, int]] = []
    max_span = timedelta(days=MAX_PRICE_HISTORY_SPAN_DAYS)
    for token_id, start_dt, end_dt in token_bounds:
        start = pd.to_datetime(start_dt, utc=True)
        end = pd.to_datetime(end_dt, utc=True)
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + max_span, end)
            request_specs.append((str(token_id), int(cursor.timestamp()), int(chunk_end.timestamp())))
            cursor = chunk_end + pd.Timedelta(seconds=1)

    semaphore = asyncio.Semaphore(settings.max_concurrency)
    rows: list[dict[str, Any]] = []

    async def fetch_chunk(
        client: ClobClient,
        token_id: str,
        start_ts: int,
        end_ts: int,
    ) -> list[dict[str, Any]]:
        async with semaphore:
            try:
                payload = await client.get_prices_history(
                    token_id=token_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    fidelity=1,
                )
            except httpx.HTTPStatusError as exc:
                response = exc.response
                if response.status_code == 400 and "interval is too long" in response.text:
                    midpoint = start_ts + max(1, math.floor((end_ts - start_ts) / 2))
                    if midpoint <= start_ts or midpoint >= end_ts:
                        return []
                    left = await fetch_chunk(client, token_id, start_ts, midpoint)
                    right = await fetch_chunk(client, token_id, midpoint + 1, end_ts)
                    return left + right
                if response.status_code in {400, 404}:
                    return []
                raise
            return _history_rows(token_id, payload)

    async with ClobClient() as clob_client:
        for batch_start in range(0, len(request_specs), PRICE_FETCH_BATCH_SIZE):
            batch = request_specs[batch_start : batch_start + PRICE_FETCH_BATCH_SIZE]
            total_batches = math.ceil(len(request_specs) / PRICE_FETCH_BATCH_SIZE)
            print(
                f"[prices] fetching batch {batch_start // PRICE_FETCH_BATCH_SIZE + 1}/{total_batches} "
                f"({len(batch)} requests)",
                flush=True,
            )
            tasks = [
                asyncio.create_task(fetch_chunk(clob_client, token_id, start_ts, end_ts))
                for token_id, start_ts, end_ts in batch
            ]
            for chunk_rows in await asyncio.gather(*tasks):
                rows.extend(chunk_rows)

    if not rows:
        return pd.DataFrame(columns=["token_id", "ts", "price"])

    frame = pd.DataFrame.from_records(rows)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["token_id", "ts", "price"]).drop_duplicates(["token_id", "ts"])
    return frame.sort_values(["token_id", "ts"]).reset_index(drop=True)


def _collect_sparse_token_bounds(
    records: list[dict[str, Any]],
    *,
    delays: tuple[int, ...],
    analysis_asof: datetime,
) -> list[tuple[str, datetime, datetime]]:
    """Build narrow price windows around modeled entry/exit timestamps only."""

    per_token: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    cutoff_ts = pd.to_datetime(analysis_asof, utc=True)
    lookback = pd.Timedelta(seconds=SPARSE_LOOKBACK_SECONDS)
    lookahead = pd.Timedelta(seconds=MAKER_REQUIRED_LAG_SECONDS)

    for record in records:
        token_id = str(record.get("token_id") or "")
        if not token_id:
            continue
        buy_ts = pd.to_datetime(record.get("buy_timestamp"), utc=True, errors="coerce")
        sell_ts = pd.to_datetime(record.get("sell_timestamp"), utc=True, errors="coerce")
        if pd.isna(buy_ts):
            continue

        windows = per_token.setdefault(token_id, [])
        for delay_seconds in delays:
            entry_target = buy_ts + pd.Timedelta(seconds=int(delay_seconds))
            entry_end = min(entry_target + lookahead, cutoff_ts)
            windows.append((entry_target - lookback, entry_end))
            if pd.notna(sell_ts):
                exit_target = sell_ts + pd.Timedelta(seconds=int(delay_seconds))
                exit_end = min(exit_target + lookahead, cutoff_ts)
                windows.append((exit_target - lookback, exit_end))

    merged_specs: list[tuple[str, datetime, datetime]] = []
    for token_id, windows in per_token.items():
        ordered = sorted(
            (pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True))
            for start, end in windows
            if pd.notna(start) and pd.notna(end)
        )
        if not ordered:
            continue
        current_start, current_end = ordered[0]
        for start, end in ordered[1:]:
            if start <= current_end + pd.Timedelta(seconds=1):
                current_end = max(current_end, end)
            else:
                merged_specs.append((token_id, current_start.to_pydatetime(), current_end.to_pydatetime()))
                current_start, current_end = start, end
        merged_specs.append((token_id, current_start.to_pydatetime(), current_end.to_pydatetime()))
    return merged_specs


def _scan_passive_fill_capped(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    *,
    target_epoch: int,
    limit_price: float,
    is_buy: bool,
) -> FillAttempt:
    """Conservative maker fill scan within a short post-signal window only."""

    if token_id is None or token_id not in index:
        return FillAttempt(None, "missing_prices", None, None, "unfilled_maker")

    times, prices = index[str(token_id)]
    position = bisect_left(times, target_epoch)
    maker_deadline = target_epoch + MAKER_REQUIRED_LAG_SECONDS
    while position < len(times) and times[position] <= maker_deadline:
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


def _entry_fill_sparse(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    *,
    target_epoch: int,
    lookup_forward_price_fn: Any,
) -> FillAttempt:
    """Return one modeled copied entry fill using sparse public prices."""

    forward = lookup_forward_price_fn(index, token_id, target_epoch)
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
        return _scan_passive_fill_capped(
            index,
            token_id,
            target_epoch=target_epoch,
            limit_price=float(last_price),
            is_buy=True,
        )

    if forward.price is not None and forward.delta_seconds is not None:
        return FillAttempt(
            price=float(forward.price),
            source=forward.source,
            delta_seconds=int(forward.delta_seconds),
            fill_epoch=target_epoch + int(forward.delta_seconds),
            fill_mode="taker_late_no_prior_price",
        )
    return FillAttempt(None, "missing_entry_price_after_delay", None, None, "unfilled")


def _exit_fill_sparse(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    *,
    target_epoch: int,
    lookup_forward_price_fn: Any,
) -> FillAttempt:
    """Return one modeled copied exit fill using sparse public prices."""

    forward = lookup_forward_price_fn(index, token_id, target_epoch)
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
        return _scan_passive_fill_capped(
            index,
            token_id,
            target_epoch=target_epoch,
            limit_price=float(last_price),
            is_buy=False,
        )

    if forward.price is not None and forward.delta_seconds is not None:
        return FillAttempt(
            price=float(forward.price),
            source=forward.source,
            delta_seconds=int(forward.delta_seconds),
            fill_epoch=target_epoch + int(forward.delta_seconds),
            fill_mode="taker_late_no_prior_price",
        )
    return FillAttempt(None, "missing_exit_price_after_delay", None, None, "unfilled")


def compute_manual_seed_pma_copy_backtest_from_frame(
    trades: pd.DataFrame,
    price_history: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    delays: tuple[int, ...],
    analysis_asof: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the wallet-exit copy backtest on one mapped PMA trade frame."""

    pairs, open_positions = build_copy_exit_pairs(trades)
    base_records: list[dict[str, Any]] = []
    if not pairs.empty:
        base_records.extend(pairs.to_dict(orient="records"))
    if not open_positions.empty:
        base_records.extend(open_positions.to_dict(orient="records"))
    if not base_records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    from research.delay_analysis import _build_price_index, _to_epoch_seconds, lookup_forward_price
    from research.costs import calculate_net_pnl, estimate_entry_only_cost

    cfg = get_settings()
    price_index = _build_price_index(price_history)
    terminal_lookup = _build_terminal_lookup(markets)
    cutoff_ts = pd.to_datetime(analysis_asof, utc=True)
    cutoff_epoch = _to_epoch_seconds(cutoff_ts)

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
            "analysis_asof": cutoff_ts,
            "resolution_ts": resolution_ts,
            "terminal_price": terminal_price,
            "terminal_price_source": terminal_info.get("terminal_price_source"),
        }

        for delay_seconds in delays:
            label = f"{int(delay_seconds)}s"
            entry_target_epoch = buy_epoch + int(delay_seconds)
            entry_until_epoch = sell_epoch + int(delay_seconds) if sell_epoch is not None else cutoff_epoch
            entry = _entry_fill_sparse(
                price_index,
                token_id,
                target_epoch=entry_target_epoch,
                lookup_forward_price_fn=lookup_forward_price,
            )

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

                exit_fill = _exit_fill_sparse(
                    price_index,
                    token_id,
                    target_epoch=exit_target_epoch,
                    lookup_forward_price_fn=lookup_forward_price,
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
                    raw_usdc = raw_unit * copied_size
                    total_cost_unit = float(entry_cost_unit) + float(exit_cost_unit)
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

    source_groups = trades.groupby("wallet_address") if not trades.empty else None
    summary_rows: list[dict[str, Any]] = []
    for wallet_id, group in trade_diagnostics.groupby("wallet_address", sort=False):
        source_group = source_groups.get_group(wallet_id) if source_groups is not None and wallet_id in source_groups.groups else pd.DataFrame()
        row: dict[str, Any] = {
            "wallet_address": wallet_id,
            "n_trades": int(len(source_group)),
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
            row[f"maker_entry_fills_{label}"] = int((group[entry_mode_col] == "maker").sum())
            row[f"maker_exit_fills_{label}"] = int((group[exit_mode_col] == "maker").sum())
        summary_rows.append(row)

    wallet_summary = pd.DataFrame.from_records(summary_rows).sort_values(
        ["combined_net_total_usdc_30s", "wallet_address"],
        ascending=[False, True],
        na_position="last",
    )

    overview = pd.DataFrame(
        [
            {
                "wallets_requested": int(trades["wallet_address"].nunique()) if not trades.empty else 0,
                "raw_trades_full_history": int(len(trades)),
                "copy_slices_total": int(len(trade_diagnostics)),
                "analysis_asof": cutoff_ts.isoformat(),
                **{
                    f"combined_net_total_usdc_{int(delay)}s": _sum_or_none(
                        trade_diagnostics[f"copy_combined_net_usdc_{int(delay)}s"]
                    )
                    for delay in delays
                },
            }
        ]
    )
    return wallet_summary.reset_index(drop=True), trade_diagnostics.reset_index(drop=True), overview


def run_manual_seed_pma_full_history_backtest(
    *,
    seed_csv: str | Path = DEFAULT_SEED_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    delays: tuple[int, ...] = (0, 5, 15, 30, 60),
    bearer_token: str | None = None,
) -> dict[str, Any]:
    """Fetch PMA full-history trades for manual seeds and run one copy backtest."""

    settings = get_settings()
    seed_wallets = load_manual_seed_wallets(seed_csv)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    delay_label = "_".join(f"{int(delay)}s" for delay in delays)
    raw_trades_file = output_root / "manual_seed_pma_activity_trades_full_history.csv"
    coverage_file = output_root / "manual_seed_pma_trade_coverage.csv"
    mapped_trades_file = output_root / "manual_seed_pma_mapped_trades_full_history.csv"
    audit_file = output_root / "manual_seed_pma_mapping_audit.csv"
    markets_file = output_root / "manual_seed_pma_markets_frame.csv"
    price_file = output_root / "manual_seed_pma_price_history.csv"
    wallet_file = output_root / f"manual_seed_pma_backtest_wallet_summary_{delay_label}_full_history.csv"
    diagnostics_file = output_root / f"manual_seed_pma_backtest_trade_diagnostics_{delay_label}_full_history.csv"
    overview_file = output_root / "manual_seed_pma_backtest_summary.csv"
    assumptions_file = output_root / "manual_seed_pma_backtest_assumptions.md"

    print(f"[run] loaded {len(seed_wallets)} seed wallets", flush=True)
    if raw_trades_file.exists() and coverage_file.exists():
        raw_trades = pd.read_csv(raw_trades_file)
        coverage = pd.read_csv(coverage_file)
        print(f"[run] loaded cached raw PMA trades from {raw_trades_file}", flush=True)
    else:
        raw_trades, coverage = fetch_full_history_pma_trades(
            seed_wallets,
            page_size=DEFAULT_PAGE_SIZE,
            max_pages=DEFAULT_MAX_PAGES,
            bearer_token=bearer_token,
        )
        _write_csv(raw_trades, raw_trades_file)
        _write_csv(coverage, coverage_file)
    print(f"[run] fetched {len(raw_trades)} PMA trade rows", flush=True)

    if mapped_trades_file.exists() and audit_file.exists() and markets_file.exists():
        mapped_trades = pd.read_csv(mapped_trades_file)
        if not mapped_trades.empty and "timestamp" in mapped_trades.columns:
            mapped_trades["timestamp"] = pd.to_datetime(mapped_trades["timestamp"], utc=True, errors="coerce")
        mapping_audit = pd.read_csv(audit_file)
        markets = pd.read_csv(markets_file)
        print(f"[run] loaded cached mapping artifacts from {mapped_trades_file}", flush=True)
    else:
        mapped_trades, mapping_audit, markets = map_pma_trades_to_public_markets(raw_trades)
        _write_csv(mapped_trades, mapped_trades_file)
        _write_csv(mapping_audit, audit_file)
        _write_csv(markets, markets_file)
    print(
        f"[run] mapped {len(mapped_trades)} trades across "
        f"{mapped_trades['token_id'].nunique() if not mapped_trades.empty else 0} tokens",
        flush=True,
    )
    analysis_asof = datetime.now(timezone.utc)

    if price_file.exists():
        price_history = pd.read_csv(price_file)
        if not price_history.empty and "ts" in price_history.columns:
            price_history["ts"] = pd.to_datetime(price_history["ts"], utc=True, errors="coerce")
        print(f"[run] loaded cached price history from {price_file}", flush=True)
    else:
        pairs, open_positions = build_copy_exit_pairs(mapped_trades)
        base_records: list[dict[str, Any]] = []
        if not pairs.empty:
            base_records.extend(pairs.to_dict(orient="records"))
        if not open_positions.empty:
            base_records.extend(open_positions.to_dict(orient="records"))
        token_bounds = _collect_sparse_token_bounds(
            base_records,
            delays=delays,
            analysis_asof=analysis_asof,
        )
        print(f"[run] built {len(token_bounds)} sparse token windows", flush=True)
        price_history = asyncio.run(_fetch_price_payloads(token_bounds, settings=settings))
        _write_csv(price_history, price_file)
    print(f"[run] fetched {len(price_history)} public price rows", flush=True)

    wallet_summary, trade_diagnostics, overview = compute_manual_seed_pma_copy_backtest_from_frame(
        mapped_trades,
        price_history,
        markets,
        delays=delays,
        analysis_asof=analysis_asof,
    )

    dashboards: list[dict[str, Any]] = []
    with PolymarketAnalyticsClient(bearer_token=bearer_token) as client:
        for row in seed_wallets.to_dict(orient="records"):
            wallet = str(row["wallet_address"]).lower()
            payload = client.get_trader_dashboard(wallet) or {}
            dashboards.append(
                {
                    "wallet_address": wallet,
                    "display_name": row.get("display_name") or wallet,
                    "pma_overall_gain_usd": payload.get("overall_gain"),
                    "pma_total_current_value_usd": payload.get("total_current_value"),
                    "pma_win_rate": payload.get("win_rate"),
                    "pma_total_positions": payload.get("total_positions"),
                    "pma_active_positions": payload.get("active_positions"),
                }
            )
    dashboard_frame = pd.DataFrame.from_records(dashboards)
    if not wallet_summary.empty:
        wallet_summary = wallet_summary.merge(dashboard_frame, on="wallet_address", how="left")

    raw_trades_path = _write_csv(raw_trades, raw_trades_file)
    coverage_path = _write_csv(coverage, coverage_file)
    _write_csv(mapped_trades, mapped_trades_file)
    audit_path = _write_csv(mapping_audit, audit_file)
    markets_path = _write_csv(markets, markets_file)
    price_path = _write_csv(price_history, price_file)
    wallet_path = _write_csv(wallet_summary, wallet_file)
    diagnostics_path = _write_csv(trade_diagnostics, diagnostics_file)
    overview_path = _write_csv(overview, overview_file)
    assumptions_path = _write_assumptions(assumptions_file)
    print(f"[run] finished; outputs written to {output_root}", flush=True)

    return {
        "seed_wallets": seed_wallets,
        "raw_trades": raw_trades,
        "coverage": coverage,
        "mapping_audit": mapping_audit,
        "markets": markets,
        "price_history": price_history,
        "wallet_summary": wallet_summary,
        "trade_diagnostics": trade_diagnostics,
        "overview": overview,
        "raw_trades_path": raw_trades_path,
        "coverage_path": coverage_path,
        "audit_path": audit_path,
        "markets_path": markets_path,
        "price_path": price_path,
        "wallet_path": wallet_path,
        "diagnostics_path": diagnostics_path,
        "overview_path": overview_path,
        "assumptions_path": assumptions_path,
    }
