"""Trade enrichment pipeline for returns, current book proxies, and cost-adjusted PnL."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import sys
from bisect import bisect_left
from datetime import datetime, timezone
from typing import Any, Sequence

import httpx
sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from clients.clob_client import ClobClient
from config.settings import get_settings
from db.models import Market, PriceHistory, WalletTradeEnriched, WalletTradeRaw
from research.costs import apply_copy_fade_costs


LOGGER = logging.getLogger(__name__)
ENRICHED_UPSERT_CHUNK_SIZE = 250


@dataclass(frozen=True)
class PriceLookup:
    """Nearest price lookup result plus provenance."""

    price: float | None
    source: str
    delta_seconds: int | None


def normalize_side(side: str | None) -> str:
    """Normalize side labels to BUY or SELL."""

    normalized = str(side or "").strip().upper()
    if normalized in {"BUY", "B"}:
        return "BUY"
    if normalized in {"SELL", "S"}:
        return "SELL"
    raise ValueError(f"Unsupported trade side: {side!r}")


def classify_liquidity_bucket(size: float | None) -> str:
    """Bucket trade sizes for coarse slippage assumptions."""

    if size is None:
        return "unknown"
    if size < 25:
        return "micro"
    if size < 100:
        return "small"
    if size < 500:
        return "medium"
    return "large"


def _to_epoch_seconds(value: datetime) -> int:
    """Convert a datetime to UTC epoch seconds."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.astimezone(timezone.utc).timestamp())


def _build_price_index(session: Session, token_ids: list[str]) -> dict[str, tuple[list[int], list[float]]]:
    """Load price history into a bisect-friendly in-memory index."""

    index: dict[str, tuple[list[int], list[float]]] = {}
    if not token_ids:
        return index

    rows = session.execute(
        select(PriceHistory.token_id, PriceHistory.ts, PriceHistory.price)
        .where(PriceHistory.token_id.in_(token_ids))
        .order_by(PriceHistory.token_id, PriceHistory.ts)
    )
    for token_id, ts, price in rows:
        times, prices = index.setdefault(token_id, ([], []))
        times.append(_to_epoch_seconds(ts))
        prices.append(float(price))
    return index


def _lookup_nearest_price(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    target_ts: int,
    exactish_threshold_seconds: int = 60,
) -> PriceLookup:
    """Return the nearest recorded price plus a provenance label."""

    if token_id is None or token_id not in index:
        return PriceLookup(price=None, source="missing_prices", delta_seconds=None)

    times, prices = index[token_id]
    if not times:
        return PriceLookup(price=None, source="missing_prices", delta_seconds=None)

    position = bisect_left(times, target_ts)
    candidates: list[tuple[int, float]] = []
    if position < len(times):
        candidates.append((abs(times[position] - target_ts), prices[position]))
    if position > 0:
        candidates.append((abs(times[position - 1] - target_ts), prices[position - 1]))
    if not candidates:
        return PriceLookup(price=None, source="missing_prices", delta_seconds=None)
    candidates.sort(key=lambda item: item[0])
    delta_seconds, price = candidates[0]
    source = (
        "price_history_exactish"
        if delta_seconds <= exactish_threshold_seconds
        else "price_history_nearest"
    )
    return PriceLookup(price=price, source=source, delta_seconds=delta_seconds)


def _extract_book_metrics(book: dict[str, Any] | None) -> dict[str, float | None]:
    """Extract best bid, best ask, spread, and midpoint from a current order book snapshot."""

    if not book or book.get("error"):
        return {
            "best_bid_at_trade": None,
            "best_ask_at_trade": None,
            "spread_at_trade": None,
            "mid_at_trade": None,
        }

    bids = book.get("bids") or []
    asks = book.get("asks") or []
    best_bid = max((float(level["price"]) for level in bids if "price" in level), default=None)
    best_ask = min((float(level["price"]) for level in asks if "price" in level), default=None)
    spread = None
    midpoint = None
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid
        midpoint = (best_ask + best_bid) / 2.0

    return {
        "best_bid_at_trade": best_bid,
        "best_ask_at_trade": best_ask,
        "spread_at_trade": spread,
        "mid_at_trade": midpoint,
    }


async def _fetch_order_books(token_ids: list[str], client: ClobClient | None = None) -> dict[str, dict[str, Any]]:
    """Fetch current public order books for all tokens seen in raw trades."""

    if not token_ids:
        return {}

    own_client = client is None
    clob_client = client or ClobClient()
    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.max_concurrency)

    async def _fetch(token_id: str) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            try:
                # Future realtime research extension:
                # this is where a market WebSocket subscriber should keep rolling public
                # order book snapshots so event-time book state can be captured directly
                # instead of using a late live lookup.
                return token_id, await clob_client.get_order_book(token_id)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    LOGGER.info("No current order book available for token %s", token_id)
                    return token_id, {}
                LOGGER.warning("Order book lookup failed for token %s: %s", token_id, exc)
                return token_id, {}
            except Exception as exc:
                LOGGER.warning("Order book lookup failed for token %s: %s", token_id, exc)
                return token_id, {}

    try:
        pairs = await asyncio.gather(*[_fetch(token_id) for token_id in token_ids])
    finally:
        if own_client:
            await clob_client.aclose()

    return dict(pairs)


def _resolve_trade_midpoint(
    history_lookup: PriceLookup,
    book_metrics: dict[str, float | None],
) -> tuple[float | None, str, bool]:
    """Resolve the trade-time midpoint using the documented priority order."""

    if history_lookup.price is not None:
        return history_lookup.price, history_lookup.source, False
    if book_metrics["mid_at_trade"] is not None:
        return book_metrics["mid_at_trade"], "live_book_approx", True
    return None, "missing_prices", False


def _enrichment_status(
    *,
    missing_price_history: bool,
    missing_market_metadata: bool,
    used_fallback_midpoint: bool,
) -> str:
    """Summarize the enrichment outcome for one trade row."""

    if not missing_price_history and not missing_market_metadata and not used_fallback_midpoint:
        return "complete"
    if missing_price_history and missing_market_metadata:
        return "missing_prices_and_market_metadata"
    if missing_price_history:
        return "missing_prices" if not used_fallback_midpoint else "partial_missing_prices"
    if missing_market_metadata:
        return "missing_market_metadata"
    return "partial"


def _normalize_wallets(wallets: Sequence[str] | None) -> list[str]:
    """Normalize an optional wallet filter."""

    if wallets is None:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for wallet in wallets:
        normalized = str(wallet or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


async def enrich_wallet_trades(
    session: Session,
    client: ClobClient | None = None,
    *,
    wallets: Sequence[str] | None = None,
    fetch_books: bool = True,
) -> int:
    """Build the enriched wallet trades table from raw trades and price history.

    When `wallets` is supplied, enrichment is restricted to that public-wallet subset.
    """

    normalized_wallets = _normalize_wallets(wallets)
    query = select(WalletTradeRaw).order_by(WalletTradeRaw.timestamp, WalletTradeRaw.trade_id)
    if normalized_wallets:
        query = query.where(WalletTradeRaw.wallet_address.in_(normalized_wallets))

    raw_trades = list(session.scalars(query))
    if not raw_trades:
        return 0

    token_ids = sorted({trade.token_id for trade in raw_trades if trade.token_id})
    price_index = _build_price_index(session, token_ids)
    books = await _fetch_order_books(token_ids, client=client) if fetch_books else {}
    market_refs = {
        str(value)
        for market_id, condition_id in session.execute(select(Market.id, Market.condition_id))
        for value in (market_id, condition_id)
        if value
    }

    rows: list[dict[str, Any]] = []
    for trade in raw_trades:
        if trade.price is None:
            LOGGER.warning("Skipping trade %s because price is missing", trade.trade_id)
            continue

        try:
            side = normalize_side(trade.side)
        except ValueError:
            LOGGER.warning("Skipping trade %s because side is invalid: %r", trade.trade_id, trade.side)
            continue

        trade_ts = _to_epoch_seconds(trade.timestamp)
        direction = 1.0 if side == "BUY" else -1.0
        book_metrics = _extract_book_metrics(books.get(trade.token_id))
        trade_lookup = _lookup_nearest_price(price_index, trade.token_id, trade_ts)
        mid_1m_lookup = _lookup_nearest_price(price_index, trade.token_id, trade_ts + 60)
        mid_5m_lookup = _lookup_nearest_price(price_index, trade.token_id, trade_ts + 5 * 60)
        mid_30m_lookup = _lookup_nearest_price(price_index, trade.token_id, trade_ts + 30 * 60)
        mid_at_trade, midpoint_source, used_fallback_midpoint = _resolve_trade_midpoint(
            trade_lookup,
            book_metrics,
        )
        missing_market_metadata = not bool(trade.market_id and str(trade.market_id) in market_refs)
        missing_horizons = [
            horizon
            for horizon, lookup in {
                "trade": trade_lookup,
                "1m": mid_1m_lookup,
                "5m": mid_5m_lookup,
                "30m": mid_30m_lookup,
            }.items()
            if lookup.price is None
        ]
        missing_price_history = bool(missing_horizons)
        missing_reason_parts: list[str] = []
        if missing_price_history:
            missing_reason_parts.append(f"price_history_missing:{','.join(missing_horizons)}")
        if missing_market_metadata:
            missing_reason_parts.append("market_metadata_not_found")
        if used_fallback_midpoint:
            missing_reason_parts.append("used_live_book_fallback_for_trade_midpoint")
        book_source = (
            "live_book_approx"
            if book_metrics["mid_at_trade"] is not None
            or book_metrics["best_bid_at_trade"] is not None
            or book_metrics["best_ask_at_trade"] is not None
            else "missing_book"
        )

        base_row: dict[str, Any] = {
            "trade_id": trade.trade_id,
            "wallet_address": trade.wallet_address,
            "market_id": trade.market_id,
            "token_id": trade.token_id,
            "timestamp": trade.timestamp.astimezone(timezone.utc)
            if trade.timestamp.tzinfo
            else trade.timestamp.replace(tzinfo=timezone.utc),
            "side": side,
            "price": float(trade.price),
            "size": trade.size,
            "best_bid_at_trade": book_metrics["best_bid_at_trade"],
            "best_ask_at_trade": book_metrics["best_ask_at_trade"],
            "spread_at_trade": book_metrics["spread_at_trade"],
            "mid_at_trade": mid_at_trade,
            # Future realtime research extension:
            # delayed forward-return tracking should hook in here if the project later
            # records timestamped event emissions or websocket snapshots in real time.
            "mid_1m": mid_1m_lookup.price,
            "mid_5m": mid_5m_lookup.price,
            "mid_30m": mid_30m_lookup.price,
            "liquidity_bucket": classify_liquidity_bucket(trade.size),
            "missing_price_history": missing_price_history,
            "missing_market_metadata": missing_market_metadata,
            "used_fallback_midpoint": used_fallback_midpoint,
            "trade_price_source": "public_trade_feed",
            "midpoint_source": midpoint_source,
            "book_source": book_source,
            "enrichment_status": _enrichment_status(
                missing_price_history=missing_price_history,
                missing_market_metadata=missing_market_metadata,
                used_fallback_midpoint=used_fallback_midpoint,
            ),
            "missing_reason": ";".join(missing_reason_parts) if missing_reason_parts else None,
            "raw_json": trade.raw_json,
        }

        for horizon in ("1m", "5m", "30m"):
            midpoint = base_row[f"mid_{horizon}"]
            base_row[f"ret_{horizon}"] = (
                direction * (float(midpoint) - float(trade.price))
                if midpoint is not None
                else None
            )

        priced_row = pd.Series(base_row)
        cost_adjusted = apply_copy_fade_costs(priced_row)
        base_row["slippage_bps_assumed"] = float(cost_adjusted["slippage_bps_assumed"])
        base_row["fees_bps_assumed"] = float(cost_adjusted["fees_bps_assumed"])
        for key in ("copy_pnl_1m", "copy_pnl_5m", "copy_pnl_30m", "fade_pnl_1m", "fade_pnl_5m", "fade_pnl_30m"):
            base_row[key] = cost_adjusted[key]

        rows.append(base_row)

    if not rows:
        return 0

    for start in range(0, len(rows), ENRICHED_UPSERT_CHUNK_SIZE):
        chunk = rows[start : start + ENRICHED_UPSERT_CHUNK_SIZE]
        stmt = insert(WalletTradeEnriched).values(chunk)
        excluded = stmt.excluded
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[WalletTradeEnriched.trade_id],
                set_={
                    "wallet_address": excluded.wallet_address,
                    "market_id": excluded.market_id,
                    "token_id": excluded.token_id,
                    "timestamp": excluded.timestamp,
                    "side": excluded.side,
                    "price": excluded.price,
                    "size": excluded.size,
                    "best_bid_at_trade": excluded.best_bid_at_trade,
                    "best_ask_at_trade": excluded.best_ask_at_trade,
                    "spread_at_trade": excluded.spread_at_trade,
                    "mid_at_trade": excluded.mid_at_trade,
                    "mid_1m": excluded.mid_1m,
                    "mid_5m": excluded.mid_5m,
                    "mid_30m": excluded.mid_30m,
                    "ret_1m": excluded.ret_1m,
                    "ret_5m": excluded.ret_5m,
                    "ret_30m": excluded.ret_30m,
                    "copy_pnl_1m": excluded.copy_pnl_1m,
                    "copy_pnl_5m": excluded.copy_pnl_5m,
                    "copy_pnl_30m": excluded.copy_pnl_30m,
                    "fade_pnl_1m": excluded.fade_pnl_1m,
                    "fade_pnl_5m": excluded.fade_pnl_5m,
                    "fade_pnl_30m": excluded.fade_pnl_30m,
                    "slippage_bps_assumed": excluded.slippage_bps_assumed,
                    "fees_bps_assumed": excluded.fees_bps_assumed,
                    "liquidity_bucket": excluded.liquidity_bucket,
                    "missing_price_history": excluded.missing_price_history,
                    "missing_market_metadata": excluded.missing_market_metadata,
                    "used_fallback_midpoint": excluded.used_fallback_midpoint,
                    "trade_price_source": excluded.trade_price_source,
                    "midpoint_source": excluded.midpoint_source,
                    "book_source": excluded.book_source,
                    "enrichment_status": excluded.enrichment_status,
                    "missing_reason": excluded.missing_reason,
                    "raw_json": excluded.raw_json,
                },
            )
        )
    session.commit()
    return len(rows)
