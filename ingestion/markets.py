"""Backfill Gamma market metadata referenced by wallet trades."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import httpx
from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from clients.gamma_client import GammaClient
from config.settings import get_settings
from db.models import Market, Token, WalletTradeRaw


LOGGER = logging.getLogger(__name__)


def _chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    """Yield fixed-size chunks."""

    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def _to_utc_datetime(value: Any) -> datetime | None:
    """Parse optional datetime strings into UTC datetimes."""

    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_json_list(value: Any) -> list[Any]:
    """Parse Gamma's stringified list fields."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _upsert_markets(session: Session, markets: list[dict[str, Any]]) -> dict[str, str]:
    """Upsert markets and return a condition ID to Gamma market ID mapping."""

    if not markets:
        return {}

    rows: list[dict[str, Any]] = []
    condition_to_market: dict[str, str] = {}
    for market in markets:
        market_id = str(market["id"])
        condition_id = market.get("conditionId")
        if condition_id:
            condition_to_market[str(condition_id)] = market_id

        rows.append(
            {
                "id": market_id,
                "question": market.get("question"),
                "slug": market.get("slug"),
                "condition_id": condition_id,
                "active": market.get("active"),
                "closed": market.get("closed"),
                "archived": market.get("archived"),
                "enable_order_book": market.get("enableOrderBook"),
                "created_at": _to_utc_datetime(market.get("createdAt")),
                "updated_at": _to_utc_datetime(market.get("updatedAt")),
                "raw_json": json.dumps(market, sort_keys=True, ensure_ascii=False),
            }
        )

    stmt = insert(Market).values(rows)
    excluded = stmt.excluded
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Market.id],
            set_={
                "question": excluded.question,
                "slug": excluded.slug,
                "condition_id": excluded.condition_id,
                "active": excluded.active,
                "closed": excluded.closed,
                "archived": excluded.archived,
                "enable_order_book": excluded.enable_order_book,
                "created_at": excluded.created_at,
                "updated_at": excluded.updated_at,
                "raw_json": excluded.raw_json,
            },
        )
    )
    session.flush()
    return condition_to_market


def _upsert_tokens(session: Session, markets: list[dict[str, Any]]) -> int:
    """Populate token metadata from market outcome lists."""

    token_rows: list[dict[str, Any]] = []
    for market in markets:
        market_id = str(market["id"])
        token_ids = _parse_json_list(market.get("clobTokenIds"))
        outcomes = _parse_json_list(market.get("outcomes"))

        for index, token_id in enumerate(token_ids):
            payload = {
                "token_id": str(token_id),
                "market_id": market_id,
                "outcome": outcomes[index] if index < len(outcomes) else None,
                "raw_json": json.dumps(
                    {
                        "token_id": token_id,
                        "market_id": market_id,
                        "conditionId": market.get("conditionId"),
                        "outcome": outcomes[index] if index < len(outcomes) else None,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            }
            token_rows.append(payload)

    if not token_rows:
        return 0

    stmt = insert(Token).values(token_rows)
    excluded = stmt.excluded
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Token.token_id],
            set_={
                "market_id": excluded.market_id,
                "outcome": excluded.outcome,
                "raw_json": excluded.raw_json,
            },
        )
    )
    session.flush()
    return len(token_rows)


async def _fetch_markets_for_conditions(
    gamma_client: GammaClient,
    condition_ids: Sequence[str],
    *,
    max_concurrency: int,
) -> dict[str, dict[str, Any]]:
    """Fetch Gamma markets keyed by Gamma market id for a set of condition ids."""

    semaphore = asyncio.Semaphore(max_concurrency)
    collected: dict[str, dict[str, Any]] = {}

    async def fetch_one(condition_id: str) -> None:
        async with semaphore:
            try:
                batch = await gamma_client.list_markets(limit=1, condition_ids=[condition_id])
            except httpx.HTTPStatusError as exc:
                LOGGER.warning("Market lookup failed for condition %s: %s", condition_id, exc)
                return
            for market in batch:
                collected[str(market["id"])] = market

    await asyncio.gather(*(fetch_one(condition_id) for condition_id in condition_ids))
    return collected


async def _fetch_markets_for_tokens(
    gamma_client: GammaClient,
    token_ids: Sequence[str],
    *,
    max_concurrency: int,
) -> dict[str, dict[str, Any]]:
    """Fetch Gamma markets keyed by Gamma market id for a set of token ids."""

    semaphore = asyncio.Semaphore(max_concurrency)
    collected: dict[str, dict[str, Any]] = {}

    async def fetch_one(token_id: str) -> None:
        async with semaphore:
            try:
                batch = await gamma_client.list_markets(limit=1, clob_token_ids=[token_id])
            except httpx.HTTPStatusError as exc:
                LOGGER.warning("Market lookup failed for token %s: %s", token_id, exc)
                return
            for market in batch:
                collected[str(market["id"])] = market

    await asyncio.gather(*(fetch_one(token_id) for token_id in token_ids))
    return collected


async def backfill_markets_for_references(
    session: Session,
    *,
    condition_ids: Sequence[str] | None = None,
    token_ids: Sequence[str] | None = None,
    client: GammaClient | None = None,
) -> dict[str, int]:
    """Fetch and store market metadata for the supplied public references only."""

    normalized_condition_ids = sorted(
        {
            str(value)
            for value in (condition_ids or [])
            if isinstance(value, str) and value.startswith("0x")
        }
    )
    normalized_token_ids = sorted(
        {
            str(value)
            for value in (token_ids or [])
            if isinstance(value, str) and value.strip()
        }
    )
    if not normalized_condition_ids and not normalized_token_ids:
        return {"markets": 0, "tokens": 0}

    own_client = client is None
    gamma_client = client or GammaClient()
    collected_markets: dict[str, dict[str, Any]] = {}
    max_concurrency = get_settings().max_concurrency

    try:
        if normalized_condition_ids:
            collected_markets.update(
                await _fetch_markets_for_conditions(
                    gamma_client,
                    normalized_condition_ids,
                    max_concurrency=max_concurrency,
                )
            )

        covered_token_ids = {
            str(token_id)
            for market in collected_markets.values()
            for token_id in _parse_json_list(market.get("clobTokenIds"))
        }
        remaining_token_ids = [
            token_id for token_id in normalized_token_ids if token_id not in covered_token_ids
        ]
        if remaining_token_ids:
            collected_markets.update(
                await _fetch_markets_for_tokens(
                    gamma_client,
                    remaining_token_ids,
                    max_concurrency=max_concurrency,
                )
            )
    finally:
        if own_client:
            await gamma_client.aclose()

    market_list = list(collected_markets.values())
    mapping = _upsert_markets(session, market_list)
    tokens_written = _upsert_tokens(session, market_list)

    for condition_id, market_id in mapping.items():
        session.execute(
            update(WalletTradeRaw)
            .where(WalletTradeRaw.market_id == condition_id)
            .values(market_id=market_id)
        )

    session.commit()
    LOGGER.info(
        "Stored %s markets and %s tokens from targeted references",
        len(market_list),
        tokens_written,
    )
    return {"markets": len(market_list), "tokens": tokens_written}


async def backfill_markets(session: Session, client: GammaClient | None = None) -> dict[str, int]:
    """Fetch and store market metadata for markets referenced in raw trades."""

    condition_ids = sorted(
        {
            row[0]
            for row in session.execute(
                select(WalletTradeRaw.market_id).where(WalletTradeRaw.market_id.is_not(None))
            )
            if isinstance(row[0], str) and row[0].startswith("0x")
        }
    )
    token_ids = sorted(
        {
            row[0]
            for row in session.execute(
                select(WalletTradeRaw.token_id).where(WalletTradeRaw.token_id.is_not(None))
            )
            if isinstance(row[0], str)
        }
    )
    if not condition_ids and not token_ids:
        return {"markets": 0, "tokens": 0}
    return await backfill_markets_for_references(
        session,
        condition_ids=condition_ids,
        token_ids=token_ids,
        client=client,
    )
