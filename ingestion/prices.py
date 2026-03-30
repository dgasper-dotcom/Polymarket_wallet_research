"""Incremental price history backfill for traded Polymarket outcome tokens."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from clients.clob_client import ClobClient
from config.settings import get_settings
from db.models import PriceHistory, WalletTradeRaw


LOGGER = logging.getLogger(__name__)
PRICE_HISTORY_UPSERT_CHUNK_SIZE = 250
PRICE_REQUEST_BATCH_SIZE = 200


def _to_utc_datetime(epoch_value: Any) -> datetime:
    """Convert price history timestamps into UTC datetimes."""

    if isinstance(epoch_value, (int, float)):
        epoch = float(epoch_value)
        if epoch > 10_000_000_000:
            epoch /= 1000.0
        return datetime.fromtimestamp(epoch, tz=timezone.utc)
    raise ValueError(f"Unsupported price history timestamp: {epoch_value!r}")


def _ensure_utc_datetime(value: datetime | None) -> datetime | None:
    """Normalize optional datetimes to UTC-aware values."""

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _history_rows(token_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the price history payload into sqlite rows."""

    points = payload.get("history", payload)
    if not isinstance(points, list):
        return []
    return [
        {
            "token_id": token_id,
            "ts": _to_utc_datetime(point["t"]),
            "price": float(point["p"]),
        }
        for point in points
        if "t" in point and "p" in point
    ]


def _upsert_price_rows(session: Session, rows: list[dict[str, Any]]) -> None:
    """Idempotently insert price history points."""

    if not rows:
        return

    for start in range(0, len(rows), PRICE_HISTORY_UPSERT_CHUNK_SIZE):
        chunk = rows[start : start + PRICE_HISTORY_UPSERT_CHUNK_SIZE]
        stmt = insert(PriceHistory).values(chunk)
        excluded = stmt.excluded
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[PriceHistory.token_id, PriceHistory.ts],
                set_={"price": excluded.price},
            )
        )
    session.flush()


async def backfill_price_history(
    session: Session,
    client: ClobClient | None = None,
    fidelity: int = 1,
    margin_seconds: int = 3600,
) -> dict[str, int]:
    """Incrementally backfill price history for all tokens seen in raw trades."""

    token_bounds = list(
        session.execute(
            select(
                WalletTradeRaw.token_id,
                func.min(WalletTradeRaw.timestamp),
                func.max(WalletTradeRaw.timestamp),
            )
            .where(WalletTradeRaw.token_id.is_not(None))
            .group_by(WalletTradeRaw.token_id)
        )
    )
    return await backfill_price_history_for_token_bounds(
        session,
        token_bounds=[
            (str(token_id), min_ts, max_ts)
            for token_id, min_ts, max_ts in token_bounds
            if token_id is not None and min_ts is not None and max_ts is not None
        ],
        client=client,
        fidelity=fidelity,
        margin_seconds=margin_seconds,
    )


async def backfill_price_history_for_token_bounds(
    session: Session,
    *,
    token_bounds: list[tuple[str, datetime, datetime]],
    client: ClobClient | None = None,
    fidelity: int = 1,
    margin_seconds: int = 3600,
) -> dict[str, int]:
    """Incrementally backfill price history for a targeted token/date subset."""

    if not token_bounds:
        return {}

    own_client = client is None
    clob_client = client or ClobClient()
    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.max_concurrency)

    request_specs: list[tuple[str, int, int]] = []

    async def _fetch_range(token_id: str, start_ts: int, end_ts: int) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            try:
                payload = await clob_client.get_prices_history(
                    token_id=token_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    fidelity=fidelity,
                )
                return token_id, payload
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {400, 403, 404}:
                    LOGGER.warning(
                        "Skipping price history for token %s because the public endpoint returned %s",
                        token_id,
                        exc.response.status_code,
                    )
                    return token_id, {"history": []}
                raise

    for token_id, min_ts, max_ts in token_bounds:
        desired_start = _ensure_utc_datetime(min_ts) - timedelta(seconds=margin_seconds)
        desired_end = _ensure_utc_datetime(max_ts) + timedelta(seconds=margin_seconds)
        existing = session.execute(
            select(func.min(PriceHistory.ts), func.max(PriceHistory.ts)).where(
                PriceHistory.token_id == token_id
            )
        ).one()
        existing_min, existing_max = existing
        existing_min = _ensure_utc_datetime(existing_min)
        existing_max = _ensure_utc_datetime(existing_max)

        ranges: list[tuple[int, int]] = []
        if existing_min is None or existing_max is None:
            ranges.append(
                (
                    int(desired_start.timestamp()),
                    int(desired_end.timestamp()),
                )
            )
        else:
            overlap = timedelta(minutes=5)
            if desired_start < existing_min:
                ranges.append(
                    (
                        int(desired_start.timestamp()),
                        int((existing_min + overlap).timestamp()),
                    )
                )
            if desired_end > existing_max:
                ranges.append(
                    (
                        int((existing_max - overlap).timestamp()),
                        int(desired_end.timestamp()),
                    )
                )

        for start_ts, end_ts in ranges:
            request_specs.append((token_id, start_ts, end_ts))

    results: dict[str, int] = {}
    try:
        for batch_start in range(0, len(request_specs), PRICE_REQUEST_BATCH_SIZE):
            batch = request_specs[batch_start : batch_start + PRICE_REQUEST_BATCH_SIZE]
            tasks = [
                asyncio.create_task(_fetch_range(token_id, start_ts, end_ts))
                for token_id, start_ts, end_ts in batch
            ]
            for token_id, payload in await asyncio.gather(*tasks):
                rows = _history_rows(token_id, payload)
                _upsert_price_rows(session, rows)
                results[token_id] = results.get(token_id, 0) + len(rows)
            session.commit()
            LOGGER.info(
                "Price history progress: processed %s/%s targeted token ranges",
                min(batch_start + len(batch), len(request_specs)),
                len(request_specs),
            )
    finally:
        if own_client:
            await clob_client.aclose()

    return results
