"""Backfill raw wallet trade history from the public Data API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from clients.profile_client import ProfileClient
from config.settings import get_settings
from db.models import WalletTradeRaw


LOGGER = logging.getLogger(__name__)


def _coerce_float(value: Any) -> float | None:
    """Safely coerce numeric values from API payloads."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_utc_datetime(value: Any) -> datetime:
    """Normalize timestamps from epoch seconds, milliseconds, or ISO-8601 strings."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            value = int(stripped)
        else:
            parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        epoch = float(value)
        if epoch > 10_000_000_000:
            epoch /= 1000.0
        return datetime.fromtimestamp(epoch, tz=timezone.utc)

    raise ValueError(f"Unsupported timestamp value: {value!r}")


def derive_trade_id(wallet: str, trade: dict[str, Any]) -> str:
    """Derive a deterministic trade ID when the public payload does not expose one."""

    explicit_id = trade.get("trade_id") or trade.get("tradeId") or trade.get("id")
    if explicit_id:
        return str(explicit_id)

    signature = "|".join(
        [
            wallet.lower(),
            str(trade.get("transactionHash") or trade.get("txHash") or ""),
            str(trade.get("asset") or trade.get("token_id") or ""),
            str(trade.get("conditionId") or trade.get("market") or ""),
            str(trade.get("side") or ""),
            str(trade.get("price") or ""),
            str(trade.get("size") or ""),
            str(trade.get("timestamp") or ""),
        ]
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def normalize_trade_record(wallet: str, trade: dict[str, Any]) -> dict[str, Any]:
    """Map the public trade payload into the raw trades table schema."""

    normalized_wallet = str(trade.get("proxyWallet") or wallet).lower()
    price = _coerce_float(trade.get("price"))
    size = _coerce_float(trade.get("size"))
    usdc_size = _coerce_float(trade.get("usdc_size"))
    if usdc_size is None and price is not None and size is not None:
        usdc_size = price * size

    return {
        "trade_id": derive_trade_id(normalized_wallet, trade),
        "wallet_address": normalized_wallet,
        # Assumption: the public profile trades endpoint exposes conditionId but not the
        # Gamma market ID, so we store conditionId here first and later normalize it.
        "market_id": trade.get("market") or trade.get("conditionId"),
        "token_id": trade.get("asset") or trade.get("token_id"),
        "side": str(trade.get("side")).upper() if trade.get("side") else None,
        "price": price,
        "size": size,
        "usdc_size": usdc_size,
        "timestamp": _to_utc_datetime(trade.get("timestamp")),
        "tx_hash": trade.get("transactionHash") or trade.get("txHash"),
        "raw_json": json.dumps(trade, sort_keys=True, ensure_ascii=False),
    }


def _upsert_raw_trades(session: Session, rows: list[dict[str, Any]]) -> None:
    """Idempotently write raw trades into sqlite."""

    if not rows:
        return

    stmt = insert(WalletTradeRaw).values(rows)
    excluded = stmt.excluded
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[WalletTradeRaw.trade_id],
            set_={
                "wallet_address": excluded.wallet_address,
                "market_id": excluded.market_id,
                "token_id": excluded.token_id,
                "side": excluded.side,
                "price": excluded.price,
                "size": excluded.size,
                "usdc_size": excluded.usdc_size,
                "timestamp": excluded.timestamp,
                "tx_hash": excluded.tx_hash,
                "raw_json": excluded.raw_json,
            },
        )
    )
    session.flush()


async def backfill_wallet_trades(
    session: Session,
    wallets: Sequence[str],
    client: ProfileClient | None = None,
) -> dict[str, int]:
    """Backfill all public trades for a list of wallets."""

    fetched = await fetch_wallet_trade_payloads(wallets=wallets, client=client)
    summary: dict[str, int] = {}
    for wallet, trades in fetched.items():
        rows = [normalize_trade_record(wallet, trade) for trade in trades]
        _upsert_raw_trades(session, rows)
        session.commit()
        summary[wallet] = len(rows)
        LOGGER.info("Stored %s raw trades for wallet %s", len(rows), wallet)

    return summary


async def fetch_wallet_trade_payloads(
    wallets: Sequence[str],
    client: ProfileClient | None = None,
) -> dict[str, list[dict]]:
    """Fetch all public trades for a list of wallets without writing to the database."""

    if not wallets:
        return {}

    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.max_concurrency)
    own_client = client is None
    profile_client = client or ProfileClient()

    async def _fetch(wallet: str) -> tuple[str, list[dict]]:
        async with semaphore:
            # Future realtime research extension:
            # replace or augment this batch fetch with wallet activity polling so newly
            # observed public trades can be appended incrementally without a full backfill.
            trades = await profile_client.get_all_user_trades(wallet)
            return wallet, trades

    try:
        fetched = await asyncio.gather(*[_fetch(wallet) for wallet in wallets])
    finally:
        if own_client:
            await profile_client.aclose()

    return {wallet: trades for wallet, trades in fetched}


def build_backfill_preview(
    trade_payloads: dict[str, list[dict]],
) -> dict[str, list[dict[str, Any]]]:
    """Build dry-run wallet, market, and token target summaries."""

    wallet_rows: list[dict[str, Any]] = []
    market_targets: dict[str, dict[str, Any]] = {}
    token_targets: dict[str, dict[str, Any]] = {}

    for wallet, trades in trade_payloads.items():
        normalized_rows = [normalize_trade_record(wallet, trade) for trade in trades]
        latest_trade = max((row["timestamp"] for row in normalized_rows), default=None)
        wallet_rows.append(
            {
                "wallet_address": wallet,
                "n_trades": len(normalized_rows),
                "n_markets": len({row["market_id"] for row in normalized_rows if row["market_id"]}),
                "most_recent_trade": latest_trade.isoformat() if latest_trade else None,
            }
        )

        for row in normalized_rows:
            market_id = row["market_id"]
            token_id = row["token_id"]
            if market_id:
                target = market_targets.setdefault(
                    str(market_id),
                    {
                        "market_id": str(market_id),
                        "n_wallets": 0,
                        "n_trades": 0,
                        "wallet_addresses": set(),
                    },
                )
                target["n_trades"] += 1
                target["wallet_addresses"].add(wallet)
                target["n_wallets"] = len(target["wallet_addresses"])
            if token_id:
                target = token_targets.setdefault(
                    str(token_id),
                    {
                        "token_id": str(token_id),
                        "n_wallets": 0,
                        "n_trades": 0,
                        "wallet_addresses": set(),
                    },
                )
                target["n_trades"] += 1
                target["wallet_addresses"].add(wallet)
                target["n_wallets"] = len(target["wallet_addresses"])

    market_rows = [
        {
            "market_id": item["market_id"],
            "n_wallets": item["n_wallets"],
            "n_trades": item["n_trades"],
            "wallet_addresses": ",".join(sorted(item["wallet_addresses"])),
        }
        for item in market_targets.values()
    ]
    token_rows = [
        {
            "token_id": item["token_id"],
            "n_wallets": item["n_wallets"],
            "n_trades": item["n_trades"],
            "wallet_addresses": ",".join(sorted(item["wallet_addresses"])),
        }
        for item in token_targets.values()
    ]

    wallet_rows.sort(key=lambda row: row["wallet_address"])
    market_rows.sort(key=lambda row: (-row["n_trades"], row["market_id"]))
    token_rows.sort(key=lambda row: (-row["n_trades"], row["token_id"]))
    return {
        "wallets": wallet_rows,
        "market_targets": market_rows,
        "token_targets": token_rows,
    }
