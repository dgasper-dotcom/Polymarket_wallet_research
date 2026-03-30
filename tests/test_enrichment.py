"""Tests for signed return enrichment math."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from clients.clob_client import ClobClient
from db.base import Base
from db.models import Market, PriceHistory, WalletTradeEnriched, WalletTradeRaw
from research.enrich_trades import enrich_wallet_trades


def _session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True)()


def test_enrichment_signed_returns_for_buy_and_sell() -> None:
    """BUY should get positive sign and SELL should get negative sign."""

    session = _session()
    base_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

    session.add_all(
        [
            WalletTradeRaw(
                trade_id="buy-trade",
                wallet_address="0x1111111111111111111111111111111111111111",
                market_id="531202",
                token_id="token-1",
                side="BUY",
                price=0.40,
                size=10,
                usdc_size=4.0,
                timestamp=base_ts,
                tx_hash="0xabc",
                raw_json="{}",
            ),
            WalletTradeRaw(
                trade_id="sell-trade",
                wallet_address="0x2222222222222222222222222222222222222222",
                market_id="531202",
                token_id="token-2",
                side="SELL",
                price=0.40,
                size=10,
                usdc_size=4.0,
                timestamp=base_ts,
                tx_hash="0xdef",
                raw_json="{}",
            ),
            PriceHistory(token_id="token-1", ts=base_ts, price=0.40),
            PriceHistory(token_id="token-1", ts=base_ts.replace(minute=1), price=0.45),
            PriceHistory(token_id="token-2", ts=base_ts, price=0.40),
            PriceHistory(token_id="token-2", ts=base_ts.replace(minute=1), price=0.45),
        ]
    )
    session.commit()

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "no book"})

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with ClobClient(base_url="https://clob.polymarket.com", transport=transport) as client:
            await enrich_wallet_trades(session, client=client)

    asyncio.run(run())

    enriched = {
        row.trade_id: row
        for row in session.query(WalletTradeEnriched).all()
    }
    assert round(enriched["buy-trade"].ret_1m, 6) == 0.05
    assert round(enriched["sell-trade"].ret_1m, 6) == -0.05


def test_enrichment_keeps_rows_when_price_history_is_missing() -> None:
    """Missing price history should not crash enrichment or drop the trade row."""

    session = _session()
    base_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    session.add(
        WalletTradeRaw(
            trade_id="missing-price-trade",
            wallet_address="0x3333333333333333333333333333333333333333",
            market_id="531202",
            token_id="token-missing",
            side="BUY",
            price=0.55,
            size=20,
            usdc_size=11.0,
            timestamp=base_ts,
            tx_hash="0xghi",
            raw_json="{}",
        )
    )
    session.add(
        Market(
            id="531202",
            question="Example market",
            slug="example-market",
            condition_id="0xabc",
            active=True,
            closed=False,
            archived=False,
            enable_order_book=True,
            created_at=base_ts,
            updated_at=base_ts,
            raw_json="{}",
        )
    )
    session.commit()

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no book"})

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with ClobClient(base_url="https://clob.polymarket.com", transport=transport) as client:
            await enrich_wallet_trades(session, client=client)

    asyncio.run(run())

    row = session.query(WalletTradeEnriched).filter_by(trade_id="missing-price-trade").one()
    assert row.missing_price_history is True
    assert row.ret_1m is None
    assert row.mid_5m is None
    assert row.midpoint_source == "missing_prices"
    assert row.enrichment_status == "missing_prices"
