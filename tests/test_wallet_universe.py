"""Tests for market-scan wallet universe discovery and cohort filtering."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from clients.gamma_client import GammaClient
from clients.profile_client import ProfileClient
from db.base import Base
from db.models import MarketTradeRaw, MarketTradeScanProgress
from research.wallet_universe import (
    build_master_wallet_universe,
    fetch_wallet_pnl_estimates,
    filter_strong_wallets,
    filter_weak_wallets,
    scan_market_trade_wallets,
)


def _session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True)()


def test_scan_market_trade_wallets_stores_public_market_trades() -> None:
    """Market scanning should persist raw trades and progress checkpoints."""

    session = _session()
    condition_id = "0x" + ("ab" * 32)
    wallet = "0x56687bf447db6ffa42ffe2204a05edaa20f55839"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gamma-api.polymarket.com" and request.url.path == "/markets":
            offset = int(request.url.params["offset"])
            if offset == 0:
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": "531202",
                            "conditionId": condition_id,
                            "question": "Example market?",
                        }
                    ],
                )
            return httpx.Response(200, json=[])

        if request.url.host == "data-api.polymarket.com" and request.url.path == "/trades":
            offset = int(request.url.params["offset"])
            if offset == 0:
                return httpx.Response(
                    200,
                    json=[
                        {
                            "proxyWallet": wallet,
                            "side": "BUY",
                            "asset": "token-1",
                            "conditionId": condition_id,
                            "size": 10,
                            "price": 0.55,
                            "timestamp": 1774100000,
                            "transactionHash": "0xtrade1",
                        }
                    ],
                )
            return httpx.Response(200, json=[])

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with GammaClient(base_url="https://gamma-api.polymarket.com", transport=transport) as gamma:
            async with ProfileClient(base_url="https://data-api.polymarket.com", transport=transport) as profile:
                summary = await scan_market_trade_wallets(
                    session,
                    gamma_client=gamma,
                    profile_client=profile,
                    market_page_size=1,
                    trade_page_size=1,
                )
        assert summary["markets_scanned"] == 1
        assert summary["trades_scanned"] == 1

    asyncio.run(run())

    assert session.scalar(select(func.count()).select_from(MarketTradeRaw)) == 1
    stored_trade = session.scalars(select(MarketTradeRaw)).one()
    assert stored_trade.wallet_address == wallet
    progress = session.get(MarketTradeScanProgress, condition_id)
    assert progress is not None
    assert progress.completed is True
    assert progress.trades_scanned == 1


def test_fetch_wallet_pnl_estimates_aggregates_public_position_rows() -> None:
    """Wallet realized-PnL estimates should combine public current and closed positions."""

    wallet = "0x56687bf447db6ffa42ffe2204a05edaa20f55839"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/positions":
            offset = int(request.url.params["offset"])
            if offset == 0:
                return httpx.Response(
                    200,
                    json=[
                        {
                            "proxyWallet": wallet,
                            "asset": "token-1",
                            "conditionId": "0x" + ("cd" * 32),
                            "realizedPnl": 5,
                            "percentRealizedPnl": 5,
                            "totalBought": 100,
                        }
                    ],
                )
            return httpx.Response(200, json=[])
        if request.url.path == "/closed-positions":
            offset = int(request.url.params["offset"])
            if offset == 0:
                return httpx.Response(
                    200,
                    json=[
                        {
                            "proxyWallet": wallet,
                            "asset": "token-2",
                            "conditionId": "0x" + ("ef" * 32),
                            "realizedPnl": 15,
                            "totalBought": 50,
                        }
                    ],
                )
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with ProfileClient(base_url="https://data-api.polymarket.com", transport=transport) as client:
            estimates = await fetch_wallet_pnl_estimates(
                [wallet],
                profile_client=client,
                max_concurrency=1,
                positions_page_size=500,
                closed_positions_page_size=50,
            )
        row = estimates.iloc[0]
        assert row["wallet_address"] == wallet
        assert round(float(row["realized_pnl_absolute"]), 6) == 20.0
        assert round(float(row["realized_pnl_percent"]), 6) == round((20 / 150) * 100, 6)
        assert bool(row["pnl_is_estimate"]) is True

    asyncio.run(run())


def test_wallet_universe_filters_use_strict_thresholds() -> None:
    """Strong/weak wallet cohorts should respect strict inequality and activity rules."""

    master = build_master_wallet_universe(_session(), now=datetime(2026, 3, 26, tzinfo=timezone.utc))
    assert master.empty

    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "wallet_address": "0x1111111111111111111111111111111111111111",
                "total_trades": 50,
                "distinct_markets": 10,
                "first_seen_trade_ts": "2026-01-01T00:00:00+00:00",
                "most_recent_trade_ts": "2026-03-10T00:00:00+00:00",
                "has_trade_in_2026_03": True,
                "has_trade_in_last_7d": False,
                "realized_pnl_percent": 10.0,
            },
            {
                "wallet_address": "0x2222222222222222222222222222222222222222",
                "total_trades": 50,
                "distinct_markets": 10,
                "first_seen_trade_ts": "2026-01-01T00:00:00+00:00",
                "most_recent_trade_ts": "2026-03-20T00:00:00+00:00",
                "has_trade_in_2026_03": True,
                "has_trade_in_last_7d": True,
                "realized_pnl_percent": 10.1,
            },
            {
                "wallet_address": "0x3333333333333333333333333333333333333333",
                "total_trades": 100,
                "distinct_markets": 12,
                "first_seen_trade_ts": "2026-01-01T00:00:00+00:00",
                "most_recent_trade_ts": "2026-03-24T00:00:00+00:00",
                "has_trade_in_2026_03": True,
                "has_trade_in_last_7d": True,
                "realized_pnl_percent": -33.0,
            },
            {
                "wallet_address": "0x4444444444444444444444444444444444444444",
                "total_trades": 100,
                "distinct_markets": 12,
                "first_seen_trade_ts": "2026-01-01T00:00:00+00:00",
                "most_recent_trade_ts": "2026-03-24T00:00:00+00:00",
                "has_trade_in_2026_03": True,
                "has_trade_in_last_7d": True,
                "realized_pnl_percent": -33.1,
            },
        ]
    )

    strong = filter_strong_wallets(frame)
    weak = filter_weak_wallets(frame)

    assert strong["wallet_address"].tolist() == ["0x2222222222222222222222222222222222222222"]
    assert weak["wallet_address"].tolist() == ["0x4444444444444444444444444444444444444444"]
