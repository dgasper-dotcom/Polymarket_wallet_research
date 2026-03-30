"""Smoke tests for public API clients using mocked HTTP responses."""

from __future__ import annotations

import asyncio
import json

import httpx

from clients.clob_client import ClobClient
from clients.gamma_client import GammaClient
from clients.profile_client import ProfileClient


def test_profile_client_paginates_all_user_trades() -> None:
    """ProfileClient should collect all pages until the API is exhausted."""

    pages = {
        0: [{"timestamp": 1, "transactionHash": "0x1"}],
        1: [{"timestamp": 2, "transactionHash": "0x2"}],
        2: [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/trades"
        offset = int(request.url.params["offset"])
        return httpx.Response(200, json=pages[offset])

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with ProfileClient(base_url="https://data-api.polymarket.com", transport=transport) as client:
            trades = await client.get_all_user_trades(
                "0x56687bf447db6ffa42ffe2204a05edaa20f55839",
                page_size=1,
            )
        assert len(trades) == 2
        assert [trade["timestamp"] for trade in trades] == [1, 2]

    asyncio.run(run())


def test_gamma_and_clob_client_smoke() -> None:
    """GammaClient and ClobClient should decode list/dict payloads correctly."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "gamma-api.polymarket.com" and request.url.path == "/markets":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "531202",
                        "conditionId": "0xabc",
                        "question": "BitBoy convicted?",
                        "clobTokenIds": json.dumps(["1", "2"]),
                        "outcomes": json.dumps(["Yes", "No"]),
                    }
                ],
            )
        if request.url.host == "gamma-api.polymarket.com" and request.url.path == "/markets/531202":
            return httpx.Response(200, json={"id": "531202", "question": "BitBoy convicted?"})
        if request.url.host == "clob.polymarket.com" and request.url.path == "/book":
            return httpx.Response(
                200,
                json={
                    "asset_id": "1",
                    "bids": [{"price": "0.45", "size": "10"}],
                    "asks": [{"price": "0.47", "size": "11"}],
                },
            )
        if request.url.host == "clob.polymarket.com" and request.url.path == "/prices-history":
            assert request.url.params["market"] == "1"
            return httpx.Response(200, json={"history": [{"t": 100, "p": 0.46}]})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with GammaClient(base_url="https://gamma-api.polymarket.com", transport=transport) as gamma:
            markets = await gamma.list_markets(limit=1)
            market = await gamma.get_market("531202")
        async with ClobClient(base_url="https://clob.polymarket.com", transport=transport) as clob:
            book = await clob.get_order_book("1")
            history = await clob.get_prices_history("1", start_ts=0, end_ts=100, fidelity=1)

        assert markets[0]["id"] == "531202"
        assert market["question"] == "BitBoy convicted?"
        assert book["asset_id"] == "1"
        assert history["history"][0]["p"] == 0.46

    asyncio.run(run())
