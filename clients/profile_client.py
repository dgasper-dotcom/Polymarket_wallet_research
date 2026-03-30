"""Async client for the public Polymarket Data API profile, trade, and position endpoints."""

from __future__ import annotations

import logging
from typing import Any

import certifi
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config.api_contracts import (
    PUBLIC_MARKET_TRADES,
    PUBLIC_USER_CLOSED_POSITIONS,
    PUBLIC_USER_POSITIONS,
    PUBLIC_USER_TRADES,
)
from config.settings import get_settings


LOGGER = logging.getLogger(__name__)


def _is_retryable_exception(exc: BaseException) -> bool:
    """Retry network errors and server-side HTTP failures only."""

    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or status_code >= 500
    return False


class ProfileClient:
    """Small async wrapper over Polymarket's public Data API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=base_url or settings.poly_data_base,
            timeout=timeout or settings.request_timeout,
            transport=transport,
            verify=certifi.where(),
            headers={
                "Accept": "application/json",
                "User-Agent": "polymarket-wallet-research/0.1",
            },
        )

    async def __aenter__(self) -> "ProfileClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP session."""

        await self._client.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_is_retryable_exception),
    )
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Perform a retried GET request and return decoded JSON."""

        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def get_user_trades(
        self, wallet: str, limit: int = 500, offset: int = 0
    ) -> list[dict]:
        """Fetch one page of public trades for a wallet.

        The query-parameter names are centralized in `config.api_contracts`.
        """

        payload = await self._get(
            PUBLIC_USER_TRADES.path,
            params={
                PUBLIC_USER_TRADES.params["wallet"]: wallet,
                PUBLIC_USER_TRADES.params["page_size"]: limit,
                PUBLIC_USER_TRADES.params["page_offset"]: offset,
            },
        )
        if not isinstance(payload, list):
            raise TypeError(f"Expected list payload for /trades, got {type(payload)!r}")
        return payload

    async def get_all_user_trades(
        self,
        wallet: str,
        page_size: int = 500,
        max_offset: int | None = 1000,
    ) -> list[dict]:
        """Fetch all discoverable public wallet trades by paging until exhaustion.

        Assumption:
        - Public wallet `/trades` pagination is subject to the same documented
          public offset ceiling as market `/trades`. The method therefore stops
          once the next page would exceed `max_offset`.
        """

        all_rows: list[dict] = []
        offset = 0
        limit = page_size
        previous_page_signature: tuple[str, ...] | None = None

        while True:
            page = await self.get_user_trades(wallet=wallet, limit=limit, offset=offset)
            if not page:
                break

            current_signature = tuple(
                str(item.get("transactionHash") or item.get("timestamp") or index)
                for index, item in enumerate(page)
            )
            if current_signature == previous_page_signature:
                LOGGER.warning("Duplicate page detected for wallet %s at offset %s", wallet, offset)
                break

            all_rows.extend(page)
            LOGGER.info(
                "Fetched %s trades for wallet %s at offset %s", len(page), wallet, offset
            )
            if len(page) < limit:
                break

            previous_page_signature = current_signature
            offset += limit
            if max_offset is not None and offset > max_offset:
                LOGGER.warning(
                    "Reached documented public /trades offset ceiling for wallet %s at offset %s",
                    wallet,
                    offset,
                )
                break

        return all_rows

    async def get_market_trades(
        self,
        market: str,
        limit: int = 500,
        offset: int = 0,
        *,
        taker_only: bool = True,
        side: str | None = None,
    ) -> list[dict]:
        """Fetch one page of public trades for a market/condition id."""

        params: dict[str, Any] = {
            PUBLIC_MARKET_TRADES.params["market"]: market,
            PUBLIC_MARKET_TRADES.params["page_size"]: limit,
            PUBLIC_MARKET_TRADES.params["page_offset"]: offset,
            PUBLIC_MARKET_TRADES.params["taker_only"]: str(taker_only).lower(),
        }
        if side:
            params[PUBLIC_MARKET_TRADES.params["side"]] = side.upper()

        payload = await self._get(PUBLIC_MARKET_TRADES.path, params=params)
        if not isinstance(payload, list):
            raise TypeError(f"Expected list payload for /trades, got {type(payload)!r}")
        return payload

    async def get_all_market_trades(
        self,
        market: str,
        *,
        page_size: int = 500,
        taker_only: bool = True,
        side: str | None = None,
        max_offset: int | None = 1000,
    ) -> list[dict]:
        """Fetch all discoverable public trades for one market.

        The public Polymarket docs currently document a hard `/trades` offset ceiling of 1000.
        The method pages until exhaustion or until the documented public offset range is reached.
        """

        all_rows: list[dict] = []
        offset = 0
        previous_page_signature: tuple[str, ...] | None = None

        while True:
            page = await self.get_market_trades(
                market=market,
                limit=page_size,
                offset=offset,
                taker_only=taker_only,
                side=side,
            )
            if not page:
                break

            current_signature = tuple(
                str(
                    item.get("id")
                    or item.get("transactionHash")
                    or item.get("timestamp")
                    or index
                )
                for index, item in enumerate(page)
            )
            if current_signature == previous_page_signature:
                LOGGER.warning("Duplicate market-trades page detected for %s at offset %s", market, offset)
                break

            all_rows.extend(page)
            if len(page) < page_size:
                break

            previous_page_signature = current_signature
            offset += page_size
            if max_offset is not None and offset > max_offset:
                LOGGER.warning(
                    "Reached documented /trades public offset ceiling for market %s at offset %s",
                    market,
                    offset,
                )
                break

        return all_rows

    async def get_user_positions(
        self,
        wallet: str,
        *,
        closed: bool = False,
        limit: int | None = None,
        offset: int = 0,
        market: str | None = None,
    ) -> list[dict]:
        """Fetch one page of current or closed public positions for a wallet."""

        endpoint = PUBLIC_USER_CLOSED_POSITIONS if closed else PUBLIC_USER_POSITIONS
        params: dict[str, Any] = {
            endpoint.params["wallet"]: wallet,
            endpoint.params["page_size"]: limit if limit is not None else (50 if closed else 500),
            endpoint.params["page_offset"]: offset,
        }
        if market:
            params[endpoint.params["market"]] = market

        payload = await self._get(endpoint.path, params=params)
        if not isinstance(payload, list):
            raise TypeError(f"Expected list payload for {endpoint.path}, got {type(payload)!r}")
        return payload

    async def get_all_user_positions(
        self,
        wallet: str,
        *,
        closed: bool = False,
        page_size: int | None = None,
        max_offset: int | None = None,
    ) -> list[dict]:
        """Fetch all public position pages for a wallet."""

        all_rows: list[dict] = []
        limit = page_size if page_size is not None else (50 if closed else 500)
        offset = 0
        previous_page_signature: tuple[str, ...] | None = None

        while True:
            page = await self.get_user_positions(
                wallet=wallet,
                closed=closed,
                limit=limit,
                offset=offset,
            )
            if not page:
                break

            current_signature = tuple(
                str(
                    item.get("id")
                    or item.get("positionId")
                    or item.get("asset")
                    or index
                )
                for index, item in enumerate(page)
            )
            if current_signature == previous_page_signature:
                LOGGER.warning(
                    "Duplicate %s-positions page detected for wallet %s at offset %s",
                    "closed" if closed else "open",
                    wallet,
                    offset,
                )
                break

            all_rows.extend(page)
            if len(page) < limit:
                break

            previous_page_signature = current_signature
            offset += limit
            if max_offset is not None and offset > max_offset:
                LOGGER.warning(
                    "Reached configured %s-positions offset ceiling for wallet %s at offset %s",
                    "closed" if closed else "open",
                    wallet,
                    offset,
                )
                break

        return all_rows
