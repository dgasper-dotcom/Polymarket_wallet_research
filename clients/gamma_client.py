"""Async client for public Polymarket Gamma market metadata endpoints."""

from __future__ import annotations

from typing import Any, Sequence

import certifi
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config.api_contracts import PUBLIC_MARKET_DETAIL, PUBLIC_MARKETS_LIST
from config.settings import get_settings


def _is_retryable_exception(exc: BaseException) -> bool:
    """Retry network errors and server-side HTTP failures only."""

    if isinstance(exc, httpx.RequestError):
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            return False
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or status_code >= 500
    return False


class GammaClient:
    """Wrapper for market discovery and metadata retrieval."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=base_url or settings.poly_gamma_base,
            timeout=timeout or settings.request_timeout,
            transport=transport,
            verify=certifi.where(),
            headers={
                "Accept": "application/json",
                "User-Agent": "polymarket-wallet-research/0.1",
            },
        )

    async def __aenter__(self) -> "GammaClient":
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
        """Perform a retried GET request."""

        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def list_markets(
        self,
        active: bool | None = None,
        closed: bool | None = None,
        limit: int = 500,
        offset: int = 0,
        condition_ids: Sequence[str] | None = None,
        clob_token_ids: Sequence[str] | None = None,
    ) -> list[dict]:
        """Fetch a market page from the public Gamma API.

        Non-obvious filter and pagination parameter names are centralized in
        `config.api_contracts`.
        """

        params: dict[str, Any] = {
            PUBLIC_MARKETS_LIST.params["page_size"]: limit,
            PUBLIC_MARKETS_LIST.params["page_offset"]: offset,
        }
        if active is not None:
            params[PUBLIC_MARKETS_LIST.params["active"]] = str(active).lower()
        if closed is not None:
            params[PUBLIC_MARKETS_LIST.params["closed"]] = str(closed).lower()
        if condition_ids:
            params[PUBLIC_MARKETS_LIST.params["condition_ids"]] = ",".join(condition_ids)
        if clob_token_ids:
            params[PUBLIC_MARKETS_LIST.params["clob_token_ids"]] = ",".join(clob_token_ids)

        payload = await self._get(PUBLIC_MARKETS_LIST.path, params=params)
        if not isinstance(payload, list):
            raise TypeError(f"Expected list payload for /markets, got {type(payload)!r}")
        return payload

    async def get_market(self, market_id: str) -> dict:
        """Fetch one market by its Gamma market ID."""

        payload = await self._get(PUBLIC_MARKET_DETAIL.path.format(market_id=market_id))
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload for /markets/{{id}}, got {type(payload)!r}")
        return payload
