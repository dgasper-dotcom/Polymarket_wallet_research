"""Async client for public Polymarket CLOB pricing endpoints."""

from __future__ import annotations

from typing import Any

import certifi
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config.api_contracts import PUBLIC_ORDER_BOOK, PUBLIC_PRICES_HISTORY
from config.settings import get_settings


def _is_retryable_exception(exc: BaseException) -> bool:
    """Retry network errors and server-side HTTP failures only."""

    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or status_code >= 500
    return False


class ClobClient:
    """Wrapper for public order book and price history endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        verify: Any | None = None,
    ) -> None:
        settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=base_url or settings.poly_clob_base,
            timeout=timeout or settings.request_timeout,
            transport=transport,
            verify=certifi.where() if verify is None else verify,
            headers={
                "Accept": "application/json",
                "User-Agent": "polymarket-wallet-research/0.1",
            },
        )

    async def __aenter__(self) -> "ClobClient":
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

    async def get_order_book(self, token_id: str) -> dict:
        """Fetch the current public order book snapshot for a token."""

        payload = await self._get(
            PUBLIC_ORDER_BOOK.path,
            params={PUBLIC_ORDER_BOOK.params["token_id"]: token_id},
        )
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict payload for /book, got {type(payload)!r}")
        return payload

    async def get_prices_history(
        self,
        token_id: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        fidelity: int | None = None,
    ) -> dict:
        """Fetch public token price history.

        The non-obvious query-parameter mapping is centralized in
        `config.api_contracts`.
        """

        params: dict[str, Any] = {PUBLIC_PRICES_HISTORY.params["token_id"]: token_id}
        if start_ts is not None:
            params[PUBLIC_PRICES_HISTORY.params["start_ts"]] = start_ts
        if end_ts is not None:
            params[PUBLIC_PRICES_HISTORY.params["end_ts"]] = end_ts
        if fidelity is not None:
            params[PUBLIC_PRICES_HISTORY.params["fidelity"]] = fidelity

        payload = await self._get(PUBLIC_PRICES_HISTORY.path, params=params)
        if not isinstance(payload, dict):
            raise TypeError(
                f"Expected dict payload for /prices-history, got {type(payload)!r}"
            )
        return payload
