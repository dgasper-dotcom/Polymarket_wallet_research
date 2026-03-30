"""Sync client for public Polymarket Analytics research endpoints.

This client is intentionally read-only and research-focused. It targets the
same public JSON endpoints that power the site UI and does not place trades or
touch authenticated Polymarket execution APIs.
"""

from __future__ import annotations

import os
import time
from typing import Any

import certifi
import httpx

from config.settings import get_settings


DEFAULT_BASE_URL = "https://polymarketanalytics.com"
DEFAULT_TIMEOUT_SECONDS = 60.0
MIN_ACTIVITY_PAGE_SIZE = 100
MAX_RETRIES = 3


class PolymarketAnalyticsClient:
    """Small wrapper around the site's public JSON endpoints."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        bearer_token: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        token = bearer_token or os.getenv("POLYMARKET_ANALYTICS_TOKEN")
        headers = {
            "Accept": "application/json,text/plain,*/*",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        timeout_config = timeout or DEFAULT_TIMEOUT_SECONDS
        self._client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_config, connect=min(timeout_config, 20.0)),
            verify=certifi.where(),
            headers=headers,
            follow_redirects=True,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""

        self._client.close()

    def __enter__(self) -> "PolymarketAnalyticsClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """Execute one GET request and decode JSON."""

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.get(path, params=params)
                response.raise_for_status()
                return response.json()
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
                last_error = exc
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(0.5 * (attempt + 1))
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code not in {500, 502, 503, 504}:
                    raise
                last_error = exc
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(0.5 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected PMA GET failure with no captured exception.")

    def get_activity_trades(
        self,
        trader_id: str,
        *,
        limit: int = 1000,
        offset: int = 0,
        sort_by: str = "trade_dttm",
        sort_desc: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch one page of trader activity trades."""

        payload = self._get(
            "/api/activity-trades",
            params={
                "trader_id": trader_id,
                "sortBy": sort_by,
                "sortDesc": str(bool(sort_desc)).lower(),
                "limit": int(limit),
                "offset": int(offset),
            },
        )
        data = payload.get("data", []) if isinstance(payload, dict) else []
        return data if isinstance(data, list) else []

    def get_all_activity_trades(
        self,
        trader_id: str,
        *,
        page_size: int = 1000,
        max_pages: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch all paginated activity trades for one wallet."""

        all_rows: list[dict[str, Any]] = []
        current_page_size = max(int(page_size), MIN_ACTIVITY_PAGE_SIZE)
        page_count = 0
        while page_count < max_pages:
            offset = len(all_rows)
            try:
                rows = self.get_activity_trades(
                    trader_id,
                    limit=current_page_size,
                    offset=offset,
                )
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError):
                if current_page_size <= MIN_ACTIVITY_PAGE_SIZE:
                    break
                current_page_size = max(MIN_ACTIVITY_PAGE_SIZE, current_page_size // 2)
                continue

            page_count += 1
            if not rows:
                break
            all_rows.extend(rows)
            if len(rows) < current_page_size:
                break
        return all_rows

    def get_trader_dashboard(self, trader_id: str) -> dict[str, Any] | None:
        """Fetch the summary dashboard payload for one wallet."""

        payload = self._get("/api/traders-dashboard", params={"trader_id": trader_id})
        data = payload.get("data", []) if isinstance(payload, dict) else []
        if isinstance(data, list) and data:
            first = data[0]
            return first if isinstance(first, dict) else None
        return None

    def get_trader_positions(self, trader_id: str) -> list[dict[str, Any]]:
        """Fetch current positions for one wallet."""

        payload = self._get("/api/traders-positions", params={"trader_id": trader_id})
        data = payload.get("data", []) if isinstance(payload, dict) else []
        return data if isinstance(data, list) else []
