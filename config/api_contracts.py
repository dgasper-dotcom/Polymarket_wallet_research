"""Centralized public Polymarket endpoint contracts used by this research project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointSpec:
    """One public endpoint contract plus the assumptions this MVP relies on."""

    name: str
    base_env_var: str
    path: str
    params: dict[str, str]
    note: str


PUBLIC_USER_TRADES = EndpointSpec(
    name="public_user_trades",
    base_env_var="POLY_DATA_BASE",
    path="/trades",
    params={
        "wallet": "user",
        "page_size": "limit",
        "page_offset": "offset",
    },
    note=(
        "Assumption: GET /trades is the correct public profile-trades endpoint, `user` expects "
        "the public proxy wallet address. Polymarket's August 26, 2025 changelog says `/trades` "
        "was tightened to max `limit=500` and max `offset=1000`, so the project defaults to "
        "500-sized pages and treats any higher-offset refusal as an API-side discoverability cap."
    ),
)

PUBLIC_MARKET_TRADES = EndpointSpec(
    name="public_market_trades",
    base_env_var="POLY_DATA_BASE",
    path="/trades",
    params={
        "market": "market",
        "page_size": "limit",
        "page_offset": "offset",
        "taker_only": "takerOnly",
        "side": "side",
    },
    note=(
        "Assumption: GET /trades with the `market` query param expects one or more condition IDs, "
        "not Gamma market ids. The public response exposes `proxyWallet`, which is treated as the "
        "discoverable public wallet field for market-scan cohorts."
    ),
)

PUBLIC_USER_POSITIONS = EndpointSpec(
    name="public_user_positions",
    base_env_var="POLY_DATA_BASE",
    path="/positions",
    params={
        "wallet": "user",
        "page_size": "limit",
        "page_offset": "offset",
        "market": "market",
    },
    note=(
        "Assumption: GET /positions is fully public and returns current positions including "
        "`realizedPnl`, `percentRealizedPnl`, and `totalBought` for the supplied public wallet."
    ),
)

PUBLIC_USER_CLOSED_POSITIONS = EndpointSpec(
    name="public_user_closed_positions",
    base_env_var="POLY_DATA_BASE",
    path="/closed-positions",
    params={
        "wallet": "user",
        "page_size": "limit",
        "page_offset": "offset",
        "market": "market",
    },
    note=(
        "Assumption: GET /closed-positions is fully public and returns closed positions including "
        "`realizedPnl` and `totalBought`. It does not expose `percentRealizedPnl`, so any wallet-"
        "level realized PnL percent still needs to be reconstructed or estimated."
    ),
)

PUBLIC_MARKETS_LIST = EndpointSpec(
    name="public_markets_list",
    base_env_var="POLY_GAMMA_BASE",
    path="/markets",
    params={
        "active": "active",
        "closed": "closed",
        "page_size": "limit",
        "page_offset": "offset",
        "condition_ids": "condition_ids",
        "clob_token_ids": "clob_token_ids",
    },
    note=(
        "Assumption: GET /markets supports filtering by `condition_ids` and `clob_token_ids`. "
        "Live testing showed single-value lookups are reliable while comma-separated multi-value "
        "lookups can return empty lists, so the ingestion flow intentionally falls back to "
        "one-by-one requests."
    ),
)

PUBLIC_MARKET_DETAIL = EndpointSpec(
    name="public_market_detail",
    base_env_var="POLY_GAMMA_BASE",
    path="/markets/{market_id}",
    params={},
    note=(
        "Assumption: GET /markets/{market_id} expects the Gamma market id, not the market "
        "condition id or token id."
    ),
)

PUBLIC_ORDER_BOOK = EndpointSpec(
    name="public_order_book",
    base_env_var="POLY_CLOB_BASE",
    path="/book",
    params={"token_id": "token_id"},
    note=(
        "Assumption: GET /book uses `token_id` for the CLOB asset id. Older, archived, or "
        "inactive tokens can legitimately return 404 even when the trade history still exists."
    ),
)

PUBLIC_PRICES_HISTORY = EndpointSpec(
    name="public_prices_history",
    base_env_var="POLY_CLOB_BASE",
    path="/prices-history",
    params={
        "token_id": "market",
        "start_ts": "startTs",
        "end_ts": "endTs",
        "fidelity": "fidelity",
    },
    note=(
        "Assumption: GET /prices-history currently expects the query key `market` even though the "
        "value we pass is the token id. Some older tokens return 400/404; the pipeline treats "
        "that as missing public history rather than a fatal error."
    ),
)

ALL_ENDPOINT_SPECS = [
    PUBLIC_USER_TRADES,
    PUBLIC_MARKET_TRADES,
    PUBLIC_USER_POSITIONS,
    PUBLIC_USER_CLOSED_POSITIONS,
    PUBLIC_MARKETS_LIST,
    PUBLIC_MARKET_DETAIL,
    PUBLIC_ORDER_BOOK,
    PUBLIC_PRICES_HISTORY,
]


def endpoint_audit_rows() -> list[dict[str, str]]:
    """Return endpoint contracts as export-friendly rows."""

    return [
        {
            "endpoint_name": spec.name,
            "base_env_var": spec.base_env_var,
            "path": spec.path,
            "params": ", ".join(f"{key}->{value}" for key, value in spec.params.items()),
            "assumption_note": spec.note,
        }
        for spec in ALL_ENDPOINT_SPECS
    ]
