"""Scan current public Polymarket markets and aggregate discovered wallet activity."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)
import pandas as pd

from clients.gamma_client import GammaClient
from clients.profile_client import ProfileClient
from config.settings import Settings, get_settings
from ingestion.wallets import is_valid_wallet_address


MARKET_PAGE_SIZE_DEFAULT = 500
TRADE_PAGE_SIZE_DEFAULT = 500
TRADES_PUBLIC_MAX_OFFSET = 1000


@dataclass
class WalletAggregate:
    """One aggregated wallet record built from public market-trade scans."""

    wallet_address: str
    total_trades: int
    distinct_markets: set[str]
    first_seen_trade_ts: datetime | None
    most_recent_trade_ts: datetime | None
    has_trade_in_recent_window: bool
    sample_name: str | None
    sample_pseudonym: str | None
    sample_market: str | None


def _to_utc_datetime(value: Any) -> datetime | None:
    """Convert timestamps from Polymarket public payloads into UTC datetimes."""

    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _coerce_float(value: Any) -> float | None:
    """Safely coerce a numeric API field."""

    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_txt(lines: list[str], path: Path) -> Path:
    """Write newline-delimited text output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _wallet_name_from_trade(trade: dict[str, Any]) -> str | None:
    """Choose the best public display label exposed on a trade row."""

    for key in ("name", "pseudonym"):
        value = str(trade.get(key) or "").strip()
        if value:
            return value
    return None


def _trade_signature(trade: dict[str, Any], wallet_address: str) -> str:
    """Build a stable dedupe signature for one public wallet-trade observation."""

    payload = {
        "wallet_address": wallet_address,
        "conditionId": trade.get("conditionId"),
        "asset": trade.get("asset"),
        "side": trade.get("side"),
        "price": trade.get("price"),
        "size": trade.get("size"),
        "timestamp": trade.get("timestamp"),
        "transactionHash": trade.get("transactionHash"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _update_wallet_aggregate(
    wallet_map: dict[str, WalletAggregate],
    *,
    trade: dict[str, Any],
    wallet_address: str,
    market_key: str,
    recent_window_start: datetime,
    recent_window_end: datetime,
) -> None:
    """Apply one discovered trade row to the wallet-level aggregate map."""

    trade_ts = _to_utc_datetime(trade.get("timestamp"))
    aggregate = wallet_map.get(wallet_address)
    if aggregate is None:
        aggregate = WalletAggregate(
            wallet_address=wallet_address,
            total_trades=0,
            distinct_markets=set(),
            first_seen_trade_ts=trade_ts,
            most_recent_trade_ts=trade_ts,
            has_trade_in_recent_window=False,
            sample_name=None,
            sample_pseudonym=None,
            sample_market=None,
        )
        wallet_map[wallet_address] = aggregate

    aggregate.total_trades += 1
    aggregate.distinct_markets.add(market_key)
    if trade_ts is not None:
        if aggregate.first_seen_trade_ts is None or trade_ts < aggregate.first_seen_trade_ts:
            aggregate.first_seen_trade_ts = trade_ts
        if aggregate.most_recent_trade_ts is None or trade_ts > aggregate.most_recent_trade_ts:
            aggregate.most_recent_trade_ts = trade_ts
        if recent_window_start <= trade_ts <= recent_window_end:
            aggregate.has_trade_in_recent_window = True

    if aggregate.sample_name is None:
        aggregate.sample_name = _wallet_name_from_trade(trade)
    if aggregate.sample_pseudonym is None:
        pseudonym = str(trade.get("pseudonym") or "").strip()
        aggregate.sample_pseudonym = pseudonym or None
    if aggregate.sample_market is None:
        title = str(trade.get("title") or "").strip()
        aggregate.sample_market = title or None


async def fetch_current_markets(
    *,
    gamma_client: GammaClient | None = None,
    max_markets: int = 1000,
    market_page_size: int = MARKET_PAGE_SIZE_DEFAULT,
) -> list[dict[str, Any]]:
    """Fetch the first N current open markets from public Gamma metadata.

    Assumption:
    - `closed=false` is the most reliable current-market filter. Live testing on
      2026-03-26 showed `active=true` alone still returned old closed markets.
    """

    own_client = gamma_client is None
    gamma = gamma_client or GammaClient()
    markets: list[dict[str, Any]] = []
    offset = 0

    try:
        while len(markets) < max_markets:
            page = await gamma.list_markets(closed=False, limit=market_page_size, offset=offset)
            if not page:
                break
            for market in page:
                if not market.get("conditionId"):
                    continue
                volume = _coerce_float(market.get("volume")) or 0.0
                liquidity = _coerce_float(market.get("liquidity")) or 0.0
                if volume <= 0.0 and liquidity <= 0.0:
                    continue
                markets.append(market)
                if len(markets) >= max_markets:
                    break
            if len(page) < market_page_size:
                break
            offset += market_page_size
    finally:
        if own_client:
            await gamma.aclose()

    return markets[:max_markets]


async def scan_current_market_wallets(
    *,
    max_markets: int = 1000,
    market_page_size: int = MARKET_PAGE_SIZE_DEFAULT,
    trade_page_size: int = TRADE_PAGE_SIZE_DEFAULT,
    taker_only: bool = False,
    recent_window_start: datetime,
    recent_window_end: datetime,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Scan current public markets and aggregate discovered wallet activity."""

    cfg = settings or get_settings()
    markets = await fetch_current_markets(
        max_markets=max_markets,
        market_page_size=market_page_size,
    )

    wallet_map: dict[str, WalletAggregate] = {}
    scanned_market_rows: list[dict[str, Any]] = []
    seen_trade_signatures: set[str] = set()
    total_trades_scanned = 0
    truncated_markets = 0
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    async with ProfileClient() as profile:
        async def scan_one_market(market: dict[str, Any]) -> dict[str, Any]:
            nonlocal total_trades_scanned, truncated_markets
            condition_id = str(market["conditionId"])
            market_id = str(market.get("id") or "")
            question = str(market.get("question") or "")
            pages_scanned = 0
            market_trade_rows = 0
            truncated = False

            async with semaphore:
                offset = 0
                while True:
                    page = await profile.get_market_trades(
                        market=condition_id,
                        limit=trade_page_size,
                        offset=offset,
                        taker_only=taker_only,
                    )
                    pages_scanned += 1
                    if not page:
                        break

                    total_trades_scanned += len(page)
                    market_trade_rows += len(page)
                    for trade in page:
                        wallet_address = str(
                            trade.get("proxyWallet")
                            or trade.get("makerProxyWallet")
                            or trade.get("takerProxyWallet")
                            or trade.get("user")
                            or ""
                        ).lower()
                        if not is_valid_wallet_address(wallet_address):
                            continue
                        signature = _trade_signature(trade, wallet_address)
                        if signature in seen_trade_signatures:
                            continue
                        seen_trade_signatures.add(signature)
                        _update_wallet_aggregate(
                            wallet_map,
                            trade=trade,
                            wallet_address=wallet_address,
                            market_key=condition_id or market_id,
                            recent_window_start=recent_window_start,
                            recent_window_end=recent_window_end,
                        )

                    if len(page) < trade_page_size:
                        break

                    next_offset = offset + trade_page_size
                    if next_offset > TRADES_PUBLIC_MAX_OFFSET:
                        truncated = True
                        truncated_markets += 1
                        break
                    offset = next_offset

            return {
                "gamma_market_id": market_id,
                "condition_id": condition_id,
                "question": question,
                "volume": _coerce_float(market.get("volume")),
                "liquidity": _coerce_float(market.get("liquidity")),
                "updated_at": market.get("updatedAt"),
                "pages_scanned": pages_scanned,
                "trade_rows_scanned": market_trade_rows,
                "truncated_by_public_offset_limit": truncated,
            }

        scanned_market_rows = await asyncio.gather(*(scan_one_market(market) for market in markets))

    wallet_rows: list[dict[str, Any]] = []
    for aggregate in wallet_map.values():
        wallet_rows.append(
            {
                "wallet_address": aggregate.wallet_address,
                "total_trades": aggregate.total_trades,
                "distinct_markets": len(aggregate.distinct_markets),
                "first_seen_trade_ts": (
                    aggregate.first_seen_trade_ts.isoformat() if aggregate.first_seen_trade_ts else None
                ),
                "most_recent_trade_ts": (
                    aggregate.most_recent_trade_ts.isoformat() if aggregate.most_recent_trade_ts else None
                ),
                "has_trade_in_2026_03_12_to_2026_03_26": aggregate.has_trade_in_recent_window,
                "sample_name": aggregate.sample_name,
                "sample_pseudonym": aggregate.sample_pseudonym,
                "sample_market": aggregate.sample_market,
            }
        )

    wallets = pd.DataFrame(wallet_rows).sort_values(
        ["total_trades", "distinct_markets", "most_recent_trade_ts", "wallet_address"],
        ascending=[False, False, False, True],
    )
    scanned_markets = pd.DataFrame(scanned_market_rows).sort_values(
        ["trade_rows_scanned", "volume", "liquidity", "condition_id"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    eligible = wallets.loc[
        (wallets["total_trades"] >= 100)
        & (wallets["has_trade_in_2026_03_12_to_2026_03_26"] == True)
    ].copy()

    return {
        "wallet_universe": wallets.reset_index(drop=True),
        "eligible_wallets": eligible.reset_index(drop=True),
        "scanned_markets": scanned_markets.reset_index(drop=True),
        "summary": {
            "markets_scanned": len(markets),
            "trade_rows_scanned": int(total_trades_scanned),
            "unique_wallets_discovered": int(len(wallets)),
            "eligible_wallets_total": int(len(eligible)),
            "markets_truncated_by_public_offset_limit": int(truncated_markets),
            "recent_window_start": recent_window_start.isoformat(),
            "recent_window_end": recent_window_end.isoformat(),
        },
    }


def assumptions_markdown() -> str:
    """Return a concise note documenting this current-market scan methodology."""

    return "\n".join(
        [
            "# Current Market Wallet Scan Assumptions",
            "",
            "- Market universe source: public Gamma `GET /markets` with `closed=false`.",
            "- This choice is intentional: live testing on 2026-03-26 showed `active=true` alone still surfaced old closed markets.",
            "- Markets with both `volume <= 0` and `liquidity <= 0` are skipped because they do not help wallet discovery.",
            "- Market trade source: public Data API `GET /trades?market=...`.",
            "- Wallet discovery field priority: `proxyWallet`, then `makerProxyWallet`, then `takerProxyWallet`, then `user`.",
            "- Trade pagination uses `limit=500` and stops after offset `1000` because Polymarket's public `/trades` endpoint documents a public offset ceiling there.",
            "- Wallet trade counts in this export are therefore observed counts within the scanned 1000-market universe and within the publicly discoverable trade pages, not guaranteed lifetime totals across all Polymarket history.",
            "- The recent-activity window is hard-coded by the caller; for this run it should be `2026-03-12T00:00:00Z` through `2026-03-26T23:59:59Z` inclusive.",
        ]
    )


async def run_current_market_wallet_scan(
    *,
    output_dir: str | Path,
    max_markets: int = 1000,
    market_page_size: int = MARKET_PAGE_SIZE_DEFAULT,
    trade_page_size: int = TRADE_PAGE_SIZE_DEFAULT,
    taker_only: bool = False,
    recent_window_start: datetime,
    recent_window_end: datetime,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the current-market wallet scan and export CSV/TXT files."""

    results = await scan_current_market_wallets(
        max_markets=max_markets,
        market_page_size=market_page_size,
        trade_page_size=trade_page_size,
        taker_only=taker_only,
        recent_window_start=recent_window_start,
        recent_window_end=recent_window_end,
        settings=settings,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    universe_path = _write_csv(results["wallet_universe"], output_root / "master_wallet_universe_top1000_current_markets.csv")
    eligible_path = _write_csv(results["eligible_wallets"], output_root / "wallets_100plus_recent_20260312_20260326.csv")
    eligible_txt_path = _write_txt(
        results["eligible_wallets"]["wallet_address"].tolist(),
        output_root / "wallets_100plus_recent_20260312_20260326.txt",
    )
    markets_path = _write_csv(results["scanned_markets"], output_root / "scanned_markets_top1000_current.csv")
    assumptions_path = output_root / "current_market_wallet_scan_assumptions.md"
    assumptions_path.write_text(assumptions_markdown(), encoding="utf-8")

    results["paths"] = {
        "wallet_universe": universe_path,
        "eligible_wallets_csv": eligible_path,
        "eligible_wallets_txt": eligible_txt_path,
        "scanned_markets": markets_path,
        "assumptions": assumptions_path,
    }
    return results


def print_current_market_wallet_scan_summary(results: dict[str, Any]) -> None:
    """Print a concise terminal summary."""

    summary = results["summary"]
    print("Current Market Wallet Scan Summary")
    print(f"Markets scanned: {summary['markets_scanned']}")
    print(f"Trade rows scanned: {summary['trade_rows_scanned']}")
    print(f"Unique wallets discovered: {summary['unique_wallets_discovered']}")
    print(f"Eligible wallets (>=100 trades and active in 2026-03-12..2026-03-26): {summary['eligible_wallets_total']}")
    print(f"Markets truncated by public /trades offset limit: {summary['markets_truncated_by_public_offset_limit']}")
