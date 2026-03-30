"""Capture recent public wallet trades for a supplied wallet cohort."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Sequence

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from clients.profile_client import ProfileClient
from config.settings import get_settings
from db.models import WalletTradeRaw
from ingestion.backfill import _upsert_raw_trades, normalize_trade_record
from ingestion.wallets import store_wallets


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _to_utc(value: str) -> datetime:
    """Parse an ISO date/time string into UTC."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_recent_wallet_trades(
    session: Session,
    *,
    wallets: Sequence[str],
    recent_window_start: datetime,
    recent_window_end: datetime,
) -> pd.DataFrame:
    """Load raw trades for the supplied wallet cohort inside the requested window."""

    if not wallets:
        return pd.DataFrame()

    query = (
        select(WalletTradeRaw)
        .where(WalletTradeRaw.wallet_address.in_(list(wallets)))
        .where(WalletTradeRaw.timestamp >= recent_window_start)
        .where(WalletTradeRaw.timestamp <= recent_window_end)
    )
    frame = pd.read_sql(query, session.bind)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.sort_values(["wallet_address", "timestamp", "trade_id"]).reset_index(drop=True)


def summarize_recent_wallet_trades(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a wallet-level summary from recent raw trades."""

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "wallet_address",
                "recent_trades",
                "recent_distinct_markets",
                "recent_distinct_tokens",
                "first_recent_trade_ts",
                "most_recent_trade_ts",
                "recent_buy_trades",
                "recent_sell_trades",
                "recent_usdc_volume",
            ]
        )

    records: list[dict[str, Any]] = []
    for wallet, group in frame.groupby("wallet_address"):
        side_series = group["side"].astype(str).str.upper()
        records.append(
            {
                "wallet_address": wallet,
                "recent_trades": int(len(group)),
                "recent_distinct_markets": int(group["market_id"].nunique(dropna=True)),
                "recent_distinct_tokens": int(group["token_id"].nunique(dropna=True)),
                "first_recent_trade_ts": group["timestamp"].min().isoformat(),
                "most_recent_trade_ts": group["timestamp"].max().isoformat(),
                "recent_buy_trades": int((side_series == "BUY").sum()),
                "recent_sell_trades": int((side_series == "SELL").sum()),
                "recent_usdc_volume": float(pd.to_numeric(group["usdc_size"], errors="coerce").fillna(0.0).sum()),
            }
        )

    return pd.DataFrame.from_records(records).sort_values(
        ["recent_trades", "recent_usdc_volume", "wallet_address"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


async def fetch_recent_wallet_trade_payloads(
    wallets: Sequence[str],
    *,
    recent_window_start: datetime,
    recent_window_end: datetime,
    profile_client: ProfileClient | None = None,
    page_size: int = 500,
    max_offset: int | None = 1000,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch only the public wallet trade pages needed for the recent window.

    Assumption:
    - Public wallet `/trades` pages are returned newest-first. Once a page
      contains timestamps older than the requested window start, deeper pages
      are older still and can be skipped.
    """

    if not wallets:
        return {}

    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.max_concurrency)
    own_client = profile_client is None
    client = profile_client or ProfileClient()

    async def _fetch(wallet: str) -> tuple[str, list[dict[str, Any]]]:
        async with semaphore:
            collected: list[dict[str, Any]] = []
            offset = 0
            previous_page_signature: tuple[str, ...] | None = None

            while True:
                page = await client.get_user_trades(wallet=wallet, limit=page_size, offset=offset)
                if not page:
                    break

                current_signature = tuple(
                    str(item.get("transactionHash") or item.get("timestamp") or index)
                    for index, item in enumerate(page)
                )
                if current_signature == previous_page_signature:
                    break

                reached_older_rows = False
                for trade in page:
                    trade_ts = pd.to_datetime(trade.get("timestamp"), utc=True, errors="coerce")
                    if pd.isna(trade_ts):
                        continue
                    if trade_ts > recent_window_end:
                        continue
                    if trade_ts < recent_window_start:
                        reached_older_rows = True
                        continue
                    collected.append(trade)

                if reached_older_rows or len(page) < page_size:
                    break

                previous_page_signature = current_signature
                offset += page_size
                if max_offset is not None and offset > max_offset:
                    break

            return wallet, collected

    try:
        fetched = await asyncio.gather(*[_fetch(wallet) for wallet in wallets])
    finally:
        if own_client:
            await client.aclose()

    return {wallet: rows for wallet, rows in fetched}


async def run_recent_wallet_trade_capture(
    session: Session,
    *,
    wallets: Sequence[str],
    recent_window_start: str,
    recent_window_end: str,
    output_dir: str | Path,
    profile_client: ProfileClient | None = None,
) -> dict[str, Any]:
    """Backfill public wallet trades for a cohort and export the recent subset."""

    normalized_wallets = store_wallets(session, wallets, source="current_market_wallet_scan")
    start_dt = _to_utc(recent_window_start)
    end_dt = _to_utc(recent_window_end)
    payloads = await fetch_recent_wallet_trade_payloads(
        normalized_wallets,
        recent_window_start=start_dt,
        recent_window_end=end_dt,
        profile_client=profile_client,
    )
    backfill_summary: dict[str, int] = {}
    for wallet, trades in payloads.items():
        rows = [normalize_trade_record(wallet, trade) for trade in trades]
        _upsert_raw_trades(session, rows)
        session.commit()
        backfill_summary[wallet] = len(rows)

    recent_trades = load_recent_wallet_trades(
        session,
        wallets=normalized_wallets,
        recent_window_start=start_dt,
        recent_window_end=end_dt,
    )
    recent_summary = summarize_recent_wallet_trades(recent_trades)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    window_label = (
        recent_window_start[:10].replace("-", "") + "_" + recent_window_end[:10].replace("-", "")
    )
    trades_path = _write_csv(
        recent_trades,
        output_root / f"recent_wallet_trades_{window_label}.csv",
    )
    summary_path = _write_csv(
        recent_summary,
        output_root / f"recent_wallet_trade_summary_{window_label}.csv",
    )
    tracked_path = output_root / "tracked_wallets.txt"
    tracked_path.write_text("\n".join(normalized_wallets) + ("\n" if normalized_wallets else ""), encoding="utf-8")

    return {
        "backfill_summary": backfill_summary,
        "recent_trades": recent_trades,
        "recent_summary": recent_summary,
        "paths": {
            "recent_trades": trades_path,
            "recent_summary": summary_path,
            "tracked_wallets": tracked_path,
        },
        "summary": {
            "wallets_requested": len(wallets),
            "wallets_tracked": len(normalized_wallets),
            "wallets_with_recent_trades": int(recent_summary["wallet_address"].nunique()),
            "recent_trade_rows": int(len(recent_trades)),
        },
    }


def print_recent_wallet_trade_capture_summary(results: dict[str, Any]) -> None:
    """Print a concise summary for terminal use."""

    summary = results["summary"]
    print("Recent Wallet Trade Capture Summary")
    print(f"Wallets requested: {summary['wallets_requested']}")
    print(f"Wallets tracked: {summary['wallets_tracked']}")
    print(f"Wallets with recent trades: {summary['wallets_with_recent_trades']}")
    print(f"Recent trade rows captured: {summary['recent_trade_rows']}")
