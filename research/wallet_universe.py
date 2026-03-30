"""Market-scan wallet-universe discovery using only public Polymarket data."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sys
from typing import Any

import httpx
sys.modules.setdefault("pyarrow", None)
import pandas as pd
from sqlalchemy import case, distinct, func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from clients.gamma_client import GammaClient
from clients.profile_client import ProfileClient
from db.models import MarketTradeRaw, MarketTradeScanProgress
from ingestion.backfill import normalize_trade_record
from ingestion.wallets import is_valid_wallet_address, store_wallets


LOGGER = logging.getLogger(__name__)

DEFAULT_MARKET_PAGE_SIZE = 500
DEFAULT_TRADE_PAGE_SIZE = 500
DEFAULT_POSITIONS_PAGE_SIZE = 500
DEFAULT_CLOSED_POSITIONS_PAGE_SIZE = 50
TRADES_PUBLIC_MAX_OFFSET = 1000


@dataclass(frozen=True)
class PnlEstimate:
    """One wallet-level realized PnL estimate built from public position endpoints."""

    wallet_address: str
    realized_pnl_absolute: float | None
    realized_pnl_percent: float | None
    estimate_basis_total_bought: float | None
    open_positions_count: int
    closed_positions_count: int
    pnl_is_estimate: bool
    pnl_estimation_method: str


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV file, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_txt(wallets: list[str], path: Path) -> Path:
    """Write one wallet address per line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(wallets) + ("\n" if wallets else ""), encoding="utf-8")
    return path


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(tz=timezone.utc)


def _to_iso8601(series: pd.Series) -> pd.Series:
    """Convert datetime-like values to ISO-8601 strings."""

    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamps.apply(lambda value: value.isoformat() if pd.notna(value) else None)


def _coerce_float(value: Any) -> float | None:
    """Safely coerce a numeric field from a public API payload."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_market_trade_rows(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize one market-trades page and keep only valid public wallet rows."""

    rows: list[dict[str, Any]] = []
    for trade in trades:
        fallback_wallet = (
            trade.get("proxyWallet")
            or trade.get("makerProxyWallet")
            or trade.get("takerProxyWallet")
            or trade.get("user")
            or ""
        )
        if not fallback_wallet or not is_valid_wallet_address(str(fallback_wallet).lower()):
            continue
        rows.append(normalize_trade_record(str(fallback_wallet), trade))
    return rows


def _upsert_market_raw_trades(session: Session, rows: list[dict[str, Any]]) -> int:
    """Idempotently write market-discovered raw trades into sqlite."""

    if not rows:
        return 0

    stmt = insert(MarketTradeRaw).values(rows)
    excluded = stmt.excluded
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[MarketTradeRaw.trade_id],
            set_={
                "wallet_address": excluded.wallet_address,
                "market_id": excluded.market_id,
                "token_id": excluded.token_id,
                "side": excluded.side,
                "price": excluded.price,
                "size": excluded.size,
                "usdc_size": excluded.usdc_size,
                "timestamp": excluded.timestamp,
                "tx_hash": excluded.tx_hash,
                "raw_json": excluded.raw_json,
            },
        )
    )
    session.flush()
    return len(rows)


def _get_or_create_progress(
    session: Session,
    *,
    condition_id: str,
    gamma_market_id: str | None,
    question: str | None,
) -> MarketTradeScanProgress:
    """Return an existing progress row or create a new resumable checkpoint row."""

    progress = session.get(MarketTradeScanProgress, condition_id)
    if progress is not None:
        if gamma_market_id and not progress.gamma_market_id:
            progress.gamma_market_id = gamma_market_id
        if question and not progress.question:
            progress.question = question
        return progress

    progress = MarketTradeScanProgress(
        condition_id=condition_id,
        gamma_market_id=gamma_market_id,
        question=question,
        next_offset=0,
        completed=False,
        truncated_by_api_limit=False,
        pages_scanned=0,
        trades_scanned=0,
        updated_at=_utc_now(),
    )
    session.add(progress)
    session.flush()
    return progress


async def scan_market_trade_wallets(
    session: Session,
    *,
    gamma_client: GammaClient | None = None,
    profile_client: ProfileClient | None = None,
    market_page_size: int = DEFAULT_MARKET_PAGE_SIZE,
    trade_page_size: int = DEFAULT_TRADE_PAGE_SIZE,
    max_markets: int | None = None,
    resume: bool = True,
) -> dict[str, int]:
    """Scan all discoverable public markets and store their public trade history.

    Notes:
    - Market discovery is paged through Gamma `/markets` until exhaustion.
    - For each market condition id, trade pages are pulled from Data API `/trades`.
    - The public `/trades` endpoint has a documented offset ceiling; if reached, the
      market is marked `truncated_by_api_limit` and treated as fully exhausted from the
      perspective of publicly discoverable pages.
    """

    own_gamma = gamma_client is None
    own_profile = profile_client is None
    gamma = gamma_client or GammaClient()
    profile = profile_client or ProfileClient()

    markets_scanned = 0
    trades_scanned = 0
    market_offset = 0

    try:
        while True:
            market_page = await gamma.list_markets(limit=market_page_size, offset=market_offset)
            if not market_page:
                break

            for market in market_page:
                condition_id = market.get("conditionId")
                if not condition_id:
                    continue

                progress = _get_or_create_progress(
                    session,
                    condition_id=str(condition_id),
                    gamma_market_id=str(market.get("id")) if market.get("id") else None,
                    question=str(market.get("question")) if market.get("question") else None,
                )
                if not resume:
                    progress.next_offset = 0
                    progress.completed = False
                    progress.truncated_by_api_limit = False
                    progress.pages_scanned = 0
                    progress.trades_scanned = 0
                    progress.last_error = None
                    progress.updated_at = _utc_now()
                if resume and progress.completed:
                    continue

                offset = int(progress.next_offset or 0)
                markets_scanned += 1
                while True:
                    try:
                        page = await profile.get_market_trades(
                            market=str(condition_id),
                            limit=trade_page_size,
                            offset=offset,
                        )
                    except httpx.HTTPStatusError as exc:
                        progress.last_error = f"{exc.response.status_code}: {exc}"
                        if exc.response.status_code == 400 and offset > TRADES_PUBLIC_MAX_OFFSET:
                            progress.completed = True
                            progress.truncated_by_api_limit = True
                            progress.updated_at = _utc_now()
                            session.commit()
                            LOGGER.warning(
                                "Stopping market %s at offset %s due to public /trades offset cap",
                                condition_id,
                                offset,
                            )
                            break
                        session.commit()
                        raise

                    raw_page_count = len(page)
                    rows = _normalize_market_trade_rows(page)
                    _upsert_market_raw_trades(session, rows)
                    discovered_wallets = [row["wallet_address"] for row in rows]
                    if discovered_wallets:
                        store_wallets(session, discovered_wallets, source="market_scan")

                    progress.pages_scanned += 1
                    progress.trades_scanned += raw_page_count
                    progress.updated_at = _utc_now()
                    trades_scanned += raw_page_count

                    if not page:
                        progress.completed = True
                        progress.next_offset = offset
                        session.commit()
                        break

                    if len(page) < trade_page_size:
                        progress.completed = True
                        progress.next_offset = offset + len(page)
                        session.commit()
                        break

                    next_offset = offset + trade_page_size
                    progress.next_offset = next_offset
                    if next_offset > TRADES_PUBLIC_MAX_OFFSET:
                        progress.completed = True
                        progress.truncated_by_api_limit = True
                        session.commit()
                        LOGGER.warning(
                            "Marked market %s as truncated at documented /trades offset ceiling %s",
                            condition_id,
                            TRADES_PUBLIC_MAX_OFFSET,
                        )
                        break

                    session.commit()
                    offset = next_offset

                if max_markets is not None and markets_scanned >= max_markets:
                    return {
                        "markets_scanned": markets_scanned,
                        "trades_scanned": trades_scanned,
                    }

            if len(market_page) < market_page_size:
                break
            market_offset += market_page_size
    finally:
        if own_gamma:
            await gamma.aclose()
        if own_profile:
            await profile.aclose()

    return {
        "markets_scanned": markets_scanned,
        "trades_scanned": trades_scanned,
    }


def build_master_wallet_universe(
    session: Session,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Aggregate market-discovered raw trades into wallet-level observed stats."""

    current_time = now or _utc_now()
    march_start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    march_end = datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc)
    recent_cutoff = current_time - timedelta(days=7)

    query = (
        select(
            MarketTradeRaw.wallet_address.label("wallet_address"),
            func.count().label("total_trades"),
            func.count(distinct(MarketTradeRaw.market_id)).label("distinct_markets"),
            func.min(MarketTradeRaw.timestamp).label("first_seen_trade_ts"),
            func.max(MarketTradeRaw.timestamp).label("most_recent_trade_ts"),
            func.max(
                case(
                    (
                        (MarketTradeRaw.timestamp >= march_start)
                        & (MarketTradeRaw.timestamp <= march_end),
                        1,
                    ),
                    else_=0,
                )
            ).label("has_trade_in_2026_03"),
            func.max(
                case((MarketTradeRaw.timestamp >= recent_cutoff, 1), else_=0)
            ).label("has_trade_in_last_7d"),
        )
        .group_by(MarketTradeRaw.wallet_address)
        .order_by(MarketTradeRaw.wallet_address)
    )
    frame = pd.read_sql(query, session.bind)
    if frame.empty:
        return frame
    frame["first_seen_trade_ts"] = _to_iso8601(frame["first_seen_trade_ts"])
    frame["most_recent_trade_ts"] = _to_iso8601(frame["most_recent_trade_ts"])
    frame["has_trade_in_2026_03"] = frame["has_trade_in_2026_03"].fillna(0).astype(int).astype(bool)
    frame["has_trade_in_last_7d"] = frame["has_trade_in_last_7d"].fillna(0).astype(int).astype(bool)
    return frame


async def _fetch_one_wallet_pnl_estimate(
    wallet: str,
    profile_client: ProfileClient,
    *,
    positions_page_size: int,
    closed_positions_page_size: int,
) -> PnlEstimate:
    """Fetch the best public realized-PnL estimate available for one wallet."""

    current_positions = await profile_client.get_all_user_positions(
        wallet,
        closed=False,
        page_size=positions_page_size,
        max_offset=10_000,
    )
    closed_positions = await profile_client.get_all_user_positions(
        wallet,
        closed=True,
        page_size=closed_positions_page_size,
        max_offset=100_000,
    )

    realized_abs = 0.0
    total_bought = 0.0
    saw_any_pnl = False
    saw_any_total_bought = False

    for row in current_positions + closed_positions:
        realized_pnl = _coerce_float(row.get("realizedPnl"))
        if realized_pnl is not None:
            realized_abs += realized_pnl
            saw_any_pnl = True
        bought = _coerce_float(row.get("totalBought"))
        if bought is not None:
            total_bought += bought
            saw_any_total_bought = True

    realized_abs_value = realized_abs if saw_any_pnl else None
    realized_pct_value = None
    if saw_any_pnl and saw_any_total_bought and total_bought > 0:
        realized_pct_value = (realized_abs / total_bought) * 100.0

    return PnlEstimate(
        wallet_address=wallet,
        realized_pnl_absolute=realized_abs_value,
        realized_pnl_percent=realized_pct_value,
        estimate_basis_total_bought=total_bought if saw_any_total_bought else None,
        open_positions_count=len(current_positions),
        closed_positions_count=len(closed_positions),
        pnl_is_estimate=True,
        pnl_estimation_method=(
            "sum(public realizedPnl across current+closed positions) / sum(public totalBought) * 100"
        ),
    )


async def fetch_wallet_pnl_estimates(
    wallets: list[str],
    *,
    profile_client: ProfileClient | None = None,
    max_concurrency: int = 8,
    positions_page_size: int = DEFAULT_POSITIONS_PAGE_SIZE,
    closed_positions_page_size: int = DEFAULT_CLOSED_POSITIONS_PAGE_SIZE,
) -> pd.DataFrame:
    """Fetch public realized-PnL estimates for all discovered wallets."""

    if not wallets:
        return pd.DataFrame(
            columns=[
                "wallet_address",
                "realized_pnl_absolute",
                "realized_pnl_percent",
                "estimate_basis_total_bought",
                "open_positions_count",
                "closed_positions_count",
                "pnl_is_estimate",
                "pnl_estimation_method",
            ]
        )

    own_client = profile_client is None
    client = profile_client or ProfileClient()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _fetch(wallet: str) -> PnlEstimate:
        async with semaphore:
            try:
                return await _fetch_one_wallet_pnl_estimate(
                    wallet,
                    client,
                    positions_page_size=positions_page_size,
                    closed_positions_page_size=closed_positions_page_size,
                )
            except Exception as exc:
                LOGGER.warning("Failed to fetch public position data for wallet %s: %s", wallet, exc)
                return PnlEstimate(
                    wallet_address=wallet,
                    realized_pnl_absolute=None,
                    realized_pnl_percent=None,
                    estimate_basis_total_bought=None,
                    open_positions_count=0,
                    closed_positions_count=0,
                    pnl_is_estimate=True,
                    pnl_estimation_method=f"public position fetch failed: {exc}",
                )

    try:
        estimates = await asyncio.gather(*[_fetch(wallet) for wallet in wallets])
    finally:
        if own_client:
            await client.aclose()

    return pd.DataFrame([estimate.__dict__ for estimate in estimates]).sort_values("wallet_address")


def filter_strong_wallets(master: pd.DataFrame) -> pd.DataFrame:
    """Apply the strict strong-wallet cohort filter."""

    if master.empty:
        return master.copy()
    pct = pd.to_numeric(master["realized_pnl_percent"], errors="coerce")
    return master[
        (master["total_trades"] >= 50)
        & (pct > 10.0)
        & (master["has_trade_in_2026_03"] == True)
    ].sort_values(
        ["realized_pnl_percent", "total_trades", "wallet_address"],
        ascending=[False, False, True],
    )


def filter_weak_wallets(master: pd.DataFrame) -> pd.DataFrame:
    """Apply the strict weak-wallet cohort filter for fade research."""

    if master.empty:
        return master.copy()
    pct = pd.to_numeric(master["realized_pnl_percent"], errors="coerce")
    return master[
        (master["total_trades"] >= 100)
        & (master["has_trade_in_last_7d"] == True)
        & (pct < -33.0)
    ].sort_values(
        ["realized_pnl_percent", "total_trades", "wallet_address"],
        ascending=[True, False, True],
    )


def assumptions_markdown(now: datetime | None = None) -> str:
    """Return a concise markdown note documenting wallet-universe assumptions."""

    current_time = now or _utc_now()
    recent_cutoff = current_time - timedelta(days=7)
    return "\n".join(
        [
            "# Wallet Universe Assumptions",
            "",
            "## Discovery Scope",
            "",
            "- Wallets are discovered only from public market trade payloads returned by `GET /trades?market=...`.",
            "- The discovery wallet field is `proxyWallet`, with fallback checks for equivalent wallet-like keys when present.",
            "- Market scans page through public Gamma `/markets` until no more pages are returned.",
            "",
            "## Public API Limits",
            "",
            "- Polymarket's trade docs and changelog are not perfectly aligned. The API reference page for `/trades` documents broad pagination, but the August 26, 2025 changelog tightened `/trades` to max `limit=500` and max `offset=1000`.",
            "- This scanner therefore defaults to `limit=500` and treats any higher-offset refusal, or reaching offset `1000`, as the end of what is publicly discoverable through that endpoint.",
            "- Very high-volume markets can therefore be truncated by the public endpoint itself.",
            "",
            "## Realized PnL",
            "",
            "- Exact wallet-level realized PnL cannot be guaranteed from public trade history alone because public trades do not fully encode inventory accounting, settlement/redemption timing, merges, or all position lifecycle events.",
            "- The project therefore exports `realized_pnl_absolute` and `realized_pnl_percent` as estimates.",
            "- `realized_pnl_absolute` estimate: sum of public `realizedPnl` across the wallet's current `/positions` rows plus public `/closed-positions` rows.",
            "- `realized_pnl_percent` estimate: `sum(realizedPnl) / sum(totalBought) * 100` across those same public rows when `totalBought` is available.",
            "- These values are best-effort public-data estimates, not audited or exact accounting statements.",
            "",
            "## Time Filters",
            "",
            "- `2026-03` activity means at least one observed market-scan trade timestamp between `2026-03-01T00:00:00Z` and `2026-03-31T23:59:59Z`.",
            f"- `past 7 days` is evaluated relative to the run time. For a run at `{current_time.isoformat()}`, the cutoff is `{recent_cutoff.isoformat()}`.",
            "",
        ]
    )


async def run_wallet_universe_scan(
    session: Session,
    *,
    output_dir: str | Path = "exports/wallet_universe",
    market_page_size: int = DEFAULT_MARKET_PAGE_SIZE,
    trade_page_size: int = DEFAULT_TRADE_PAGE_SIZE,
    positions_page_size: int = DEFAULT_POSITIONS_PAGE_SIZE,
    closed_positions_page_size: int = DEFAULT_CLOSED_POSITIONS_PAGE_SIZE,
    max_markets: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Scan markets, discover wallets, estimate realized PnL, and export cohorts."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    scan_stats = await scan_market_trade_wallets(
        session,
        market_page_size=market_page_size,
        trade_page_size=trade_page_size,
        max_markets=max_markets,
        resume=resume,
    )

    master = build_master_wallet_universe(session)
    wallets = master["wallet_address"].tolist() if not master.empty else []
    pnl_estimates = await fetch_wallet_pnl_estimates(
        wallets,
        max_concurrency=8,
        positions_page_size=positions_page_size,
        closed_positions_page_size=closed_positions_page_size,
    )

    if master.empty:
        combined = master.copy()
    else:
        combined = master.merge(pnl_estimates, on="wallet_address", how="left")

    strong_wallets = filter_strong_wallets(combined)
    weak_wallets = filter_weak_wallets(combined)

    assumptions_path = output_root / "wallet_universe_assumptions.md"
    assumptions_path.write_text(assumptions_markdown(), encoding="utf-8")

    paths = {
        "master_wallet_universe": _write_csv(combined, output_root / "master_wallet_universe.csv"),
        "strong_wallets_csv": _write_csv(strong_wallets, output_root / "strong_wallets.csv"),
        "weak_wallets_csv": _write_csv(weak_wallets, output_root / "weak_wallets.csv"),
        "strong_wallets_txt": _write_txt(strong_wallets["wallet_address"].tolist(), output_root / "strong_wallets.txt"),
        "weak_wallets_txt": _write_txt(weak_wallets["wallet_address"].tolist(), output_root / "weak_wallets.txt"),
        "assumptions": assumptions_path,
    }

    progress_summary = pd.read_sql(
        select(
            func.count().label("scanned_markets_total"),
            func.sum(case((MarketTradeScanProgress.truncated_by_api_limit == True, 1), else_=0)).label("markets_truncated_by_api_limit"),
            func.sum(MarketTradeScanProgress.trades_scanned).label("trades_scanned_total"),
        ),
        session.bind,
    )

    summary = {
        "scanned_markets_total": int(progress_summary["scanned_markets_total"].iloc[0] or 0),
        "markets_scanned_this_run": int(scan_stats["markets_scanned"]),
        "scanned_trades_total": int(progress_summary["trades_scanned_total"].iloc[0] or 0),
        "trades_scanned_this_run": int(scan_stats["trades_scanned"]),
        "unique_wallets_total": int(len(combined)),
        "strong_wallets_total": int(len(strong_wallets)),
        "weak_wallets_total": int(len(weak_wallets)),
        "markets_truncated_by_api_limit": int(progress_summary["markets_truncated_by_api_limit"].iloc[0] or 0),
    }

    return {
        "master_wallet_universe": combined,
        "strong_wallets": strong_wallets,
        "weak_wallets": weak_wallets,
        "summary": summary,
        "paths": paths,
    }


def print_wallet_universe_summary(results: dict[str, Any]) -> None:
    """Print a concise console summary for the wallet-universe scan."""

    summary = results["summary"]
    print("Wallet Universe Scan Summary")
    print(f"Scanned markets total: {summary['scanned_markets_total']}")
    print(f"Scanned trades total: {summary['scanned_trades_total']}")
    print(f"Unique wallets discovered: {summary['unique_wallets_total']}")
    print(f"Strong wallets: {summary['strong_wallets_total']}")
    print(f"Weak wallets: {summary['weak_wallets_total']}")
    print(f"Markets truncated by public /trades limit: {summary['markets_truncated_by_api_limit']}")
