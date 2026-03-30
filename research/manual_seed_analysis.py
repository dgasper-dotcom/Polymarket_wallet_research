"""Seed-wallet-only enrichment, delay review, and long-hold evidence analysis."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Sequence

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import WalletTradeEnriched, WalletTradeRaw
from ingestion.markets import backfill_markets_for_references
from ingestion.prices import backfill_price_history_for_token_bounds
from research.all_active_reevaluation import compute_open_position_evidence
from research.copy_follow_expiry import _build_terminal_lookup, load_markets_frame
from research.delay_analysis import (
    _delay_event_study,
    compute_delay_trade_metrics,
    load_price_history_frame,
    persist_delay_metrics,
    summarize_wallet_delay_metrics,
)
from research.enrich_trades import enrich_wallet_trades
from research.event_study import compute_event_study_from_frame
from research.resolved_expiry_report import compute_resolved_expiry_positions_from_frame


DEFAULT_MANUAL_SEED_CSV = "data/manual_seed_wallets.csv"
DEFAULT_OUTPUT_DIR = "exports/manual_seed_analysis"
PRICE_HISTORY_CHUNK_SIZE = 500


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Persist one DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _normalize_wallets(values: Sequence[str]) -> list[str]:
    """Normalize wallet ids while preserving input order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        wallet = str(value or "").strip().lower()
        if not wallet or wallet in seen:
            continue
        seen.add(wallet)
        ordered.append(wallet)
    return ordered


def load_manual_seed_frame(path: str | Path = DEFAULT_MANUAL_SEED_CSV) -> pd.DataFrame:
    """Load resolved manual seed metadata."""

    seed_path = Path(path)
    if not seed_path.exists():
        return pd.DataFrame(
            columns=["display_name", "wallet_address", "status", "priority_group", "notes", "address_source"]
        )
    frame = pd.read_csv(seed_path)
    if frame.empty:
        return frame
    frame["wallet_address"] = frame["wallet_address"].astype(str).str.strip().str.lower()
    frame["status"] = frame["status"].astype(str).str.strip().str.lower()
    frame = frame.loc[(frame["status"] == "resolved") & (frame["wallet_address"] != "")].copy()
    frame = frame.drop_duplicates(subset=["wallet_address"], keep="first")
    return frame.sort_values(["priority_group", "display_name", "wallet_address"]).reset_index(drop=True)


def _analysis_cutoff(raw_trades: pd.DataFrame, explicit_cutoff: str | None = None) -> pd.Timestamp:
    """Resolve the seed-analysis cutoff timestamp."""

    if explicit_cutoff:
        parsed = pd.to_datetime(explicit_cutoff, utc=True, errors="coerce")
        if parsed is not None and not pd.isna(parsed):
            return parsed
    if not raw_trades.empty and "timestamp" in raw_trades.columns:
        parsed = pd.to_datetime(raw_trades["timestamp"], utc=True, errors="coerce").dropna()
        if not parsed.empty:
            return parsed.max()
    return pd.Timestamp.now(tz="UTC")


def _load_seed_raw_trades(session: Session, wallets: Sequence[str]) -> pd.DataFrame:
    """Load raw trades for the selected wallet subset."""

    normalized = _normalize_wallets(wallets)
    if not normalized:
        return pd.DataFrame()
    query = (
        select(WalletTradeRaw)
        .where(WalletTradeRaw.wallet_address.in_(normalized))
        .order_by(WalletTradeRaw.wallet_address, WalletTradeRaw.timestamp, WalletTradeRaw.trade_id)
    )
    frame = pd.read_sql(query, session.bind)
    if frame.empty:
        return frame
    frame["wallet_address"] = frame["wallet_address"].astype(str).str.strip().str.lower()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["wallet_address", "timestamp"]).reset_index(drop=True)


def _load_seed_enriched_trades(session: Session, wallets: Sequence[str]) -> pd.DataFrame:
    """Load enriched trades for the selected wallet subset."""

    normalized = _normalize_wallets(wallets)
    if not normalized:
        return pd.DataFrame()
    query = (
        select(WalletTradeEnriched)
        .where(WalletTradeEnriched.wallet_address.in_(normalized))
        .order_by(WalletTradeEnriched.wallet_address, WalletTradeEnriched.timestamp, WalletTradeEnriched.trade_id)
    )
    frame = pd.read_sql(query, session.bind)
    if frame.empty:
        return frame
    frame["wallet_address"] = frame["wallet_address"].astype(str).str.strip().str.lower()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["wallet_address", "timestamp"]).reset_index(drop=True)


def _chunked_price_history(
    session: Session,
    *,
    token_ids: Sequence[str],
    start_ts: Any | None = None,
    end_ts: Any | None = None,
    chunk_size: int = PRICE_HISTORY_CHUNK_SIZE,
) -> pd.DataFrame:
    """Load price history in SQLite-safe chunks for the seed subset."""

    normalized = [str(token_id) for token_id in token_ids if token_id]
    if not normalized:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for start in range(0, len(normalized), chunk_size):
        chunk = normalized[start : start + chunk_size]
        frames.append(load_price_history_frame(session, token_ids=chunk, start_ts=start_ts, end_ts=end_ts))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["token_id", "ts"], keep="last")


def _seed_raw_coverage(seed_frame: pd.DataFrame, raw_trades: pd.DataFrame) -> pd.DataFrame:
    """Build raw-trade coverage stats for the manual seeds."""

    base = seed_frame.rename(columns={"wallet_address": "wallet_id"}).copy()
    if raw_trades.empty:
        base["raw_trades"] = 0
        base["raw_markets"] = 0
        base["raw_tokens"] = 0
        base["first_raw_trade"] = None
        base["last_raw_trade"] = None
        return base

    grouped = (
        raw_trades.groupby("wallet_address", sort=False)
        .agg(
            raw_trades=("trade_id", "size"),
            raw_markets=("market_id", lambda values: pd.Series(values).nunique(dropna=True)),
            raw_tokens=("token_id", lambda values: pd.Series(values).nunique(dropna=True)),
            first_raw_trade=("timestamp", "min"),
            last_raw_trade=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"wallet_address": "wallet_id"})
    )
    grouped["first_raw_trade"] = pd.to_datetime(grouped["first_raw_trade"], utc=True, errors="coerce").apply(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    grouped["last_raw_trade"] = pd.to_datetime(grouped["last_raw_trade"], utc=True, errors="coerce").apply(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    return base.merge(grouped, on="wallet_id", how="left")


def _build_token_bounds(raw_trades: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Build token start/end bounds for targeted price-history backfill."""

    if raw_trades.empty:
        return []
    valid = raw_trades.dropna(subset=["token_id", "timestamp"]).copy()
    if valid.empty:
        return []
    bounds = (
        valid.groupby("token_id", sort=False)
        .agg(min_ts=("timestamp", "min"), max_ts=("timestamp", "max"))
        .reset_index()
    )
    return [
        (str(row.token_id), pd.Timestamp(row.min_ts), pd.Timestamp(row.max_ts))
        for row in bounds.itertuples(index=False)
        if row.token_id is not None and pd.notna(row.min_ts) and pd.notna(row.max_ts)
    ]


def build_manual_seed_wallet_overview(
    seed_frame: pd.DataFrame,
    raw_coverage: pd.DataFrame,
    delay_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    expiry_summary: pd.DataFrame,
    open_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Join seed metadata with subset delay, event, and hold evidence outputs."""

    base = seed_frame.rename(columns={"wallet_address": "wallet_id"}).copy()
    delay = delay_summary.rename(columns={"wallet_address": "wallet_id"}) if not delay_summary.empty else pd.DataFrame()
    event = event_summary.rename(columns={"wallet_address": "wallet_id"}) if not event_summary.empty else pd.DataFrame()
    expiry = expiry_summary.copy()
    open_frame = open_summary.copy()

    for frame in (delay, event, expiry, open_frame, raw_coverage):
        if not frame.empty and "wallet_id" in frame.columns:
            frame["wallet_id"] = frame["wallet_id"].astype(str).str.strip().str.lower()

    merged = raw_coverage.copy() if not raw_coverage.empty else base.copy()
    if not delay.empty:
        merged = merged.merge(delay, on="wallet_id", how="left", suffixes=("", "_delay"))
    if not event.empty:
        merged = merged.merge(event, on="wallet_id", how="left", suffixes=("", "_event"))
    if not expiry.empty:
        merged = merged.merge(expiry, on="wallet_id", how="left", suffixes=("", "_expiry"))
    if not open_frame.empty:
        merged = merged.merge(open_frame, on="wallet_id", how="left", suffixes=("", "_open"))
    return merged.sort_values(["priority_group", "display_name", "wallet_id"]).reset_index(drop=True)


def build_manual_seed_report(
    *,
    seed_frame: pd.DataFrame,
    raw_trades: pd.DataFrame,
    delay_summary: pd.DataFrame,
    event_summary: pd.DataFrame,
    expiry_summary: pd.DataFrame,
    open_summary: pd.DataFrame,
    analysis_cutoff: pd.Timestamp,
    market_backfill_counts: dict[str, int],
    price_backfill_counts: dict[str, int],
    enriched_rows: int,
) -> str:
    """Render a concise Markdown report for the manual-seed subset."""

    top_delay = (
        delay_summary.sort_values("avg_copy_pnl_net_5m_delay_30s", ascending=False, na_position="last")
        .head(10)
        .to_dict(orient="records")
        if not delay_summary.empty
        else []
    )
    top_open = (
        open_summary.sort_values("unresolved_open_mtm_total_net_usdc", ascending=False, na_position="last")
        .head(10)
        .to_dict(orient="records")
        if not open_summary.empty
        else []
    )
    lines = [
        "# Manual Seed Wallet Analysis",
        "",
        "This seed-only pipeline refreshes enrichment and then reports two separate views:",
        "",
        "- short-horizon delayed copy evidence from the enriched trades table",
        "- long-hold evidence split into observed expiry holds vs unresolved open positions",
        "",
        "## Scope",
        "",
        f"- manual seeds loaded: `{len(seed_frame)}`",
        f"- raw trade rows analyzed: `{len(raw_trades)}`",
        f"- analysis cutoff: `{analysis_cutoff.isoformat()}`",
        f"- targeted Gamma markets stored this run: `{market_backfill_counts.get('markets', 0)}`",
        f"- targeted token rows stored this run: `{market_backfill_counts.get('tokens', 0)}`",
        f"- token price-history rows stored this run: `{sum(price_backfill_counts.values())}`",
        f"- enriched raw trades processed this run: `{enriched_rows}`",
        "",
        "## Observed Expiry-Hold Evidence",
        "",
        f"- wallets with at least one observed hold-to-expiry slice: `{int((pd.to_numeric(expiry_summary.get('held_to_expiry_observed_slices'), errors='coerce').fillna(0) > 0).sum()) if not expiry_summary.empty else 0}`",
        f"- total observed hold-to-expiry slices: `{int(pd.to_numeric(expiry_summary.get('held_to_expiry_observed_slices'), errors='coerce').fillna(0).sum()) if not expiry_summary.empty else 0}`",
        "",
        "## Unresolved Open-Position Evidence",
        "",
        f"- wallets with unresolved open slices: `{int((pd.to_numeric(open_summary.get('unresolved_open_slices'), errors='coerce').fillna(0) > 0).sum()) if not open_summary.empty else 0}`",
        f"- total unresolved open slices: `{int(pd.to_numeric(open_summary.get('unresolved_open_slices'), errors='coerce').fillna(0).sum()) if not open_summary.empty else 0}`",
        "",
        "## Top Delay Candidates (30s net edge)",
        "",
    ]
    if not top_delay:
        lines.append("- No enriched seed wallets had valid 30s delay metrics.")
    else:
        for row in top_delay:
            lines.append(
                f"- `{row['wallet_address']}` | edge30 `{row.get('avg_copy_pnl_net_5m_delay_30s')}` | "
                f"tradability `{row.get('tradability_label')}` | markets `{row.get('n_markets')}`"
            )

    lines.extend(["", "## Top Open-Position MTM Wallets", ""])
    if not top_open:
        lines.append("- No unresolved open-position MTM evidence was available.")
    else:
        for row in top_open:
            lines.append(
                f"- `{row['wallet_id']}` | unresolved `{row.get('unresolved_open_slices')}` | "
                f"mtm_total `{row.get('unresolved_open_mtm_total_net_usdc')}` | "
                f"avg_hold_days `{row.get('unresolved_open_avg_holding_days')}` | "
                f"fwd7 `{row.get('unresolved_open_avg_forward_7d_net')}` | "
                f"fwd30 `{row.get('unresolved_open_avg_forward_30d_net')}`"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Observed expiry-hold slices are the strongest long-hold evidence because the market has already publicly resolved.",
            "- Unresolved open slices are weaker but still useful: they capture whether the wallet keeps positions open for long periods and whether those open lots mark positively or negatively under current public prices.",
            "- Delay metrics remain 5-minute forward proxies, so use them as a short-horizon friction check, not as proof of long-hold expiry behavior.",
        ]
    )
    return "\n".join(lines)


async def run_manual_seed_analysis(
    session: Session,
    *,
    seed_csv: str | Path = DEFAULT_MANUAL_SEED_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    analysis_cutoff: str | None = None,
    refresh_markets: bool = False,
    refresh_prices: bool = False,
    fetch_books: bool = False,
) -> dict[str, Any]:
    """Run enrichment plus long-hold and delay review for the manual seed wallets."""

    seed_frame = load_manual_seed_frame(seed_csv)
    wallets = seed_frame["wallet_address"].astype(str).str.lower().tolist()
    if not wallets:
        raise ValueError("No resolved manual seed wallets were found.")

    raw_trades = _load_seed_raw_trades(session, wallets)
    condition_ids = sorted(
        {
            str(value)
            for value in raw_trades.get("market_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
            if value.startswith("0x")
        }
    )
    token_ids = sorted(
        {
            str(value)
            for value in raw_trades.get("token_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
            if value
        }
    )
    market_backfill_counts = {"markets": 0, "tokens": 0}
    if refresh_markets and (condition_ids or token_ids):
        market_backfill_counts = await backfill_markets_for_references(
            session,
            condition_ids=condition_ids,
            token_ids=token_ids,
        )
        raw_trades = _load_seed_raw_trades(session, wallets)

    token_bounds = _build_token_bounds(raw_trades)
    price_backfill_counts: dict[str, int] = {}
    if refresh_prices and token_bounds:
        price_backfill_counts = await backfill_price_history_for_token_bounds(
            session,
            token_bounds=token_bounds,
        )

    enriched_rows = await enrich_wallet_trades(session, wallets=wallets, fetch_books=fetch_books)
    enriched = _load_seed_enriched_trades(session, wallets)

    if enriched.empty:
        delay_trades = pd.DataFrame()
        delay_summary = pd.DataFrame()
        delay_event_study = pd.DataFrame()
        event_summary = pd.DataFrame()
        trade_diagnostics = pd.DataFrame()
    else:
        analysis_end = _analysis_cutoff(raw_trades, analysis_cutoff)
        price_history = _chunked_price_history(
            session,
            token_ids=enriched["token_id"].dropna().astype(str).unique().tolist(),
            end_ts=analysis_end,
        )
        delay_trades = compute_delay_trade_metrics(enriched, price_history)
        persist_delay_metrics(session, delay_trades)
        delay_summary = summarize_wallet_delay_metrics(delay_trades)
        delay_event_study = _delay_event_study(delay_trades)
        event_summary, trade_diagnostics = compute_event_study_from_frame(delay_trades, stringify_datetimes=True)

    analysis_end = _analysis_cutoff(raw_trades, analysis_cutoff)
    session.commit()
    terminal_lookup = _build_terminal_lookup(load_markets_frame(session))
    expiry_summary, resolved_trades, expiry_overview, expiry_diagnostics = compute_resolved_expiry_positions_from_frame(
        raw_trades,
        terminal_lookup,
        analysis_cutoff=analysis_end.isoformat(),
        return_diagnostics=True,
    )
    paired_counts = {
        str(row["wallet_id"]).lower(): int(
            0
            if pd.isna(pd.to_numeric(row.get("wallet_sell_closed_slices"), errors="coerce"))
            else pd.to_numeric(row.get("wallet_sell_closed_slices"), errors="coerce")
        )
        for row in expiry_summary.to_dict(orient="records")
        if row.get("wallet_id")
    }
    open_wallet_summary, unresolved_open_trades = compute_open_position_evidence(
        expiry_diagnostics,
        terminal_lookup={},
        analysis_cutoff=analysis_end.isoformat(),
        collect_trade_records=True,
        paired_counts=paired_counts,
    )

    raw_coverage = _seed_raw_coverage(seed_frame, raw_trades)
    wallet_overview = build_manual_seed_wallet_overview(
        seed_frame,
        raw_coverage,
        delay_summary,
        event_summary,
        expiry_summary,
        open_wallet_summary,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_coverage": _write_csv(raw_coverage, output_root / "manual_seed_raw_trade_coverage.csv"),
        "delay_summary": _write_csv(delay_summary, output_root / "manual_seed_delay_wallet_summary.csv"),
        "delay_event_study": _write_csv(delay_event_study, output_root / "manual_seed_delay_event_study.csv"),
        "event_summary": _write_csv(event_summary, output_root / "manual_seed_event_study_summary.csv"),
        "trade_diagnostics": _write_csv(trade_diagnostics, output_root / "manual_seed_trade_diagnostics.csv"),
        "expiry_summary": _write_csv(expiry_summary, output_root / "manual_seed_resolved_expiry_wallet_summary.csv"),
        "resolved_trades": _write_csv(resolved_trades, output_root / "manual_seed_resolved_expiry_trades.csv"),
        "expiry_overview": _write_csv(expiry_overview, output_root / "manual_seed_resolved_expiry_overview.csv"),
        "expiry_diagnostics": _write_csv(expiry_diagnostics, output_root / "manual_seed_resolved_expiry_diagnostics.csv"),
        "open_summary": _write_csv(open_wallet_summary, output_root / "manual_seed_open_position_evidence.csv"),
        "unresolved_trades": _write_csv(unresolved_open_trades, output_root / "manual_seed_unresolved_open_trades.csv"),
        "wallet_overview": _write_csv(wallet_overview, output_root / "manual_seed_wallet_overview.csv"),
    }

    report_path = output_root / "manual_seed_report.md"
    report_path.write_text(
        build_manual_seed_report(
            seed_frame=seed_frame,
            raw_trades=raw_trades,
            delay_summary=delay_summary,
            event_summary=event_summary,
            expiry_summary=expiry_summary,
            open_summary=open_wallet_summary,
            analysis_cutoff=analysis_end,
            market_backfill_counts=market_backfill_counts,
            price_backfill_counts=price_backfill_counts,
            enriched_rows=enriched_rows,
        ),
        encoding="utf-8",
    )
    paths["report"] = report_path

    return {
        "seed_frame": seed_frame,
        "raw_trades": raw_trades,
        "raw_coverage": raw_coverage,
        "delay_trades": delay_trades,
        "delay_summary": delay_summary,
        "delay_event_study": delay_event_study,
        "event_summary": event_summary,
        "trade_diagnostics": trade_diagnostics,
        "expiry_summary": expiry_summary,
        "resolved_trades": resolved_trades,
        "expiry_overview": expiry_overview,
        "expiry_diagnostics": expiry_diagnostics,
        "open_wallet_summary": open_wallet_summary,
        "unresolved_open_trades": unresolved_open_trades,
        "wallet_overview": wallet_overview,
        "paths": paths,
    }


def print_manual_seed_analysis_summary(results: dict[str, Any]) -> None:
    """Print a concise terminal summary for the seed-only analysis."""

    seed_frame = results["seed_frame"]
    wallet_overview = results["wallet_overview"]
    open_summary = results["open_wallet_summary"]
    print("Manual Seed Analysis")
    print(f"Seed wallets: {len(seed_frame)}")
    print(f"Wallets with raw trades: {int(pd.to_numeric(wallet_overview.get('raw_trades'), errors='coerce').fillna(0).gt(0).sum())}")
    print(f"Wallets with unresolved open slices: {int(pd.to_numeric(open_summary.get('unresolved_open_slices'), errors='coerce').fillna(0).gt(0).sum()) if not open_summary.empty else 0}")
    if not wallet_overview.empty:
        columns = [
            column
            for column in [
                "display_name",
                "wallet_id",
                "priority_group",
                "raw_trades",
                "avg_copy_pnl_net_5m_delay_30s",
                "tradability_label",
                "held_to_expiry_observed_slices",
                "unresolved_open_slices",
                "unresolved_open_mtm_total_net_usdc",
                "unresolved_open_avg_holding_days",
                "unresolved_open_avg_forward_7d_net",
                "unresolved_open_avg_forward_30d_net",
            ]
            if column in wallet_overview.columns
        ]
        print(wallet_overview[columns].to_string(index=False))
