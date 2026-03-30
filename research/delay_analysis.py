"""Delay and realistic cost analysis built on top of enriched wallet trades."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
import logging
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from db.models import PriceHistory, WalletTradeEnriched
from research.costs import (
    calculate_net_pnl,
    estimate_break_even_cost,
    estimate_total_cost,
)
from research.event_study import load_enriched_trades
from research.wallet_scoring import score_wallets


LOGGER = logging.getLogger(__name__)

DELAY_SECONDS = (0, 5, 15, 30, 60)
DELAY_LABELS = {
    0: "0s",
    5: "5s",
    15: "15s",
    30: "30s",
    60: "60s",
}
DELAY_COLUMN_SUFFIX = {
    5: "_delay_5s",
    15: "_delay_15s",
    30: "_delay_30s",
    60: "_delay_60s",
}
UPSERT_CHUNK_SIZE = 250


@dataclass(frozen=True)
class ForwardPriceLookup:
    """First-available price lookup at or after a target timestamp."""

    price: float | None
    source: str
    delta_seconds: int | None


def _to_epoch_seconds(value: pd.Timestamp) -> int:
    """Convert a pandas timestamp to UTC epoch seconds."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.tz_convert("UTC").timestamp())


def _build_price_index(price_history: pd.DataFrame) -> dict[str, tuple[list[int], list[float]]]:
    """Build a forward-lookup-friendly in-memory price index."""

    if price_history.empty:
        return {}

    frame = price_history.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["token_id", "ts", "price"]).sort_values(["token_id", "ts"])

    index: dict[str, tuple[list[int], list[float]]] = {}
    for row in frame.itertuples(index=False):
        token_id = str(row.token_id)
        times, prices = index.setdefault(token_id, ([], []))
        times.append(_to_epoch_seconds(row.ts))
        prices.append(float(row.price))
    return index


def lookup_forward_price(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    target_ts: int,
    exactish_threshold_seconds: int = 60,
) -> ForwardPriceLookup:
    """Return the first public price point at or after the target timestamp."""

    if token_id is None or token_id not in index:
        return ForwardPriceLookup(price=None, source="missing_prices", delta_seconds=None)

    times, prices = index[token_id]
    if not times:
        return ForwardPriceLookup(price=None, source="missing_prices", delta_seconds=None)

    position = bisect_left(times, target_ts)
    if position >= len(times):
        return ForwardPriceLookup(price=None, source="missing_prices", delta_seconds=None)

    delta_seconds = times[position] - target_ts
    source = (
        "price_history_exactish"
        if delta_seconds <= exactish_threshold_seconds
        else "price_history_after_delay"
    )
    return ForwardPriceLookup(price=prices[position], source=source, delta_seconds=delta_seconds)


def classify_tradability(
    *,
    base_net_pnl: float | None,
    net_pnl_delay_15s: float | None,
    net_pnl_delay_30s: float | None,
    optimistic_net_pnl: float | None = None,
    mode_consistency: float | None = None,
    settings: Settings | None = None,
) -> str:
    """Label whether a wallet still looks usable after delay and costs."""

    cfg = settings or get_settings()
    consistency_ok = mode_consistency is None or mode_consistency >= cfg.multi_oos_mode_consistency_threshold
    if base_net_pnl is not None and base_net_pnl > 0 and consistency_ok:
        if (net_pnl_delay_15s is not None and net_pnl_delay_15s > 0) or (
            net_pnl_delay_30s is not None and net_pnl_delay_30s > 0
        ):
            return "tradable"
    if (
        (base_net_pnl is not None and base_net_pnl > 0)
        or (optimistic_net_pnl is not None and optimistic_net_pnl > 0)
        or (net_pnl_delay_15s is not None and net_pnl_delay_15s > 0)
    ):
        return "borderline"
    return "not_tradable"


def _delay_column(strategy: str, *, delay_seconds: int, net: bool) -> str:
    """Return the configured column name for one delay-specific metric."""

    if delay_seconds == 0:
        return f"{strategy}_pnl_net_5m" if net else f"{strategy}_pnl_5m"
    suffix = DELAY_COLUMN_SUFFIX[delay_seconds]
    return f"{strategy}_pnl_net_5m{suffix}" if net else f"{strategy}_pnl_5m{suffix}"


def _raw_baseline_pnl(row: pd.Series, strategy: str) -> float | None:
    """Return the baseline raw 5m PnL for copy or fade."""

    ret_5m = pd.to_numeric(pd.Series([row.get("ret_5m")]), errors="coerce").iloc[0]
    if pd.isna(ret_5m):
        return None
    return float(ret_5m) if strategy == "copy" else float(-ret_5m)


def _mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    """Return the mean of a column if it exists and has valid data."""

    if column not in frame.columns:
        return None
    return estimate_break_even_cost(frame[column])


def _positive_share(frame: pd.DataFrame, column: str) -> float | None:
    """Return the fraction of positive values for one column."""

    if column not in frame.columns:
        return None
    valid = pd.to_numeric(frame[column], errors="coerce").dropna()
    if valid.empty:
        return None
    return float((valid > 0).mean())


def _simple_slope(xs: list[float], ys: list[float | None]) -> float | None:
    """Compute a lightweight slope estimate without adding dependencies."""

    points = [(float(x), float(y)) for x, y in zip(xs, ys) if y is not None and pd.notna(y)]
    if len(points) < 2:
        return None
    x_mean = sum(x for x, _ in points) / len(points)
    y_mean = sum(y for _, y in points) / len(points)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return numerator / denominator


def _retention(base_value: float | None, delayed_value: float | None) -> float | None:
    """Return edge retention relative to the zero-delay baseline."""

    if base_value is None or delayed_value is None or base_value == 0:
        return None
    return delayed_value / base_value


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Persist one DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_price_history_frame(
    session: Session,
    token_ids: list[str] | None = None,
    *,
    start_ts: Any | None = None,
    end_ts: Any | None = None,
) -> pd.DataFrame:
    """Load public price history for the supplied tokens."""

    query = select(PriceHistory)
    if token_ids:
        query = query.where(PriceHistory.token_id.in_(token_ids))
    if start_ts is not None:
        query = query.where(PriceHistory.ts >= pd.to_datetime(start_ts, utc=True))
    if end_ts is not None:
        query = query.where(PriceHistory.ts <= pd.to_datetime(end_ts, utc=True))
    return pd.read_sql(query, session.bind)


def compute_delay_trade_metrics(
    enriched: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Compute delay and net-PnL columns from enriched trades and cached prices."""

    cfg = settings or get_settings()
    frame = enriched.copy()
    if frame.empty:
        return frame

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    price_index = _build_price_index(price_history)

    for column in (
        "copy_pnl_net_5m",
        "fade_pnl_net_5m",
        "copy_pnl_5m_delay_5s",
        "copy_pnl_5m_delay_15s",
        "copy_pnl_5m_delay_30s",
        "copy_pnl_5m_delay_60s",
        "fade_pnl_5m_delay_5s",
        "fade_pnl_5m_delay_15s",
        "fade_pnl_5m_delay_30s",
        "fade_pnl_5m_delay_60s",
        "copy_pnl_net_5m_delay_5s",
        "copy_pnl_net_5m_delay_15s",
        "copy_pnl_net_5m_delay_30s",
        "copy_pnl_net_5m_delay_60s",
        "fade_pnl_net_5m_delay_5s",
        "fade_pnl_net_5m_delay_15s",
        "fade_pnl_net_5m_delay_30s",
        "fade_pnl_net_5m_delay_60s",
    ):
        if column not in frame.columns:
            frame[column] = pd.NA

    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        series = pd.Series(row)
        if pd.isna(series.get("timestamp")) or pd.isna(series.get("price")):
            records.append({"trade_id": row["trade_id"]})
            continue

        trade_ts = _to_epoch_seconds(pd.Timestamp(series["timestamp"]))
        price = float(series["price"])
        direction = 1.0 if str(series.get("side")).upper() == "BUY" else -1.0
        update: dict[str, Any] = {"trade_id": row["trade_id"]}

        baseline_cost = estimate_total_cost(series, entry_price=price, settings=cfg)
        for strategy in ("copy", "fade"):
            baseline_raw = _raw_baseline_pnl(series, strategy)
            update[_delay_column(strategy, delay_seconds=0, net=True)] = calculate_net_pnl(
                baseline_raw,
                baseline_cost["total_cost"],
            )

        for delay_seconds in (5, 15, 30, 60):
            entry_lookup = lookup_forward_price(
                price_index,
                str(series.get("token_id")) if pd.notna(series.get("token_id")) else None,
                trade_ts + delay_seconds,
            )
            exit_lookup = lookup_forward_price(
                price_index,
                str(series.get("token_id")) if pd.notna(series.get("token_id")) else None,
                trade_ts + delay_seconds + 5 * 60,
            )
            if entry_lookup.price is None or exit_lookup.price is None:
                update[_delay_column("copy", delay_seconds=delay_seconds, net=False)] = None
                update[_delay_column("fade", delay_seconds=delay_seconds, net=False)] = None
                update[_delay_column("copy", delay_seconds=delay_seconds, net=True)] = None
                update[_delay_column("fade", delay_seconds=delay_seconds, net=True)] = None
                continue

            copy_raw = direction * (float(exit_lookup.price) - float(entry_lookup.price))
            fade_raw = -copy_raw
            update[_delay_column("copy", delay_seconds=delay_seconds, net=False)] = copy_raw
            update[_delay_column("fade", delay_seconds=delay_seconds, net=False)] = fade_raw

            delayed_cost = estimate_total_cost(series, entry_price=float(entry_lookup.price), settings=cfg)
            update[_delay_column("copy", delay_seconds=delay_seconds, net=True)] = calculate_net_pnl(
                copy_raw,
                delayed_cost["total_cost"],
            )
            update[_delay_column("fade", delay_seconds=delay_seconds, net=True)] = calculate_net_pnl(
                fade_raw,
                delayed_cost["total_cost"],
            )

        records.append(update)

    updates = pd.DataFrame.from_records(records)
    if updates.empty:
        return frame
    merged = frame.drop(
        columns=[
            column
            for column in updates.columns
            if column != "trade_id" and column in frame.columns
        ]
    ).merge(updates, on="trade_id", how="left")
    return merged


def _delay_event_study(trades: pd.DataFrame) -> pd.DataFrame:
    """Build a long-form per-wallet delay event-study table."""

    rows: list[dict[str, Any]] = []
    for wallet, group in trades.groupby("wallet_address"):
        for strategy in ("copy", "fade"):
            for delay_seconds in DELAY_SECONDS:
                raw_col = _delay_column(strategy, delay_seconds=delay_seconds, net=False)
                net_col = _delay_column(strategy, delay_seconds=delay_seconds, net=True)
                avg_pnl = _mean_or_none(group, raw_col)
                avg_net_pnl = _mean_or_none(group, net_col)
                hit_rate = _positive_share(group, net_col)
                rows.append(
                    {
                        "wallet_address": wallet,
                        "strategy_mode": strategy,
                        "delay_seconds": delay_seconds,
                        "delay_label": DELAY_LABELS[delay_seconds],
                        "avg_pnl": avg_pnl,
                        "avg_net_pnl": avg_net_pnl,
                        "hit_rate": hit_rate,
                    }
                )
    return pd.DataFrame.from_records(rows)


def summarize_wallet_delay_metrics(
    trades: pd.DataFrame,
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Summarize delay, retention, break-even cost, and tradability by wallet."""

    cfg = settings or get_settings()
    if trades.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for wallet, group in trades.groupby("wallet_address"):
        avg_copy_raw_0 = _mean_or_none(group, "ret_5m")
        avg_fade_raw_0 = None if avg_copy_raw_0 is None else -avg_copy_raw_0
        avg_copy_net_0 = _mean_or_none(group, "copy_pnl_net_5m")
        avg_fade_net_0 = _mean_or_none(group, "fade_pnl_net_5m")
        avg_copy_net_15 = _mean_or_none(group, "copy_pnl_net_5m_delay_15s")
        avg_copy_net_30 = _mean_or_none(group, "copy_pnl_net_5m_delay_30s")
        avg_fade_net_15 = _mean_or_none(group, "fade_pnl_net_5m_delay_15s")
        avg_fade_net_30 = _mean_or_none(group, "fade_pnl_net_5m_delay_30s")

        dominant_mode = "copy"
        if avg_copy_raw_0 is None and avg_fade_raw_0 is not None:
            dominant_mode = "fade"
        elif avg_copy_raw_0 is not None and avg_fade_raw_0 is not None and avg_fade_raw_0 > avg_copy_raw_0:
            dominant_mode = "fade"

        dominant_base_raw = avg_copy_raw_0 if dominant_mode == "copy" else avg_fade_raw_0
        dominant_raw_15 = _mean_or_none(group, _delay_column(dominant_mode, delay_seconds=15, net=False))
        dominant_raw_30 = _mean_or_none(group, _delay_column(dominant_mode, delay_seconds=30, net=False))
        dominant_raw_60 = _mean_or_none(group, _delay_column(dominant_mode, delay_seconds=60, net=False))
        dominant_base_net = avg_copy_net_0 if dominant_mode == "copy" else avg_fade_net_0
        dominant_net_15 = avg_copy_net_15 if dominant_mode == "copy" else avg_fade_net_15
        dominant_net_30 = avg_copy_net_30 if dominant_mode == "copy" else avg_fade_net_30

        optimistic_series = group.apply(
            lambda row: calculate_net_pnl(
                _raw_baseline_pnl(row, dominant_mode),
                estimate_total_cost(row, entry_price=float(row["price"]), scenario="optimistic", settings=cfg)["total_cost"],
            ),
            axis=1,
        )
        optimistic_net = estimate_break_even_cost(optimistic_series)

        rows.append(
            {
                "wallet_address": wallet,
                "n_trades": int(len(group)),
                "n_markets": int(group["market_id"].nunique(dropna=True)),
                "dominant_mode": dominant_mode,
                "avg_copy_pnl_5m": avg_copy_raw_0,
                "avg_fade_pnl_5m": avg_fade_raw_0,
                "avg_copy_pnl_net_5m": avg_copy_net_0,
                "avg_fade_pnl_net_5m": avg_fade_net_0,
                "avg_copy_pnl_net_5m_delay_15s": avg_copy_net_15,
                "avg_copy_pnl_net_5m_delay_30s": avg_copy_net_30,
                "avg_fade_pnl_net_5m_delay_15s": avg_fade_net_15,
                "avg_fade_pnl_net_5m_delay_30s": avg_fade_net_30,
                "break_even_cost_copy_5m": estimate_break_even_cost(group["ret_5m"]),
                "break_even_cost_fade_5m": estimate_break_even_cost(-pd.to_numeric(group["ret_5m"], errors="coerce")),
                "delay_decay_rate": _simple_slope(
                    list(DELAY_SECONDS),
                    [
                        dominant_base_raw,
                        _mean_or_none(group, _delay_column(dominant_mode, delay_seconds=5, net=False)),
                        dominant_raw_15,
                        dominant_raw_30,
                        dominant_raw_60,
                    ],
                ),
                "edge_retention_15s": _retention(dominant_base_raw, dominant_raw_15),
                "edge_retention_30s": _retention(dominant_base_raw, dominant_raw_30),
                "edge_retention_60s": _retention(dominant_base_raw, dominant_raw_60),
                "tradability_label": classify_tradability(
                    base_net_pnl=dominant_base_net,
                    net_pnl_delay_15s=dominant_net_15,
                    net_pnl_delay_30s=dominant_net_30,
                    optimistic_net_pnl=optimistic_net,
                    settings=cfg,
                ),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("wallet_address").reset_index(drop=True)


def build_portfolio_delay_performance(
    trades: pd.DataFrame,
    selected_wallets: pd.DataFrame,
) -> pd.DataFrame:
    """Build equal-weight event-level portfolio summaries across delay settings."""

    if trades.empty or selected_wallets.empty:
        return pd.DataFrame(
            columns=[
                "strategy_mode",
                "delay_seconds",
                "delay_label",
                "n_wallets",
                "n_events",
                "avg_net_return",
                "hit_rate",
            ]
        )

    rows: list[dict[str, Any]] = []
    for strategy in ("copy", "fade"):
        wallets = selected_wallets[selected_wallets["recommended_mode"] == strategy]["wallet_address"].tolist()
        if not wallets:
            continue
        subset = trades[trades["wallet_address"].isin(wallets)].copy()
        for delay_seconds in DELAY_SECONDS:
            column = _delay_column(strategy, delay_seconds=delay_seconds, net=True)
            valid = pd.to_numeric(subset[column], errors="coerce").dropna() if column in subset.columns else pd.Series(dtype=float)
            rows.append(
                {
                    "strategy_mode": strategy,
                    "delay_seconds": delay_seconds,
                    "delay_label": DELAY_LABELS[delay_seconds],
                    "n_wallets": len(wallets),
                    "n_events": int(len(valid)),
                    "avg_net_return": float(valid.mean()) if not valid.empty else None,
                    "hit_rate": float((valid > 0).mean()) if not valid.empty else None,
                }
            )
    return pd.DataFrame.from_records(rows)


def persist_delay_metrics(session: Session, trades: pd.DataFrame) -> int:
    """Persist computed delay/net columns back into existing enriched rows."""

    tracked_columns = [
        "copy_pnl_5m_delay_5s",
        "copy_pnl_5m_delay_15s",
        "copy_pnl_5m_delay_30s",
        "copy_pnl_5m_delay_60s",
        "fade_pnl_5m_delay_5s",
        "fade_pnl_5m_delay_15s",
        "fade_pnl_5m_delay_30s",
        "fade_pnl_5m_delay_60s",
        "copy_pnl_net_5m",
        "fade_pnl_net_5m",
        "copy_pnl_net_5m_delay_5s",
        "copy_pnl_net_5m_delay_15s",
        "copy_pnl_net_5m_delay_30s",
        "copy_pnl_net_5m_delay_60s",
        "fade_pnl_net_5m_delay_5s",
        "fade_pnl_net_5m_delay_15s",
        "fade_pnl_net_5m_delay_30s",
        "fade_pnl_net_5m_delay_60s",
    ]
    available_columns = ["trade_id"] + [column for column in tracked_columns if column in trades.columns]
    if len(available_columns) == 1:
        return 0

    frame = trades[available_columns].copy()
    frame = frame.where(pd.notna(frame), None)
    payload = frame.to_dict(orient="records")
    if not payload:
        return 0

    updated = 0
    for start in range(0, len(payload), UPSERT_CHUNK_SIZE):
        chunk = payload[start:start + UPSERT_CHUNK_SIZE]
        session.bulk_update_mappings(WalletTradeEnriched, chunk)
        updated += len(chunk)
    session.flush()
    return updated


def run_delay_analysis(
    session: Session,
    *,
    output_dir: str | Path = "exports/delay_analysis",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run delay and cost analysis using the existing enriched dataset."""

    cfg = settings or get_settings()
    enriched = load_enriched_trades(session)
    if enriched.empty:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame()
        paths = {
            "trade_delay_diagnostics": _write_csv(empty, output_root / "trade_delay_diagnostics.csv"),
            "wallet_delay_event_study": _write_csv(empty, output_root / "wallet_delay_event_study.csv"),
            "wallet_delay_summary": _write_csv(empty, output_root / "wallet_delay_summary.csv"),
            "portfolio_delay_performance": _write_csv(empty, output_root / "portfolio_delay_performance.csv"),
        }
        return {
            "trades": empty,
            "wallet_delay_event_study": empty,
            "wallet_delay_summary": empty,
            "portfolio_delay_performance": empty,
            "paths": paths,
        }

    token_ids = sorted({str(token_id) for token_id in enriched["token_id"].dropna().tolist()})
    price_history = load_price_history_frame(session, token_ids=token_ids)
    trades = compute_delay_trade_metrics(enriched, price_history, settings=cfg)
    updated_rows = persist_delay_metrics(session, trades)
    # Release sqlite's write lock before the downstream scoring/event-study helpers
    # open fresh reads against wallet_trades_enriched.
    session.commit()
    LOGGER.info("Updated delay and net-PnL columns for %s enriched trades", updated_rows)

    wallet_delay_event_study = _delay_event_study(trades)
    wallet_delay_summary = summarize_wallet_delay_metrics(trades, settings=cfg)
    selected_wallets = score_wallets(session)
    portfolio_delay_performance = build_portfolio_delay_performance(trades, selected_wallets)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    from reports.delay_plots import generate_delay_analysis_plots

    plot_paths = generate_delay_analysis_plots(
        wallet_delay_summary=wallet_delay_summary,
        wallet_delay_event_study=wallet_delay_event_study,
        portfolio_delay_performance=portfolio_delay_performance,
        output_dir=output_root / "plots",
    )
    paths = {
        "trade_delay_diagnostics": _write_csv(trades, output_root / "trade_delay_diagnostics.csv"),
        "wallet_delay_event_study": _write_csv(wallet_delay_event_study, output_root / "wallet_delay_event_study.csv"),
        "wallet_delay_summary": _write_csv(wallet_delay_summary, output_root / "wallet_delay_summary.csv"),
        "portfolio_delay_performance": _write_csv(
            portfolio_delay_performance,
            output_root / "portfolio_delay_performance.csv",
        ),
    }

    return {
        "trades": trades,
        "wallet_delay_event_study": wallet_delay_event_study,
        "wallet_delay_summary": wallet_delay_summary,
        "portfolio_delay_performance": portfolio_delay_performance,
        "plot_paths": plot_paths,
        "paths": paths,
    }


def print_delay_summary(results: dict[str, Any]) -> None:
    """Print a concise console summary for delay analysis."""

    summary: pd.DataFrame = results["wallet_delay_summary"]
    portfolio: pd.DataFrame = results["portfolio_delay_performance"]

    print("Delay Analysis Summary")
    if summary.empty:
        print("No enriched trades are available for delay analysis.")
        return

    label_counts = (
        summary["tradability_label"]
        .value_counts()
        .rename_axis("tradability_label")
        .reset_index(name="wallet_count")
    )
    print("Tradability Labels")
    print(label_counts.to_string(index=False))
    print("Top Wallets by Break-Even Cost")
    print(
        summary[
            [
                "wallet_address",
                "dominant_mode",
                "break_even_cost_copy_5m",
                "break_even_cost_fade_5m",
                "edge_retention_30s",
                "tradability_label",
            ]
        ]
        .sort_values(
            by=["break_even_cost_copy_5m", "break_even_cost_fade_5m"],
            ascending=False,
            na_position="last",
        )
        .head(10)
        .to_string(index=False)
    )
    if not portfolio.empty:
        print("Portfolio Delay Performance")
        print(portfolio.to_string(index=False))
