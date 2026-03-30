"""Wallet-level event study summaries."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import WalletTradeEnriched


SUMMARY_COLUMNS = [
    "wallet_address",
    "n_trades",
    "n_markets",
    "most_recent_trade",
    "avg_ret_1m",
    "avg_ret_5m",
    "avg_ret_30m",
    "median_ret_5m",
    "std_ret_5m",
    "t_stat_ret_5m",
    "avg_copy_pnl_1m",
    "avg_copy_pnl_5m",
    "avg_copy_pnl_30m",
    "avg_fade_pnl_1m",
    "avg_fade_pnl_5m",
    "avg_fade_pnl_30m",
    "median_copy_pnl_5m",
    "median_fade_pnl_5m",
    "copy_hit_rate_1m",
    "copy_hit_rate_5m",
    "copy_hit_rate_30m",
    "fade_hit_rate_1m",
    "fade_hit_rate_5m",
    "fade_hit_rate_30m",
    "fraction_top_market",
]
EXTRA_DELAY_SUMMARY_COLUMNS = [
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
]


def load_enriched_trades(session: Session) -> pd.DataFrame:
    """Load enriched trades into a DataFrame."""

    query = select(WalletTradeEnriched)
    return pd.read_sql(query, session.bind)


def _to_iso8601(series: pd.Series) -> pd.Series:
    """Convert datetime-like values to ISO-8601 strings without pandas string dtype."""

    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamps.apply(lambda value: value.isoformat() if pd.notna(value) else None)


def _numeric(series: pd.Series) -> pd.Series:
    """Coerce a pandas Series to numeric values."""

    return pd.to_numeric(series, errors="coerce")


def _hit_rate(series: pd.Series) -> float | None:
    """Return the fraction of positive outcomes."""

    valid = _numeric(series).dropna()
    if valid.empty:
        return None
    return float((valid > 0).mean())


def _t_stat(series: pd.Series) -> float | None:
    """Compute a lightweight t-stat style summary when data is sufficient."""

    valid = _numeric(series).dropna()
    if len(valid) < 5:
        return None
    std = valid.std(ddof=1)
    if pd.isna(std) or std == 0:
        return None
    return float(valid.mean() / (std / (len(valid) ** 0.5)))


def _prepare_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort a trade-level frame before aggregation."""

    diagnostics = frame.copy()
    if diagnostics.empty:
        return diagnostics
    diagnostics = diagnostics.sort_values(["wallet_address", "timestamp", "trade_id"]).reset_index(drop=True)
    diagnostics["timestamp"] = pd.to_datetime(diagnostics["timestamp"], utc=True, errors="coerce")
    return diagnostics


def compute_event_study_from_frame(
    frame: pd.DataFrame,
    *,
    stringify_datetimes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute wallet summary and trade-level diagnostics from an in-memory frame."""

    diagnostics = _prepare_diagnostics(frame)
    if diagnostics.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS), diagnostics

    records: list[dict[str, Any]] = []
    for wallet, group in diagnostics.groupby("wallet_address"):
        top_market_fraction = (
            float(group["market_id"].value_counts(normalize=True, dropna=True).max())
            if group["market_id"].notna().any()
            else 1.0
        )
        ret_5m = _numeric(group["ret_5m"]).dropna()
        record = {
            "wallet_address": wallet,
            "n_trades": int(len(group)),
            "n_markets": int(group["market_id"].nunique(dropna=True)),
            "most_recent_trade": group["timestamp"].max(),
            "avg_ret_1m": _numeric(group["ret_1m"]).mean(),
            "avg_ret_5m": _numeric(group["ret_5m"]).mean(),
            "avg_ret_30m": _numeric(group["ret_30m"]).mean(),
            "median_ret_5m": _numeric(group["ret_5m"]).median(),
            "std_ret_5m": float(ret_5m.std(ddof=1)) if len(ret_5m) >= 2 else None,
            "t_stat_ret_5m": _t_stat(group["ret_5m"]),
            "avg_copy_pnl_1m": _numeric(group["copy_pnl_1m"]).mean(),
            "avg_copy_pnl_5m": _numeric(group["copy_pnl_5m"]).mean(),
            "avg_copy_pnl_30m": _numeric(group["copy_pnl_30m"]).mean(),
            "avg_fade_pnl_1m": _numeric(group["fade_pnl_1m"]).mean(),
            "avg_fade_pnl_5m": _numeric(group["fade_pnl_5m"]).mean(),
            "avg_fade_pnl_30m": _numeric(group["fade_pnl_30m"]).mean(),
            "median_copy_pnl_5m": _numeric(group["copy_pnl_5m"]).median(),
            "median_fade_pnl_5m": _numeric(group["fade_pnl_5m"]).median(),
            "copy_hit_rate_1m": _hit_rate(group["copy_pnl_1m"]),
            "copy_hit_rate_5m": _hit_rate(group["copy_pnl_5m"]),
            "copy_hit_rate_30m": _hit_rate(group["copy_pnl_30m"]),
            "fade_hit_rate_1m": _hit_rate(group["fade_pnl_1m"]),
            "fade_hit_rate_5m": _hit_rate(group["fade_pnl_5m"]),
            "fade_hit_rate_30m": _hit_rate(group["fade_pnl_30m"]),
            "fraction_top_market": top_market_fraction,
        }
        for column in EXTRA_DELAY_SUMMARY_COLUMNS:
            if column not in group.columns:
                continue
            record[f"avg_{column}"] = _numeric(group[column]).mean()
            record[f"hit_rate_{column}"] = _hit_rate(group[column])
        records.append(record)

    summary = pd.DataFrame.from_records(records).sort_values(
        by=["avg_copy_pnl_5m", "avg_fade_pnl_5m", "wallet_address"],
        ascending=[False, False, True],
        na_position="last",
    )
    if stringify_datetimes:
        summary["most_recent_trade"] = _to_iso8601(summary["most_recent_trade"])
        diagnostics["timestamp"] = _to_iso8601(diagnostics["timestamp"])
    return summary, diagnostics


def compute_event_study_outputs(session: Session) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute wallet summary and trade-level diagnostic outputs."""

    return compute_event_study_from_frame(load_enriched_trades(session), stringify_datetimes=True)


def build_event_study(session: Session, output_dir: str | Path = "artifacts/event_study") -> pd.DataFrame:
    """Aggregate wallet-level event study results and persist them as CSV."""

    summary, diagnostics = compute_event_study_outputs(session)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _ = diagnostics  # keep name explicit for readability below
    summary.to_csv(output_path / "wallet_event_study_summary.csv", index=False)
    diagnostics.to_csv(output_path / "wallet_trade_diagnostics.csv", index=False)
    return summary
