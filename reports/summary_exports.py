"""Console summaries and CSV exports for small-batch research runs."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.api_contracts import endpoint_audit_rows
from db.models import WalletTradeRaw
from research.event_study import compute_event_study_outputs
from research.wallet_scoring import score_wallets


def _ensure_output_dir(output_dir: str | Path) -> Path:
    """Create and return the export directory."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_iso8601(series: pd.Series) -> pd.Series:
    """Convert datetime-like values to ISO-8601 strings without pandas string dtype."""

    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamps.apply(lambda value: value.isoformat() if pd.notna(value) else None)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Persist a DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def export_endpoint_audit(output_dir: str | Path = "exports") -> Path:
    """Export the centralized endpoint audit table."""

    output_path = _ensure_output_dir(output_dir) / "endpoint_audit.csv"
    frame = pd.DataFrame.from_records(endpoint_audit_rows())
    return _write_csv(frame, output_path)


def build_raw_wallet_activity_summary(session: Session) -> pd.DataFrame:
    """Summarize raw wallet activity counts for inspection."""

    raw = pd.read_sql(select(WalletTradeRaw), session.bind)
    columns = ["wallet_address", "n_trades", "n_markets", "most_recent_trade"]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    summary = (
        raw.groupby("wallet_address", dropna=False)
        .agg(
            n_trades=("trade_id", "count"),
            n_markets=("market_id", lambda series: series.dropna().nunique()),
            most_recent_trade=("timestamp", "max"),
        )
        .reset_index()
        .sort_values(by=["n_trades", "wallet_address"], ascending=[False, True])
    )
    summary["most_recent_trade"] = _to_iso8601(summary["most_recent_trade"])
    return summary[columns]


def build_wallet_diagnostics(session: Session) -> pd.DataFrame:
    """Build the per-wallet diagnostics export requested for research review."""

    raw_summary = build_raw_wallet_activity_summary(session)
    event_summary, _ = compute_event_study_outputs(session)
    score_summary = score_wallets(session)

    diagnostics = raw_summary.copy()
    if diagnostics.empty:
        return pd.DataFrame(
            columns=[
                "wallet_address",
                "n_trades",
                "n_markets",
                "most_recent_trade",
                "avg_ret_1m",
                "avg_ret_5m",
                "avg_ret_30m",
                "avg_copy_pnl_5m",
                "avg_fade_pnl_5m",
                "median_copy_pnl_5m",
                "median_fade_pnl_5m",
                "fraction_top_market",
                "overall_copy_score",
                "overall_fade_score",
                "recommended_mode",
                "score_confidence",
            ]
        )

    if not event_summary.empty:
        diagnostics = diagnostics.merge(
            event_summary[
                [
                    "wallet_address",
                    "avg_ret_1m",
                    "avg_ret_5m",
                    "avg_ret_30m",
                    "avg_copy_pnl_5m",
                    "avg_fade_pnl_5m",
                    "median_copy_pnl_5m",
                    "median_fade_pnl_5m",
                    "fraction_top_market",
                ]
            ],
            on="wallet_address",
            how="left",
        )
    else:
        for column in (
            "avg_ret_1m",
            "avg_ret_5m",
            "avg_ret_30m",
            "avg_copy_pnl_5m",
            "avg_fade_pnl_5m",
            "median_copy_pnl_5m",
            "median_fade_pnl_5m",
            "fraction_top_market",
        ):
            diagnostics[column] = pd.NA

    if not score_summary.empty:
        diagnostics = diagnostics.merge(
            score_summary[
                [
                    "wallet_address",
                    "overall_copy_score",
                    "overall_fade_score",
                    "recommended_mode",
                    "score_confidence",
                ]
            ],
            on="wallet_address",
            how="left",
        )
    else:
        diagnostics["overall_copy_score"] = pd.NA
        diagnostics["overall_fade_score"] = pd.NA
        diagnostics["recommended_mode"] = pd.NA
        diagnostics["score_confidence"] = pd.NA

    return diagnostics[
        [
            "wallet_address",
            "n_trades",
            "n_markets",
            "most_recent_trade",
            "avg_ret_1m",
            "avg_ret_5m",
            "avg_ret_30m",
            "avg_copy_pnl_5m",
            "avg_fade_pnl_5m",
            "median_copy_pnl_5m",
            "median_fade_pnl_5m",
            "fraction_top_market",
            "overall_copy_score",
            "overall_fade_score",
            "recommended_mode",
            "score_confidence",
        ]
    ]


def export_backfill_dry_run_preview(
    preview: dict[str, list[dict[str, Any]]],
    output_dir: str | Path = "exports",
) -> dict[str, Path]:
    """Write dry-run wallet, market, and token target CSVs."""

    directory = _ensure_output_dir(output_dir)
    outputs = {
        "wallets": _write_csv(
            pd.DataFrame.from_records(preview.get("wallets", [])),
            directory / "dry_run_wallets.csv",
        ),
        "market_targets": _write_csv(
            pd.DataFrame.from_records(preview.get("market_targets", [])),
            directory / "dry_run_market_targets.csv",
        ),
        "token_targets": _write_csv(
            pd.DataFrame.from_records(preview.get("token_targets", [])),
            directory / "dry_run_token_targets.csv",
        ),
    }
    return outputs


def export_research_summary(
    session: Session,
    output_dir: str | Path = "exports",
) -> dict[str, pd.DataFrame | Path]:
    """Export wallet activity, diagnostics, top-score tables, and mode counts."""

    directory = _ensure_output_dir(output_dir)
    raw_summary = build_raw_wallet_activity_summary(session)
    diagnostics = build_wallet_diagnostics(session)
    scores = score_wallets(session)

    top_copy = (
        diagnostics.dropna(subset=["overall_copy_score"])
        .sort_values(by="overall_copy_score", ascending=False)
        .head(10)[
            ["wallet_address", "overall_copy_score", "recommended_mode", "score_confidence", "n_trades", "n_markets"]
        ]
        if not diagnostics.empty
        else pd.DataFrame(
            columns=["wallet_address", "overall_copy_score", "recommended_mode", "score_confidence", "n_trades", "n_markets"]
        )
    )
    top_fade = (
        diagnostics.dropna(subset=["overall_fade_score"])
        .sort_values(by="overall_fade_score", ascending=False)
        .head(10)[
            ["wallet_address", "overall_fade_score", "recommended_mode", "score_confidence", "n_trades", "n_markets"]
        ]
        if not diagnostics.empty
        else pd.DataFrame(
            columns=["wallet_address", "overall_fade_score", "recommended_mode", "score_confidence", "n_trades", "n_markets"]
        )
    )

    if scores.empty:
        mode_counts = pd.DataFrame(
            {
                "recommended_mode": ["copy", "fade", "ignore"],
                "wallet_count": [0, 0, 0],
            }
        )
    else:
        counts = scores["recommended_mode"].value_counts()
        mode_counts = pd.DataFrame(
            {
                "recommended_mode": ["copy", "fade", "ignore"],
                "wallet_count": [int(counts.get("copy", 0)), int(counts.get("fade", 0)), int(counts.get("ignore", 0))],
            }
        )

    outputs: dict[str, pd.DataFrame | Path] = {
        "raw_summary": raw_summary,
        "diagnostics": diagnostics,
        "top_copy": top_copy,
        "top_fade": top_fade,
        "mode_counts": mode_counts,
        "raw_summary_path": _write_csv(raw_summary, directory / "wallet_activity_summary.csv"),
        "diagnostics_path": _write_csv(diagnostics, directory / "wallet_diagnostics.csv"),
        "top_copy_path": _write_csv(top_copy, directory / "top_copy_wallets.csv"),
        "top_fade_path": _write_csv(top_fade, directory / "top_fade_wallets.csv"),
        "mode_counts_path": _write_csv(mode_counts, directory / "recommended_mode_counts.csv"),
        "endpoint_audit_path": export_endpoint_audit(output_dir=directory),
    }
    return outputs


def print_backfill_dry_run_preview(preview: dict[str, list[dict[str, Any]]]) -> None:
    """Print a concise dry-run preview for small wallet batches."""

    wallets = pd.DataFrame.from_records(preview.get("wallets", []))
    market_targets = pd.DataFrame.from_records(preview.get("market_targets", []))
    token_targets = pd.DataFrame.from_records(preview.get("token_targets", []))

    print("Dry-Run Preview")
    print(f"Wallets to fetch: {len(wallets)}")
    if not wallets.empty:
        print(wallets.to_string(index=False))
    print(f"Market metadata targets: {len(market_targets)}")
    if not market_targets.empty:
        print(market_targets.head(10).to_string(index=False))
    print(f"Token price history targets: {len(token_targets)}")
    if not token_targets.empty:
        print(token_targets.head(10).to_string(index=False))


def print_research_summary(summary: dict[str, pd.DataFrame | Path], title: str) -> None:
    """Print raw wallet activity plus score-driven summary tables."""

    raw_summary = summary["raw_summary"]
    top_copy = summary["top_copy"]
    top_fade = summary["top_fade"]
    mode_counts = summary["mode_counts"]

    print(title)
    print("Wallet Activity")
    if isinstance(raw_summary, pd.DataFrame) and not raw_summary.empty:
        print(raw_summary.to_string(index=False))
    else:
        print("No wallet activity available.")

    print("Top Copy Wallets")
    if isinstance(top_copy, pd.DataFrame) and not top_copy.empty:
        print(top_copy.to_string(index=False))
    else:
        print("Copy-score ranking unavailable until enrichment/scoring exists.")

    print("Top Fade Wallets")
    if isinstance(top_fade, pd.DataFrame) and not top_fade.empty:
        print(top_fade.to_string(index=False))
    else:
        print("Fade-score ranking unavailable until enrichment/scoring exists.")

    print("Recommended Mode Counts")
    if isinstance(mode_counts, pd.DataFrame):
        print(mode_counts.to_string(index=False))
