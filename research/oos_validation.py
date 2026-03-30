"""Out-of-sample validation helpers built on top of enriched wallet trades."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from research.event_study import compute_event_study_from_frame, load_enriched_trades
from research.wallet_scoring import score_wallet_summary


HORIZONS = ("1m", "5m", "30m")


@dataclass(frozen=True)
class TimeSplitResult:
    """Train/test split metadata plus partitioned trade frames."""

    method: str
    cutoff_ts: pd.Timestamp | None
    train: pd.DataFrame
    test: pd.DataFrame


def _to_iso8601(series: pd.Series) -> pd.Series:
    """Convert datetime-like values to ISO strings for stable CSV output."""

    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamps.apply(lambda value: value.isoformat() if pd.notna(value) else None)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV file, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _safe_t_stat(series: pd.Series) -> float | None:
    """Simple t-stat helper for wallet-level summary values."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) < 5:
        return None
    std = valid.std(ddof=1)
    if pd.isna(std) or std == 0:
        return None
    return float(valid.mean() / (std / math.sqrt(len(valid))))


def _as_int(value: Any) -> int:
    """Convert scalars to int while treating null-like values as zero."""

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0
    return int(numeric)


def _classify_horizon_mode(copy_value: Any, fade_value: Any) -> str:
    """Map one horizon's average copy/fade PnL into copy, fade, or ignore."""

    copy_metric = pd.to_numeric(pd.Series([copy_value]), errors="coerce").iloc[0]
    fade_metric = pd.to_numeric(pd.Series([fade_value]), errors="coerce").iloc[0]
    if pd.isna(copy_metric) and pd.isna(fade_metric):
        return "ignore"
    if pd.notna(copy_metric) and copy_metric > 0 and (pd.isna(fade_metric) or copy_metric > fade_metric):
        return "copy"
    if pd.notna(fade_metric) and fade_metric > 0 and (pd.isna(copy_metric) or fade_metric > copy_metric):
        return "fade"
    return "ignore"


def label_stability(
    train_mode: str | None,
    test_mode: str | None,
    train_n_trades: int,
    test_n_trades: int,
    test_n_markets: int,
    settings: Settings | None = None,
) -> str:
    """Assign a stability label using train/test mode agreement and sample sufficiency."""

    cfg = settings or get_settings()
    if train_n_trades < cfg.oos_train_min_trades:
        return "insufficient_data"
    if test_n_trades < cfg.oos_test_min_trades or test_n_markets < cfg.oos_test_min_markets:
        return "insufficient_data"
    if train_mode == test_mode:
        return "stable"
    return "unstable"


def split_enriched_frame(
    frame: pd.DataFrame,
    *,
    split_date: str | None = None,
    train_fraction: float = 0.70,
) -> TimeSplitResult:
    """Split enriched trades into train/test partitions using strict time ordering.

    For fraction-based splits, the cutoff is moved to the first timestamp in the test
    partition and the train set uses timestamps strictly earlier than that cutoff.
    This prevents same-timestamp rows from leaking across the boundary.
    """

    data = frame.copy()
    if data.empty:
        return TimeSplitResult(method="empty", cutoff_ts=None, train=data, test=data)

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values(["timestamp", "trade_id"]).reset_index(drop=True)
    if data.empty:
        return TimeSplitResult(method="empty", cutoff_ts=None, train=data, test=data)

    if split_date:
        cutoff = pd.Timestamp(split_date, tz="UTC")
        train = data[data["timestamp"] < cutoff].copy()
        test = data[data["timestamp"] >= cutoff].copy()
        return TimeSplitResult(method=f"date:{cutoff.isoformat()}", cutoff_ts=cutoff, train=train, test=test)

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    if len(data) == 1:
        cutoff = data.iloc[0]["timestamp"]
        return TimeSplitResult(method=f"fraction:{train_fraction:.2f}", cutoff_ts=cutoff, train=data.iloc[:0].copy(), test=data.copy())

    raw_boundary = max(1, min(len(data) - 1, int(math.ceil(len(data) * train_fraction))))
    cutoff = data.iloc[raw_boundary]["timestamp"]
    train = data[data["timestamp"] < cutoff].copy()
    test = data[data["timestamp"] >= cutoff].copy()
    return TimeSplitResult(method=f"fraction:{train_fraction:.2f}", cutoff_ts=cutoff, train=train, test=test)


def _merge_metrics_and_scores(metrics: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Combine event-study metrics and wallet scores for one partition."""

    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "wallet_address",
                "n_trades",
                "n_markets",
                "most_recent_trade",
                "avg_ret_1m",
                "avg_ret_5m",
                "avg_ret_30m",
                "avg_copy_pnl_1m",
                "avg_copy_pnl_5m",
                "avg_copy_pnl_30m",
                "avg_fade_pnl_1m",
                "avg_fade_pnl_5m",
                "avg_fade_pnl_30m",
                "copy_hit_rate_1m",
                "copy_hit_rate_5m",
                "copy_hit_rate_30m",
                "fade_hit_rate_1m",
                "fade_hit_rate_5m",
                "fade_hit_rate_30m",
                "fraction_top_market",
                "overall_copy_score",
                "overall_fade_score",
                "score_confidence",
                "recommended_mode",
            ]
        )

    merged = metrics.merge(
        scores[
            [
                "wallet_address",
                "overall_copy_score",
                "overall_fade_score",
                "score_confidence",
                "recommended_mode",
            ]
        ],
        on="wallet_address",
        how="left",
    )
    for horizon in HORIZONS:
        merged[f"mode_{horizon}"] = merged.apply(
            lambda row: _classify_horizon_mode(
                row.get(f"avg_copy_pnl_{horizon}"),
                row.get(f"avg_fade_pnl_{horizon}"),
            ),
            axis=1,
        )
    return merged.sort_values(["wallet_address"]).reset_index(drop=True)


def _apply_train_eligibility(
    metrics: pd.DataFrame,
    split: TimeSplitResult,
    settings: Settings,
) -> pd.DataFrame:
    """Flag train wallets that meet the configurable selection requirements."""

    result = metrics.copy()
    train_end = split.train["timestamp"].max() if not split.train.empty else None
    most_recent_trade = pd.to_datetime(result["most_recent_trade"], utc=True, errors="coerce")

    result["passes_min_trades"] = result["n_trades"].fillna(0) >= settings.oos_train_min_trades
    result["passes_min_markets"] = result["n_markets"].fillna(0) >= settings.oos_train_min_markets
    result["passes_concentration"] = (
        result["fraction_top_market"].fillna(1.0) <= settings.oos_train_max_top_market_fraction
    )
    if train_end is None or pd.isna(train_end):
        result["passes_recent_activity"] = False
    else:
        recent_cutoff = train_end - pd.Timedelta(days=settings.oos_recent_activity_days)
        result["passes_recent_activity"] = most_recent_trade >= recent_cutoff

    result["is_eligible_for_selection"] = (
        result["passes_min_trades"]
        & result["passes_min_markets"]
        & result["passes_concentration"]
        & result["passes_recent_activity"]
    )
    return result


def _select_train_wallets(
    train_metrics: pd.DataFrame,
    *,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select top copy and fade wallets from the eligible train set only."""

    eligible = train_metrics[train_metrics["is_eligible_for_selection"]].copy()
    copy = (
        eligible[eligible["recommended_mode"] == "copy"]
        .sort_values(["overall_copy_score", "avg_copy_pnl_5m", "wallet_address"], ascending=[False, False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    fade = (
        eligible[eligible["recommended_mode"] == "fade"]
        .sort_values(["overall_fade_score", "avg_fade_pnl_5m", "wallet_address"], ascending=[False, False, True])
        .head(top_n)
        .reset_index(drop=True)
    )
    return copy, fade


def _prefix_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Add a prefix to all columns except wallet address."""

    renamed = {}
    for column in frame.columns:
        if column == "wallet_address":
            continue
        renamed[column] = f"{prefix}{column}"
    return frame.rename(columns=renamed)


def _warning_flags(row: pd.Series, settings: Settings) -> str:
    """Build compact warning flags for selected-wallet diagnostics."""

    flags: list[str] = []
    if pd.notna(row.get("train_fraction_top_market")) and row["train_fraction_top_market"] > settings.oos_train_max_top_market_fraction:
        flags.append("high_train_concentration")
    if pd.notna(row.get("test_fraction_top_market")) and row["test_fraction_top_market"] > settings.oos_train_max_top_market_fraction:
        flags.append("high_test_concentration")
    if pd.notna(row.get("test_n_trades")) and row["test_n_trades"] < settings.oos_test_min_trades:
        flags.append("low_test_trades")
    if pd.notna(row.get("test_n_markets")) and row["test_n_markets"] < settings.oos_test_min_markets:
        flags.append("low_test_markets")
    if pd.notna(row.get("test_depends_on_single_market")) and bool(row["test_depends_on_single_market"]):
        flags.append("test_single_market_dependence")
    return ";".join(flags)


def _build_comparison(
    train_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    selected_copy: pd.DataFrame,
    selected_fade: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Combine train/test metrics and attach stability and warning labels."""

    comparison = _prefix_columns(train_metrics, "train_").merge(
        _prefix_columns(test_metrics, "test_"),
        on="wallet_address",
        how="outer",
    )

    selection_map = {wallet: "copy" for wallet in selected_copy["wallet_address"].tolist()}
    selection_map.update({wallet: "fade" for wallet in selected_fade["wallet_address"].tolist()})
    comparison["selected_group"] = comparison["wallet_address"].map(selection_map)
    comparison["test_depends_on_single_market"] = (
        comparison["test_fraction_top_market"].fillna(1.0) > settings.oos_train_max_top_market_fraction
    )
    comparison["stability_label"] = comparison.apply(
        lambda row: label_stability(
            train_mode=row.get("train_mode_5m"),
            test_mode=row.get("test_mode_5m"),
            train_n_trades=_as_int(row.get("train_n_trades")),
            test_n_trades=_as_int(row.get("test_n_trades")),
            test_n_markets=_as_int(row.get("test_n_markets")),
            settings=settings,
        ),
        axis=1,
    )
    comparison["warning_flags"] = comparison.apply(lambda row: _warning_flags(row, settings), axis=1)
    return comparison.sort_values(["wallet_address"]).reset_index(drop=True)


def _build_selected_test_results(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return the selected-wallet test result table requested by the user."""

    result = comparison[comparison["selected_group"].isin(["copy", "fade"])].copy()
    if result.empty:
        return pd.DataFrame(
            columns=[
                "wallet_address",
                "selected_group",
                "train_recommended_mode",
                "test_recommended_mode",
                "train_fraction_top_market",
                "test_fraction_top_market",
                "test_depends_on_single_market",
                "train_n_trades",
                "test_n_trades",
                "train_n_markets",
                "test_n_markets",
                "train_mode_1m",
                "train_mode_5m",
                "train_mode_30m",
                "test_mode_1m",
                "test_mode_5m",
                "test_mode_30m",
                "stability_label",
                "warning_flags",
            ]
        )

    result["train_selected_hit_rate_5m"] = result.apply(
        lambda row: row["train_copy_hit_rate_5m"]
        if row["selected_group"] == "copy"
        else row["train_fade_hit_rate_5m"],
        axis=1,
    )
    result["test_selected_hit_rate_5m"] = result.apply(
        lambda row: row["test_copy_hit_rate_5m"]
        if row["selected_group"] == "copy"
        else row["test_fade_hit_rate_5m"],
        axis=1,
    )
    return result[
        [
            "wallet_address",
            "selected_group",
            "train_recommended_mode",
            "test_recommended_mode",
            "train_fraction_top_market",
            "test_fraction_top_market",
            "test_depends_on_single_market",
            "train_n_trades",
            "test_n_trades",
            "train_n_markets",
            "test_n_markets",
            "train_avg_copy_pnl_5m",
            "test_avg_copy_pnl_5m",
            "train_avg_fade_pnl_5m",
            "test_avg_fade_pnl_5m",
            "train_selected_hit_rate_5m",
            "test_selected_hit_rate_5m",
            "train_mode_1m",
            "train_mode_5m",
            "train_mode_30m",
            "test_mode_1m",
            "test_mode_5m",
            "test_mode_30m",
            "stability_label",
            "warning_flags",
        ]
    ]


def _build_portfolio_summary(selected_results: pd.DataFrame, strategy_mode: str) -> pd.DataFrame:
    """Summarize selected-wallet test performance with equal wallet weights.

    The MVP uses equal weight across wallets by summarizing wallet-level mean event
    returns, not a synchronized tradable portfolio time series.
    """

    frame = selected_results[selected_results["selected_group"] == strategy_mode].copy()
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        pnl_col = f"test_avg_{strategy_mode}_pnl_{horizon}"
        hit_col = f"test_{strategy_mode}_hit_rate_{horizon}"
        if pnl_col not in frame.columns:
            pnl_col = f"test_avg_{strategy_mode}_pnl_5m"
        wallet_means = pd.to_numeric(frame.get(pnl_col, pd.Series(dtype=float)), errors="coerce").dropna()
        wallet_hits = pd.to_numeric(frame.get(hit_col, pd.Series(dtype=float)), errors="coerce").dropna()
        event_count = int(pd.to_numeric(frame.get("test_n_trades", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        rows.append(
            {
                "strategy_mode": strategy_mode,
                "horizon": horizon,
                "n_wallets": int(frame["wallet_address"].nunique()) if not frame.empty else 0,
                "n_events_total": event_count,
                "avg_event_return": float(wallet_means.mean()) if not wallet_means.empty else None,
                "median_event_return": float(wallet_means.median()) if not wallet_means.empty else None,
                "hit_rate": float(wallet_hits.mean()) if not wallet_hits.empty else None,
                "std_event_return": float(wallet_means.std(ddof=1)) if len(wallet_means) >= 2 else None,
                "t_stat_event_return": _safe_t_stat(wallet_means),
            }
        )
    return pd.DataFrame.from_records(rows)


def _stringify_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime-like columns in a frame to ISO strings for export."""

    result = frame.copy()
    for column in result.columns:
        if column.endswith("most_recent_trade") or column == "timestamp" or column.endswith("_ts"):
            try:
                converted = pd.to_datetime(result[column], utc=True, errors="coerce")
            except (TypeError, ValueError):
                continue
            if converted.notna().any():
                result[column] = _to_iso8601(result[column])
    return result


def run_oos_validation_from_frame(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path = "exports/oos_validation",
    split_date: str | None = None,
    train_fraction: float | None = None,
    top_n: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the full OOS validation workflow from an in-memory enriched trade frame."""

    cfg = settings or get_settings()
    effective_fraction = cfg.oos_train_fraction if train_fraction is None else train_fraction
    split = split_enriched_frame(frame, split_date=split_date, train_fraction=effective_fraction)
    return run_oos_validation_on_split(
        split,
        output_dir=output_dir,
        top_n=top_n,
        settings=cfg,
    )


def run_oos_validation_on_split(
    split: TimeSplitResult,
    *,
    output_dir: str | Path = "exports/oos_validation",
    top_n: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the OOS workflow for a pre-computed train/test split."""

    cfg = settings or get_settings()
    effective_top_n = cfg.oos_select_top_n if top_n is None else top_n

    train_summary, _ = compute_event_study_from_frame(split.train, stringify_datetimes=False)
    test_summary, _ = compute_event_study_from_frame(split.test, stringify_datetimes=False)
    train_scores = score_wallet_summary(train_summary)
    test_scores = score_wallet_summary(test_summary)
    train_metrics = _apply_train_eligibility(_merge_metrics_and_scores(train_summary, train_scores), split, cfg)
    test_metrics = _merge_metrics_and_scores(test_summary, test_scores)
    selected_copy, selected_fade = _select_train_wallets(train_metrics, top_n=effective_top_n)
    comparison = _build_comparison(train_metrics, test_metrics, selected_copy, selected_fade, cfg)
    selected_test = _build_selected_test_results(comparison)

    test_copy_portfolio = _build_portfolio_summary(
        comparison[comparison["selected_group"] == "copy"],
        strategy_mode="copy",
    )
    test_fade_portfolio = _build_portfolio_summary(
        comparison[comparison["selected_group"] == "fade"],
        strategy_mode="fade",
    )

    export_root = Path(output_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    from reports.oos_plots import generate_oos_validation_plots

    plot_paths = generate_oos_validation_plots(
        selected_results=selected_test,
        copy_portfolio=test_copy_portfolio,
        fade_portfolio=test_fade_portfolio,
        output_dir=export_root / "plots",
    )
    paths = {
        "wallet_train_metrics": _write_csv(
            _stringify_datetime_columns(train_metrics),
            export_root / "wallet_train_metrics.csv",
        ),
        "wallet_test_metrics": _write_csv(
            _stringify_datetime_columns(test_metrics),
            export_root / "wallet_test_metrics.csv",
        ),
        "wallet_oos_comparison": _write_csv(
            _stringify_datetime_columns(comparison),
            export_root / "wallet_oos_comparison.csv",
        ),
        "train_selected_copy_wallets": _write_csv(
            _stringify_datetime_columns(selected_copy),
            export_root / "train_selected_copy_wallets.csv",
        ),
        "train_selected_fade_wallets": _write_csv(
            _stringify_datetime_columns(selected_fade),
            export_root / "train_selected_fade_wallets.csv",
        ),
        "selected_wallets_test_results": _write_csv(
            _stringify_datetime_columns(selected_test),
            export_root / "selected_wallets_test_results.csv",
        ),
        "test_portfolio_copy_summary": _write_csv(
            test_copy_portfolio,
            export_root / "test_portfolio_copy_summary.csv",
        ),
        "test_portfolio_fade_summary": _write_csv(
            test_fade_portfolio,
            export_root / "test_portfolio_fade_summary.csv",
        ),
    }

    return {
        "split": split,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "comparison": comparison,
        "selected_copy": selected_copy,
        "selected_fade": selected_fade,
        "selected_test": selected_test,
        "test_copy_portfolio": test_copy_portfolio,
        "test_fade_portfolio": test_fade_portfolio,
        "plot_paths": plot_paths,
        "paths": paths,
    }


def run_oos_validation(
    session: Session,
    *,
    output_dir: str | Path = "exports/oos_validation",
    split_date: str | None = None,
    train_fraction: float | None = None,
    top_n: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the OOS workflow from the enriched-trades table."""

    frame = load_enriched_trades(session)
    return run_oos_validation_from_frame(
        frame,
        output_dir=output_dir,
        split_date=split_date,
        train_fraction=train_fraction,
        top_n=top_n,
        settings=settings,
    )


def print_oos_summary(results: dict[str, Any]) -> None:
    """Print a concise console summary for OOS validation."""

    split: TimeSplitResult = results["split"]
    selected_copy: pd.DataFrame = results["selected_copy"]
    selected_fade: pd.DataFrame = results["selected_fade"]
    selected_test: pd.DataFrame = results["selected_test"]
    copy_portfolio: pd.DataFrame = results["test_copy_portfolio"]
    fade_portfolio: pd.DataFrame = results["test_fade_portfolio"]

    print("OOS Validation Summary")
    print(f"Split method: {split.method}")
    print(
        f"Train trades: {len(split.train)} | Test trades: {len(split.test)} | "
        f"Cutoff: {split.cutoff_ts.isoformat() if split.cutoff_ts is not None else 'n/a'}"
    )
    print(f"Selected copy wallets: {len(selected_copy)} | Selected fade wallets: {len(selected_fade)}")

    if not selected_test.empty:
        stability = (
            selected_test["stability_label"]
            .fillna("unknown")
            .value_counts()
            .rename_axis("stability_label")
            .reset_index(name="wallet_count")
        )
        print("Selected Wallet Stability")
        print(stability.to_string(index=False))
    else:
        print("Selected Wallet Stability")
        print("No wallets passed training-time selection.")

    for label, portfolio in (("Copy", copy_portfolio), ("Fade", fade_portfolio)):
        print(f"Test Portfolio {label}")
        if portfolio.empty:
            print("No selected wallets.")
            continue
        print(portfolio.to_string(index=False))
