"""Robustness testing across multiple out-of-sample split schemes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any, Iterable

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from research.delay_analysis import DELAY_LABELS, DELAY_SECONDS, classify_tradability
from research.event_study import load_enriched_trades
from research.oos_validation import (
    TimeSplitResult,
    run_oos_validation_on_split,
    split_enriched_frame,
)


@dataclass(frozen=True)
class MultiSplitSpec:
    """One split configuration used by multi-run OOS validation."""

    split_id: str
    split_kind: str
    train_fraction: float | None = None
    cutoff_ts: pd.Timestamp | None = None


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort and normalize enriched trades before generating split specs."""

    data = frame.copy()
    if data.empty:
        return data
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values(["timestamp", "trade_id"]).reset_index(drop=True)
    return data


def _fraction_label(value: float) -> str:
    """Turn 0.7 into a compact split label fragment."""

    return f"{int(round(value * 100)):02d}"


def generate_multi_split_specs(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    ratio_splits: Iterable[float] = (0.60, 0.70, 0.80),
    include_random: bool = False,
    random_splits: int = 0,
    random_seed: int = 42,
) -> list[MultiSplitSpec]:
    """Build ratio, rolling, and optional random-index split definitions.

    Split generation is deterministic except for the optional random-index family.
    Random-index splits still respect time ordering because the sampled index is
    converted into a timestamp cutoff and the train set remains strictly earlier.
    """

    data = _prepare_frame(frame)
    if data.empty or len(data) < 2:
        return []

    specs: list[MultiSplitSpec] = []
    seen: set[tuple[str, str]] = set()

    def add_spec(spec: MultiSplitSpec) -> None:
        key = (
            spec.split_kind,
            spec.cutoff_ts.isoformat() if spec.cutoff_ts is not None else f"{spec.train_fraction:.4f}",
        )
        if key in seen or len(specs) >= n_splits:
            return
        seen.add(key)
        specs.append(spec)

    for fraction in ratio_splits:
        if not 0.0 < fraction < 1.0:
            continue
        add_spec(
            MultiSplitSpec(
                split_id=f"ratio_{_fraction_label(fraction)}_{_fraction_label(1.0 - fraction)}",
                split_kind="ratio",
                train_fraction=float(fraction),
            )
        )

    remaining = max(0, n_splits - len(specs))
    random_count = min(random_splits, remaining) if include_random else 0
    rolling_count = max(0, remaining - random_count)

    if rolling_count > 0:
        for index in range(1, rolling_count + 1):
            pct = 0.50 + (0.35 * index / (rolling_count + 1))
            boundary = max(1, min(len(data) - 1, int(round(pct * (len(data) - 1)))))
            cutoff = pd.Timestamp(data.iloc[boundary]["timestamp"])
            add_spec(
                MultiSplitSpec(
                    split_id=f"rolling_{index:02d}",
                    split_kind="rolling",
                    cutoff_ts=cutoff,
                )
            )

    if random_count > 0:
        rng = random.Random(random_seed)
        low = max(1, int(len(data) * 0.45))
        high = max(low + 1, min(len(data) - 1, int(len(data) * 0.90)))
        candidates = list(range(low, high))
        rng.shuffle(candidates)
        picked = candidates[:random_count]
        for index, boundary in enumerate(sorted(picked), start=1):
            cutoff = pd.Timestamp(data.iloc[boundary]["timestamp"])
            add_spec(
                MultiSplitSpec(
                    split_id=f"random_index_{index:02d}",
                    split_kind="random_index",
                    cutoff_ts=cutoff,
                )
            )

    return specs[:n_splits]


def materialize_multi_split(frame: pd.DataFrame, spec: MultiSplitSpec) -> TimeSplitResult:
    """Convert one multi-split spec into a concrete train/test partition."""

    if spec.split_kind == "ratio":
        if spec.train_fraction is None:
            raise ValueError("ratio split requires train_fraction")
        split = split_enriched_frame(frame, train_fraction=spec.train_fraction)
    else:
        if spec.cutoff_ts is None:
            raise ValueError("timestamp-based split requires cutoff_ts")
        split = split_enriched_frame(frame, split_date=spec.cutoff_ts.isoformat())
    return TimeSplitResult(
        method=spec.split_id,
        cutoff_ts=split.cutoff_ts,
        train=split.train,
        test=split.test,
    )


def classify_wallet_robustness(
    *,
    observed_splits: int,
    selected_splits: int,
    selection_frequency: float | None,
    positive_test_frequency: float | None,
    mode_consistency: float | None,
    settings: Settings | None = None,
) -> str:
    """Assign a cross-split robustness label.

    The defaults are conservative:
    - `robust` requires frequent selection, mostly-positive test follow-through,
      and a consistent dominant mode across splits.
    - `inconsistent` is reserved for wallets whose selected mode flips often.
    - `fragile` means the wallet shows up often enough to evaluate but lacks the
      consistency or positive follow-through needed for a robust label.
    - `insufficient_data` covers wallets that were rarely observed or rarely selected.
    """

    cfg = settings or get_settings()
    if observed_splits < cfg.multi_oos_min_observed_splits or selected_splits < cfg.multi_oos_min_selected_splits:
        return "insufficient_data"
    if mode_consistency is not None and mode_consistency < cfg.multi_oos_inconsistent_mode_threshold:
        return "inconsistent"
    if (
        selection_frequency is not None
        and positive_test_frequency is not None
        and mode_consistency is not None
        and selection_frequency >= cfg.multi_oos_robust_selection_frequency
        and positive_test_frequency >= cfg.multi_oos_robust_positive_test_frequency
        and mode_consistency >= cfg.multi_oos_mode_consistency_threshold
    ):
        return "robust"
    return "fragile"


def _safe_mean(series: pd.Series) -> float | None:
    """Return a numeric mean or null for empty data."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _safe_std(series: pd.Series) -> float | None:
    """Return a numeric standard deviation or null for empty data."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) < 2:
        return None
    return float(valid.std(ddof=1))


def _safe_frequency(series: pd.Series) -> float | None:
    """Return the share of truthy entries among non-null values."""

    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.astype(bool).mean())


def _mode_consistency(series: pd.Series) -> float | None:
    """Return the share captured by the dominant non-null mode."""

    valid = series.dropna()
    valid = valid[valid.astype(str) != ""]
    if valid.empty:
        return None
    return float(valid.value_counts(normalize=True).max())


def _to_iso8601(series: pd.Series) -> pd.Series:
    """Stringify datetimes for CSV output."""

    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamps.apply(lambda value: value.isoformat() if pd.notna(value) else None)


def _stringify_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert train/test boundary columns to ISO strings."""

    result = frame.copy()
    for column in result.columns:
        if column.endswith("_start") or column.endswith("_end") or column.endswith("_trade") or column.endswith("_ts"):
            converted = pd.to_datetime(result[column], utc=True, errors="coerce")
            if converted.notna().any():
                result[column] = _to_iso8601(result[column])
    return result


def _extract_portfolio_value(frame: pd.DataFrame, strategy_mode: str, horizon: str, column: str) -> float | None:
    """Fetch one portfolio metric from a per-strategy summary table."""

    if frame.empty:
        return None
    row = frame[(frame["strategy_mode"] == strategy_mode) & (frame["horizon"] == horizon)]
    if row.empty:
        return None
    value = pd.to_numeric(row.iloc[0][column], errors="coerce")
    return None if pd.isna(value) else float(value)


def _selected_net_column(delay_seconds: int) -> tuple[str, str]:
    """Return copy/fade net test columns for one delay bucket."""

    if delay_seconds == 0:
        return "test_avg_copy_pnl_net_5m", "test_avg_fade_pnl_net_5m"
    suffix = f"_delay_{delay_seconds}s"
    return (
        f"test_avg_copy_pnl_net_5m{suffix}",
        f"test_avg_fade_pnl_net_5m{suffix}",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate frames via records to avoid dtype edge-case warnings."""

    records: list[dict[str, Any]] = []
    for frame in frames:
        if frame.empty:
            continue
        records.extend(frame.to_dict(orient="records"))
    return pd.DataFrame.from_records(records)


def _selected_mode_net_series(frame: pd.DataFrame, delay_seconds: int = 0) -> pd.Series:
    """Resolve the selected-mode net-PnL series for one delay bucket."""

    copy_col, fade_col = _selected_net_column(delay_seconds)
    if copy_col not in frame.columns:
        frame[copy_col] = pd.NA
    if fade_col not in frame.columns:
        frame[fade_col] = pd.NA
    return frame.apply(
        lambda row: row[copy_col]
        if row.get("selected_group") == "copy"
        else row[fade_col]
        if row.get("selected_group") == "fade"
        else pd.NA,
        axis=1,
    )


def _build_split_delay_portfolio_frame(comparison: pd.DataFrame, split_id: str, split_kind: str) -> pd.DataFrame:
    """Build split-level portfolio net performance across delay buckets."""

    if comparison.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for strategy_mode in ("copy", "fade"):
        subset = comparison[comparison["selected_group"] == strategy_mode].copy()
        for delay_seconds in DELAY_SECONDS:
            copy_col, fade_col = _selected_net_column(delay_seconds)
            selected_col = copy_col if strategy_mode == "copy" else fade_col
            if selected_col not in subset.columns:
                values = pd.Series(dtype=float)
            else:
                values = pd.to_numeric(subset[selected_col], errors="coerce").dropna()
            rows.append(
                {
                    "split_id": split_id,
                    "split_kind": split_kind,
                    "strategy_mode": strategy_mode,
                    "delay_seconds": delay_seconds,
                    "delay_label": DELAY_LABELS[delay_seconds],
                    "n_wallets": int(subset["wallet_address"].nunique()) if not subset.empty else 0,
                    "avg_net_return": float(values.mean()) if not values.empty else None,
                    "hit_rate": float((values > 0).mean()) if not values.empty else None,
                    "positive_split": float(values.mean()) > 0 if not values.empty else pd.NA,
                }
            )
    return pd.DataFrame.from_records(rows)


def _aggregate_wallet_robustness(
    comparison_frame: pd.DataFrame,
    *,
    total_splits: int,
    settings: Settings,
) -> pd.DataFrame:
    """Summarize wallet behavior across many split runs."""

    if comparison_frame.empty:
        return pd.DataFrame(
            columns=[
                "wallet_address",
                "observed_splits",
                "selected_splits",
                "selection_frequency",
                "positive_test_frequency",
                "net_positive_test_frequency",
                "mode_consistency",
                "avg_test_copy_pnl_5m",
                "avg_test_fade_pnl_5m",
                "std_test_copy_pnl_5m",
                "std_test_fade_pnl_5m",
                "avg_net_test_copy_pnl_5m",
                "avg_net_test_fade_pnl_5m",
                "tradability_label",
                "robustness_label",
            ]
        )

    frame = comparison_frame.copy()
    frame["selected_test_pnl_5m"] = frame.apply(
        lambda row: row["test_avg_copy_pnl_5m"]
        if row.get("selected_group") == "copy"
        else row["test_avg_fade_pnl_5m"]
        if row.get("selected_group") == "fade"
        else pd.NA,
        axis=1,
    )
    selected_numeric = pd.to_numeric(frame["selected_test_pnl_5m"], errors="coerce")
    frame["selected_test_positive"] = selected_numeric.apply(
        lambda value: (value > 0) if pd.notna(value) else pd.NA
    )
    frame["selected_net_test_pnl_5m"] = _selected_mode_net_series(frame, delay_seconds=0)
    frame["selected_net_test_pnl_5m_delay_15s"] = _selected_mode_net_series(frame, delay_seconds=15)
    frame["selected_net_test_pnl_5m_delay_30s"] = _selected_mode_net_series(frame, delay_seconds=30)
    selected_net_numeric = pd.to_numeric(frame["selected_net_test_pnl_5m"], errors="coerce")
    frame["selected_net_test_positive"] = selected_net_numeric.apply(
        lambda value: (value > 0) if pd.notna(value) else pd.NA
    )

    records: list[dict[str, Any]] = []
    for wallet, group in frame.groupby("wallet_address"):
        observed_splits = int(group["split_id"].nunique())
        selected = group[group["selected_group"].notna()].copy()
        selected_splits = int(selected["split_id"].nunique())
        selection_frequency = selected_splits / total_splits if total_splits else None
        positive_test_frequency = _safe_frequency(selected["selected_test_positive"])
        net_positive_test_frequency = _safe_frequency(selected["selected_net_test_positive"])
        mode_consistency = _mode_consistency(
            selected["selected_group"]
            if not selected.empty
            else group["train_recommended_mode"].replace({"ignore": pd.NA})
        )
        robustness_label = classify_wallet_robustness(
            observed_splits=observed_splits,
            selected_splits=selected_splits,
            selection_frequency=selection_frequency,
            positive_test_frequency=positive_test_frequency,
            mode_consistency=mode_consistency,
            settings=settings,
        )
        dominant_mode = None
        if not selected.empty:
            mode_counts = selected["selected_group"].value_counts()
            dominant_mode = str(mode_counts.index[0]) if not mode_counts.empty else None
        selected_base_net = _safe_mean(selected["selected_net_test_pnl_5m"]) if not selected.empty else None
        selected_net_15 = _safe_mean(selected["selected_net_test_pnl_5m_delay_15s"]) if not selected.empty else None
        selected_net_30 = _safe_mean(selected["selected_net_test_pnl_5m_delay_30s"]) if not selected.empty else None
        records.append(
            {
                "wallet_address": wallet,
                "observed_splits": observed_splits,
                "selected_splits": selected_splits,
                "selection_frequency": selection_frequency,
                "positive_test_frequency": positive_test_frequency,
                "net_positive_test_frequency": net_positive_test_frequency,
                "mode_consistency": mode_consistency,
                "avg_test_copy_pnl_5m": _safe_mean(group["test_avg_copy_pnl_5m"]),
                "avg_test_fade_pnl_5m": _safe_mean(group["test_avg_fade_pnl_5m"]),
                "std_test_copy_pnl_5m": _safe_std(group["test_avg_copy_pnl_5m"]),
                "std_test_fade_pnl_5m": _safe_std(group["test_avg_fade_pnl_5m"]),
                "avg_net_test_copy_pnl_5m": _safe_mean(group["test_avg_copy_pnl_net_5m"]) if "test_avg_copy_pnl_net_5m" in group.columns else None,
                "avg_net_test_fade_pnl_5m": _safe_mean(group["test_avg_fade_pnl_net_5m"]) if "test_avg_fade_pnl_net_5m" in group.columns else None,
                "tradability_label": classify_tradability(
                    base_net_pnl=selected_base_net,
                    net_pnl_delay_15s=selected_net_15,
                    net_pnl_delay_30s=selected_net_30,
                    mode_consistency=mode_consistency,
                    settings=settings,
                ) if dominant_mode is not None else "not_tradable",
                "robustness_label": robustness_label,
            }
        )

    result = pd.DataFrame.from_records(records)
    return result.sort_values(
        by=["selection_frequency", "positive_test_frequency", "wallet_address"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _aggregate_portfolio_robustness(split_portfolio_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize portfolio performance across all splits."""

    if split_portfolio_frame.empty:
        return pd.DataFrame(
            columns=[
                "strategy_mode",
                "horizon",
                "n_splits",
                "avg_portfolio_return",
                "std_portfolio_return",
                "positive_split_frequency",
            ]
        )

    records: list[dict[str, Any]] = []
    for (strategy_mode, horizon), group in split_portfolio_frame.groupby(["strategy_mode", "horizon"]):
        returns = pd.to_numeric(group["avg_event_return"], errors="coerce").dropna()
        records.append(
            {
                "strategy_mode": strategy_mode,
                "horizon": horizon,
                "n_splits": int(group["split_id"].nunique()),
                "avg_portfolio_return": float(returns.mean()) if not returns.empty else None,
                "std_portfolio_return": float(returns.std(ddof=1)) if len(returns) >= 2 else None,
                "positive_split_frequency": float((returns > 0).mean()) if not returns.empty else None,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["strategy_mode", "horizon"]).reset_index(drop=True)


def _aggregate_delay_robustness(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize selected-mode net performance across delays for each wallet."""

    if comparison_frame.empty:
        return pd.DataFrame(
            columns=[
                "wallet_address",
                "delay_seconds",
                "delay_label",
                "avg_selected_net_test_pnl_5m",
                "positive_split_frequency",
                "selected_splits",
            ]
        )

    frame = comparison_frame.copy()
    frame = frame[frame["selected_group"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "wallet_address",
                "delay_seconds",
                "delay_label",
                "avg_selected_net_test_pnl_5m",
                "positive_split_frequency",
                "selected_splits",
            ]
        )

    rows: list[dict[str, Any]] = []
    for wallet, group in frame.groupby("wallet_address"):
        for delay_seconds in DELAY_SECONDS:
            selected_series = pd.to_numeric(_selected_mode_net_series(group, delay_seconds=delay_seconds), errors="coerce")
            valid = selected_series.dropna()
            rows.append(
                {
                    "wallet_address": wallet,
                    "delay_seconds": delay_seconds,
                    "delay_label": DELAY_LABELS[delay_seconds],
                    "avg_selected_net_test_pnl_5m": float(valid.mean()) if not valid.empty else None,
                    "positive_split_frequency": float((valid > 0).mean()) if not valid.empty else None,
                    "selected_splits": int(group["split_id"].nunique()),
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(["wallet_address", "delay_seconds"]).reset_index(drop=True)


def _aggregate_portfolio_delay_performance(split_delay_portfolio_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split-level portfolio delay performance across all splits."""

    if split_delay_portfolio_frame.empty:
        return pd.DataFrame(
            columns=[
                "strategy_mode",
                "delay_seconds",
                "delay_label",
                "n_splits",
                "avg_net_return",
                "hit_rate",
                "positive_split_frequency",
            ]
        )

    rows: list[dict[str, Any]] = []
    for (strategy_mode, delay_seconds), group in split_delay_portfolio_frame.groupby(["strategy_mode", "delay_seconds"]):
        returns = pd.to_numeric(group["avg_net_return"], errors="coerce").dropna()
        hit_rates = pd.to_numeric(group["hit_rate"], errors="coerce").dropna()
        rows.append(
            {
                "strategy_mode": strategy_mode,
                "delay_seconds": int(delay_seconds),
                "delay_label": DELAY_LABELS[int(delay_seconds)],
                "n_splits": int(group["split_id"].nunique()),
                "avg_net_return": float(returns.mean()) if not returns.empty else None,
                "hit_rate": float(hit_rates.mean()) if not hit_rates.empty else None,
                "positive_split_frequency": float((returns > 0).mean()) if not returns.empty else None,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["strategy_mode", "delay_seconds"]).reset_index(drop=True)


def _build_split_run_summary(split_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn split-level metadata rows into a DataFrame."""

    frame = pd.DataFrame.from_records(split_rows)
    if frame.empty:
        return frame
    return _stringify_datetime_columns(frame).sort_values("split_id").reset_index(drop=True)


def run_multi_oos_validation_from_frame(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path = "exports/multi_oos",
    n_splits: int | None = None,
    ratio_splits: Iterable[float] = (0.60, 0.70, 0.80),
    include_random: bool = False,
    random_splits: int = 0,
    random_seed: int | None = None,
    top_n: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run many OOS split schemes and aggregate robustness outputs."""

    cfg = settings or get_settings()
    effective_n_splits = cfg.multi_oos_n_splits if n_splits is None else n_splits
    effective_seed = cfg.multi_oos_random_seed if random_seed is None else random_seed
    specs = generate_multi_split_specs(
        frame,
        n_splits=effective_n_splits,
        ratio_splits=ratio_splits,
        include_random=include_random,
        random_splits=random_splits,
        random_seed=effective_seed,
    )

    export_root = Path(output_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    split_results: list[dict[str, Any]] = []
    comparison_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    split_portfolio_frames: list[pd.DataFrame] = []
    split_delay_portfolio_frames: list[pd.DataFrame] = []
    split_summary_rows: list[dict[str, Any]] = []

    prepared_frame = _prepare_frame(frame)
    for spec in specs:
        split = materialize_multi_split(prepared_frame, spec)
        split_output_dir = export_root / "splits" / spec.split_id
        result = run_oos_validation_on_split(
            split,
            output_dir=split_output_dir,
            top_n=top_n,
            settings=cfg,
        )
        split_results.append({"spec": spec, "result": result})

        comparison = result["comparison"].copy()
        comparison["split_id"] = spec.split_id
        comparison["split_kind"] = spec.split_kind
        comparison_frames.append(comparison)

        selected = result["selected_test"].copy()
        if not selected.empty:
            selected["split_id"] = spec.split_id
            selected["split_kind"] = spec.split_kind
            selected_frames.append(selected)

        for portfolio_name in ("test_copy_portfolio", "test_fade_portfolio"):
            portfolio = result[portfolio_name].copy()
            if portfolio.empty:
                continue
            portfolio["split_id"] = spec.split_id
            portfolio["split_kind"] = spec.split_kind
            split_portfolio_frames.append(portfolio)
        delay_portfolio = _build_split_delay_portfolio_frame(comparison, spec.split_id, spec.split_kind)
        if not delay_portfolio.empty:
            split_delay_portfolio_frames.append(delay_portfolio)

        split_summary_rows.append(
            {
                "split_id": spec.split_id,
                "split_kind": spec.split_kind,
                "train_start": split.train["timestamp"].min() if not split.train.empty else None,
                "train_end": split.train["timestamp"].max() if not split.train.empty else None,
                "test_start": split.test["timestamp"].min() if not split.test.empty else None,
                "test_end": split.test["timestamp"].max() if not split.test.empty else None,
                "n_train_trades": len(split.train),
                "n_test_trades": len(split.test),
                "n_selected_copy_wallets": len(result["selected_copy"]),
                "n_selected_fade_wallets": len(result["selected_fade"]),
                "selected_copy_wallets": ";".join(result["selected_copy"]["wallet_address"].tolist()),
                "selected_fade_wallets": ";".join(result["selected_fade"]["wallet_address"].tolist()),
                "portfolio_copy_avg_return_5m": _extract_portfolio_value(
                    result["test_copy_portfolio"], "copy", "5m", "avg_event_return"
                ),
                "portfolio_fade_avg_return_5m": _extract_portfolio_value(
                    result["test_fade_portfolio"], "fade", "5m", "avg_event_return"
                ),
            }
        )

    split_run_summary = _build_split_run_summary(split_summary_rows)
    all_comparisons = _concat_frames(comparison_frames)
    all_selected = _concat_frames(selected_frames)
    split_portfolio_performance = _concat_frames(split_portfolio_frames)
    split_delay_portfolio_performance = _concat_frames(split_delay_portfolio_frames)
    wallet_robustness = _aggregate_wallet_robustness(
        all_comparisons,
        total_splits=len(specs),
        settings=cfg,
    )
    delay_robustness = _aggregate_delay_robustness(all_comparisons)
    portfolio_robustness = _aggregate_portfolio_robustness(split_portfolio_performance)
    portfolio_delay_performance = _aggregate_portfolio_delay_performance(split_delay_portfolio_performance)

    from reports.multi_oos_plots import generate_multi_oos_plots

    plot_paths = generate_multi_oos_plots(
        wallet_robustness=wallet_robustness,
        split_portfolio_performance=split_portfolio_performance,
        delay_robustness=delay_robustness,
        portfolio_delay_performance=portfolio_delay_performance,
        output_dir=export_root / "plots",
    )

    paths = {
        "split_run_summary": _write_csv(split_run_summary, export_root / "split_run_summary.csv"),
        "split_selected_wallets": _write_csv(_stringify_datetime_columns(all_selected), export_root / "split_selected_wallets.csv"),
        "split_portfolio_performance": _write_csv(split_portfolio_performance, export_root / "split_portfolio_performance.csv"),
        "wallet_robustness_summary": _write_csv(wallet_robustness, export_root / "wallet_robustness_summary.csv"),
        "portfolio_robustness_summary": _write_csv(portfolio_robustness, export_root / "portfolio_robustness_summary.csv"),
        "delay_robustness_summary": _write_csv(delay_robustness, export_root / "delay_robustness_summary.csv"),
        "portfolio_delay_performance": _write_csv(
            portfolio_delay_performance,
            export_root / "portfolio_delay_performance.csv",
        ),
    }

    return {
        "specs": specs,
        "split_results": split_results,
        "split_run_summary": split_run_summary,
        "split_selected_wallets": all_selected,
        "split_portfolio_performance": split_portfolio_performance,
        "split_delay_portfolio_performance": split_delay_portfolio_performance,
        "wallet_robustness": wallet_robustness,
        "delay_robustness": delay_robustness,
        "portfolio_robustness": portfolio_robustness,
        "portfolio_delay_performance": portfolio_delay_performance,
        "plot_paths": plot_paths,
        "paths": paths,
    }


def run_multi_oos_validation(
    session: Session,
    *,
    output_dir: str | Path = "exports/multi_oos",
    n_splits: int | None = None,
    ratio_splits: Iterable[float] = (0.60, 0.70, 0.80),
    include_random: bool = False,
    random_splits: int = 0,
    random_seed: int | None = None,
    top_n: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run multi-split OOS validation from the enriched-trades table."""

    frame = load_enriched_trades(session)
    return run_multi_oos_validation_from_frame(
        frame,
        output_dir=output_dir,
        n_splits=n_splits,
        ratio_splits=ratio_splits,
        include_random=include_random,
        random_splits=random_splits,
        random_seed=random_seed,
        top_n=top_n,
        settings=settings,
    )


def print_multi_oos_summary(results: dict[str, Any]) -> None:
    """Print a concise summary of multi-split robustness results."""

    split_summary: pd.DataFrame = results["split_run_summary"]
    wallet_robustness: pd.DataFrame = results["wallet_robustness"]
    portfolio_robustness: pd.DataFrame = results["portfolio_robustness"]

    print("Multi-Split OOS Summary")
    print(f"Executed splits: {len(split_summary)}")

    if not wallet_robustness.empty:
        label_counts = (
            wallet_robustness["robustness_label"]
            .value_counts()
            .rename_axis("robustness_label")
            .reset_index(name="wallet_count")
        )
        print("Wallet Robustness Labels")
        print(label_counts.to_string(index=False))
        print("Top Wallets by Selection Frequency")
        print(
            wallet_robustness[
                [
                    "wallet_address",
                    "selection_frequency",
                    "positive_test_frequency",
                    "mode_consistency",
                    "robustness_label",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("Wallet Robustness Labels")
        print("No wallet robustness results were produced.")

    if not portfolio_robustness.empty:
        print("Portfolio Robustness")
        print(portfolio_robustness.to_string(index=False))
    else:
        print("Portfolio Robustness")
        print("No portfolio robustness rows were produced.")
