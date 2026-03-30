"""Plot helpers for out-of-sample validation exports."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

sys.modules.setdefault("pyarrow", None)

import matplotlib.pyplot as plt
import pandas as pd


def generate_oos_validation_plots(
    selected_results: pd.DataFrame,
    copy_portfolio: pd.DataFrame,
    fade_portfolio: pd.DataFrame,
    output_dir: str | Path = "exports/oos_validation/plots",
) -> list[Path]:
    """Generate simple train-vs-test and portfolio summary charts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if not selected_results.empty:
        frame = selected_results.copy()
        frame["wallet_label"] = frame["wallet_address"].str.slice(0, 10) + "..."
        x = range(len(frame))
        width = 0.38

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        comparisons = [
            ("train_avg_copy_pnl_5m", "test_avg_copy_pnl_5m", "Avg Copy PnL 5m"),
            ("train_avg_fade_pnl_5m", "test_avg_fade_pnl_5m", "Avg Fade PnL 5m"),
            ("train_selected_hit_rate_5m", "test_selected_hit_rate_5m", "Selected Hit Rate 5m"),
        ]
        for axis, (train_col, test_col, title) in zip(axes, comparisons):
            train_values = pd.to_numeric(frame[train_col], errors="coerce").fillna(0.0)
            test_values = pd.to_numeric(frame[test_col], errors="coerce").fillna(0.0)
            axis.bar([value - width / 2 for value in x], train_values, width=width, label="train", color="#1f77b4")
            axis.bar([value + width / 2 for value in x], test_values, width=width, label="test", color="#ff7f0e")
            axis.set_xticks(list(x))
            axis.set_xticklabels(frame["wallet_label"], rotation=25, ha="right")
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
        axes[0].legend()
        fig.tight_layout()
        target = output_path / "selected_wallets_train_vs_test_5m.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        generated.append(target)

    portfolio_records: list[dict[str, object]] = []
    for frame in (copy_portfolio, fade_portfolio):
        if frame.empty:
            continue
        portfolio_records.extend(frame.to_dict(orient="records"))
    portfolio = pd.DataFrame.from_records(portfolio_records)
    if not portfolio.empty:
        pivot = portfolio.pivot(index="horizon", columns="strategy_mode", values="avg_event_return")
        pivot = pivot.apply(pd.to_numeric, errors="coerce").astype(float).fillna(0.0)
        horizons = list(pivot.index)
        x = range(len(horizons))
        width = 0.38

        fig, axis = plt.subplots(figsize=(10, 5))
        copy_values = pivot["copy"] if "copy" in pivot else pd.Series([0.0] * len(horizons), index=horizons)
        fade_values = pivot["fade"] if "fade" in pivot else pd.Series([0.0] * len(horizons), index=horizons)
        axis.bar([value - width / 2 for value in x], copy_values, width=width, label="copy", color="#2ca02c")
        axis.bar([value + width / 2 for value in x], fade_values, width=width, label="fade", color="#d62728")
        axis.set_xticks(list(x))
        axis.set_xticklabels(horizons)
        axis.set_title("Test Portfolio Average Event Return")
        axis.set_ylabel("avg event return")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        target = output_path / "test_portfolio_summary.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        generated.append(target)

    return generated
