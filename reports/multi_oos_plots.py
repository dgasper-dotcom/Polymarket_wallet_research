"""Aggregate plots for multi-split OOS robustness runs."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

sys.modules.setdefault("pyarrow", None)

import matplotlib.pyplot as plt
import pandas as pd


def generate_multi_oos_plots(
    *,
    wallet_robustness: pd.DataFrame,
    split_portfolio_performance: pd.DataFrame,
    delay_robustness: pd.DataFrame,
    portfolio_delay_performance: pd.DataFrame,
    output_dir: str | Path = "exports/multi_oos/plots",
) -> list[Path]:
    """Generate cross-split robustness charts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if not wallet_robustness.empty:
        copy_values = pd.to_numeric(wallet_robustness["avg_test_copy_pnl_5m"], errors="coerce").dropna()
        fade_values = pd.to_numeric(wallet_robustness["avg_test_fade_pnl_5m"], errors="coerce").dropna()
        fig, axis = plt.subplots(figsize=(8, 5))
        if not copy_values.empty:
            axis.hist(copy_values, bins=min(20, max(8, len(copy_values))), alpha=0.6, label="copy", color="#1f77b4")
        if not fade_values.empty:
            axis.hist(fade_values, bins=min(20, max(8, len(fade_values))), alpha=0.6, label="fade", color="#ff7f0e")
        axis.set_title("Wallet Test PnL Across Splits (5m)")
        axis.set_xlabel("avg test pnl 5m")
        axis.set_ylabel("wallet count")
        axis.legend()
        fig.tight_layout()
        target = output_path / "wallet_test_pnl_histogram_5m.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        generated.append(target)

        top_wallets = wallet_robustness.head(20).copy()
        top_wallets["wallet_label"] = top_wallets["wallet_address"].str.slice(0, 10) + "..."
        fig, axis = plt.subplots(figsize=(12, 5))
        axis.bar(top_wallets["wallet_label"], pd.to_numeric(top_wallets["selection_frequency"], errors="coerce").fillna(0.0), color="#2ca02c")
        axis.set_title("Wallet Selection Frequency Across Splits")
        axis.set_ylabel("selection frequency")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        target = output_path / "wallet_selection_frequency.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        generated.append(target)

    if not split_portfolio_performance.empty:
        frame = split_portfolio_performance.copy()
        focus = frame[frame["horizon"] == "5m"].copy()
        if not focus.empty:
            fig, axis = plt.subplots(figsize=(12, 5))
            for strategy_mode, color in (("copy", "#2ca02c"), ("fade", "#d62728")):
                subset = focus[focus["strategy_mode"] == strategy_mode].copy()
                if subset.empty:
                    continue
                subset = subset.sort_values("split_id")
                axis.plot(
                    subset["split_id"],
                    pd.to_numeric(subset["avg_event_return"], errors="coerce"),
                    marker="o",
                    label=strategy_mode,
                    color=color,
                )
            axis.set_title("Portfolio Performance Across Splits (5m)")
            axis.set_xlabel("split id")
            axis.set_ylabel("avg event return")
            axis.tick_params(axis="x", rotation=45)
            axis.grid(alpha=0.25)
            axis.legend()
            fig.tight_layout()
            target = output_path / "portfolio_performance_across_splits_5m.png"
            fig.savefig(target, dpi=150)
            plt.close(fig)
            generated.append(target)

    if not portfolio_delay_performance.empty:
        fig, axis = plt.subplots(figsize=(10, 5))
        for strategy_mode, color in (("copy", "#2ca02c"), ("fade", "#d62728")):
            subset = portfolio_delay_performance[portfolio_delay_performance["strategy_mode"] == strategy_mode]
            if subset.empty:
                continue
            axis.plot(
                subset["delay_seconds"],
                pd.to_numeric(subset["avg_net_return"], errors="coerce"),
                marker="o",
                label=strategy_mode,
                color=color,
            )
        axis.set_title("Portfolio Net Return vs Delay Across Splits")
        axis.set_xlabel("delay seconds")
        axis.set_ylabel("avg net return")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        target = output_path / "portfolio_delay_performance.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        generated.append(target)

    return generated
