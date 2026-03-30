"""Plot helpers for delay and net-PnL analysis."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

sys.modules.setdefault("pyarrow", None)

import matplotlib.pyplot as plt
import pandas as pd


def generate_delay_analysis_plots(
    *,
    wallet_delay_summary: pd.DataFrame,
    wallet_delay_event_study: pd.DataFrame,
    portfolio_delay_performance: pd.DataFrame,
    output_dir: str | Path = "exports/delay_analysis/plots",
) -> list[Path]:
    """Generate simple delay-analysis charts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if not wallet_delay_event_study.empty:
        top_wallets = wallet_delay_summary.head(6)["wallet_address"].tolist() if not wallet_delay_summary.empty else []
        subset = wallet_delay_event_study[
            wallet_delay_event_study["wallet_address"].isin(top_wallets)
        ].copy() if top_wallets else wallet_delay_event_study.head(12).copy()
        if not subset.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            for axis, value_column, title in (
                (axes[0], "avg_pnl", "Gross PnL vs Delay"),
                (axes[1], "avg_net_pnl", "Net PnL vs Delay"),
            ):
                for (wallet, strategy_mode), group in subset.groupby(["wallet_address", "strategy_mode"]):
                    axis.plot(
                        group["delay_seconds"],
                        pd.to_numeric(group[value_column], errors="coerce"),
                        marker="o",
                        label=f"{wallet[:8]} {strategy_mode}",
                    )
                axis.set_title(title)
                axis.set_xlabel("delay seconds")
                axis.grid(alpha=0.25)
            axes[0].set_ylabel("pnl")
            axes[1].legend(fontsize=8, ncol=2)
            fig.tight_layout()
            target = output_path / "wallet_pnl_vs_delay.png"
            fig.savefig(target, dpi=150)
            plt.close(fig)
            generated.append(target)

    if not portfolio_delay_performance.empty:
        fig, axis = plt.subplots(figsize=(10, 5))
        for strategy_mode, color in (("copy", "#2ca02c"), ("fade", "#d62728")):
            group = portfolio_delay_performance[portfolio_delay_performance["strategy_mode"] == strategy_mode]
            if group.empty:
                continue
            axis.plot(
                group["delay_seconds"],
                pd.to_numeric(group["avg_net_return"], errors="coerce"),
                marker="o",
                label=strategy_mode,
                color=color,
            )
        axis.set_title("Portfolio Net PnL vs Delay")
        axis.set_xlabel("delay seconds")
        axis.set_ylabel("avg net return")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        target = output_path / "portfolio_pnl_vs_delay.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        generated.append(target)

    if not wallet_delay_summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        retention = pd.to_numeric(wallet_delay_summary["edge_retention_30s"], errors="coerce").dropna()
        break_even = pd.concat(
            [
                pd.to_numeric(wallet_delay_summary["break_even_cost_copy_5m"], errors="coerce"),
                pd.to_numeric(wallet_delay_summary["break_even_cost_fade_5m"], errors="coerce"),
            ],
            ignore_index=True,
        ).dropna()

        axes[0].hist(retention, bins=min(20, max(8, len(retention))) if not retention.empty else 10, color="#1f77b4", alpha=0.8)
        axes[0].set_title("Edge Retention at 30s")
        axes[0].set_xlabel("retention")
        axes[0].set_ylabel("wallet count")

        axes[1].hist(break_even, bins=min(20, max(8, len(break_even))) if not break_even.empty else 10, color="#ff7f0e", alpha=0.8)
        axes[1].set_title("Break-Even Cost")
        axes[1].set_xlabel("cost level")
        axes[1].set_ylabel("wallet count")

        fig.tight_layout()
        target = output_path / "retention_and_break_even_histograms.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        generated.append(target)

    return generated
