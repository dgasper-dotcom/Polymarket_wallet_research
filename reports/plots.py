"""Simple histogram plots for top-ranked wallets."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

sys.modules.setdefault("pyarrow", None)

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import WalletTradeEnriched
from research.wallet_scoring import score_wallets


def generate_top_wallet_plots(
    session: Session,
    output_dir: str | Path = "artifacts/plots",
    top_n: int = 5,
) -> list[Path]:
    """Generate histogram plots for the highest-ranked wallets."""

    scores = score_wallets(session)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if scores.empty:
        return []

    scores = scores.copy()
    scores["sort_score"] = scores[["overall_copy_score", "overall_fade_score"]].max(axis=1)
    top_wallets = scores.sort_values("sort_score", ascending=False).head(top_n)["wallet_address"]
    enriched = pd.read_sql(select(WalletTradeEnriched), session.bind)

    generated: list[Path] = []
    for wallet in top_wallets:
        wallet_frame = enriched[enriched["wallet_address"] == wallet]
        if wallet_frame.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, column, title in zip(
            axes,
            ["ret_5m", "copy_pnl_5m", "fade_pnl_5m"],
            ["ret_5m", "copy_pnl_5m", "fade_pnl_5m"],
        ):
            values = pd.to_numeric(wallet_frame[column], errors="coerce").dropna()
            bins = min(30, max(10, len(values))) if not values.empty else 10
            axis.hist(values, bins=bins, color="#1f77b4", alpha=0.8)
            axis.set_title(f"{wallet[:10]}... {title}")
            axis.set_xlabel(title)
            axis.set_ylabel("count")

        fig.tight_layout()
        target = output_path / f"{wallet}_histograms.png"
        fig.savefig(target, dpi=150)
        plt.close(fig)
        generated.append(target)

    return generated
