"""Leaderboard export helpers."""

from __future__ import annotations

from pathlib import Path
import sys

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy.orm import Session

from research.wallet_scoring import score_wallets


LEADERBOARD_COLUMNS = [
    "wallet_address",
    "n_trades",
    "n_markets",
    "avg_copy_pnl_5m",
    "avg_fade_pnl_5m",
    "overall_copy_score",
    "overall_fade_score",
    "recommended_mode",
    "score_confidence",
]


def export_wallet_leaderboard(
    session: Session, output_path: str | Path = "artifacts/reports/wallet_leaderboard.csv"
) -> pd.DataFrame:
    """Export the wallet leaderboard to CSV."""

    leaderboard = score_wallets(session)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if leaderboard.empty:
        leaderboard.to_csv(path, index=False)
        return leaderboard

    export_frame = leaderboard[LEADERBOARD_COLUMNS].copy()
    export_frame.to_csv(path, index=False)
    return export_frame
