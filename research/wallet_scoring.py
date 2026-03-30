"""Wallet ranking logic based on copy/fade research metrics."""

from __future__ import annotations

import math
import sys

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy.orm import Session

from research.event_study import compute_event_study_outputs


MIN_TRADES_FOR_RECOMMENDATION = 5
SCORE_DOMINANCE_MARGIN = 0.50
# The composite scores are z-score style. Requiring a 0.50 gap means the winning
# mode should be roughly half a standard deviation better than the alternative.
# This keeps the MVP conservative when the public data is sparse or noisy.


def _safe_mean(series: pd.Series) -> float | None:
    """Mean with numeric coercion."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _hit_rate(series: pd.Series) -> float | None:
    """Fraction of positive observations."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float((valid > 0).mean())


def _zscore(series: pd.Series) -> pd.Series:
    """Stable z-score helper that returns zeros when dispersion is absent."""

    numeric = pd.to_numeric(series, errors="coerce")
    filled = numeric.fillna(numeric.mean())
    std = filled.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (filled - filled.mean()) / std


def classify_score_confidence(
    n_trades: int,
    n_markets: int,
    fraction_top_market: float | None,
) -> str:
    """Label score confidence from sample size, breadth, and concentration."""

    concentration = 1.0 if fraction_top_market is None else float(fraction_top_market)
    if n_trades >= 25 and n_markets >= 5 and concentration <= 0.50:
        return "high"
    if n_trades >= 10 and n_markets >= 3 and concentration <= 0.70:
        return "medium"
    return "low"


def recommend_mode(
    copy_score: float,
    fade_score: float,
    n_trades: int,
) -> str:
    """Apply the conservative copy/fade recommendation thresholds."""

    if n_trades < MIN_TRADES_FOR_RECOMMENDATION:
        return "ignore"
    if copy_score > 0 and (copy_score - fade_score) >= SCORE_DOMINANCE_MARGIN:
        return "copy"
    if fade_score > 0 and (fade_score - copy_score) >= SCORE_DOMINANCE_MARGIN:
        return "fade"
    return "ignore"


def score_wallet_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute wallet scores and recommended modes from a wallet-summary frame.

    Composite score design:
    - PnL means dominate the score because we care about directional edge.
    - Hit rate adds robustness.
    - Trade count and market breadth reward repeatability.
    - Concentration penalizes wallets whose signal depends too much on one market.
    """

    if summary.empty:
        return pd.DataFrame(
            columns=[
                "wallet_address",
                "n_trades",
                "n_markets",
                "avg_copy_pnl_5m",
                "avg_copy_pnl_30m",
                "avg_fade_pnl_5m",
                "avg_fade_pnl_30m",
                "copy_hit_rate_5m",
                "fade_hit_rate_5m",
                "fraction_top_market",
                "concentration_score",
                "overall_copy_score",
                "overall_fade_score",
                "score_confidence",
                "recommended_mode",
            ]
        )

    result = summary[
        [
            "wallet_address",
            "n_trades",
            "n_markets",
            "avg_copy_pnl_5m",
            "avg_copy_pnl_30m",
            "avg_fade_pnl_5m",
            "avg_fade_pnl_30m",
            "copy_hit_rate_5m",
            "fade_hit_rate_5m",
            "fraction_top_market",
        ]
    ].copy()
    result["concentration_score"] = 1.0 - result["fraction_top_market"].fillna(1.0)
    result["log_n_trades"] = result["n_trades"].map(lambda value: math.log1p(value))

    copy_score = (
        0.30 * _zscore(result["avg_copy_pnl_5m"])
        + 0.30 * _zscore(result["avg_copy_pnl_30m"])
        + 0.15 * _zscore(result["copy_hit_rate_5m"])
        + 0.10 * _zscore(result["log_n_trades"])
        + 0.10 * _zscore(result["n_markets"])
        + 0.05 * _zscore(result["concentration_score"])
    )
    fade_score = (
        0.30 * _zscore(result["avg_fade_pnl_5m"])
        + 0.30 * _zscore(result["avg_fade_pnl_30m"])
        + 0.15 * _zscore(result["fade_hit_rate_5m"])
        + 0.10 * _zscore(result["log_n_trades"])
        + 0.10 * _zscore(result["n_markets"])
        + 0.05 * _zscore(result["concentration_score"])
    )
    result["overall_copy_score"] = copy_score
    result["overall_fade_score"] = fade_score

    result["score_confidence"] = result.apply(
        lambda row: classify_score_confidence(
            n_trades=int(row["n_trades"]),
            n_markets=int(row["n_markets"]),
            fraction_top_market=float(row["fraction_top_market"]),
        ),
        axis=1,
    )
    result["recommended_mode"] = result.apply(
        lambda row: recommend_mode(
            copy_score=float(row["overall_copy_score"]),
            fade_score=float(row["overall_fade_score"]),
            n_trades=int(row["n_trades"]),
        ),
        axis=1,
    )
    result = result.drop(columns=["log_n_trades"]).sort_values(
        by=["overall_copy_score", "overall_fade_score"],
        ascending=False,
        na_position="last",
    )
    return result


def score_wallets(session: Session) -> pd.DataFrame:
    """Compute wallet scores and recommended modes from the enriched-trades table."""

    summary, _ = compute_event_study_outputs(session)
    return score_wallet_summary(summary)
