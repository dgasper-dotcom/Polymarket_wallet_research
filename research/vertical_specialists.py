"""Vertical-specialist wallet analysis built from existing recent-trade exports."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd

from research.good_wallet_analysis import (
    DEFAULT_FEATURES_CSV,
    DEFAULT_RECENT_TRADES_CSV,
    _safe_ratio,
    load_recent_trades,
    load_wallet_features,
)


DEFAULT_OUTPUT_DIR = "exports/vertical_specialists"

SPORTS_KEYWORDS = (
    "nba", "nhl", "nfl", "mlb", "mls", "soccer", "football", "basketball", "baseball",
    "hockey", "tennis", "golf", "ufc", "mma", "boxing", "playoffs", "division", "seed",
    "premier-league", "premier league", "la-liga", "serie-a", "ligue-1", "ligue 1",
    "champions league", "champions-league", "march madness", "ncaa", "ballon d'or",
    "ballon-d-or", "hart memorial trophy", "art ross", "stanley cup", "super bowl",
    "world cup", "pga", "atp", "wta",
)

POLITICS_KEYWORDS = (
    "election", "president", "presidential", "prime minister", "prime-minister", "senate",
    "governor", "mayor", "democratic", "republican", "nominee", "referendum",
    "parliament", "parliamentary", "minister", "leader", "trump", "biden", "ceasefire",
    "iran", "russia", "ukraine", "cuba", "israel", "gaza", "war", "forces enter",
    "supreme court", "supreme-court", "congress", "starmer", "netanyahu", "putin",
)

CULTURE_KEYWORDS = (
    "oscar", "grammy", "emmy", "actor", "actress", "album", "song", "music", "movie",
    "film", "tv", "television", "box office", "box-office", "celebrity", "netflix",
    "reality show", "reality-show", "award", "bachelor", "drag race", "met gala",
)

CRYPTO_KEYWORDS = (
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "doge", "xrp", "crypto",
    "token", "coin", "blockchain", "airdrop",
)

MACRO_KEYWORDS = (
    "fed", "inflation", "cpi", "rates", "rate cut", "rate-cut", "recession", "gdp",
    "payroll", "jobs report", "jobs-report", "unemployment", "treasury", "yield",
    "stock", "s&p", "sp500", "nasdaq", "dow", "earnings",
)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write one CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def classify_vertical(event_slug: str | None, title: str | None) -> str:
    """Map one market to a broad research vertical using explainable rules."""

    text = f"{event_slug or ''} {title or ''}".strip().lower()
    if not text:
        return "other"

    for keyword in SPORTS_KEYWORDS:
        if keyword in text:
            return "sports"
    for keyword in POLITICS_KEYWORDS:
        if keyword in text:
            return "politics"
    for keyword in CRYPTO_KEYWORDS:
        if keyword in text:
            return "crypto"
    for keyword in MACRO_KEYWORDS:
        if keyword in text:
            return "macro"
    for keyword in CULTURE_KEYWORDS:
        if keyword in text:
            return "culture"
    return "other"


def build_vertical_wallet_summary(
    *,
    features: pd.DataFrame,
    recent_trades: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per wallet with broad vertical concentration metrics."""

    if recent_trades.empty:
        return pd.DataFrame()

    trades = recent_trades.copy()
    trades["vertical"] = trades.apply(
        lambda row: classify_vertical(row.get("event_slug"), row.get("title")),
        axis=1,
    )
    trades["wallet_address"] = trades["wallet_address"].astype(str).str.lower()

    feature_lookup = (
        features.set_index("wallet_id", drop=False).to_dict(orient="index")
        if not features.empty and "wallet_id" in features.columns
        else {}
    )

    records: list[dict[str, Any]] = []
    for wallet, group in trades.groupby("wallet_address", sort=False):
        feature_row = feature_lookup.get(wallet, {})
        vertical_counts = group["vertical"].value_counts()
        dominant_vertical = vertical_counts.index[0] if not vertical_counts.empty else None
        dominant_count = int(vertical_counts.iloc[0]) if not vertical_counts.empty else 0
        secondary_count = int(vertical_counts.iloc[1]) if len(vertical_counts) > 1 else 0
        daily_counts = group.groupby("trade_date").size().sort_values(ascending=False)

        pending = pd.to_numeric(pd.Series([feature_row.get("pending_open_copy_slices")]), errors="coerce").iloc[0]
        paired = pd.to_numeric(pd.Series([feature_row.get("paired_copy_slices")]), errors="coerce").iloc[0]
        pending_value = None if pd.isna(pending) else float(pending)
        paired_value = None if pd.isna(paired) else float(paired)
        open_share = _safe_ratio(
            pending_value,
            (pending_value or 0.0) + (paired_value or 0.0),
        )
        sell_share = _safe_ratio(
            feature_row.get("recent_sell_trades"),
            feature_row.get("recent_trades_window"),
        )
        if sell_share is None:
            sell_share = float(group["side"].eq("SELL").mean())
        observed_name = (
            group["sample_name_observed"].dropna().iloc[0]
            if "sample_name_observed" in group.columns and group["sample_name_observed"].notna().any()
            else None
        )

        record = {
            "wallet_address": wallet,
            "sample_name": feature_row.get("sample_name") or observed_name,
            "recent_trades_window": feature_row.get("recent_trades_window") or len(group),
            "observed_recent_trades": int(len(group)),
            "active_days_observed": int(group["trade_date"].nunique()),
            "avg_trades_per_active_day_observed": float(len(group) / max(group["trade_date"].nunique(), 1)),
            "max_trades_in_one_day_observed": int(daily_counts.iloc[0]) if not daily_counts.empty else None,
            "dominant_vertical": dominant_vertical,
            "dominant_vertical_trades": dominant_count,
            "dominant_vertical_share": float(dominant_count / len(group)) if len(group) else None,
            "top2_vertical_share": float((dominant_count + secondary_count) / len(group)) if len(group) else None,
            "distinct_verticals": int(vertical_counts.shape[0]),
            "sports_share": float(vertical_counts.get("sports", 0) / len(group)),
            "politics_share": float(vertical_counts.get("politics", 0) / len(group)),
            "culture_share": float(vertical_counts.get("culture", 0) / len(group)),
            "crypto_share": float(vertical_counts.get("crypto", 0) / len(group)),
            "macro_share": float(vertical_counts.get("macro", 0) / len(group)),
            "other_share": float(vertical_counts.get("other", 0) / len(group)),
            "sell_share_observed": sell_share,
            "open_copy_slice_share": open_share,
            "avg_copy_edge_net_15s": feature_row.get("avg_copy_edge_net_15s"),
            "avg_copy_edge_net_30s": feature_row.get("avg_copy_edge_net_30s"),
            "avg_copy_edge_net_60s_proxy": feature_row.get("avg_copy_edge_net_60s_proxy"),
            "fast_exit_share_30s": feature_row.get("fast_exit_share_30s"),
            "median_holding_seconds": feature_row.get("median_holding_seconds"),
            "rolling_3d_positive_share": feature_row.get("rolling_3d_positive_share"),
            "repeat_oos_test_positive_windows": feature_row.get("repeat_oos_test_positive_windows"),
            "repeat_oos_test_valid_slices_total": feature_row.get("repeat_oos_test_valid_slices_total"),
            "realized_pnl_abs": feature_row.get("realized_pnl_abs"),
        }
        records.append(record)

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    return result.sort_values(
        ["dominant_vertical_share", "avg_copy_edge_net_30s", "wallet_address"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def rank_vertical_specialists(
    summary: pd.DataFrame,
    *,
    min_recent_trades: int = 20,
    min_vertical_share: float = 0.55,
    min_edge_30s: float = 0.0,
    max_sell_share: float = 0.45,
    max_fast_exit_share_30s: float = 0.05,
) -> pd.DataFrame:
    """Filter and rank vertical-specialist wallets using simple explainable rules."""

    if summary.empty:
        return pd.DataFrame()

    filtered = summary.copy()
    filtered = filtered[
        pd.to_numeric(filtered["observed_recent_trades"], errors="coerce").fillna(0) >= min_recent_trades
    ]
    filtered = filtered[
        pd.to_numeric(filtered["dominant_vertical_share"], errors="coerce").fillna(0) >= min_vertical_share
    ]
    filtered = filtered[
        pd.to_numeric(filtered["avg_copy_edge_net_30s"], errors="coerce").fillna(-999) > min_edge_30s
    ]
    filtered = filtered[
        pd.to_numeric(filtered["sell_share_observed"], errors="coerce").fillna(1.0) <= max_sell_share
    ]
    filtered = filtered[
        pd.to_numeric(filtered["fast_exit_share_30s"], errors="coerce").fillna(1.0) <= max_fast_exit_share_30s
    ]
    if filtered.empty:
        return filtered

    filtered = filtered.copy()
    filtered["specialist_score"] = (
        pd.to_numeric(filtered["avg_copy_edge_net_30s"], errors="coerce").fillna(0) * 20.0
        + pd.to_numeric(filtered["dominant_vertical_share"], errors="coerce").fillna(0) * 4.0
        + pd.to_numeric(filtered["open_copy_slice_share"], errors="coerce").fillna(0) * 2.0
        + pd.to_numeric(filtered["rolling_3d_positive_share"], errors="coerce").fillna(0)
        - pd.to_numeric(filtered["sell_share_observed"], errors="coerce").fillna(1.0) * 2.5
        - pd.to_numeric(filtered["avg_trades_per_active_day_observed"], errors="coerce").fillna(0) * 0.03
    )
    return filtered.sort_values(
        ["dominant_vertical", "specialist_score", "dominant_vertical_share", "avg_copy_edge_net_30s"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_vertical_top_wallets(candidates: pd.DataFrame, *, top_n: int = 5) -> pd.DataFrame:
    """Return the top-N ranked wallets inside each vertical."""

    if candidates.empty:
        return pd.DataFrame()
    return (
        candidates.groupby("dominant_vertical", group_keys=False, sort=True)
        .head(top_n)
        .reset_index(drop=True)
    )


def build_vertical_report(
    *,
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    top_wallets: pd.DataFrame,
) -> str:
    """Render a short Markdown report."""

    lines = [
        "# Vertical Specialist Wallets",
        "",
        "This report classifies recent wallet activity into broad verticals and ranks",
        "wallets that look specialized rather than all-purpose.",
        "",
    ]

    if summary.empty:
        lines.append("No recent wallet data was available.")
        return "\n".join(lines)

    lines.extend(
        [
            "## Universe",
            "",
            f"- Wallets scored: {len(summary)}",
            f"- Candidate specialists after filters: {len(candidates)}",
            "",
            "## Rules",
            "",
            "- Minimum recent trades: `20`",
            "- Minimum dominant-vertical share: `55%`",
            "- `30s` delayed copy edge must stay positive",
            "- Sell share must stay at or below `45%`",
            "- Fast-exit share at `30s` must stay at or below `5%`",
            "",
            "## Top Wallets By Vertical",
            "",
        ]
    )

    for vertical, group in top_wallets.groupby("dominant_vertical", sort=True):
        lines.append(f"### {vertical.title()}")
        lines.append("")
        for row in group.to_dict(orient="records"):
            lines.append(
                f"- `{row['wallet_address']}` ({row.get('sample_name') or 'no name'}) | "
                f"score `{row.get('specialist_score')}` | "
                f"vertical share `{row.get('dominant_vertical_share')}` | "
                f"edge 30s `{row.get('avg_copy_edge_net_30s')}` | "
                f"sell share `{row.get('sell_share_observed')}` | "
                f"trades/day `{row.get('avg_trades_per_active_day_observed')}`"
            )
        lines.append("")

    if top_wallets.empty:
        lines.append("No strong specialists passed the filters.")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Vertical labels are rule-based and derived from event slug / title keywords.",
            "- This is a wallet-style screen, not a live trading model.",
            "- Culture may have fewer candidates simply because fewer recent trades in the captured sample map cleanly to that vertical.",
        ]
    )
    return "\n".join(lines)


def run_vertical_specialist_analysis(
    *,
    features_csv: str | Path = DEFAULT_FEATURES_CSV,
    recent_trades_csv: str | Path = DEFAULT_RECENT_TRADES_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the vertical-specialist wallet analysis."""

    features = load_wallet_features(features_csv)
    recent_trades = load_recent_trades(recent_trades_csv)
    summary = build_vertical_wallet_summary(features=features, recent_trades=recent_trades)
    candidates = rank_vertical_specialists(summary)
    top_wallets = build_vertical_top_wallets(candidates)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "vertical_specialist_report.md"
    report_path.write_text(
        build_vertical_report(summary=summary, candidates=candidates, top_wallets=top_wallets),
        encoding="utf-8",
    )

    paths = {
        "summary": _write_csv(summary, output_root / "vertical_wallet_summary.csv"),
        "candidates": _write_csv(candidates, output_root / "vertical_specialist_candidates.csv"),
        "top_wallets": _write_csv(top_wallets, output_root / "vertical_top_wallets.csv"),
        "report": report_path,
    }
    return {
        "summary": summary,
        "candidates": candidates,
        "top_wallets": top_wallets,
        "paths": paths,
    }


def print_vertical_specialist_summary(results: dict[str, Any]) -> None:
    """Print a concise console summary."""

    summary = results["summary"]
    top_wallets = results["top_wallets"]

    print("Vertical Specialist Wallets")
    print(f"Wallets scored: {len(summary)}")
    print(f"Candidate specialists: {len(results['candidates'])}")
    if top_wallets.empty:
        print("No specialists passed the current filters.")
        return
    print(
        top_wallets[
            [
                "dominant_vertical",
                "wallet_address",
                "sample_name",
                "specialist_score",
                "dominant_vertical_share",
                "avg_copy_edge_net_30s",
                "sell_share_observed",
            ]
        ].to_string(index=False)
    )
