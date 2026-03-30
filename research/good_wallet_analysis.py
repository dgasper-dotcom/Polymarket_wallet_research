"""Focused analysis for a short list of promising followable wallets.

This module stays in the research layer. It summarizes wallet behavior from
existing public-data exports and looks for "mirror" wallets that appear to
place very similar recent trades to a chosen anchor wallet.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd


DEFAULT_FEATURES_CSV = "data/wallet_features.csv"
DEFAULT_RECENT_TRADES_CSV = (
    "exports/recent_wallet_trade_capture_top1000_cohort/"
    "recent_wallet_trades_20260312_20260326.csv"
)
DEFAULT_OUTPUT_DIR = "exports/good_wallet_analysis"
DEFAULT_GOOD_WALLETS = (
    "0x53ecc53e7a69aad0e6dda60264cc2e363092df91",
    "0x77fd7aec1952ea7d042a6eec83bc4782f67db6c8",
)
DEFAULT_MIRROR_ANCHOR = DEFAULT_GOOD_WALLETS[0]


def _read_csv(path: str | Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read one CSV file while tolerating missing paths."""

    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=usecols or [])
    return pd.read_csv(csv_path, usecols=usecols)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write one DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _normalize_wallets(wallets: list[str] | tuple[str, ...]) -> list[str]:
    """Normalize wallet identifiers."""

    seen: set[str] = set()
    ordered: list[str] = []
    for wallet in wallets:
        normalized = str(wallet or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """Return a numeric ratio or None."""

    if numerator is None or denominator is None:
        return None
    denominator_float = float(denominator)
    if denominator_float == 0:
        return None
    return float(numerator) / denominator_float


def load_wallet_features(path: str | Path = DEFAULT_FEATURES_CSV) -> pd.DataFrame:
    """Load wallet-level feature exports."""

    frame = _read_csv(path)
    if frame.empty:
        return frame
    if "wallet_id" in frame.columns:
        frame["wallet_id"] = frame["wallet_id"].astype(str).str.strip().str.lower()
    return frame


def _extract_raw_trade_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse a few raw JSON fields from recent trade rows."""

    if frame.empty or "raw_json" not in frame.columns:
        return frame

    extracted: list[dict[str, Any]] = []
    for raw in frame["raw_json"].tolist():
        payload: dict[str, Any]
        if not isinstance(raw, str) or not raw.strip():
            payload = {}
        else:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
        extracted.append(
            {
                "event_slug": payload.get("eventSlug") or payload.get("slug"),
                "title": payload.get("title"),
                "outcome": payload.get("outcome"),
                "sample_name_observed": payload.get("name"),
                "sample_pseudonym_observed": payload.get("pseudonym"),
            }
        )

    return pd.concat([frame.reset_index(drop=True), pd.DataFrame.from_records(extracted)], axis=1)


def load_recent_trades(path: str | Path = DEFAULT_RECENT_TRADES_CSV) -> pd.DataFrame:
    """Load recent wallet-trade exports and normalize core fields."""

    usecols = [
        "wallet_address",
        "market_id",
        "token_id",
        "side",
        "price",
        "size",
        "usdc_size",
        "timestamp",
        "raw_json",
    ]
    frame = _read_csv(path, usecols=usecols)
    if frame.empty:
        return frame

    frame["wallet_address"] = frame["wallet_address"].astype(str).str.strip().str.lower()
    frame["side"] = frame["side"].astype(str).str.strip().str.upper()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
    frame["usdc_size"] = pd.to_numeric(frame["usdc_size"], errors="coerce")
    frame = frame.dropna(subset=["wallet_address", "timestamp"]).reset_index(drop=True)
    frame["trade_date"] = frame["timestamp"].dt.floor("D")
    frame["epoch_seconds"] = (frame["timestamp"].astype("int64") // 10**9).astype(int)
    return _extract_raw_trade_fields(frame)


def _top_value_summary(group: pd.DataFrame, column: str) -> tuple[str | None, int | None, float | None]:
    """Return top value, count, and share for one categorical column."""

    if column not in group.columns or group.empty:
        return None, None, None
    counts = group[column].dropna().astype(str).value_counts()
    if counts.empty:
        return None, None, None
    top_value = counts.index[0]
    top_count = int(counts.iloc[0])
    top_share = float(top_count / len(group))
    return top_value, top_count, top_share


def build_good_wallet_summary(
    wallets: list[str] | tuple[str, ...],
    *,
    features: pd.DataFrame,
    recent_trades: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per selected wallet with wallet-level behavior notes."""

    normalized_wallets = _normalize_wallets(wallets)
    if not normalized_wallets:
        return pd.DataFrame()

    feature_subset = (
        features[features["wallet_id"].isin(normalized_wallets)].copy()
        if not features.empty and "wallet_id" in features.columns
        else pd.DataFrame()
    )
    feature_lookup = feature_subset.set_index("wallet_id", drop=False).to_dict(orient="index") if not feature_subset.empty else {}
    trade_subset = recent_trades[recent_trades["wallet_address"].isin(normalized_wallets)].copy() if not recent_trades.empty else pd.DataFrame()

    records: list[dict[str, Any]] = []
    for wallet in normalized_wallets:
        group = trade_subset[trade_subset["wallet_address"] == wallet].copy()
        feature_row = feature_lookup.get(wallet, {})

        top_event_slug, top_event_count, top_event_share = _top_value_summary(group, "event_slug")
        top_title, top_title_count, top_title_share = _top_value_summary(group, "title")
        top_market, top_market_count, top_market_share = _top_value_summary(group, "market_id")
        top_outcome, _, _ = _top_value_summary(group, "outcome")

        daily_counts = (
            group.groupby("trade_date").size().sort_values(ascending=False)
            if not group.empty
            else pd.Series(dtype=float)
        )
        intertrade_seconds = (
            group.sort_values("timestamp")["timestamp"].diff().dt.total_seconds().dropna()
            if not group.empty
            else pd.Series(dtype=float)
        )

        pending = pd.to_numeric(pd.Series([feature_row.get("pending_open_copy_slices")]), errors="coerce").iloc[0]
        paired = pd.to_numeric(pd.Series([feature_row.get("paired_copy_slices")]), errors="coerce").iloc[0]
        pending_value = None if pd.isna(pending) else float(pending)
        paired_value = None if pd.isna(paired) else float(paired)
        open_share = _safe_ratio(
            pending_value,
            (pending_value or 0.0) + (paired_value or 0.0),
        )
        observed_name = (
            group["sample_name_observed"].dropna().iloc[0]
            if not group.empty and "sample_name_observed" in group.columns and group["sample_name_observed"].notna().any()
            else None
        )
        observed_pseudonym = (
            group["sample_pseudonym_observed"].dropna().iloc[0]
            if not group.empty and "sample_pseudonym_observed" in group.columns and group["sample_pseudonym_observed"].notna().any()
            else None
        )
        sell_share_observed = (
            float(group["side"].eq("SELL").mean())
            if not group.empty
            else feature_row.get("recent_sell_trades") / feature_row.get("recent_trades_window")
            if feature_row.get("recent_trades_window")
            else None
        )

        records.append(
            {
                "wallet_address": wallet,
                "sample_name": feature_row.get("sample_name") or observed_name,
                "sample_pseudonym": feature_row.get("sample_pseudonym") or observed_pseudonym,
                "recent_trades_window": feature_row.get("recent_trades_window") if feature_row else len(group),
                "observed_recent_trades": int(len(group)),
                "active_days_observed": int(group["trade_date"].nunique()) if not group.empty else None,
                "avg_trades_per_active_day_observed": (
                    float(len(group) / max(group["trade_date"].nunique(), 1))
                    if not group.empty
                    else None
                ),
                "max_trades_in_one_day_observed": int(daily_counts.iloc[0]) if not daily_counts.empty else None,
                "sell_share_observed": sell_share_observed,
                "buy_share_observed": (
                    float(group["side"].eq("BUY").mean()) if not group.empty else None
                ),
                "distinct_markets_observed": int(group["market_id"].nunique(dropna=True)) if not group.empty else None,
                "distinct_event_slugs_observed": int(group["event_slug"].nunique(dropna=True)) if not group.empty else None,
                "top_event_slug": top_event_slug,
                "top_event_slug_trade_count": top_event_count,
                "top_event_slug_trade_share": top_event_share,
                "top_title": top_title,
                "top_title_trade_count": top_title_count,
                "top_title_trade_share": top_title_share,
                "top_market_id": top_market,
                "top_market_trade_count": top_market_count,
                "top_market_trade_share": top_market_share,
                "top_outcome": top_outcome,
                "first_recent_trade_ts": (
                    group["timestamp"].min().isoformat() if not group.empty else None
                ),
                "most_recent_trade_ts": (
                    group["timestamp"].max().isoformat() if not group.empty else None
                ),
                "avg_intertrade_seconds": (
                    float(intertrade_seconds.mean()) if not intertrade_seconds.empty else None
                ),
                "median_intertrade_seconds": (
                    float(intertrade_seconds.median()) if not intertrade_seconds.empty else None
                ),
                "avg_position_size_usdc": feature_row.get("avg_position_size_usdc"),
                "avg_holding_seconds": feature_row.get("avg_holding_seconds"),
                "median_holding_seconds": feature_row.get("median_holding_seconds"),
                "avg_copy_edge_net_15s": feature_row.get("avg_copy_edge_net_15s"),
                "avg_copy_edge_net_30s": feature_row.get("avg_copy_edge_net_30s"),
                "avg_copy_edge_net_60s_proxy": feature_row.get("avg_copy_edge_net_60s_proxy"),
                "fast_exit_share_30s": feature_row.get("fast_exit_share_30s"),
                "rolling_3d_positive_share": feature_row.get("rolling_3d_positive_share"),
                "repeat_oos_test_positive_windows": feature_row.get("repeat_oos_test_positive_windows"),
                "repeat_oos_test_valid_slices_total": feature_row.get("repeat_oos_test_valid_slices_total"),
                "realized_pnl_abs": feature_row.get("realized_pnl_abs"),
                "open_copy_slice_share": open_share,
                "pending_open_copy_slices": pending_value,
                "paired_copy_slices": paired_value,
            }
        )

    summary = pd.DataFrame.from_records(records)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["avg_copy_edge_net_30s", "open_copy_slice_share", "sell_share_observed", "wallet_address"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_wallet_top_events(recent_trades: pd.DataFrame, wallets: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Return per-wallet top event / title rows for quick inspection."""

    normalized_wallets = _normalize_wallets(wallets)
    trade_subset = recent_trades[recent_trades["wallet_address"].isin(normalized_wallets)].copy()
    if trade_subset.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for wallet, group in trade_subset.groupby("wallet_address", sort=False):
        total_trades = len(group)
        event_counts = (
            group.groupby(["event_slug", "title"], dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        for (event_slug, title), count in event_counts.items():
            records.append(
                {
                    "wallet_address": wallet,
                    "event_slug": event_slug,
                    "title": title,
                    "trade_count": int(count),
                    "trade_share": float(count / total_trades),
                }
            )
    result = pd.DataFrame.from_records(records)
    return result.sort_values(
        ["wallet_address", "trade_count", "event_slug"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_wallet_daily_activity(recent_trades: pd.DataFrame, wallets: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Return one row per wallet per day."""

    normalized_wallets = _normalize_wallets(wallets)
    trade_subset = recent_trades[recent_trades["wallet_address"].isin(normalized_wallets)].copy()
    if trade_subset.empty:
        return pd.DataFrame()

    grouped = (
        trade_subset.groupby(["wallet_address", "trade_date"], as_index=False)
        .agg(
            trades=("wallet_address", "size"),
            buys=("side", lambda values: int((values == "BUY").sum())),
            sells=("side", lambda values: int((values == "SELL").sum())),
            distinct_markets=("market_id", "nunique"),
            distinct_events=("event_slug", "nunique"),
            usdc_volume=("usdc_size", "sum"),
        )
        .sort_values(["wallet_address", "trade_date"], ascending=[True, True])
        .reset_index(drop=True)
    )
    grouped["trade_date"] = pd.to_datetime(grouped["trade_date"], utc=True, errors="coerce").apply(
        lambda value: value.date().isoformat() if pd.notna(value) else None
    )
    return grouped


def _prepare_anchor_pairs(
    anchor_wallet: str,
    recent_trades: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare anchor trades and non-anchor trades for mirror matching."""

    anchor = recent_trades[recent_trades["wallet_address"] == anchor_wallet].copy()
    others = recent_trades[recent_trades["wallet_address"] != anchor_wallet].copy()
    required = ["token_id", "side", "timestamp", "price", "wallet_address"]
    anchor = anchor.dropna(subset=required)
    others = others.dropna(subset=required)
    if anchor.empty or others.empty:
        return anchor, others

    anchor = anchor.reset_index(drop=True)
    anchor["anchor_trade_id"] = anchor.index.astype(str)
    return anchor, others


def build_mirror_wallet_candidates(
    anchor_wallet: str,
    *,
    recent_trades: pd.DataFrame,
    features: pd.DataFrame,
    price_tolerance: float = 0.001,
) -> pd.DataFrame:
    """Rank wallets whose recent trades most closely resemble an anchor wallet.

    Matching rules are descriptive and conservative:
    - same token and same side
    - pick the nearest trade from each candidate wallet for each anchor trade
    - report overlap at 5s / 15s / 30s windows
    - separately flag "same-size" matches for likely mirrored orders
    """

    anchor_wallet_normalized = str(anchor_wallet).strip().lower()
    anchor, others = _prepare_anchor_pairs(anchor_wallet_normalized, recent_trades)
    if anchor.empty or others.empty:
        return pd.DataFrame()

    merged = anchor.merge(
        others,
        on=["token_id", "side"],
        how="inner",
        suffixes=("_anchor", "_candidate"),
    )
    if merged.empty:
        return pd.DataFrame()

    merged["time_diff_s"] = (
        merged["epoch_seconds_candidate"] - merged["epoch_seconds_anchor"]
    ).abs()
    merged["price_diff_abs"] = (merged["price_candidate"] - merged["price_anchor"]).abs()
    merged["size_ratio"] = (
        merged["usdc_size_candidate"] / merged["usdc_size_anchor"]
    ).where(pd.to_numeric(merged["usdc_size_anchor"], errors="coerce") > 0)
    merged = merged[merged["time_diff_s"] <= 30].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["log_size_distance"] = pd.to_numeric(merged["size_ratio"], errors="coerce").map(
        lambda value: abs(math.log(float(value))) if pd.notna(value) and float(value) > 0 else 999.0
    )
    merged = merged.sort_values(
        ["wallet_address_candidate", "anchor_trade_id", "time_diff_s", "price_diff_abs", "log_size_distance"],
        ascending=[True, True, True, True, True],
    )
    best = merged.drop_duplicates(subset=["wallet_address_candidate", "anchor_trade_id"], keep="first").copy()

    feature_lookup = (
        features.set_index("wallet_id", drop=False).to_dict(orient="index")
        if not features.empty and "wallet_id" in features.columns
        else {}
    )

    anchor_total = len(anchor)
    records: list[dict[str, Any]] = []
    for wallet, group in best.groupby("wallet_address_candidate", sort=False):
        match_5s = group[group["time_diff_s"] <= 5]
        match_15s = group[group["time_diff_s"] <= 15]
        match_30s = group[group["time_diff_s"] <= 30]
        exact_5s = match_5s[match_5s["price_diff_abs"] <= price_tolerance]
        exact_15s = match_15s[match_15s["price_diff_abs"] <= price_tolerance]
        exact_30s = match_30s[match_30s["price_diff_abs"] <= price_tolerance]
        same_size_5s = exact_5s[exact_5s["size_ratio"].between(0.9, 1.1, inclusive="both")]

        feature_row = feature_lookup.get(wallet, {})
        records.append(
            {
                "anchor_wallet": anchor_wallet_normalized,
                "candidate_wallet": wallet,
                "candidate_sample_name": feature_row.get("sample_name"),
                "matched_anchor_trades_5s": int(len(match_5s)),
                "matched_anchor_trades_15s": int(len(match_15s)),
                "matched_anchor_trades_30s": int(len(match_30s)),
                "exact_price_matches_5s": int(len(exact_5s)),
                "exact_price_matches_15s": int(len(exact_15s)),
                "exact_price_matches_30s": int(len(exact_30s)),
                "same_size_matches_5s": int(len(same_size_5s)),
                "match_share_5s": float(len(match_5s) / anchor_total),
                "match_share_15s": float(len(match_15s) / anchor_total),
                "match_share_30s": float(len(match_30s) / anchor_total),
                "median_time_diff_s": float(group["time_diff_s"].median()),
                "median_price_diff_abs": float(group["price_diff_abs"].median()),
                "median_size_ratio": (
                    float(pd.to_numeric(group["size_ratio"], errors="coerce").dropna().median())
                    if pd.to_numeric(group["size_ratio"], errors="coerce").dropna().shape[0] > 0
                    else None
                ),
                "shared_event_slugs_30s": int(group["event_slug_anchor"].nunique(dropna=True)),
                "candidate_recent_trades_window": feature_row.get("recent_trades_window"),
                "candidate_sell_share": _safe_ratio(
                    feature_row.get("recent_sell_trades"),
                    feature_row.get("recent_trades_window"),
                ),
                "candidate_avg_copy_edge_net_30s": feature_row.get("avg_copy_edge_net_30s"),
            }
        )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    return result.sort_values(
        [
            "exact_price_matches_15s",
            "same_size_matches_5s",
            "matched_anchor_trades_30s",
            "median_time_diff_s",
        ],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_markdown_report(
    *,
    wallet_summary: pd.DataFrame,
    top_events: pd.DataFrame,
    mirror_candidates: pd.DataFrame,
    mirror_anchor: str,
) -> str:
    """Render a concise Markdown summary."""

    lines = [
        "# Good Wallet Analysis",
        "",
        "This report summarizes the currently preferred slower-hold wallets and checks",
        "whether the anchor wallet appears to have a close recent-trade mirror.",
        "",
    ]

    if wallet_summary.empty:
        lines.append("No wallet data was available.")
        return "\n".join(lines)

    lines.extend(
        [
            "## Wallet Summary",
            "",
            f"- Wallets analyzed: {len(wallet_summary)}",
            f"- Mirror anchor: `{mirror_anchor}`",
            "",
        ]
    )

    for row in wallet_summary.to_dict(orient="records"):
        lines.append(
            f"- `{row['wallet_address']}` ({row.get('sample_name') or 'no name'}) | "
            f"trades `{row.get('observed_recent_trades')}` | "
            f"active days `{row.get('active_days_observed')}` | "
            f"sell share `{row.get('sell_share_observed')}` | "
            f"open share `{row.get('open_copy_slice_share')}` | "
            f"edge 30s `{row.get('avg_copy_edge_net_30s')}` | "
            f"median hold sec `{row.get('median_holding_seconds')}` | "
            f"top event `{row.get('top_event_slug')}`"
        )

    lines.extend(["", "## Top Event Focus", ""])
    for wallet, group in top_events.groupby("wallet_address", sort=False):
        lines.append(f"- `{wallet}`")
        for row in group.head(5).to_dict(orient="records"):
            lines.append(
                f"  {row.get('event_slug') or 'unknown'} | trades `{row.get('trade_count')}` | "
                f"share `{row.get('trade_share')}`"
            )

    lines.extend(["", "## Mirror Candidates", ""])
    if mirror_candidates.empty:
        lines.append(f"- No mirror candidates were found for `{mirror_anchor}`.")
    else:
        top = mirror_candidates.head(10)
        strongest = top.iloc[0]
        if int(strongest.get("exact_price_matches_15s") or 0) <= 2:
            lines.append(
                f"- No strong recent mirror was found for `{mirror_anchor}`. "
                "The top overlap candidate still shares only a very small number of near-synchronous trades."
            )
        for row in top.to_dict(orient="records"):
            lines.append(
                f"- `{row['candidate_wallet']}` ({row.get('candidate_sample_name') or 'no name'}) | "
                f"exact 15s `{row.get('exact_price_matches_15s')}` | "
                f"same-size 5s `{row.get('same_size_matches_5s')}` | "
                f"share 30s `{row.get('match_share_30s')}` | "
                f"median dt `{row.get('median_time_diff_s')}`"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Mirror detection is descriptive only. It looks for same-token, same-side trades placed near each other in time.",
            "- A low overlap score does not prove the wallet is unique. It only says there is no strong recent duplicate pattern in the captured sample.",
            "- The analysis uses the existing recent-trade export window and does not introduce new lookahead logic.",
        ]
    )
    return "\n".join(lines)


def run_good_wallet_analysis(
    *,
    wallets: list[str] | tuple[str, ...] = DEFAULT_GOOD_WALLETS,
    mirror_anchor: str = DEFAULT_MIRROR_ANCHOR,
    features_csv: str | Path = DEFAULT_FEATURES_CSV,
    recent_trades_csv: str | Path = DEFAULT_RECENT_TRADES_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run a focused good-wallet analysis from existing research exports."""

    normalized_wallets = _normalize_wallets(wallets)
    features = load_wallet_features(features_csv)
    recent_trades = load_recent_trades(recent_trades_csv)
    wallet_summary = build_good_wallet_summary(
        normalized_wallets,
        features=features,
        recent_trades=recent_trades,
    )
    top_events = build_wallet_top_events(recent_trades, normalized_wallets)
    daily_activity = build_wallet_daily_activity(recent_trades, normalized_wallets)
    mirror_candidates = build_mirror_wallet_candidates(
        str(mirror_anchor).strip().lower(),
        recent_trades=recent_trades,
        features=features,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "good_wallet_analysis_report.md"
    report_path.write_text(
        build_markdown_report(
            wallet_summary=wallet_summary,
            top_events=top_events,
            mirror_candidates=mirror_candidates,
            mirror_anchor=str(mirror_anchor).strip().lower(),
        ),
        encoding="utf-8",
    )

    paths = {
        "wallet_summary": _write_csv(wallet_summary, output_root / "good_wallet_summary.csv"),
        "top_events": _write_csv(top_events, output_root / "good_wallet_top_events.csv"),
        "daily_activity": _write_csv(daily_activity, output_root / "good_wallet_daily_activity.csv"),
        "mirror_candidates": _write_csv(
            mirror_candidates,
            output_root / "mirror_wallet_candidates.csv",
        ),
        "report": report_path,
    }
    return {
        "wallet_summary": wallet_summary,
        "top_events": top_events,
        "daily_activity": daily_activity,
        "mirror_candidates": mirror_candidates,
        "paths": paths,
    }


def print_good_wallet_analysis_summary(results: dict[str, Any]) -> None:
    """Print a concise console summary."""

    wallet_summary = results["wallet_summary"]
    mirror_candidates = results["mirror_candidates"]

    print("Good Wallet Analysis")
    print(f"Wallets analyzed: {len(wallet_summary)}")
    if not wallet_summary.empty:
        print(
            wallet_summary[
                [
                    "wallet_address",
                    "sample_name",
                    "observed_recent_trades",
                    "active_days_observed",
                    "sell_share_observed",
                    "open_copy_slice_share",
                    "avg_copy_edge_net_30s",
                    "median_holding_seconds",
                    "top_event_slug",
                ]
            ].to_string(index=False)
        )

    print("Mirror Candidates")
    if mirror_candidates.empty:
        print("No mirror candidates found.")
        return
    print(
        mirror_candidates[
            [
                "candidate_wallet",
                "candidate_sample_name",
                "exact_price_matches_15s",
                "same_size_matches_5s",
                "match_share_30s",
                "median_time_diff_s",
            ]
        ].head(10).to_string(index=False)
    )
