"""Re-evaluate the full 1067-wallet active universe with expiry-hold evidence."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd

from db.session import get_session
from research.active_watchlist import (
    DEFAULT_CURRENT_ACTIVE_CSV,
    DEFAULT_VERTICAL_SUMMARY_CSV,
    MANUAL_EXCLUDE_WALLETS,
    _style_candidate_mask,
    score_watchlist,
)
from research.copy_follow_expiry import _build_terminal_lookup, load_markets_frame
from research.copy_follow_wallet_exit import build_copy_exit_pairs
from research.costs import calculate_net_pnl, estimate_entry_only_cost
from research.delay_analysis import _build_price_index, _to_epoch_seconds, load_price_history_frame, lookup_forward_price
from research.resolved_expiry_report import (
    DEFAULT_ANALYSIS_CUTOFF,
    DEFAULT_FEATURES_CSV,
    DEFAULT_RECENT_TRADES_CSV,
    compute_resolved_expiry_positions_from_frame,
    load_long_window_recent_trades,
)


DEFAULT_OUTPUT_DIR = "exports/all_active_1067_reevaluation"
PRICE_HISTORY_CHUNK_SIZE = 750


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read one CSV file while tolerating missing paths."""

    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write one CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _normalize_wallet_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Normalize one wallet-id column."""

    if frame.empty or column not in frame.columns:
        return frame
    normalized = frame.copy()
    normalized[column] = normalized[column].astype(str).str.strip().str.lower()
    normalized = normalized.loc[normalized[column] != ""].copy()
    return normalized


def _safe_ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    """Return a ratio or None."""

    if numerator is None or denominator is None:
        return None
    denominator_float = float(denominator)
    if denominator_float == 0:
        return None
    return float(numerator) / denominator_float


def _load_price_history_chunked(
    token_ids: list[str],
    *,
    end_ts: Any | None = None,
    chunk_size: int = PRICE_HISTORY_CHUNK_SIZE,
) -> pd.DataFrame:
    """Load price history in SQLite-safe chunks."""

    unique_token_ids = [str(token_id) for token_id in token_ids if token_id]
    if not unique_token_ids:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    with get_session() as session:
        for start in range(0, len(unique_token_ids), chunk_size):
            chunk = unique_token_ids[start : start + chunk_size]
            frames.append(load_price_history_frame(session, token_ids=chunk, end_ts=end_ts))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["token_id", "ts"], keep="last")


def load_all_active_frame(
    *,
    current_active_csv: str | Path = DEFAULT_CURRENT_ACTIVE_CSV,
    features_csv: str | Path = DEFAULT_FEATURES_CSV,
    vertical_summary_csv: str | Path = DEFAULT_VERTICAL_SUMMARY_CSV,
) -> pd.DataFrame:
    """Load the full active wallet universe and merge feature exports onto it."""

    current = _normalize_wallet_column(_read_csv(current_active_csv), "wallet_address")
    features = _normalize_wallet_column(_read_csv(features_csv), "wallet_id")
    vertical = _normalize_wallet_column(_read_csv(vertical_summary_csv), "wallet_address")
    if current.empty:
        return pd.DataFrame()

    frame = current.rename(
        columns={
            "wallet_address": "wallet_id",
            "total_trades": "total_trades_scan_universe",
            "distinct_markets": "distinct_markets_scan_universe",
            "first_seen_trade_ts": "first_seen_trade_ts_scan_universe",
            "most_recent_trade_ts": "most_recent_trade_ts_scan_universe",
        }
    ).copy()
    frame["wallet_id"] = frame["wallet_id"].astype(str).str.lower()
    frame["is_current_active"] = True

    if not features.empty:
        frame = frame.merge(features, on="wallet_id", how="left", suffixes=("_scan", ""))
    if not vertical.empty:
        frame = frame.merge(
            vertical[
                [
                    "wallet_address",
                    "dominant_vertical",
                    "dominant_vertical_share",
                    "avg_trades_per_active_day_observed",
                ]
            ],
            left_on="wallet_id",
            right_on="wallet_address",
            how="left",
        )

    recent_trades_window = pd.to_numeric(frame.get("recent_trades_window"), errors="coerce")
    sell_trades = pd.to_numeric(frame.get("recent_sell_trades"), errors="coerce")
    frame["sell_share"] = sell_trades / recent_trades_window
    pending = pd.to_numeric(frame.get("pending_open_copy_slices"), errors="coerce")
    paired = pd.to_numeric(frame.get("paired_copy_slices"), errors="coerce")
    frame["open_share"] = pending / (pending + paired)
    frame["holding_days"] = pd.to_numeric(frame.get("median_holding_seconds"), errors="coerce") / 86400.0
    frame["manual_excluded"] = frame["wallet_id"].isin(MANUAL_EXCLUDE_WALLETS)
    return frame


def merge_expiry_behavior(
    frame: pd.DataFrame,
    *,
    recent_trades_csv: str | Path = DEFAULT_RECENT_TRADES_CSV,
    analysis_cutoff: str = DEFAULT_ANALYSIS_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge long-horizon expiry-hold behavior into the active universe."""

    if frame.empty:
        return frame.copy(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    wallets = frame["wallet_id"].astype(str).str.lower().tolist()
    recent_trades = load_long_window_recent_trades(recent_trades_csv=recent_trades_csv, wallets=wallets)
    with get_session() as session:
        terminal_lookup = _build_terminal_lookup(load_markets_frame(session))

    expiry_summary, resolved_trades, overview, diagnostics = compute_resolved_expiry_positions_from_frame(
        recent_trades,
        terminal_lookup,
        analysis_cutoff=analysis_cutoff,
        return_diagnostics=True,
    )
    base = pd.DataFrame({"wallet_id": wallets}).drop_duplicates().reset_index(drop=True)
    expiry_summary = base.merge(expiry_summary, on="wallet_id", how="left")

    trade_counts = (
        recent_trades.groupby("wallet_address", sort=False).size().astype(int).rename("long_window_trade_rows")
        if not recent_trades.empty
        else pd.Series(dtype=int)
    )
    expiry_summary = expiry_summary.merge(
        trade_counts.rename_axis("wallet_id").reset_index(),
        on="wallet_id",
        how="left",
    )
    for column in (
        "wallet_sell_closed_slices",
        "held_to_expiry_observed_slices",
        "unresolved_open_slices",
        "resolved_behavior_slices",
        "total_behavior_slices",
        "long_window_trade_rows",
    ):
        if column in expiry_summary.columns:
            expiry_summary[column] = pd.to_numeric(expiry_summary[column], errors="coerce").fillna(0).astype(int)

    merged = frame.merge(expiry_summary, on="wallet_id", how="left", suffixes=("", "_expiry"))
    return merged, resolved_trades, overview, diagnostics


def _lookup_latest_price_at_or_before(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    target_ts: int,
) -> tuple[float | None, str, int | None]:
    """Return the latest cached public price at or before one timestamp."""

    if token_id is None or token_id not in index:
        return None, "missing_prices", None

    times, prices = index[token_id]
    if not times:
        return None, "missing_prices", None

    position = bisect_right(times, target_ts) - 1
    if position < 0:
        return None, "missing_prices", None
    age_seconds = target_ts - times[position]
    return prices[position], "price_history_latest_before_cutoff", age_seconds


def compute_open_position_evidence(
    recent_trades: pd.DataFrame,
    terminal_lookup: dict[str, dict[str, Any]],
    *,
    analysis_cutoff: str = DEFAULT_ANALYSIS_CUTOFF,
    collect_trade_records: bool = True,
    paired_counts: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute MTM and forward evidence for unresolved open positions.

    Scope:
    - Only BUY lots that remain open by the analysis cutoff.
    - Exclude lots whose market publicly resolved by the cutoff; those belong to
      the resolved-expiry evidence table instead.
    - MTM and forward 7d/30d metrics use public price history where available,
      with current Gamma outcomePrices as a fallback mark source only for MTM.
    """

    if recent_trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    cutoff = pd.to_datetime(analysis_cutoff, utc=True, errors="coerce")
    if cutoff is None or pd.isna(cutoff):
        cutoff = pd.to_datetime(recent_trades["timestamp"], utc=True, errors="coerce").max()
    cutoff_epoch = _to_epoch_seconds(cutoff)

    paired_counts = paired_counts or {}

    if "resolved_status" in recent_trades.columns:
        open_frame = recent_trades.copy()
        open_frame["wallet_address"] = open_frame["wallet_address"].astype(str).str.lower()
        open_frame["token_id"] = open_frame["token_id"].astype(str)
        open_frame["buy_timestamp"] = pd.to_datetime(open_frame["buy_timestamp"], utc=True, errors="coerce")
        open_frame["buy_price_signal"] = pd.to_numeric(open_frame["buy_price_signal"], errors="coerce")
        open_frame["copied_size"] = pd.to_numeric(open_frame["copied_size"], errors="coerce")
        open_frame["resolution_ts"] = pd.to_datetime(open_frame.get("resolution_ts"), utc=True, errors="coerce")
        open_frame = open_frame.loc[open_frame["resolved_status"] == "unresolved_open"].copy()
        open_frame = open_frame.dropna(
            subset=["wallet_address", "buy_timestamp", "token_id", "buy_price_signal", "copied_size"]
        )
        if open_frame.empty:
            return pd.DataFrame(), pd.DataFrame()
    else:
        pairs, open_positions = build_copy_exit_pairs(recent_trades)
        if open_positions.empty:
            return pd.DataFrame(), pd.DataFrame()

        open_frame = open_positions.copy()
        open_frame["wallet_address"] = open_frame["wallet_address"].astype(str).str.lower()
        open_frame["token_id"] = open_frame["token_id"].astype(str)
        open_frame["buy_timestamp"] = pd.to_datetime(open_frame["buy_timestamp"], utc=True, errors="coerce")
        open_frame["buy_price_signal"] = pd.to_numeric(open_frame["buy_price_signal"], errors="coerce")
        open_frame["copied_size"] = pd.to_numeric(open_frame["copied_size"], errors="coerce")
        open_frame = open_frame.dropna(
            subset=["wallet_address", "buy_timestamp", "token_id", "buy_price_signal", "copied_size"]
        )
        if open_frame.empty:
            return pd.DataFrame(), pd.DataFrame()

        def _terminal_value(token_id: str, key: str) -> Any:
            return (terminal_lookup.get(token_id) or {}).get(key)

        open_frame["resolution_ts"] = pd.to_datetime(
            open_frame["token_id"].map(lambda token_id: _terminal_value(token_id, "resolution_ts")),
            utc=True,
            errors="coerce",
        )
        open_frame = open_frame.loc[
            open_frame["resolution_ts"].isna() | (open_frame["resolution_ts"] > cutoff)
        ].copy()
        if open_frame.empty:
            return pd.DataFrame(), pd.DataFrame()

        if not paired_counts:
            paired_counts = (
                pairs.groupby("wallet_address", sort=False).size().astype(int).to_dict()
                if not pairs.empty
                else {}
            )

    open_frame["entry_cost"] = open_frame.apply(
        lambda row: float(
            estimate_entry_only_cost(
                pd.Series(row),
                entry_price=float(row["buy_price_signal"]),
            )["total_cost"]
        ),
        axis=1,
    )
    open_frame["holding_days_open"] = (cutoff - open_frame["buy_timestamp"]).dt.total_seconds() / 86400.0

    wallet_stats: dict[str, dict[str, float | int]] = {}

    def _ensure_wallet(wallet_id: str) -> dict[str, float | int]:
        return wallet_stats.setdefault(
            wallet_id,
            {
                "unresolved_open_slices": 0,
                "mtm_total_net_usdc_sum": 0.0,
                "mtm_net_sum": 0.0,
                "mtm_net_count": 0,
                "mtm_positive_count": 0,
                "holding_days_sum": 0.0,
                "holding_days_count": 0,
                "forward_7d_net_sum": 0.0,
                "forward_7d_count": 0,
                "forward_7d_positive_count": 0,
                "forward_30d_net_sum": 0.0,
                "forward_30d_count": 0,
                "forward_30d_positive_count": 0,
            },
        )

    unresolved_records: list[dict[str, Any]] = []
    token_ids = sorted(open_frame["token_id"].dropna().astype(str).unique())
    for start in range(0, len(token_ids), PRICE_HISTORY_CHUNK_SIZE):
        chunk_ids = token_ids[start : start + PRICE_HISTORY_CHUNK_SIZE]
        subset = open_frame.loc[open_frame["token_id"].isin(chunk_ids)].copy()
        if subset.empty:
            continue

        price_history = _load_price_history_chunked(chunk_ids, end_ts=cutoff, chunk_size=PRICE_HISTORY_CHUNK_SIZE)
        price_index = _build_price_index(price_history)

        for row in subset.to_dict(orient="records"):
            wallet_id = str(row["wallet_address"])
            token_id = str(row["token_id"])
            terminal_info = terminal_lookup.get(token_id) or {
                "terminal_price": row.get("terminal_price"),
                "terminal_price_source": row.get("terminal_price_source"),
                "resolution_ts": row.get("resolution_ts"),
            }
            buy_ts = pd.to_datetime(row["buy_timestamp"], utc=True, errors="coerce")
            buy_epoch = _to_epoch_seconds(buy_ts)
            buy_price = float(row["buy_price_signal"])
            copied_size = float(row["copied_size"])
            entry_cost = float(row["entry_cost"])
            holding_days = float(row["holding_days_open"])

            stats = _ensure_wallet(wallet_id)
            stats["unresolved_open_slices"] += 1
            stats["holding_days_sum"] += holding_days
            stats["holding_days_count"] += 1

            mtm_price, mtm_source, mtm_age_seconds = _lookup_latest_price_at_or_before(price_index, token_id, cutoff_epoch)
            if mtm_price is None and terminal_info.get("terminal_price") is not None:
                mtm_price = float(terminal_info.get("terminal_price"))
                mtm_source = "gamma_outcome_prices_current_fallback"
                mtm_age_seconds = None

            raw_mtm_unit = None if mtm_price is None else float(mtm_price) - buy_price
            net_mtm_unit = calculate_net_pnl(raw_mtm_unit, entry_cost) if raw_mtm_unit is not None else None
            raw_mtm_usdc = raw_mtm_unit * copied_size if raw_mtm_unit is not None else None
            net_mtm_usdc = calculate_net_pnl(raw_mtm_usdc, entry_cost * copied_size) if raw_mtm_usdc is not None else None
            if net_mtm_unit is not None:
                stats["mtm_net_sum"] += float(net_mtm_unit)
                stats["mtm_net_count"] += 1
                if float(net_mtm_unit) > 0:
                    stats["mtm_positive_count"] += 1
            if net_mtm_usdc is not None:
                stats["mtm_total_net_usdc_sum"] += float(net_mtm_usdc)

            trade_record = None
            if collect_trade_records:
                trade_record = {
                    "wallet_address": wallet_id,
                    "token_id": token_id,
                    "market_id": row.get("market_id"),
                    "signal_trade_id": row.get("signal_trade_id"),
                    "buy_timestamp": buy_ts,
                    "buy_price_signal": buy_price,
                    "copied_size": copied_size,
                    "analysis_cutoff": cutoff,
                    "resolution_ts": row.get("resolution_ts"),
                    "unresolved_status": "open_unresolved",
                    "holding_days_open": holding_days,
                    "mtm_price": mtm_price,
                    "mtm_price_source": mtm_source,
                    "mtm_price_age_seconds": mtm_age_seconds,
                    "mtm_pnl_raw": raw_mtm_unit,
                    "mtm_pnl_net": net_mtm_unit,
                    "mtm_pnl_raw_usdc": raw_mtm_usdc,
                    "mtm_pnl_net_usdc": net_mtm_usdc,
                }

            for horizon_days in (7, 30):
                label = f"{horizon_days}d"
                target_epoch = buy_epoch + int(horizon_days * 86400)
                if target_epoch > cutoff_epoch:
                    if trade_record is not None:
                        trade_record[f"forward_price_{label}"] = None
                        trade_record[f"forward_price_source_{label}"] = "insufficient_elapsed_time"
                        trade_record[f"forward_pnl_raw_{label}"] = None
                        trade_record[f"forward_pnl_net_{label}"] = None
                        trade_record[f"forward_pnl_raw_usdc_{label}"] = None
                        trade_record[f"forward_pnl_net_usdc_{label}"] = None
                    continue

                forward = lookup_forward_price(price_index, token_id, target_epoch)
                matched_epoch = (
                    None if forward.price is None or forward.delta_seconds is None else target_epoch + int(forward.delta_seconds)
                )
                if forward.price is None or matched_epoch is None or matched_epoch > cutoff_epoch:
                    if trade_record is not None:
                        trade_record[f"forward_price_{label}"] = None
                        trade_record[f"forward_price_source_{label}"] = "missing_forward_price_before_cutoff"
                        trade_record[f"forward_pnl_raw_{label}"] = None
                        trade_record[f"forward_pnl_net_{label}"] = None
                        trade_record[f"forward_pnl_raw_usdc_{label}"] = None
                        trade_record[f"forward_pnl_net_usdc_{label}"] = None
                    continue

                raw_forward = float(forward.price) - buy_price
                net_forward = calculate_net_pnl(raw_forward, entry_cost)
                raw_forward_usdc = raw_forward * copied_size
                net_forward_usdc = calculate_net_pnl(raw_forward_usdc, entry_cost * copied_size)
                stats[f"forward_{label}_net_sum"] += float(net_forward)
                stats[f"forward_{label}_count"] += 1
                if float(net_forward) > 0:
                    stats[f"forward_{label}_positive_count"] += 1
                if trade_record is not None:
                    trade_record[f"forward_price_{label}"] = float(forward.price)
                    trade_record[f"forward_price_source_{label}"] = forward.source
                    trade_record[f"forward_pnl_raw_{label}"] = raw_forward
                    trade_record[f"forward_pnl_net_{label}"] = net_forward
                    trade_record[f"forward_pnl_raw_usdc_{label}"] = raw_forward_usdc
                    trade_record[f"forward_pnl_net_usdc_{label}"] = net_forward_usdc

            if trade_record is not None:
                unresolved_records.append(trade_record)

    if not wallet_stats:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for wallet_id, stats in wallet_stats.items():
        unresolved_count = int(stats["unresolved_open_slices"])
        mtm_count = int(stats["mtm_net_count"])
        holding_count = int(stats["holding_days_count"])
        forward_7d_count = int(stats["forward_7d_count"])
        forward_30d_count = int(stats["forward_30d_count"])
        paired_count = int(paired_counts.get(wallet_id, 0))
        summary_rows.append(
            {
                "wallet_id": wallet_id,
                "unresolved_open_slices": unresolved_count,
                "paired_wallet_sell_slices": paired_count,
                "unresolved_open_share": _safe_ratio(unresolved_count, unresolved_count + paired_count),
                "unresolved_open_mtm_total_net_usdc": float(stats["mtm_total_net_usdc_sum"]),
                "unresolved_open_mtm_avg_net": (
                    float(stats["mtm_net_sum"]) / mtm_count if mtm_count else None
                ),
                "unresolved_open_mtm_positive_share": (
                    float(stats["mtm_positive_count"]) / mtm_count if mtm_count else None
                ),
                "unresolved_open_avg_holding_days": (
                    float(stats["holding_days_sum"]) / holding_count if holding_count else None
                ),
                "unresolved_open_valid_forward_7d": forward_7d_count,
                "unresolved_open_avg_forward_7d_net": (
                    float(stats["forward_7d_net_sum"]) / forward_7d_count if forward_7d_count else None
                ),
                "unresolved_open_forward_7d_hit_rate": (
                    float(stats["forward_7d_positive_count"]) / forward_7d_count if forward_7d_count else None
                ),
                "unresolved_open_valid_forward_30d": forward_30d_count,
                "unresolved_open_avg_forward_30d_net": (
                    float(stats["forward_30d_net_sum"]) / forward_30d_count if forward_30d_count else None
                ),
                "unresolved_open_forward_30d_hit_rate": (
                    float(stats["forward_30d_positive_count"]) / forward_30d_count if forward_30d_count else None
                ),
            }
        )

    wallet_summary = pd.DataFrame.from_records(summary_rows).sort_values(
        ["unresolved_open_mtm_total_net_usdc", "unresolved_open_slices", "wallet_id"],
        ascending=[False, False, True],
        na_position="last",
    )
    unresolved_frame = pd.DataFrame.from_records(unresolved_records) if collect_trade_records else pd.DataFrame()
    return wallet_summary, unresolved_frame


def classify_all_active_wallets(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply prior style rules plus expiry/open-position reinterpretation."""

    if frame.empty:
        return frame.copy()

    classified = frame.copy()
    if "holding_days" not in classified.columns:
        classified["holding_days"] = pd.to_numeric(
            classified.get("median_holding_seconds"), errors="coerce"
        ) / 86400.0
    if "dominant_vertical_share" not in classified.columns:
        classified["dominant_vertical_share"] = 0.0
    if "avg_trades_per_active_day_observed" not in classified.columns:
        classified["avg_trades_per_active_day_observed"] = pd.NA
    classified["meets_prior_style_filters"] = _style_candidate_mask(classified)
    classified = score_watchlist(classified)
    classified["has_observed_hold_to_expiry"] = (
        pd.to_numeric(classified.get("held_to_expiry_observed_slices"), errors="coerce").fillna(0) > 0
    )
    classified["has_meaningful_unresolved_open"] = (
        pd.to_numeric(classified.get("unresolved_open_slices"), errors="coerce").fillna(0) >= 10
    )
    classified["has_long_window_public_trades"] = (
        pd.to_numeric(classified.get("long_window_trade_rows"), errors="coerce").fillna(0) > 0
    )
    classified["reevaluation_bucket"] = "fails_prior_or_insufficient"
    classified.loc[
        classified["meets_prior_style_filters"] & classified["has_observed_hold_to_expiry"],
        "reevaluation_bucket",
    ] = "confirmed_by_observed_expiry_holds"
    classified.loc[
        classified["meets_prior_style_filters"]
        & (~classified["has_observed_hold_to_expiry"])
        & classified["has_meaningful_unresolved_open"],
        "reevaluation_bucket",
    ] = "candidate_with_many_unresolved_open_positions"
    classified.loc[
        (~classified["has_long_window_public_trades"]),
        "reevaluation_bucket",
    ] = "insufficient_public_long_window_data"
    hold_share = pd.to_numeric(classified.get("hold_to_expiry_share_observed"), errors="coerce")
    if not isinstance(hold_share, pd.Series):
        hold_share = pd.Series(0.0, index=classified.index, dtype=float)
    hold_share = hold_share.fillna(0.0)
    classified["hold_preference_score"] = (
        pd.to_numeric(classified.get("watchlist_score"), errors="coerce").fillna(0.0)
        + hold_share * 3.0
        + pd.to_numeric(classified.get("unresolved_open_slices"), errors="coerce").fillna(0.0).clip(upper=2000) * 0.002
    )
    return classified.sort_values(
        ["meets_prior_style_filters", "hold_preference_score", "wallet_id"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_summary_markdown(frame: pd.DataFrame) -> str:
    """Render a concise Markdown summary."""

    bucket_counts = frame["reevaluation_bucket"].value_counts(dropna=False).to_dict() if not frame.empty else {}
    candidate_subset = frame.loc[
        frame["reevaluation_bucket"].isin(
            ["confirmed_by_observed_expiry_holds", "candidate_with_many_unresolved_open_positions"]
        )
    ].copy()
    top_rows = candidate_subset.head(20).to_dict(orient="records")
    lines = [
        "# All-1067 Active Wallet Re-Evaluation",
        "",
        "This report re-screens the full active 1067-wallet cohort using the",
        "existing slow-copy standards plus two extra hold-through-resolution views:",
        "",
        "- observed buys that remained open until public market resolution",
        "- open buys that still had not resolved by the sample cutoff",
        "",
        "## Prior Standards Kept",
        "",
        "- `avg_copy_edge_net_30s > 0.01`",
        "- `fast_exit_share_30s <= 5%`",
        "- `median_holding_seconds > 1 hour`",
        "- `rolling_3d_positive_share >= 0.40`",
        "- `5 <= recent_trades_window <= 250`",
        "- `sell_share <= 40%`",
        "- `open_share >= 60%`",
        "- manually excluded wallets remain excluded",
        "",
        "## Bucket Counts",
        "",
        f"- `confirmed_by_observed_expiry_holds`: `{bucket_counts.get('confirmed_by_observed_expiry_holds', 0)}`",
        f"- `candidate_with_many_unresolved_open_positions`: `{bucket_counts.get('candidate_with_many_unresolved_open_positions', 0)}`",
        f"- `insufficient_public_long_window_data`: `{bucket_counts.get('insufficient_public_long_window_data', 0)}`",
        f"- `fails_prior_or_insufficient`: `{bucket_counts.get('fails_prior_or_insufficient', 0)}`",
        "",
        "## Top Reevaluated Candidates",
        "",
    ]
    if not top_rows:
        lines.append("- No wallets passed the prior style filter under the expanded all-1067 re-evaluation.")
    else:
        for row in top_rows:
            lines.append(
                f"- `{row['wallet_id']}` | bucket `{row.get('reevaluation_bucket')}` | "
                f"edge30 `{row.get('avg_copy_edge_net_30s')}` | "
                f"sell `{row.get('sell_share')}` | "
                f"open `{row.get('open_share')}` | "
                f"held_to_expiry `{row.get('held_to_expiry_observed_slices')}` | "
                f"unresolved `{row.get('unresolved_open_slices')}` | "
                f"mtm_total `{row.get('unresolved_open_mtm_total_net_usdc')}` | "
                f"avg_hold_days `{row.get('unresolved_open_avg_holding_days')}` | "
                f"fwd7 `{row.get('unresolved_open_avg_forward_7d_net')}` | "
                f"fwd30 `{row.get('unresolved_open_avg_forward_30d_net')}` | "
                f"score `{row.get('hold_preference_score')}`"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `confirmed_by_observed_expiry_holds` is the strongest bucket, but it will be sparse because many current-event positions have not resolved yet.",
            "- `candidate_with_many_unresolved_open_positions` is the main long-horizon bucket for still-active traders whose behavior looks like hold-through-resolution but whose markets remain open.",
            "- `insufficient_public_long_window_data` means the wallet is in the active scan universe but lacks enough public wallet-history rows in the long-window capture for this hold-style review.",
        ]
    )
    return "\n".join(lines)


def run_all_active_reevaluation(
    *,
    current_active_csv: str | Path = DEFAULT_CURRENT_ACTIVE_CSV,
    features_csv: str | Path = DEFAULT_FEATURES_CSV,
    vertical_summary_csv: str | Path = DEFAULT_VERTICAL_SUMMARY_CSV,
    recent_trades_csv: str | Path = DEFAULT_RECENT_TRADES_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    analysis_cutoff: str = DEFAULT_ANALYSIS_CUTOFF,
) -> dict[str, Any]:
    """Run the all-active 1067-wallet reevaluation."""

    frame = load_all_active_frame(
        current_active_csv=current_active_csv,
        features_csv=features_csv,
        vertical_summary_csv=vertical_summary_csv,
    )
    merged, resolved_trades, _, diagnostics = merge_expiry_behavior(
        frame,
        recent_trades_csv=recent_trades_csv,
        analysis_cutoff=analysis_cutoff,
    )
    paired_counts = {}
    for row in merged[["wallet_id", "wallet_sell_closed_slices"]].dropna(subset=["wallet_id"]).to_dict(orient="records"):
        sell_count = pd.to_numeric(row.get("wallet_sell_closed_slices"), errors="coerce")
        paired_counts[str(row["wallet_id"]).lower()] = int(0 if pd.isna(sell_count) else sell_count)
    open_wallet_summary, unresolved_open_trades = compute_open_position_evidence(
        diagnostics,
        terminal_lookup={},
        analysis_cutoff=analysis_cutoff,
        collect_trade_records=False,
        paired_counts=paired_counts,
    )
    base_wallets = pd.DataFrame({"wallet_id": frame["wallet_id"].astype(str).str.lower().tolist()}).drop_duplicates()
    open_wallet_summary = base_wallets.merge(open_wallet_summary, on="wallet_id", how="left")
    merged = merged.merge(open_wallet_summary, on="wallet_id", how="left", suffixes=("", "_open"))
    unresolved_count_for_share = pd.to_numeric(
        merged.get("unresolved_open_slices_open", merged.get("unresolved_open_slices")),
        errors="coerce",
    )
    paired_count_for_share = pd.to_numeric(merged.get("wallet_sell_closed_slices"), errors="coerce")
    merged["unresolved_open_share"] = unresolved_count_for_share / (unresolved_count_for_share + paired_count_for_share)
    if "unresolved_open_slices_open" in merged.columns:
        merged["unresolved_open_slices"] = unresolved_count_for_share
    classified = classify_all_active_wallets(merged)
    candidates = classified.loc[
        classified["reevaluation_bucket"].isin(
            ["confirmed_by_observed_expiry_holds", "candidate_with_many_unresolved_open_positions"]
        )
    ].copy()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    all_path = _write_csv(classified, output_root / "all_active_1067_reevaluation.csv")
    candidate_path = _write_csv(candidates, output_root / "all_active_1067_candidates.csv")
    resolved_path = _write_csv(resolved_trades, output_root / "all_active_1067_resolved_expiry_trades.csv")
    unresolved_path = _write_csv(unresolved_open_trades, output_root / "all_active_1067_unresolved_open_trades.csv")
    open_wallet_path = _write_csv(open_wallet_summary, output_root / "all_active_1067_open_position_evidence.csv")
    report_path = output_root / "all_active_1067_summary.md"
    report_path.write_text(build_summary_markdown(classified), encoding="utf-8")
    return {
        "classified": classified,
        "candidates": candidates,
        "resolved_trades": resolved_trades,
        "unresolved_open_trades": unresolved_open_trades,
        "open_wallet_summary": open_wallet_summary,
        "paths": {
            "all": all_path,
            "candidates": candidate_path,
            "resolved_trades": resolved_path,
            "unresolved_open_trades": unresolved_path,
            "open_wallet_summary": open_wallet_path,
            "report": report_path,
        },
    }


def print_all_active_reevaluation_summary(results: dict[str, Any]) -> None:
    """Print a concise terminal summary."""

    classified = results["classified"]
    candidates = results["candidates"]
    counts = classified["reevaluation_bucket"].value_counts(dropna=False).to_dict() if not classified.empty else {}
    print("All-1067 Active Re-Evaluation")
    print(f"Wallet rows: {len(classified)}")
    print(f"Confirmed by observed expiry holds: {counts.get('confirmed_by_observed_expiry_holds', 0)}")
    print(
        "Candidates with many unresolved open positions: "
        f"{counts.get('candidate_with_many_unresolved_open_positions', 0)}"
    )
    print(f"Insufficient public long-window data: {counts.get('insufficient_public_long_window_data', 0)}")
    if not candidates.empty:
        columns = [
            column
            for column in [
                "wallet_id",
                "sample_name",
                "reevaluation_bucket",
                "avg_copy_edge_net_30s",
                "sell_share",
                "open_share",
                "held_to_expiry_observed_slices",
                "unresolved_open_slices",
                "hold_preference_score",
            ]
            if column in candidates.columns
        ]
        print(candidates.head(20)[columns].to_string(index=False))
