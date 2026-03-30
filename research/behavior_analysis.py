"""Behavior-level trade feature extraction and rule-based conditional research."""

from __future__ import annotations

from bisect import bisect_right
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

sys.modules.setdefault("pyarrow", None)

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from db.models import Market, PriceHistory, TradeFeature
from research.delay_analysis import (
    compute_delay_trade_metrics,
    load_price_history_frame,
    persist_delay_metrics,
)
from research.event_study import load_enriched_trades


LOGGER = logging.getLogger(__name__)

FEATURE_GROUP_COLUMNS = (
    "size_bucket",
    "price_zone",
    "pre_trade_trend_state",
    "market_phase",
    "liquidity_bucket",
    "trade_type_cluster",
)
PORTFOLIO_SELECTION_MIN_TRADES = 5
PORTFOLIO_MIN_DELAY_RETENTION = 0.25
SIZE_ZSCORE_LARGE = 0.75
SIZE_ZSCORE_SMALL = -0.75


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a CSV file, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _to_epoch_seconds(value: pd.Timestamp) -> int:
    """Convert a timestamp-like value to UTC epoch seconds."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.tz_convert("UTC").timestamp())


def _to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse a timestamp-like value into UTC or return None."""

    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp


def _build_price_index(price_history: pd.DataFrame) -> dict[str, tuple[list[int], list[float]]]:
    """Build a token -> (times, prices) lookup table for past-price queries."""

    if price_history.empty:
        return {}

    frame = price_history.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["token_id", "ts", "price"]).sort_values(["token_id", "ts"])
    index: dict[str, tuple[list[int], list[float]]] = {}
    for row in frame.itertuples(index=False):
        token_id = str(row.token_id)
        times, prices = index.setdefault(token_id, ([], []))
        times.append(_to_epoch_seconds(row.ts))
        prices.append(float(row.price))
    return index


def _lookup_price_at_or_before(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    target_ts: int,
) -> float | None:
    """Return the most recent public price at or before the target timestamp."""

    if token_id is None or token_id not in index:
        return None
    times, prices = index[token_id]
    if not times:
        return None
    position = bisect_right(times, target_ts) - 1
    if position < 0:
        return None
    return prices[position]


def _window_prices(
    index: dict[str, tuple[list[int], list[float]]],
    token_id: str | None,
    start_ts: int,
    end_ts: int,
) -> list[float]:
    """Return public prices inside one closed time window."""

    if token_id is None or token_id not in index:
        return []
    times, prices = index[token_id]
    if not times:
        return []
    left = bisect_right(times, start_ts - 1)
    right = bisect_right(times, end_ts)
    return prices[left:right]


def _safe_mean(series: pd.Series) -> float | None:
    """Return the numeric mean or None when the series has no valid data."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def _hit_rate(series: pd.Series) -> float | None:
    """Return the fraction of positive observations."""

    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return None
    return float((valid > 0).mean())


def _retention(base_value: float | None, delayed_value: float | None) -> float | None:
    """Return delayed edge retention relative to the baseline edge."""

    if base_value is None or delayed_value is None or base_value == 0:
        return None
    return delayed_value / base_value


def _choose_mode(copy_value: float | None, fade_value: float | None) -> str:
    """Choose copy, fade, or ignore from two feature-group averages."""

    if copy_value is None and fade_value is None:
        return "ignore"
    if copy_value is not None and copy_value > 0 and (fade_value is None or copy_value > fade_value):
        return "copy"
    if fade_value is not None and fade_value > 0 and (copy_value is None or fade_value > copy_value):
        return "fade"
    return "ignore"


def _load_market_frame(session: Session) -> pd.DataFrame:
    """Load market metadata needed for market-age and resolution-time features."""

    query = select(
        Market.id,
        Market.condition_id,
        Market.created_at,
        Market.updated_at,
        Market.closed,
        Market.raw_json,
    )
    return pd.read_sql(query, session.bind)


def _parse_market_resolution_ts(raw_json: Any, updated_at: Any, closed: Any) -> pd.Timestamp | None:
    """Parse a best-effort market resolution time from public Gamma market JSON.

    Assumptions:
    - Gamma raw market JSON is not fully stable across market vintages.
    - We first try explicit end/resolution style keys.
    - If the market is closed and no explicit end is present, we fall back to
      `updated_at` as a weak late-stage proxy instead of fabricating a time.
    """

    payload: dict[str, Any] = {}
    if isinstance(raw_json, str) and raw_json.strip():
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = {}
    elif isinstance(raw_json, dict):
        payload = raw_json

    for key in (
        "endDate",
        "end_date",
        "resolutionDate",
        "resolution_date",
        "resolveBy",
        "resolve_by",
        "closeTime",
        "close_time",
    ):
        timestamp = _to_utc_timestamp(payload.get(key))
        if timestamp is not None:
            return timestamp

    if bool(closed):
        return _to_utc_timestamp(updated_at)
    return None


def _build_market_lookup(markets: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build a market lookup keyed by both market id and condition id."""

    lookup: dict[str, dict[str, Any]] = {}
    if markets.empty:
        return lookup

    frame = markets.copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True, errors="coerce")
    for row in frame.to_dict(orient="records"):
        resolution_ts = _parse_market_resolution_ts(
            row.get("raw_json"),
            row.get("updated_at"),
            row.get("closed"),
        )
        record = {
            "created_at": row.get("created_at"),
            "resolution_ts": resolution_ts,
        }
        for key in (row.get("id"), row.get("condition_id")):
            if key:
                lookup[str(key)] = record
    return lookup


def _classify_size_bucket(standardized_size: float | None) -> str:
    """Bucket standardized size into coarse, human-readable bins."""

    if standardized_size is None or pd.isna(standardized_size):
        return "unknown"
    if standardized_size >= SIZE_ZSCORE_LARGE:
        return "large"
    if standardized_size <= SIZE_ZSCORE_SMALL:
        return "small"
    return "medium"


def _classify_price_zone(price_distance_from_mid: float | None) -> str:
    """Bucket prices by distance from the 0.5 midpoint."""

    if price_distance_from_mid is None or pd.isna(price_distance_from_mid):
        return "unknown"
    if price_distance_from_mid <= 0.10:
        return "centered"
    if price_distance_from_mid <= 0.25:
        return "balanced"
    return "extreme"


def _classify_pre_trade_trend(
    momentum_1m: float | None,
    momentum_5m: float | None,
    volatility_5m: float | None,
) -> str:
    """Classify the pre-trade market state as trending or ranging."""

    if momentum_5m is None or pd.isna(momentum_5m):
        return "unknown"

    vol_floor = 0.01 if volatility_5m is None or pd.isna(volatility_5m) else max(float(volatility_5m), 0.01)
    one_minute = 0.0 if momentum_1m is None or pd.isna(momentum_1m) else float(momentum_1m)
    five_minute = float(momentum_5m)

    if five_minute >= vol_floor and one_minute >= 0:
        return "uptrend"
    if five_minute <= -vol_floor and one_minute <= 0:
        return "downtrend"
    if abs(five_minute) < vol_floor:
        return "ranging"
    return "transitioning"


def _classify_market_phase(
    trade_ts: pd.Timestamp,
    created_at: pd.Timestamp | None,
    resolution_ts: pd.Timestamp | None,
) -> tuple[str, float | None]:
    """Bucket trades into early/mid/late market phases when timestamps exist."""

    if created_at is None or resolution_ts is None or pd.isna(created_at) or pd.isna(resolution_ts):
        return "unknown", None
    if resolution_ts <= trade_ts or resolution_ts <= created_at:
        return "late", 0.0

    total_seconds = (resolution_ts - created_at).total_seconds()
    if total_seconds <= 0:
        return "unknown", None
    elapsed = max(0.0, (trade_ts - created_at).total_seconds())
    progress = min(1.0, elapsed / total_seconds)
    time_to_resolution_minutes = (resolution_ts - trade_ts).total_seconds() / 60.0
    if progress < 0.33:
        return "early", time_to_resolution_minutes
    if progress < 0.66:
        return "mid", time_to_resolution_minutes
    return "late", time_to_resolution_minutes


def _trend_alignment(side: str | None, pre_trade_trend_state: str | None) -> bool | None:
    """Return whether the trade direction follows the observed pre-trade trend."""

    normalized_side = str(side or "").upper()
    if pre_trade_trend_state == "uptrend":
        return normalized_side == "BUY"
    if pre_trade_trend_state == "downtrend":
        return normalized_side == "SELL"
    return None


def _compute_trade_cluster_density(frame: pd.DataFrame) -> pd.Series:
    """Count nearby trades from the same wallet and market within +/- 5 minutes."""

    if frame.empty:
        return pd.Series(dtype="Int64")

    working = frame.copy()
    working["_cluster_group"] = working["market_id"].fillna(working["token_id"]).fillna("unknown_market")
    working["_epoch_ts"] = working["timestamp"].map(_to_epoch_seconds)
    densities = pd.Series(index=working.index, dtype="Int64")

    for (_, _), group in working.groupby(["wallet_address", "_cluster_group"], dropna=False):
        ordered = group.sort_values("_epoch_ts")
        epochs = ordered["_epoch_ts"].tolist()
        indexes = ordered.index.tolist()
        left = 0
        right = 0
        for position, trade_ts in enumerate(epochs):
            while epochs[left] < trade_ts - 300:
                left += 1
            while right + 1 < len(epochs) and epochs[right + 1] <= trade_ts + 300:
                right += 1
            densities.loc[indexes[position]] = max(0, right - left)
    return densities


def classify_trade_behavior(row: pd.Series) -> str:
    """Assign one coarse behavior label using transparent rule-based logic."""

    market_phase = str(row.get("market_phase") or "unknown")
    trend_state = str(row.get("pre_trade_trend_state") or "unknown")
    price_zone = str(row.get("price_zone") or "unknown")
    size_bucket = str(row.get("size_bucket") or "unknown")
    alignment = row.get("trend_alignment")
    cluster_density = pd.to_numeric(pd.Series([row.get("trade_cluster_density_5m")]), errors="coerce").iloc[0]

    if (
        market_phase == "late"
        and price_zone == "extreme"
        and trend_state in {"uptrend", "downtrend"}
        and alignment is True
    ):
        return "late_fomo_chaser"

    if (
        market_phase == "early"
        and price_zone in {"centered", "balanced"}
        and size_bucket in {"medium", "large"}
        and trend_state in {"ranging", "transitioning", "unknown"}
    ):
        return "early_positioning"

    if (
        pd.notna(cluster_density)
        and float(cluster_density) >= 2.0
        and size_bucket in {"small", "medium"}
        and market_phase in {"early", "mid", "unknown"}
    ):
        return "passive_accumulator"

    if (
        trend_state in {"uptrend", "downtrend"}
        and alignment is True
        and size_bucket in {"medium", "large"}
    ):
        return "aggressive_momentum_chaser"

    return "other"


def extract_trade_features_frame(
    enriched: pd.DataFrame,
    price_history: pd.DataFrame,
    markets: pd.DataFrame,
) -> pd.DataFrame:
    """Extract trade-level behavior features from enriched trades and public prices."""

    frame = enriched.copy()
    if frame.empty:
        return frame

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["trade_id", "wallet_address", "timestamp", "price"]).reset_index(drop=True)
    if frame.empty:
        return frame

    frame["trade_notional"] = pd.to_numeric(frame["size"], errors="coerce") * pd.to_numeric(frame["price"], errors="coerce")
    notional_proxy = pd.to_numeric(frame["trade_notional"], errors="coerce")
    logged_notional = notional_proxy.map(
        lambda value: pd.NA if pd.isna(value) or value <= 0 else math.log1p(float(value))
    )
    logged_valid = pd.to_numeric(logged_notional, errors="coerce")
    std = logged_valid.std(ddof=0)
    if pd.isna(std) or std == 0:
        frame["standardized_size"] = 0.0
    else:
        frame["standardized_size"] = (logged_valid - logged_valid.mean()) / std

    market_medians = frame.groupby("market_id", dropna=True)["trade_notional"].median()
    token_medians = frame.groupby("token_id", dropna=True)["trade_notional"].median()

    def _relative_size(row: pd.Series) -> float | None:
        median = market_medians.get(row.get("market_id"))
        if median is None or pd.isna(median) or median <= 0:
            median = token_medians.get(row.get("token_id"))
        trade_notional = pd.to_numeric(pd.Series([row.get("trade_notional")]), errors="coerce").iloc[0]
        if pd.isna(trade_notional) or median is None or pd.isna(median) or median <= 0:
            return None
        return float(trade_notional / median)

    frame["size_to_liquidity_ratio"] = frame.apply(_relative_size, axis=1)
    frame["price_distance_from_mid"] = (pd.to_numeric(frame["price"], errors="coerce") - 0.5).abs()
    frame["size_bucket"] = frame["standardized_size"].apply(_classify_size_bucket)
    frame["price_zone"] = frame["price_distance_from_mid"].apply(_classify_price_zone)

    price_index = _build_price_index(price_history)
    market_lookup = _build_market_lookup(markets)

    recent_momentum_1m: list[float | None] = []
    recent_momentum_5m: list[float | None] = []
    short_term_volatility_5m: list[float | None] = []
    market_phases: list[str] = []
    time_to_resolution_minutes: list[float | None] = []
    trend_states: list[str] = []
    trend_alignments: list[bool | None] = []

    for row in frame.itertuples(index=False):
        trade_ts = _to_epoch_seconds(row.timestamp)
        reference_price = (
            float(row.mid_at_trade)
            if hasattr(row, "mid_at_trade") and pd.notna(row.mid_at_trade)
            else float(row.price)
        )
        price_1m_ago = _lookup_price_at_or_before(price_index, row.token_id, trade_ts - 60)
        price_5m_ago = _lookup_price_at_or_before(price_index, row.token_id, trade_ts - 5 * 60)
        prices_5m_window = _window_prices(price_index, row.token_id, trade_ts - 5 * 60, trade_ts)

        momentum_1m = None if price_1m_ago is None else reference_price - price_1m_ago
        momentum_5m = None if price_5m_ago is None else reference_price - price_5m_ago
        volatility_5m = (
            float(pd.Series(prices_5m_window, dtype=float).std(ddof=0))
            if len(prices_5m_window) >= 2
            else None
        )

        market_record = market_lookup.get(str(row.market_id)) if row.market_id is not None else None
        created_at = None if market_record is None else market_record.get("created_at")
        resolution_ts = None if market_record is None else market_record.get("resolution_ts")
        market_phase, minutes_to_resolution = _classify_market_phase(row.timestamp, created_at, resolution_ts)
        trend_state = _classify_pre_trade_trend(momentum_1m, momentum_5m, volatility_5m)
        alignment = _trend_alignment(row.side, trend_state)

        recent_momentum_1m.append(momentum_1m)
        recent_momentum_5m.append(momentum_5m)
        short_term_volatility_5m.append(volatility_5m)
        market_phases.append(market_phase)
        time_to_resolution_minutes.append(minutes_to_resolution)
        trend_states.append(trend_state)
        trend_alignments.append(alignment)

    frame["recent_momentum_1m"] = recent_momentum_1m
    frame["recent_momentum_5m"] = recent_momentum_5m
    frame["short_term_volatility_5m"] = short_term_volatility_5m
    frame["market_phase"] = market_phases
    frame["time_to_resolution_minutes"] = time_to_resolution_minutes
    frame["pre_trade_trend_state"] = trend_states
    frame["trend_alignment"] = trend_alignments
    frame["trade_cluster_density_5m"] = _compute_trade_cluster_density(frame)
    frame["trade_type_cluster"] = frame.apply(lambda row: classify_trade_behavior(pd.Series(row)), axis=1)
    return frame


def persist_trade_features(session: Session, trade_features: pd.DataFrame) -> int:
    """Upsert the derived trade_features table from one feature frame."""

    columns = [
        "trade_id",
        "wallet_address",
        "market_id",
        "token_id",
        "timestamp",
        "trade_notional",
        "standardized_size",
        "size_to_liquidity_ratio",
        "price_distance_from_mid",
        "recent_momentum_1m",
        "recent_momentum_5m",
        "short_term_volatility_5m",
        "time_to_resolution_minutes",
        "trade_cluster_density_5m",
        "size_bucket",
        "price_zone",
        "pre_trade_trend_state",
        "market_phase",
        "trend_alignment",
        "trade_type_cluster",
    ]
    if trade_features.empty:
        return 0

    payload_frame = trade_features[columns].copy()
    payload_frame = payload_frame.where(pd.notna(payload_frame), None)
    payload = payload_frame.to_dict(orient="records")
    if not payload:
        return 0

    updated = 0
    for start in range(0, len(payload), 250):
        chunk = payload[start:start + 250]
        statement = insert(TradeFeature).values(chunk)
        update_map = {
            column: getattr(statement.excluded, column)
            for column in columns
            if column != "trade_id"
        }
        session.execute(
            statement.on_conflict_do_update(
                index_elements=["trade_id"],
                set_=update_map,
            )
        )
        updated += len(chunk)
    session.flush()
    return updated


def summarize_feature_performance(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate conditional performance by feature bucket instead of by wallet."""

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "feature_value",
                "n_trades",
                "n_wallets",
                "n_markets",
                "avg_copy_pnl_5m",
                "avg_fade_pnl_5m",
                "avg_copy_pnl_net_5m",
                "avg_fade_pnl_net_5m",
                "avg_copy_pnl_net_5m_delay_30s",
                "avg_fade_pnl_net_5m_delay_30s",
                "copy_delay_sensitivity_30s",
                "fade_delay_sensitivity_30s",
                "copy_edge_retention_30s",
                "fade_edge_retention_30s",
                "recommended_mode",
                "selected_for_behavior_portfolio",
            ]
        )

    records: list[dict[str, Any]] = []
    for feature_name in FEATURE_GROUP_COLUMNS:
        if feature_name not in trades.columns:
            continue
        groupable = trades.dropna(subset=[feature_name]).copy()
        if groupable.empty:
            continue
        for feature_value, group in groupable.groupby(feature_name, dropna=False):
            avg_copy_net = _safe_mean(group["copy_pnl_net_5m"])
            avg_fade_net = _safe_mean(group["fade_pnl_net_5m"])
            avg_copy_delay_30s = _safe_mean(group["copy_pnl_net_5m_delay_30s"])
            avg_fade_delay_30s = _safe_mean(group["fade_pnl_net_5m_delay_30s"])
            recommended_mode = _choose_mode(avg_copy_net, avg_fade_net)
            selected_for_behavior_portfolio = False
            if feature_name == "trade_type_cluster" and len(group) >= PORTFOLIO_SELECTION_MIN_TRADES:
                if recommended_mode == "copy" and avg_copy_net is not None and avg_copy_net > 0:
                    retention = _retention(avg_copy_net, avg_copy_delay_30s)
                    selected_for_behavior_portfolio = (
                        avg_copy_delay_30s is not None
                        and avg_copy_delay_30s > 0
                        and retention is not None
                        and retention >= PORTFOLIO_MIN_DELAY_RETENTION
                    )
                if recommended_mode == "fade" and avg_fade_net is not None and avg_fade_net > 0:
                    retention = _retention(avg_fade_net, avg_fade_delay_30s)
                    selected_for_behavior_portfolio = (
                        avg_fade_delay_30s is not None
                        and avg_fade_delay_30s > 0
                        and retention is not None
                        and retention >= PORTFOLIO_MIN_DELAY_RETENTION
                    )

            records.append(
                {
                    "feature_name": feature_name,
                    "feature_value": feature_value,
                    "n_trades": int(len(group)),
                    "n_wallets": int(group["wallet_address"].nunique(dropna=True)),
                    "n_markets": int(group["market_id"].nunique(dropna=True)),
                    "avg_copy_pnl_5m": _safe_mean(group["copy_pnl_5m"]),
                    "avg_fade_pnl_5m": _safe_mean(group["fade_pnl_5m"]),
                    "avg_copy_pnl_net_5m": avg_copy_net,
                    "avg_fade_pnl_net_5m": avg_fade_net,
                    "avg_copy_pnl_net_5m_delay_30s": avg_copy_delay_30s,
                    "avg_fade_pnl_net_5m_delay_30s": avg_fade_delay_30s,
                    "copy_delay_sensitivity_30s": (
                        None if avg_copy_net is None or avg_copy_delay_30s is None else avg_copy_delay_30s - avg_copy_net
                    ),
                    "fade_delay_sensitivity_30s": (
                        None if avg_fade_net is None or avg_fade_delay_30s is None else avg_fade_delay_30s - avg_fade_net
                    ),
                    "copy_edge_retention_30s": _retention(avg_copy_net, avg_copy_delay_30s),
                    "fade_edge_retention_30s": _retention(avg_fade_net, avg_fade_delay_30s),
                    "recommended_mode": recommended_mode,
                    "selected_for_behavior_portfolio": selected_for_behavior_portfolio,
                }
            )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    return result.sort_values(
        ["feature_name", "selected_for_behavior_portfolio", "n_trades", "feature_value"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def build_trade_type_clusters_export(trades: pd.DataFrame) -> pd.DataFrame:
    """Build the trade-level diagnostic export centered on behavior labels."""

    if trades.empty:
        return pd.DataFrame()

    columns = [
        "trade_id",
        "wallet_address",
        "market_id",
        "token_id",
        "timestamp",
        "trade_notional",
        "standardized_size",
        "size_to_liquidity_ratio",
        "price_distance_from_mid",
        "recent_momentum_1m",
        "recent_momentum_5m",
        "short_term_volatility_5m",
        "time_to_resolution_minutes",
        "trade_cluster_density_5m",
        "size_bucket",
        "price_zone",
        "pre_trade_trend_state",
        "market_phase",
        "trend_alignment",
        "liquidity_bucket",
        "trade_type_cluster",
        "copy_pnl_5m",
        "fade_pnl_5m",
        "copy_pnl_net_5m",
        "fade_pnl_net_5m",
        "copy_pnl_net_5m_delay_30s",
        "fade_pnl_net_5m_delay_30s",
    ]
    available = [column for column in columns if column in trades.columns]
    result = trades[available].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce").apply(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    return result.sort_values(["trade_type_cluster", "timestamp", "trade_id"]).reset_index(drop=True)


def build_wallet_behavior_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """Break down each wallet into the behavior types that drive gains or losses."""

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "wallet_address",
                "trade_type_cluster",
                "n_trades",
                "share_of_wallet_trades",
                "avg_copy_pnl_net_5m",
                "avg_fade_pnl_net_5m",
                "total_copy_pnl_net_5m",
                "total_fade_pnl_net_5m",
                "copy_pnl_share_of_wallet",
                "fade_pnl_share_of_wallet",
                "dominant_mode",
                "cluster_role",
            ]
        )

    total_trades_by_wallet = trades.groupby("wallet_address").size()
    wallet_copy_totals = pd.to_numeric(trades["copy_pnl_net_5m"], errors="coerce").groupby(trades["wallet_address"]).sum(min_count=1)
    wallet_fade_totals = pd.to_numeric(trades["fade_pnl_net_5m"], errors="coerce").groupby(trades["wallet_address"]).sum(min_count=1)

    records: list[dict[str, Any]] = []
    grouped = trades.groupby(["wallet_address", "trade_type_cluster"], dropna=False)
    for (wallet_address, trade_type_cluster), group in grouped:
        total_copy = _safe_mean(group["copy_pnl_net_5m"])
        total_fade = _safe_mean(group["fade_pnl_net_5m"])
        sum_copy = pd.to_numeric(group["copy_pnl_net_5m"], errors="coerce").sum(min_count=1)
        sum_fade = pd.to_numeric(group["fade_pnl_net_5m"], errors="coerce").sum(min_count=1)
        dominant_mode = _choose_mode(total_copy, total_fade)

        cluster_role = "neutral"
        if dominant_mode == "copy":
            if pd.notna(sum_copy) and sum_copy > 0:
                cluster_role = "copy_edge_driver"
            elif pd.notna(sum_copy) and sum_copy < 0:
                cluster_role = "copy_drag"
        elif dominant_mode == "fade":
            if pd.notna(sum_fade) and sum_fade > 0:
                cluster_role = "fade_edge_driver"
            elif pd.notna(sum_fade) and sum_fade < 0:
                cluster_role = "fade_drag"

        wallet_copy_total = wallet_copy_totals.get(wallet_address)
        wallet_fade_total = wallet_fade_totals.get(wallet_address)
        records.append(
            {
                "wallet_address": wallet_address,
                "trade_type_cluster": trade_type_cluster,
                "n_trades": int(len(group)),
                "share_of_wallet_trades": float(len(group) / total_trades_by_wallet[wallet_address]),
                "avg_copy_pnl_net_5m": total_copy,
                "avg_fade_pnl_net_5m": total_fade,
                "total_copy_pnl_net_5m": float(sum_copy) if pd.notna(sum_copy) else None,
                "total_fade_pnl_net_5m": float(sum_fade) if pd.notna(sum_fade) else None,
                "copy_pnl_share_of_wallet": (
                    None
                    if wallet_copy_total is None or pd.isna(wallet_copy_total) or wallet_copy_total == 0 or pd.isna(sum_copy)
                    else float(sum_copy / wallet_copy_total)
                ),
                "fade_pnl_share_of_wallet": (
                    None
                    if wallet_fade_total is None or pd.isna(wallet_fade_total) or wallet_fade_total == 0 or pd.isna(sum_fade)
                    else float(sum_fade / wallet_fade_total)
                ),
                "dominant_mode": dominant_mode,
                "cluster_role": cluster_role,
            }
        )

    result = pd.DataFrame.from_records(records)
    return result.sort_values(
        ["wallet_address", "n_trades", "trade_type_cluster"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _needs_delay_refresh(enriched: pd.DataFrame) -> bool:
    """Return whether net-PnL delay columns appear to be unpopulated."""

    required_columns = [
        "copy_pnl_net_5m",
        "fade_pnl_net_5m",
        "copy_pnl_net_5m_delay_30s",
        "fade_pnl_net_5m_delay_30s",
    ]
    for column in required_columns:
        if column not in enriched.columns:
            return True
        if pd.to_numeric(enriched[column], errors="coerce").notna().any():
            continue
        return True
    return False


def run_behavior_analysis(
    session: Session,
    *,
    output_dir: str | Path = "exports/behavior_analysis",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run behavior-level feature extraction and conditional signal analysis."""

    _ = settings or get_settings()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    enriched = load_enriched_trades(session)
    if enriched.empty:
        empty = pd.DataFrame()
        paths = {
            "feature_performance_summary": _write_csv(empty, output_root / "feature_performance_summary.csv"),
            "trade_type_clusters": _write_csv(empty, output_root / "trade_type_clusters.csv"),
            "wallet_behavior_breakdown": _write_csv(empty, output_root / "wallet_behavior_breakdown.csv"),
        }
        return {
            "trade_features": empty,
            "feature_performance_summary": empty,
            "trade_type_clusters": empty,
            "wallet_behavior_breakdown": empty,
            "paths": paths,
        }

    token_ids = sorted({str(token_id) for token_id in enriched["token_id"].dropna().tolist()})
    price_history = load_price_history_frame(session, token_ids=token_ids)
    if _needs_delay_refresh(enriched):
        LOGGER.info("Delay/net columns missing or empty; refreshing them before behavior analysis")
        enriched = compute_delay_trade_metrics(enriched, price_history)
        persist_delay_metrics(session, enriched)

    markets = _load_market_frame(session)
    trade_features = extract_trade_features_frame(enriched, price_history, markets)
    persisted = persist_trade_features(session, trade_features)
    LOGGER.info("Upserted %s trade_features rows", persisted)

    feature_performance_summary = summarize_feature_performance(trade_features)
    trade_type_clusters = build_trade_type_clusters_export(trade_features)
    wallet_behavior_breakdown = build_wallet_behavior_breakdown(trade_features)

    paths = {
        "feature_performance_summary": _write_csv(
            feature_performance_summary,
            output_root / "feature_performance_summary.csv",
        ),
        "trade_type_clusters": _write_csv(
            trade_type_clusters,
            output_root / "trade_type_clusters.csv",
        ),
        "wallet_behavior_breakdown": _write_csv(
            wallet_behavior_breakdown,
            output_root / "wallet_behavior_breakdown.csv",
        ),
    }
    return {
        "trade_features": trade_features,
        "feature_performance_summary": feature_performance_summary,
        "trade_type_clusters": trade_type_clusters,
        "wallet_behavior_breakdown": wallet_behavior_breakdown,
        "paths": paths,
    }


def print_behavior_summary(results: dict[str, Any]) -> None:
    """Print a concise console summary for behavior-level research output."""

    feature_summary = results["feature_performance_summary"]
    clusters = results["trade_type_clusters"]
    wallet_breakdown = results["wallet_behavior_breakdown"]

    print("Behavior Analysis Summary")
    if clusters.empty:
        print("No enriched trades available for behavior analysis.")
        return

    cluster_counts = (
        clusters["trade_type_cluster"]
        .value_counts(dropna=False)
        .rename_axis("trade_type_cluster")
        .reset_index(name="n_trades")
    )
    print("Trade Type Counts")
    print(cluster_counts.head(10).to_string(index=False))

    if not feature_summary.empty:
        top_groups = feature_summary.sort_values(
            ["selected_for_behavior_portfolio", "n_trades"],
            ascending=[False, False],
        ).head(10)
        print("Top Feature Groups")
        print(
            top_groups[
                [
                    "feature_name",
                    "feature_value",
                    "n_trades",
                    "avg_copy_pnl_net_5m",
                    "avg_fade_pnl_net_5m",
                    "recommended_mode",
                    "selected_for_behavior_portfolio",
                ]
            ].to_string(index=False)
        )

    if not wallet_breakdown.empty:
        top_wallet_drivers = wallet_breakdown[wallet_breakdown["cluster_role"].str.contains("driver", na=False)].head(10)
        if not top_wallet_drivers.empty:
            print("Top Wallet Behavior Drivers")
            print(
                top_wallet_drivers[
                    [
                        "wallet_address",
                        "trade_type_cluster",
                        "n_trades",
                        "dominant_mode",
                        "cluster_role",
                    ]
                ].to_string(index=False)
            )
