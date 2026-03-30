"""Simple research-only execution cost assumptions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config.settings import Settings, get_settings


DEFAULT_SPREAD = 0.01
DEFAULT_FEES_BPS = 0.0
DEFAULT_SLIPPAGE_BPS = 10.0
SLIPPAGE_BY_BUCKET = {
    "micro": 20.0,
    "small": 12.0,
    "medium": 8.0,
    "large": 5.0,
    "unknown": DEFAULT_SLIPPAGE_BPS,
}


def _value(row: pd.Series, key: str, default: Any = None) -> Any:
    """Fetch a Series value while tolerating missing keys."""

    return row[key] if key in row and pd.notna(row[key]) else default


def estimate_polymarket_fee(
    price: float,
    fee_enabled: bool,
    *,
    fee_k: float | None = None,
    flat_fee_bps: float | None = None,
    settings: Settings | None = None,
) -> float:
    """Estimate one event-level Polymarket fee in price units.

    Default behavior uses the research approximation:
    `fee = k * price * (1 - price)`.
    When `flat_fee_bps` is supplied, it overrides the curved fee model.
    """

    if not fee_enabled:
        return 0.0

    cfg = settings or get_settings()
    effective_flat_fee_bps = cfg.flat_fee_bps if flat_fee_bps is None else flat_fee_bps
    if effective_flat_fee_bps is not None:
        return max(float(price), 0.0) * float(effective_flat_fee_bps) / 10_000.0

    effective_fee_k = cfg.polymarket_fee_k if fee_k is None else fee_k
    bounded_price = min(max(float(price), 0.0), 1.0)
    return float(effective_fee_k) * bounded_price * (1.0 - bounded_price)


def estimate_total_cost(
    row: pd.Series,
    *,
    entry_price: float | None = None,
    scenario: str | None = None,
    settings: Settings | None = None,
) -> dict[str, float | str]:
    """Estimate scenario-based total research cost in price units.

    Scenario definitions:
    - optimistic: spread + slippage
    - base: spread + slippage + fee
    - conservative: spread + slippage + fee + extra_penalty
    """

    cfg = settings or get_settings()
    selected_scenario = (scenario or cfg.cost_scenario).strip().lower()
    if selected_scenario not in {"optimistic", "base", "conservative"}:
        raise ValueError("scenario must be optimistic, base, or conservative")

    price = float(entry_price if entry_price is not None else _value(row, "price", 0.5))
    spread = _value(row, "spread_at_trade")
    if spread is None:
        best_bid = _value(row, "best_bid_at_trade")
        best_ask = _value(row, "best_ask_at_trade")
        if best_bid is not None and best_ask is not None:
            spread = float(best_ask) - float(best_bid)
    spread_cost = max(float(spread if spread is not None else DEFAULT_SPREAD), 0.0)

    liquidity_bucket = str(_value(row, "liquidity_bucket", "unknown"))
    slippage_bps = float(
        _value(row, "slippage_bps_assumed", SLIPPAGE_BY_BUCKET.get(liquidity_bucket, DEFAULT_SLIPPAGE_BPS))
    )
    slippage_cost = max(float(price), 0.0) * slippage_bps / 10_000.0
    fee_enabled = selected_scenario in {"base", "conservative"}
    fee_cost = estimate_polymarket_fee(price, fee_enabled=fee_enabled, settings=cfg)
    extra_penalty = float(cfg.extra_cost_penalty) if selected_scenario == "conservative" else 0.0

    total_cost = spread_cost + slippage_cost
    if selected_scenario in {"base", "conservative"}:
        total_cost += fee_cost
    if selected_scenario == "conservative":
        total_cost += extra_penalty

    return {
        "scenario": selected_scenario,
        "spread_cost": spread_cost,
        "slippage_cost": slippage_cost,
        "fee_cost": fee_cost,
        "extra_penalty": extra_penalty,
        "total_cost": total_cost,
    }


def estimate_entry_only_cost(
    row: pd.Series,
    *,
    entry_price: float | None = None,
    scenario: str | None = None,
    settings: Settings | None = None,
) -> dict[str, float | str]:
    """Estimate a one-way entry cost for delayed-follow strategies held to expiry.

    This differs from the round-trip-style helper used for short forward horizons:
    - only half-spread is charged on entry;
    - only entry slippage is charged;
    - only one event-level fee estimate is charged when enabled.

    This is the research approximation used when we enter after observing another
    wallet's trade and then hold the copied position through market resolution
    rather than selling back through the book before expiry.
    """

    cfg = settings or get_settings()
    selected_scenario = (scenario or cfg.cost_scenario).strip().lower()
    if selected_scenario not in {"optimistic", "base", "conservative"}:
        raise ValueError("scenario must be optimistic, base, or conservative")

    price = float(entry_price if entry_price is not None else _value(row, "price", 0.5))
    spread = _value(row, "spread_at_trade")
    if spread is None:
        best_bid = _value(row, "best_bid_at_trade")
        best_ask = _value(row, "best_ask_at_trade")
        if best_bid is not None and best_ask is not None:
            spread = float(best_ask) - float(best_bid)
    half_spread_cost = max(float(spread if spread is not None else DEFAULT_SPREAD), 0.0) / 2.0

    liquidity_bucket = str(_value(row, "liquidity_bucket", "unknown"))
    slippage_bps = float(
        _value(row, "slippage_bps_assumed", SLIPPAGE_BY_BUCKET.get(liquidity_bucket, DEFAULT_SLIPPAGE_BPS))
    )
    slippage_cost = max(float(price), 0.0) * slippage_bps / 10_000.0
    fee_enabled = selected_scenario in {"base", "conservative"}
    fee_cost = estimate_polymarket_fee(price, fee_enabled=fee_enabled, settings=cfg)
    extra_penalty = float(cfg.extra_cost_penalty) if selected_scenario == "conservative" else 0.0

    total_cost = half_spread_cost + slippage_cost
    if selected_scenario in {"base", "conservative"}:
        total_cost += fee_cost
    if selected_scenario == "conservative":
        total_cost += extra_penalty

    return {
        "scenario": selected_scenario,
        "half_spread_cost": half_spread_cost,
        "slippage_cost": slippage_cost,
        "fee_cost": fee_cost,
        "extra_penalty": extra_penalty,
        "total_cost": total_cost,
    }


def calculate_net_pnl(raw_pnl: float | None, total_cost: float | None) -> float | None:
    """Subtract total cost from raw PnL while tolerating missing values."""

    if raw_pnl is None or total_cost is None:
        return None
    return float(raw_pnl) - float(total_cost)


def estimate_break_even_cost(values: pd.Series) -> float | None:
    """Return the mean raw edge, interpreted as the break-even cost level."""

    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def estimate_trade_costs(row: pd.Series) -> dict:
    """Estimate a round-trip research cost in price units.

    The model is intentionally simple:
    1. Half-spread on entry.
    2. Half-spread on exit.
    3. Symmetric slippage on entry and exit.
    4. Optional symmetric fee estimate on entry and exit.
    """

    price = float(_value(row, "price", 0.5))
    spread = _value(row, "spread_at_trade")
    if spread is None:
        best_bid = _value(row, "best_bid_at_trade")
        best_ask = _value(row, "best_ask_at_trade")
        if best_bid is not None and best_ask is not None:
            spread = float(best_ask) - float(best_bid)
    spread = max(float(spread if spread is not None else DEFAULT_SPREAD), 0.0)

    liquidity_bucket = str(_value(row, "liquidity_bucket", "unknown"))
    slippage_bps = float(
        _value(row, "slippage_bps_assumed", SLIPPAGE_BY_BUCKET.get(liquidity_bucket, DEFAULT_SLIPPAGE_BPS))
    )
    fees_bps = float(_value(row, "fees_bps_assumed", DEFAULT_FEES_BPS))

    half_spread = spread / 2.0
    slippage_cost = price * slippage_bps / 10_000.0
    fee_cost = price * fees_bps / 10_000.0
    entry_cost = half_spread + slippage_cost + fee_cost
    exit_cost = half_spread + slippage_cost + fee_cost

    return {
        "entry_cost": entry_cost,
        "exit_cost": exit_cost,
        "round_trip_cost": entry_cost + exit_cost,
        "half_spread_cost": half_spread,
        "slippage_bps_assumed": slippage_bps,
        "fees_bps_assumed": fees_bps,
    }


def apply_copy_fade_costs(row: pd.Series) -> dict:
    """Apply the cost model to copy and fade strategies for each horizon."""

    costs = estimate_trade_costs(row)
    round_trip_cost = costs["round_trip_cost"]
    outputs: dict[str, float | None] = dict(costs)

    for horizon in ("1m", "5m", "30m"):
        raw_ret = _value(row, f"ret_{horizon}")
        if raw_ret is None:
            outputs[f"copy_pnl_{horizon}"] = None
            outputs[f"fade_pnl_{horizon}"] = None
            continue

        raw_ret = float(raw_ret)
        outputs[f"copy_pnl_{horizon}"] = raw_ret - round_trip_cost
        outputs[f"fade_pnl_{horizon}"] = -raw_ret - round_trip_cost

    return outputs
