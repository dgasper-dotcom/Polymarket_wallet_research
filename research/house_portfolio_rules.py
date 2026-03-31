"""Shared cap and concentration helpers for the unified house portfolio."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable


def allowed_house_notional(
    *,
    signaled_notional: float,
    current_position_notional: float,
    current_event_notional: float,
    supporting_wallets: Iterable[str],
    wallet_open_notional: dict[str, float],
    max_position_notional_usdc: float | None,
    max_event_notional_usdc: float | None,
    max_wallet_open_notional_usdc: float | None,
) -> tuple[float, float, list[str]]:
    """Return executed/skipped notional plus the binding cap names.

    Wallet cap is enforced on equal-split attribution across supporting wallets.
    """

    signaled = max(0.0, float(signaled_notional or 0.0))
    wallets = sorted({str(wallet) for wallet in supporting_wallets if str(wallet).strip()})

    limits: list[tuple[str, float]] = [("signal", signaled)]

    if max_position_notional_usdc is not None:
        limits.append(
            (
                "position_cap",
                max(0.0, float(max_position_notional_usdc) - float(current_position_notional or 0.0)),
            )
        )

    if max_event_notional_usdc is not None:
        limits.append(
            (
                "event_cap",
                max(0.0, float(max_event_notional_usdc) - float(current_event_notional or 0.0)),
            )
        )

    if max_wallet_open_notional_usdc is not None and wallets:
        wallet_limits: list[float] = []
        for wallet in wallets:
            remaining_wallet = max(
                0.0,
                float(max_wallet_open_notional_usdc) - float(wallet_open_notional.get(wallet, 0.0)),
            )
            wallet_limits.append(remaining_wallet * len(wallets))
        limits.append(("wallet_cap", min(wallet_limits) if wallet_limits else signaled))

    executed = min(limit for _, limit in limits) if limits else signaled
    executed = max(0.0, min(signaled, executed))
    skipped = max(0.0, signaled - executed)

    eps = 1e-9
    binding_caps = [
        name
        for name, limit in limits
        if name != "signal" and limit <= executed + eps and signaled > limit + eps
    ]
    return executed, skipped, binding_caps


def apply_wallet_open_notional_delta(
    wallet_open_notional: dict[str, float],
    supporting_wallets: Iterable[str],
    delta_notional: float,
) -> dict[str, float]:
    """Apply equal-share wallet attribution for the given executed notional."""

    wallets = sorted({str(wallet) for wallet in supporting_wallets if str(wallet).strip()})
    if not wallets:
        return {}
    per_wallet = float(delta_notional or 0.0) / len(wallets)
    attribution = {wallet: per_wallet for wallet in wallets}
    for wallet, value in attribution.items():
        wallet_open_notional[wallet] = float(wallet_open_notional.get(wallet, 0.0)) + value
    return attribution


def release_wallet_open_notional(
    wallet_open_notional: dict[str, float],
    wallet_attribution: dict[str, float] | None,
) -> None:
    """Release previously attributed wallet notional when a position closes."""

    if not wallet_attribution:
        return
    for wallet, value in wallet_attribution.items():
        remaining = float(wallet_open_notional.get(wallet, 0.0)) - float(value or 0.0)
        if remaining <= 1e-9:
            wallet_open_notional.pop(wallet, None)
        else:
            wallet_open_notional[wallet] = remaining


def positive_contribution_shares(
    rows: list[dict[str, object]],
    *,
    value_key: str = "combined_net_pnl_usdc",
) -> dict[str, float]:
    """Return top-1 / top-5 share metrics over positive contributions only."""

    positive_values = sorted(
        [float(row.get(value_key) or 0.0) for row in rows if float(row.get(value_key) or 0.0) > 0.0],
        reverse=True,
    )
    total_positive = sum(positive_values)
    if total_positive <= 0:
        return {
            "positive_count": 0.0,
            "total_positive_pnl_usdc": 0.0,
            "top1_positive_share": 0.0,
            "top5_positive_share": 0.0,
        }
    top1 = positive_values[0]
    top5 = sum(positive_values[:5])
    return {
        "positive_count": float(len(positive_values)),
        "total_positive_pnl_usdc": total_positive,
        "top1_positive_share": top1 / total_positive,
        "top5_positive_share": top5 / total_positive,
    }


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def aggregate_wallet_contributions(
    position_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate combined/realized/MTM PnL by wallet from attributed house positions."""

    wallet_rollup: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "positions": 0.0,
            "entry_notional_usdc": 0.0,
            "realized_net_pnl_usdc": 0.0,
            "open_mtm_net_pnl_usdc": 0.0,
            "combined_net_pnl_usdc": 0.0,
        }
    )
    for row in position_rows:
        attribution = row.get("wallet_notional_attribution") or {}
        entry_notional = _safe_float(row.get("entry_notional_usdc"))
        realized = _safe_float(row.get("realized_net_pnl_usdc"))
        mtm = _safe_float(row.get("mtm_net_pnl_usdc"))
        if entry_notional <= 0 or not attribution:
            continue
        for wallet, attributed_notional in attribution.items():
            share = _safe_float(attributed_notional) / entry_notional if entry_notional > 0 else 0.0
            roll = wallet_rollup[str(wallet)]
            roll["positions"] += 1.0
            roll["entry_notional_usdc"] += _safe_float(attributed_notional)
            roll["realized_net_pnl_usdc"] += realized * share
            roll["open_mtm_net_pnl_usdc"] += mtm * share
            roll["combined_net_pnl_usdc"] += (realized + mtm) * share
    rows = [{"wallet": wallet, **metrics} for wallet, metrics in wallet_rollup.items()]
    return sorted(rows, key=lambda item: item["combined_net_pnl_usdc"], reverse=True)
