"""Build a unified paper-tracking model for copy-ready wallets.

This layer is intentionally not a backtest. It uses the already-selected
wallet set and the consolidated house tape to answer operational questions:

- What positions would the unified house currently be tracking?
- Which raw wallet signals were absorbed as reinforcements?
- Which signals were skipped because they conflicted with an existing market
  exposure?
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from research.manual_seed_consolidation import (
    DEFAULT_ACTION_BUCKET,
    DEFAULT_CLUSTER_WINDOW_HOURS,
    DEFAULT_MAPPED_TRADES_CSV,
    DEFAULT_WALLET_CSV,
    run_manual_seed_consolidation,
)
from research.house_portfolio_rules import (
    allowed_house_notional,
    apply_wallet_open_notional_delta,
    release_wallet_open_notional,
)


DEFAULT_OUTPUT_DIR = "exports/manual_seed_paper_tracking"


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> Path:
    if not rows and not fieldnames:
        raise ValueError(f"refusing to write empty csv without headers: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0].keys()))
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return path


def _capped_notional(
    signaled_notional: float,
    current_notional: float,
    max_position_notional_usdc: float | None,
) -> tuple[float, float]:
    """Return executed and skipped notional under a per-position cap."""

    if max_position_notional_usdc is None:
        return signaled_notional, 0.0
    remaining = max(0.0, float(max_position_notional_usdc) - current_notional)
    executed = min(signaled_notional, remaining)
    skipped = max(0.0, signaled_notional - executed)
    return executed, skipped


def _build_house_positions(
    tape_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return _build_house_positions_with_cap(tape_rows, max_position_notional_usdc=None)


def _build_house_positions_with_cap(
    tape_rows: list[dict[str, Any]],
    *,
    max_position_notional_usdc: float | None,
    max_event_notional_usdc: float | None = None,
    max_wallet_open_notional_usdc: float | None = None,
    max_total_open_notional_usdc: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    open_positions: dict[str, dict[str, Any]] = {}
    closed_positions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    event_open_notional: dict[str, float] = {}
    wallet_open_notional: dict[str, float] = {}
    total_open_notional = 0.0

    for row in sorted(tape_rows, key=lambda item: item["first_ts"]):
        token_id = row["token_id"]
        action = row["action"]
        wallets = tuple(json.loads(row["supporting_wallets"]))
        current = open_positions.get(token_id)
        signaled_notional = float(row["total_notional_usdc"] or 0.0)
        event_key = str(row.get("event_title") or "")

        if action == "open_long":
            executed_notional, skipped_notional, binding_caps = allowed_house_notional(
                signaled_notional=signaled_notional,
                current_position_notional=0.0,
                current_event_notional=event_open_notional.get(event_key, 0.0),
                current_total_open_notional=total_open_notional,
                supporting_wallets=wallets,
                wallet_open_notional=wallet_open_notional,
                max_position_notional_usdc=max_position_notional_usdc,
                max_event_notional_usdc=max_event_notional_usdc,
                max_wallet_open_notional_usdc=max_wallet_open_notional_usdc,
                max_total_open_notional_usdc=max_total_open_notional_usdc,
            )
            if executed_notional <= 0:
                reason = "position_cap_reached"
                if "event_cap" in binding_caps:
                    reason = "event_cap_reached"
                elif "wallet_cap" in binding_caps:
                    reason = "wallet_cap_reached"
                elif "book_cap" in binding_caps:
                    reason = "book_cap_reached"
                cap_rows.append(
                    {
                        "cluster_id": row["cluster_id"],
                        "action": action,
                        "reason": reason,
                        "binding_caps": ",".join(binding_caps),
                        "token_id": token_id,
                        "market_id": row["market_id"],
                        "event_title": row["event_title"],
                        "outcome": row["outcome"],
                        "signaled_notional_usdc": signaled_notional,
                        "executed_notional_usdc": executed_notional,
                        "skipped_notional_usdc": skipped_notional,
                    }
                )
                continue
            wallet_attribution = apply_wallet_open_notional_delta(wallet_open_notional, wallets, executed_notional)
            event_open_notional[event_key] = float(event_open_notional.get(event_key, 0.0)) + executed_notional
            total_open_notional += executed_notional
            open_positions[token_id] = {
                "token_id": token_id,
                "market_id": row["market_id"],
                "event_title": row["event_title"],
                "outcome": row["outcome"],
                "opened_at": row["first_ts"],
                "last_signal_at": row["last_ts"],
                "close_ts": "",
                "status": "open",
                "opening_cluster_id": row["cluster_id"],
                "closing_cluster_id": "",
                "signal_cluster_count": 1,
                "reinforcement_count": 0,
                "raw_trade_count": int(row["trade_count"]),
                "supporting_wallet_count": int(row["unique_wallet_count"]),
                "supporting_wallets": set(wallets),
                "wallet_notional_attribution": wallet_attribution,
                "total_signaled_notional_usdc": signaled_notional,
                "executed_notional_usdc": executed_notional,
                "suppressed_notional_usdc": skipped_notional,
                "avg_open_signal_price": row["avg_signal_price"],
            }
            if skipped_notional > 0:
                reason = "position_cap_partial_open"
                if "event_cap" in binding_caps:
                    reason = "event_cap_partial_open"
                elif "wallet_cap" in binding_caps:
                    reason = "wallet_cap_partial_open"
                elif "book_cap" in binding_caps:
                    reason = "book_cap_partial_open"
                cap_rows.append(
                    {
                        "cluster_id": row["cluster_id"],
                        "action": action,
                        "reason": reason,
                        "binding_caps": ",".join(binding_caps),
                        "token_id": token_id,
                        "market_id": row["market_id"],
                        "event_title": row["event_title"],
                        "outcome": row["outcome"],
                        "signaled_notional_usdc": signaled_notional,
                        "executed_notional_usdc": executed_notional,
                        "skipped_notional_usdc": skipped_notional,
                    }
                )
        elif action == "reinforce_long" and current is not None:
            executed_notional, skipped_notional, binding_caps = allowed_house_notional(
                signaled_notional=signaled_notional,
                current_position_notional=float(current.get("executed_notional_usdc") or 0.0),
                current_event_notional=event_open_notional.get(event_key, 0.0),
                current_total_open_notional=total_open_notional,
                supporting_wallets=wallets,
                wallet_open_notional=wallet_open_notional,
                max_position_notional_usdc=max_position_notional_usdc,
                max_event_notional_usdc=max_event_notional_usdc,
                max_wallet_open_notional_usdc=max_wallet_open_notional_usdc,
                max_total_open_notional_usdc=max_total_open_notional_usdc,
            )
            if executed_notional <= 0:
                reason = "position_cap_reached"
                if "event_cap" in binding_caps:
                    reason = "event_cap_reached"
                elif "wallet_cap" in binding_caps:
                    reason = "wallet_cap_reached"
                elif "book_cap" in binding_caps:
                    reason = "book_cap_reached"
                cap_rows.append(
                    {
                        "cluster_id": row["cluster_id"],
                        "action": action,
                        "reason": reason,
                        "binding_caps": ",".join(binding_caps),
                        "token_id": token_id,
                        "market_id": row["market_id"],
                        "event_title": row["event_title"],
                        "outcome": row["outcome"],
                        "signaled_notional_usdc": signaled_notional,
                        "executed_notional_usdc": executed_notional,
                        "skipped_notional_usdc": skipped_notional,
                    }
                )
                continue
            current["last_signal_at"] = row["last_ts"]
            current["signal_cluster_count"] += 1
            current["reinforcement_count"] += 1
            current["raw_trade_count"] += int(row["trade_count"])
            current["total_signaled_notional_usdc"] += signaled_notional
            current["executed_notional_usdc"] += executed_notional
            current["suppressed_notional_usdc"] += skipped_notional
            current["supporting_wallets"].update(wallets)
            current["supporting_wallet_count"] = len(current["supporting_wallets"])
            current_wallet_attribution = dict(current.get("wallet_notional_attribution") or {})
            delta_wallet_attr = apply_wallet_open_notional_delta(wallet_open_notional, wallets, executed_notional)
            for wallet, value in delta_wallet_attr.items():
                current_wallet_attribution[wallet] = float(current_wallet_attribution.get(wallet, 0.0)) + value
            current["wallet_notional_attribution"] = current_wallet_attribution
            event_open_notional[event_key] = float(event_open_notional.get(event_key, 0.0)) + executed_notional
            total_open_notional += executed_notional
            if skipped_notional > 0:
                reason = "position_cap_partial_reinforce"
                if "event_cap" in binding_caps:
                    reason = "event_cap_partial_reinforce"
                elif "wallet_cap" in binding_caps:
                    reason = "wallet_cap_partial_reinforce"
                elif "book_cap" in binding_caps:
                    reason = "book_cap_partial_reinforce"
                cap_rows.append(
                    {
                        "cluster_id": row["cluster_id"],
                        "action": action,
                        "reason": reason,
                        "binding_caps": ",".join(binding_caps),
                        "token_id": token_id,
                        "market_id": row["market_id"],
                        "event_title": row["event_title"],
                        "outcome": row["outcome"],
                        "signaled_notional_usdc": signaled_notional,
                        "executed_notional_usdc": executed_notional,
                        "skipped_notional_usdc": skipped_notional,
                    }
                )
        elif action == "close_long" and current is not None:
            current["last_signal_at"] = row["last_ts"]
            current["close_ts"] = row["first_ts"]
            current["status"] = "closed"
            current["closing_cluster_id"] = row["cluster_id"]
            current["raw_trade_count"] += int(row["trade_count"])
            current["signal_cluster_count"] += 1
            event_open_notional[event_key] = max(
                0.0,
                float(event_open_notional.get(event_key, 0.0)) - float(current.get("executed_notional_usdc") or 0.0),
            )
            if event_open_notional[event_key] <= 1e-9:
                event_open_notional.pop(event_key, None)
            release_wallet_open_notional(wallet_open_notional, current.get("wallet_notional_attribution"))
            total_open_notional = max(
                0.0,
                total_open_notional - float(current.get("executed_notional_usdc") or 0.0),
            )
            current["supporting_wallets"] = tuple(sorted(current["supporting_wallets"]))
            closed_positions.append(current)
            del open_positions[token_id]
        elif action == "market_conflict_skip":
            conflicts.append(
                {
                    "cluster_id": row["cluster_id"],
                    "first_ts": row["first_ts"],
                    "market_id": row["market_id"],
                    "event_title": row["event_title"],
                    "outcome": row["outcome"],
                    "token_id": row["token_id"],
                    "trade_count": row["trade_count"],
                    "unique_wallet_count": row["unique_wallet_count"],
                    "supporting_wallets": row["supporting_wallets"],
                    "total_notional_usdc": row["total_notional_usdc"],
                }
            )

    open_rows: list[dict[str, Any]] = []
    for row in sorted(open_positions.values(), key=lambda item: item["opened_at"]):
        row["supporting_wallets"] = json.dumps(sorted(row["supporting_wallets"]))
        row["wallet_notional_attribution"] = json.dumps(
            {key: round(float(value), 10) for key, value in sorted((row.get("wallet_notional_attribution") or {}).items())}
        )
        open_rows.append(row)

    closed_rows: list[dict[str, Any]] = []
    for row in sorted(closed_positions, key=lambda item: item["opened_at"]):
        row["supporting_wallets"] = json.dumps(sorted(row["supporting_wallets"]))
        row["wallet_notional_attribution"] = json.dumps(
            {key: round(float(value), 10) for key, value in sorted((row.get("wallet_notional_attribution") or {}).items())}
        )
        closed_rows.append(row)

    return open_rows, closed_rows, conflicts, cap_rows


def run_paper_tracking_model(
    wallet_csv: str | Path = DEFAULT_WALLET_CSV,
    mapped_trades_csv: str | Path = DEFAULT_MAPPED_TRADES_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cluster_window_hours: int = DEFAULT_CLUSTER_WINDOW_HOURS,
    action_bucket: str | None = DEFAULT_ACTION_BUCKET,
    max_position_notional_usdc: float | None = None,
    max_event_notional_usdc: float | None = None,
    max_wallet_open_notional_usdc: float | None = None,
    max_total_open_notional_usdc: float | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    consolidated = run_manual_seed_consolidation(
        wallet_csv=wallet_csv,
        mapped_trades_csv=mapped_trades_csv,
        output_dir=output_root / "consolidated",
        cluster_window_hours=cluster_window_hours,
        action_bucket=action_bucket,
    )

    tape_rows = consolidated["house_tape"]
    open_rows, closed_rows, conflict_rows, cap_rows = _build_house_positions_with_cap(
        tape_rows,
        max_position_notional_usdc=max_position_notional_usdc,
        max_event_notional_usdc=max_event_notional_usdc,
        max_wallet_open_notional_usdc=max_wallet_open_notional_usdc,
        max_total_open_notional_usdc=max_total_open_notional_usdc,
    )

    position_fields = [
        "token_id",
        "market_id",
        "event_title",
        "outcome",
        "opened_at",
        "last_signal_at",
        "close_ts",
        "status",
        "opening_cluster_id",
        "closing_cluster_id",
        "signal_cluster_count",
        "reinforcement_count",
        "raw_trade_count",
        "supporting_wallet_count",
        "supporting_wallets",
        "wallet_notional_attribution",
        "total_signaled_notional_usdc",
        "executed_notional_usdc",
        "suppressed_notional_usdc",
        "avg_open_signal_price",
    ]
    conflict_fields = [
        "cluster_id",
        "first_ts",
        "market_id",
        "event_title",
        "outcome",
        "token_id",
        "trade_count",
        "unique_wallet_count",
        "supporting_wallets",
        "total_notional_usdc",
    ]
    cap_fields = [
        "cluster_id",
        "action",
        "reason",
        "binding_caps",
        "token_id",
        "market_id",
        "event_title",
        "outcome",
        "signaled_notional_usdc",
        "executed_notional_usdc",
        "skipped_notional_usdc",
    ]

    open_path = _write_csv(output_root / "current_house_positions.csv", open_rows, fieldnames=position_fields)
    closed_path = _write_csv(output_root / "closed_house_positions.csv", closed_rows, fieldnames=position_fields)
    conflict_path = _write_csv(output_root / "skipped_market_conflicts.csv", conflict_rows, fieldnames=conflict_fields)
    cap_path = _write_csv(output_root / "skipped_position_cap_records.csv", cap_rows, fieldnames=cap_fields)

    raw_trade_count = len(consolidated["trades"])
    tape_count = len(tape_rows)
    open_count = len(open_rows)
    closed_count = len(closed_rows)
    conflict_count = len(conflict_rows)
    cap_skip_count = len(cap_rows)
    reinforce_count = sum(1 for row in tape_rows if row["action"] == "reinforce_long")
    open_action_count = sum(1 for row in tape_rows if row["action"] == "open_long")
    duplicate_absorbed = max(0, raw_trade_count - open_action_count - reinforce_count - conflict_count)
    avg_wallet_support = (
        sum(int(row["supporting_wallet_count"]) for row in open_rows) / open_count
        if open_count
        else 0.0
    )

    summary_lines = [
        "# Paper Tracking Model Summary\n",
        "\n",
        "This is the operational paper-tracking layer for the unified house portfolio. It is not a historical backtest. It merges copy-ready wallet signals into one portfolio state so repeated same-thesis bets do not triple-size exposure.\n",
        "\n",
        f"- Wallet bucket tracked: `{action_bucket or 'all'}`\n",
        f"- Raw mapped trades consumed: `{raw_trade_count}`\n",
        f"- Consolidated house tape actions: `{tape_count}`\n",
        f"- Current open house positions: `{open_count}`\n",
        f"- Closed house positions: `{closed_count}`\n",
        f"- Reinforcement actions: `{reinforce_count}`\n",
        f"- Market-conflict actions skipped: `{conflict_count}`\n",
        f"- Position-cap skip / partial-fill records: `{cap_skip_count}`\n",
        (
            f"- Per-position notional cap: `{max_position_notional_usdc:.2f} USDC`\n"
            if max_position_notional_usdc is not None
            else ""
        ),
        (
            f"- Per-event notional cap: `{max_event_notional_usdc:.2f} USDC`\n"
            if max_event_notional_usdc is not None
            else ""
        ),
        (
            f"- Per-wallet open-notional cap: `{max_wallet_open_notional_usdc:.2f} USDC`\n"
            if max_wallet_open_notional_usdc is not None
            else ""
        ),
        (
            f"- Total concurrent house-notional cap: `{max_total_open_notional_usdc:.2f} USDC`\n"
            if max_total_open_notional_usdc is not None
            else ""
        ),
        f"- Average supporting wallets per open house position: `{avg_wallet_support:.2f}`\n",
        f"- Approximate duplicate signal absorption count: `{duplicate_absorbed}`\n",
        "\n",
        "## Interpretation\n",
        "- `current_house_positions.csv` is the file to monitor going forward.\n",
        "- `closed_house_positions.csv` shows where the unified house tape would have exited based on source wallet sells.\n",
        "- `skipped_market_conflicts.csv` shows signals ignored because another wallet wanted the opposite outcome in a market where the house was already exposed.\n",
        "- `skipped_position_cap_records.csv` shows signals that were fully blocked or partially clipped by the per-position cap.\n",
        "- Reinforcements are recorded, but they do not automatically open a second independent position.\n",
    ]

    summary_path = output_root / "paper_tracking_model_summary.md"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")

    return {
        "open_rows": open_rows,
        "closed_rows": closed_rows,
        "conflict_rows": conflict_rows,
        "summary_path": summary_path,
        "open_path": open_path,
        "closed_path": closed_path,
        "conflict_path": conflict_path,
        "cap_path": cap_path,
        "consolidated": consolidated,
    }
