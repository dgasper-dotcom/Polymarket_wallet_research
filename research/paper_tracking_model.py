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


def _build_house_positions(
    tape_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    open_positions: dict[str, dict[str, Any]] = {}
    closed_positions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for row in sorted(tape_rows, key=lambda item: item["first_ts"]):
        token_id = row["token_id"]
        action = row["action"]
        wallets = tuple(json.loads(row["supporting_wallets"]))
        current = open_positions.get(token_id)

        if action == "open_long":
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
                "total_signaled_notional_usdc": float(row["total_notional_usdc"] or 0.0),
                "avg_open_signal_price": row["avg_signal_price"],
            }
        elif action == "reinforce_long" and current is not None:
            current["last_signal_at"] = row["last_ts"]
            current["signal_cluster_count"] += 1
            current["reinforcement_count"] += 1
            current["raw_trade_count"] += int(row["trade_count"])
            current["total_signaled_notional_usdc"] += float(row["total_notional_usdc"] or 0.0)
            current["supporting_wallets"].update(wallets)
            current["supporting_wallet_count"] = len(current["supporting_wallets"])
        elif action == "close_long" and current is not None:
            current["last_signal_at"] = row["last_ts"]
            current["close_ts"] = row["first_ts"]
            current["status"] = "closed"
            current["closing_cluster_id"] = row["cluster_id"]
            current["raw_trade_count"] += int(row["trade_count"])
            current["signal_cluster_count"] += 1
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
        open_rows.append(row)

    closed_rows: list[dict[str, Any]] = []
    for row in sorted(closed_positions, key=lambda item: item["opened_at"]):
        row["supporting_wallets"] = json.dumps(sorted(row["supporting_wallets"]))
        closed_rows.append(row)

    return open_rows, closed_rows, conflicts


def run_paper_tracking_model(
    wallet_csv: str | Path = DEFAULT_WALLET_CSV,
    mapped_trades_csv: str | Path = DEFAULT_MAPPED_TRADES_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cluster_window_hours: int = DEFAULT_CLUSTER_WINDOW_HOURS,
    action_bucket: str | None = DEFAULT_ACTION_BUCKET,
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
    open_rows, closed_rows, conflict_rows = _build_house_positions(tape_rows)

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
        "total_signaled_notional_usdc",
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

    open_path = _write_csv(output_root / "current_house_positions.csv", open_rows, fieldnames=position_fields)
    closed_path = _write_csv(output_root / "closed_house_positions.csv", closed_rows, fieldnames=position_fields)
    conflict_path = _write_csv(output_root / "skipped_market_conflicts.csv", conflict_rows, fieldnames=conflict_fields)

    raw_trade_count = len(consolidated["trades"])
    tape_count = len(tape_rows)
    open_count = len(open_rows)
    closed_count = len(closed_rows)
    conflict_count = len(conflict_rows)
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
        f"- Average supporting wallets per open house position: `{avg_wallet_support:.2f}`\n",
        f"- Approximate duplicate signal absorption count: `{duplicate_absorbed}`\n",
        "\n",
        "## Interpretation\n",
        "- `current_house_positions.csv` is the file to monitor going forward.\n",
        "- `closed_house_positions.csv` shows where the unified house tape would have exited based on source wallet sells.\n",
        "- `skipped_market_conflicts.csv` shows signals ignored because another wallet wanted the opposite outcome in a market where the house was already exposed.\n",
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
        "consolidated": consolidated,
    }
