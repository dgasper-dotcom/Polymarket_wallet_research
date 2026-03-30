"""Consolidate overlapping wallet signals into one house portfolio tape.

This research layer is designed to prevent self-competition:

- multiple wallets buying the same token around the same time should not open
  multiple independent house positions;
- overlapping same-side trades are collapsed into one entry cluster;
- repeated same-side activity while the house is already in the position is
  marked as reinforcement rather than a fresh position.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any


DEFAULT_WALLET_CSV = "exports/manual_seed_copy_ready_v2/manual_seed_copy_ready_monitor_avoid.csv"
DEFAULT_MAPPED_TRADES_CSV = (
    "exports/manual_seed_pma_full_history_backtest_no_lucky/"
    "manual_seed_pma_mapped_trades_full_history.csv"
)
DEFAULT_OUTPUT_DIR = "exports/manual_seed_consolidation"
DEFAULT_CLUSTER_WINDOW_HOURS = 24
DEFAULT_ACTION_BUCKET = "copy_ready"


@dataclass(frozen=True)
class Cluster:
    """One overlap cluster for a token and side."""

    cluster_id: str
    side: str
    token_id: str
    market_id: str
    event_title: str
    outcome: str
    first_ts: datetime
    last_ts: datetime
    trade_count: int
    unique_wallet_count: int
    supporting_wallets: tuple[str, ...]
    total_size: float
    total_notional_usdc: float
    avg_signal_price: float | None


def _read_wallets(path: str | Path, action_bucket: str | None = None) -> list[str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        wallets: list[str] = []
        for row in csv.DictReader(handle):
            wallet = (row.get("wallet_id") or row.get("wallet_address") or "").strip().lower()
            if not wallet:
                continue
            if action_bucket and (row.get("action_bucket") or "").strip() != action_bucket:
                continue
            wallets.append(wallet)
        return wallets


def _read_trades(path: str | Path, wallets: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            wallet = (row.get("wallet_address") or "").lower()
            if wallet not in wallets:
                continue
            ts_raw = row.get("timestamp")
            if not ts_raw:
                continue
            cloned = dict(row)
            cloned["_timestamp"] = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            cloned["_size"] = float(row["size"]) if row.get("size") not in ("", None) else 0.0
            cloned["_usdc_size"] = float(row["usdc_size"]) if row.get("usdc_size") not in ("", None) else 0.0
            cloned["_price"] = float(row["price"]) if row.get("price") not in ("", None) else None
            rows.append(cloned)
    rows.sort(key=lambda row: (row.get("token_id") or "", row.get("side") or "", row["_timestamp"]))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"refusing to write empty csv: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _cluster_trades(
    rows: list[dict[str, Any]],
    cluster_window_hours: int,
) -> list[Cluster]:
    clusters: list[Cluster] = []
    window = timedelta(hours=cluster_window_hours)

    current: list[dict[str, Any]] = []
    current_token = ""
    current_side = ""

    def flush() -> None:
        nonlocal current
        if not current:
            return
        wallets = tuple(sorted({(row.get("wallet_address") or "").lower() for row in current}))
        total_size = sum(row["_size"] for row in current)
        total_notional = sum(row["_usdc_size"] for row in current)
        price_numerator = sum((row["_price"] or 0.0) * row["_usdc_size"] for row in current if row["_price"] is not None)
        avg_price = price_numerator / total_notional if total_notional > 0 else None
        first = current[0]
        cluster_id = "|".join(
            [
                current_side,
                first.get("token_id") or "",
                first["_timestamp"].isoformat(),
                str(len(clusters)),
            ]
        )
        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                side=current_side,
                token_id=first.get("token_id") or "",
                market_id=first.get("market_id") or "",
                event_title=first.get("event_title") or "",
                outcome=first.get("outcome") or "",
                first_ts=current[0]["_timestamp"],
                last_ts=current[-1]["_timestamp"],
                trade_count=len(current),
                unique_wallet_count=len(wallets),
                supporting_wallets=wallets,
                total_size=total_size,
                total_notional_usdc=total_notional,
                avg_signal_price=avg_price,
            )
        )
        current = []

    for row in rows:
        token_id = row.get("token_id") or ""
        side = row.get("side") or ""
        if not current:
            current = [row]
            current_token = token_id
            current_side = side
            continue
        gap_ok = row["_timestamp"] - current[-1]["_timestamp"] <= window
        same_bucket = token_id == current_token and side == current_side
        if same_bucket and gap_ok:
            current.append(row)
            continue
        flush()
        current = [row]
        current_token = token_id
        current_side = side
    flush()
    return clusters


def _cluster_to_row(cluster: Cluster) -> dict[str, Any]:
    return {
        "cluster_id": cluster.cluster_id,
        "side": cluster.side,
        "token_id": cluster.token_id,
        "market_id": cluster.market_id,
        "event_title": cluster.event_title,
        "outcome": cluster.outcome,
        "first_ts": cluster.first_ts.isoformat(),
        "last_ts": cluster.last_ts.isoformat(),
        "trade_count": cluster.trade_count,
        "unique_wallet_count": cluster.unique_wallet_count,
        "supporting_wallets": json.dumps(cluster.supporting_wallets),
        "total_size": cluster.total_size,
        "total_notional_usdc": cluster.total_notional_usdc,
        "avg_signal_price": cluster.avg_signal_price,
    }


def _build_house_tape(clusters: list[Cluster]) -> list[dict[str, Any]]:
    """Turn clustered same-token activity into a unified house signal tape."""

    tape: list[dict[str, Any]] = []
    open_tokens: dict[str, Cluster] = {}
    open_markets: dict[str, Cluster] = {}

    for cluster in sorted(clusters, key=lambda item: item.first_ts):
        state = open_tokens.get(cluster.token_id)
        if cluster.side == "BUY":
            market_state = open_markets.get(cluster.market_id)
            if state is not None:
                action = "reinforce_long"
            elif market_state is not None and market_state.token_id != cluster.token_id:
                action = "market_conflict_skip"
            else:
                open_tokens[cluster.token_id] = cluster
                open_markets[cluster.market_id] = cluster
                action = "open_long"
        elif cluster.side == "SELL":
            if state is None:
                action = "orphan_sell"
            else:
                action = "close_long"
                del open_tokens[cluster.token_id]
                current_market_state = open_markets.get(cluster.market_id)
                if current_market_state is not None and current_market_state.token_id == cluster.token_id:
                    del open_markets[cluster.market_id]
        else:
            action = "ignore"

        tape.append(
            {
                "cluster_id": cluster.cluster_id,
                "action": action,
                "side": cluster.side,
                "token_id": cluster.token_id,
                "market_id": cluster.market_id,
                "event_title": cluster.event_title,
                "outcome": cluster.outcome,
                "first_ts": cluster.first_ts.isoformat(),
                "last_ts": cluster.last_ts.isoformat(),
                "trade_count": cluster.trade_count,
                "unique_wallet_count": cluster.unique_wallet_count,
                "supporting_wallets": json.dumps(cluster.supporting_wallets),
                "total_notional_usdc": cluster.total_notional_usdc,
                "avg_signal_price": cluster.avg_signal_price,
            }
        )
    return tape


def run_manual_seed_consolidation(
    wallet_csv: str | Path = DEFAULT_WALLET_CSV,
    mapped_trades_csv: str | Path = DEFAULT_MAPPED_TRADES_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cluster_window_hours: int = DEFAULT_CLUSTER_WINDOW_HOURS,
    action_bucket: str | None = DEFAULT_ACTION_BUCKET,
) -> dict[str, Any]:
    """Collapse overlapping wallet trades into one house signal stream."""

    output_root = Path(output_dir)
    wallets = _read_wallets(wallet_csv, action_bucket=action_bucket)
    wallet_set = set(wallets)
    trades = _read_trades(mapped_trades_csv, wallet_set)
    clusters = _cluster_trades(trades, cluster_window_hours)
    cluster_rows = [_cluster_to_row(cluster) for cluster in clusters]
    cluster_path = _write_csv(output_root / "consolidated_signal_clusters.csv", cluster_rows)

    tape_rows = _build_house_tape(clusters)
    tape_path = _write_csv(output_root / "house_signal_tape.csv", tape_rows)

    buy_clusters = [cluster for cluster in clusters if cluster.side == "BUY"]
    duplicate_buy_trades = sum(max(0, cluster.trade_count - 1) for cluster in buy_clusters)
    clusters_with_overlap = sum(1 for cluster in buy_clusters if cluster.unique_wallet_count >= 2)
    clusters_with_heavy_overlap = sum(1 for cluster in buy_clusters if cluster.unique_wallet_count >= 3)
    avg_wallets_per_buy_cluster = (
        sum(cluster.unique_wallet_count for cluster in buy_clusters) / len(buy_clusters)
        if buy_clusters
        else 0.0
    )
    open_actions = sum(1 for row in tape_rows if row["action"] == "open_long")
    reinforce_actions = sum(1 for row in tape_rows if row["action"] == "reinforce_long")
    conflict_actions = sum(1 for row in tape_rows if row["action"] == "market_conflict_skip")
    close_actions = sum(1 for row in tape_rows if row["action"] == "close_long")

    summary_lines = [
        "# Manual Seed Consolidation Summary\n",
        "\n",
        "This report collapses overlapping wallet trades into one house portfolio tape so the strategy does not self-compete or triple-size the same thesis.\n",
        "\n",
        f"- Input wallets: `{len(wallets)}`\n",
        f"- Action bucket filter: `{action_bucket or 'all'}`\n",
        f"- Raw mapped trades used: `{len(trades)}`\n",
        f"- Cluster window: `{cluster_window_hours}h`\n",
        f"- Consolidated signal clusters: `{len(clusters)}`\n",
        f"- Buy clusters: `{len(buy_clusters)}`\n",
        f"- Duplicate raw buy trades absorbed: `{duplicate_buy_trades}`\n",
        f"- Buy clusters with 2+ supporting wallets: `{clusters_with_overlap}`\n",
        f"- Buy clusters with 3+ supporting wallets: `{clusters_with_heavy_overlap}`\n",
        f"- Average supporting wallets per buy cluster: `{avg_wallets_per_buy_cluster:.3f}`\n",
        f"- House opens after consolidation: `{open_actions}`\n",
        f"- Reinforcement signals while already long: `{reinforce_actions}`\n",
        f"- Opposite-outcome market conflicts skipped: `{conflict_actions}`\n",
        f"- House closes: `{close_actions}`\n",
        "\n",
        "## House Tape Action Counts\n",
    ]
    action_counts: dict[str, int] = {}
    for row in tape_rows:
        action_counts[row["action"]] = action_counts.get(row["action"], 0) + 1
    for action, count in sorted(action_counts.items()):
        summary_lines.append(f"- `{action}`: `{count}`\n")

    summary_lines.extend(
        [
            "\n",
            "## Interpretation\n",
            "- `open_long` means the house portfolio had no existing position in that token and opens one.\n",
            "- `reinforce_long` means another wallet added to the same token while the house was already long; this is treated as overlap evidence rather than a fresh independent bet.\n",
            "- `close_long` means a clustered sell signal arrived after a prior house entry and closes the unified position.\n",
            "- `market_conflict_skip` means another wallet bought a different outcome in a market where the house was already exposed; the unified portfolio skips that signal to avoid self-competition.\n",
            "- `orphan_sell` means the source wallets sold a token that the house tape did not currently hold under this simple one-position-per-token rule.\n",
        ]
    )

    summary_path = output_root / "consolidation_summary.md"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")

    return {
        "wallets": wallets,
        "trades": trades,
        "clusters": cluster_rows,
        "cluster_path": cluster_path,
        "house_tape": tape_rows,
        "tape_path": tape_path,
        "summary_path": summary_path,
    }
