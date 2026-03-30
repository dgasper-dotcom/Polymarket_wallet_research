"""Rebuild the manual-seed watchlist under a follower-accessible-edge lens.

This module intentionally does not require long holding periods as a gating
criterion. A wallet can be copy-ready if:

- the follower copy evidence is positive, and
- the behavior is not HFT / execution-only.

It also produces an investor-readiness summary for the current copy-ready
subset using historical and time-split copy PnL.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "exports/manual_seed_copy_ready"
DEFAULT_SEED_CSV = "data/manual_seed_wallets.csv"
DEFAULT_LONG_BETTING_RERANK_CSV = "exports/manual_seed_long_betting_rerank/manual_seed_long_betting_rerank.csv"
DEFAULT_PMA_WALLET_SUMMARY_CSV = (
    "exports/manual_seed_pma_full_history_backtest_no_lucky/"
    "manual_seed_pma_backtest_wallet_summary_0s_5s_15s_30s_60s_full_history.csv"
)
DEFAULT_PMA_TRADE_DIAGNOSTICS_CSV = (
    "exports/manual_seed_pma_full_history_backtest_no_lucky/"
    "manual_seed_pma_backtest_trade_diagnostics_0s_5s_15s_30s_60s_full_history.csv"
)
DEFAULT_MANUAL_SEED_PNL_CSV = "exports/manual_seed_analysis/manual_seed_wallet_pnl_summary.csv"
DEFAULT_WALLET_FEATURES_CSV = "data/wallet_features.csv"

DELAYS = ("0s", "5s", "15s", "30s", "60s")
RATIO_SPLITS = (0.60, 0.70, 0.80)


@dataclass(frozen=True)
class WalletDecision:
    """One wallet decision row."""

    wallet_id: str
    display_name: str
    action_bucket: str
    action_rationale: str
    is_hft_like: bool
    is_copy_positive: bool
    pma_copy_combined_net_30s: float | None
    avg_copy_edge_net_30s: float | None
    combined_realized_plus_mtm_pnl_usdc_est: float | None
    combined_realized_pnl_usdc_est: float | None
    unresolved_open_mtm_net_pnl_usdc: float | None
    unresolved_open_forward_30d_net_pnl_usdc: float | None
    recent_trades_window: float | None
    recent_sell_share: float | None
    fast_exit_share_30s: float | None
    median_holding_seconds: float | None
    notes: str


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_keyed(path: str | Path, key: str) -> dict[str, dict[str, str]]:
    keyed: dict[str, dict[str, str]] = {}
    for row in _read_rows(path):
        keyed[(row.get(key) or "").lower()] = row
    return keyed


def _to_float(value: Any) -> float | None:
    if value in ("", None, "None", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _safe_sum(rows: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = _to_float(row.get(key))
        if value is not None:
            total += value
    return total


def _ratio_split(
    rows: list[dict[str, Any]],
    fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], datetime]:
    if not rows:
        raise ValueError("cannot split an empty row set")
    split_index = max(1, int(len(rows) * fraction))
    split_index = min(split_index, len(rows) - 1)
    cutoff_ts = rows[split_index]["_buy_ts"]
    train = [row for row in rows if row["_buy_ts"] < cutoff_ts]
    test = [row for row in rows if row["_buy_ts"] >= cutoff_ts]
    return train, test, cutoff_ts


def _wallet_name(seed_row: dict[str, str]) -> str:
    return seed_row.get("display_name") or seed_row.get("wallet_address") or ""


def _is_hft_like(
    fast_exit_share_30s: float | None,
    median_holding_seconds: float | None,
    recent_trades_window: float | None,
) -> bool:
    if fast_exit_share_30s is not None and fast_exit_share_30s > 0.20:
        return True
    if median_holding_seconds is not None and median_holding_seconds < 300:
        return True
    if recent_trades_window is not None and recent_trades_window > 5000:
        return True
    return False


def _recent_sell_share(feature_row: dict[str, str]) -> float | None:
    recent_trades = _to_float(feature_row.get("recent_trades_window"))
    recent_sells = _to_float(feature_row.get("recent_sell_trades"))
    if recent_trades is None or recent_trades <= 0 or recent_sells is None:
        return None
    return recent_sells / recent_trades


def _classify_wallet(
    seed_row: dict[str, str],
    rerank_row: dict[str, str] | None,
    pnl_row: dict[str, str] | None,
    feature_row: dict[str, str] | None,
) -> WalletDecision:
    wallet_id = (seed_row.get("wallet_address") or "").lower()
    display_name = _wallet_name(seed_row)
    notes = seed_row.get("notes") or ""

    rerank_row = rerank_row or {}
    pnl_row = pnl_row or {}
    feature_row = feature_row or {}

    pma_copy_30 = _to_float(rerank_row.get("pma_copy_combined_net_30s"))
    avg_copy_30 = _to_float(feature_row.get("avg_copy_edge_net_30s"))
    combined = _to_float(pnl_row.get("combined_realized_plus_mtm_pnl_usdc_est"))
    realized = _to_float(pnl_row.get("combined_realized_pnl_usdc_est"))
    mtm = _to_float(pnl_row.get("unresolved_open_mtm_net_pnl_usdc"))
    forward_30d = _to_float(pnl_row.get("unresolved_open_forward_30d_net_pnl_usdc"))
    recent_trades = _to_float(feature_row.get("recent_trades_window"))
    recent_sell_share = _recent_sell_share(feature_row)
    fast_exit_share_30s = _to_float(feature_row.get("fast_exit_share_30s"))
    median_holding_seconds = _to_float(feature_row.get("median_holding_seconds"))

    is_hft_like = _is_hft_like(fast_exit_share_30s, median_holding_seconds, recent_trades)
    is_copy_positive = (
        (pma_copy_30 is not None and pma_copy_30 > 0)
        or (avg_copy_30 is not None and avg_copy_30 > 0)
    )
    is_copy_negative = (
        pma_copy_30 is not None
        and pma_copy_30 < 0
        and (avg_copy_30 is None or avg_copy_30 <= 0)
    )

    action_bucket = "monitor"
    rationale = "mixed_signal"

    if "crowded" in notes.lower() or display_name.lower() == "kickstand7":
        action_bucket = "monitor"
        rationale = "crowded_reference_not_primary"
    elif is_hft_like:
        action_bucket = "avoid"
        rationale = "hft_or_execution_only"
    elif display_name.lower() == "melody626":
        action_bucket = "avoid"
        rationale = "copyability_failed"
    elif is_copy_negative and (combined is None or combined <= 0):
        action_bucket = "avoid"
        rationale = "copyability_failed"
    elif is_copy_positive:
        action_bucket = "copy_ready"
        rationale = "copyable_and_not_hft"
    elif forward_30d is not None and forward_30d > 0 and not is_hft_like:
        action_bucket = "monitor"
        rationale = "positive_forward_but_copy_incomplete"
    else:
        action_bucket = "avoid"
        rationale = "insufficient_copyable_edge"

    # Explicit overrides from current research interpretation.
    if wallet_id == "0x53ecc53e7a69aad0e6dda60264cc2e363092df91":
        action_bucket = "copy_ready"
        rationale = "best_overall_copyable_candidate"
    if wallet_id == "0x77fd7aec1952ea7d042a6eec83bc4782f67db6c8":
        action_bucket = "monitor"
        rationale = "copy_signal_conflict_needs_more_validation"
    if wallet_id in {
        "0xe36296a42555b95e95880412387e954d84b0bd00",  # SnowballHustle
        "0xe7590338d435112c032e3ea51ff3d08a27a1e7ca",  # PetrGrepl
        "0x312bcca3bc77bdc1d37dc6db5b9c1493de61cafe",  # ELICHOU
        "0x5d2f49295387e01a49f0a3e59449ceed791c4adb",  # MikeMoore
        "0x3c4c03892f47d3166ee049a48a73d4743a17dd95",  # RobertoRubio
        "0xc483ee2ce773ae281131382ecc6285c968b88ac8",  # sbinnala
        "0x68d1b156197fc516c56fc95d325b3716322c3c4d",  # aikko
    } and not is_hft_like and is_copy_positive:
        action_bucket = "copy_ready"
        rationale = "copyable_and_not_hft"

    return WalletDecision(
        wallet_id=wallet_id,
        display_name=display_name,
        action_bucket=action_bucket,
        action_rationale=rationale,
        is_hft_like=is_hft_like,
        is_copy_positive=is_copy_positive,
        pma_copy_combined_net_30s=pma_copy_30,
        avg_copy_edge_net_30s=avg_copy_30,
        combined_realized_plus_mtm_pnl_usdc_est=combined,
        combined_realized_pnl_usdc_est=realized,
        unresolved_open_mtm_net_pnl_usdc=mtm,
        unresolved_open_forward_30d_net_pnl_usdc=forward_30d,
        recent_trades_window=recent_trades,
        recent_sell_share=recent_sell_share,
        fast_exit_share_30s=fast_exit_share_30s,
        median_holding_seconds=median_holding_seconds,
        notes=notes,
    )


def _decision_to_row(decision: WalletDecision) -> dict[str, Any]:
    return {
        "display_name": decision.display_name,
        "wallet_id": decision.wallet_id,
        "action_bucket": decision.action_bucket,
        "action_rationale": decision.action_rationale,
        "is_hft_like": decision.is_hft_like,
        "is_copy_positive": decision.is_copy_positive,
        "pma_copy_combined_net_30s": decision.pma_copy_combined_net_30s,
        "avg_copy_edge_net_30s": decision.avg_copy_edge_net_30s,
        "combined_realized_plus_mtm_pnl_usdc_est": decision.combined_realized_plus_mtm_pnl_usdc_est,
        "combined_realized_pnl_usdc_est": decision.combined_realized_pnl_usdc_est,
        "unresolved_open_mtm_net_pnl_usdc": decision.unresolved_open_mtm_net_pnl_usdc,
        "unresolved_open_forward_30d_net_pnl_usdc": decision.unresolved_open_forward_30d_net_pnl_usdc,
        "recent_trades_window": decision.recent_trades_window,
        "recent_sell_share": decision.recent_sell_share,
        "fast_exit_share_30s": decision.fast_exit_share_30s,
        "median_holding_seconds": decision.median_holding_seconds,
        "notes": decision.notes,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty csv: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_txt(path: Path, wallet_ids: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{wallet}\n" for wallet in wallet_ids), encoding="utf-8")
    return path


def _portfolio_row(rows: list[dict[str, str]], split_label: str, cutoff: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "split": split_label,
        "cutoff_ts": cutoff,
        "n_slices": len(rows),
        "n_wallets": len({(row.get("wallet_address") or "").lower() for row in rows}),
        "first_buy_ts": rows[0]["buy_timestamp"] if rows else "",
        "last_buy_ts": rows[-1]["buy_timestamp"] if rows else "",
    }
    for delay in DELAYS:
        record[f"combined_net_total_usdc_{delay}"] = _safe_sum(rows, f"copy_combined_net_usdc_{delay}")
        record[f"realized_net_total_usdc_{delay}"] = _safe_sum(rows, f"copy_pnl_net_usdc_{delay}")
        record[f"unrealized_mtm_total_usdc_{delay}"] = _safe_sum(rows, f"copy_unrealized_mtm_net_usdc_{delay}")
    return record


def _select_trade_rows(
    trade_rows: list[dict[str, str]],
    wallets: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in trade_rows:
        wallet = (row.get("wallet_address") or "").lower()
        if wallet not in wallets:
            continue
        ts_raw = row.get("buy_timestamp")
        if not ts_raw:
            continue
        cloned = dict(row)
        cloned["_buy_ts"] = _to_iso_timestamp(ts_raw)
        selected.append(cloned)
    selected.sort(key=lambda item: item["_buy_ts"])
    return selected


def _subset_concentration(wallet_summary: list[dict[str, str]], wallets: set[str]) -> tuple[float | None, str | None]:
    subset = [row for row in wallet_summary if (row.get("wallet_address") or "").lower() in wallets]
    positives = [(row.get("display_name") or row.get("wallet_address") or "", _to_float(row.get("combined_net_total_usdc_30s")) or 0.0) for row in subset]
    total = sum(value for _, value in positives if value > 0)
    if total <= 0:
        return None, None
    name, top_value = max(positives, key=lambda item: item[1])
    if top_value <= 0:
        return None, None
    return top_value / total, name


def _search_best_subset(
    trade_rows: list[dict[str, Any]],
    wallet_summary: list[dict[str, str]],
    copy_ready_wallets: list[str],
) -> list[dict[str, Any]]:
    """Explore small copy-ready subsets for positive OOS robustness.

    This is an exploratory search, not production validation.
    """

    candidates: list[dict[str, Any]] = []
    for size in range(2, min(5, len(copy_ready_wallets)) + 1):
        for subset in combinations(copy_ready_wallets, size):
            subset_set = set(subset)
            subset_rows = _select_trade_rows(trade_rows, subset_set)
            if len(subset_rows) < 100:
                continue
            positive_splits = 0
            split_records: dict[str, float] = {}
            for fraction in RATIO_SPLITS:
                train, test, _ = _ratio_split(subset_rows, fraction)
                test_30s = _safe_sum(test, "copy_combined_net_usdc_30s")
                split_records[f"test_30s_{int(fraction*100)}_{int((1-fraction)*100)}"] = test_30s
                if test_30s > 0:
                    positive_splits += 1
            full_30s = _safe_sum(subset_rows, "copy_combined_net_usdc_30s")
            concentration, leader = _subset_concentration(wallet_summary, subset_set)
            candidates.append(
                {
                    "wallet_ids": ",".join(subset),
                    "subset_size": size,
                    "full_history_combined_net_30s": full_30s,
                    "positive_oos_splits": positive_splits,
                    "top_wallet_positive_contribution_share_30s": concentration,
                    "top_wallet_positive_contribution_name": leader,
                    **split_records,
                }
            )
    candidates.sort(
        key=lambda row: (
            -(row["positive_oos_splits"] or 0),
            -(row["full_history_combined_net_30s"] or 0.0),
            row["top_wallet_positive_contribution_share_30s"]
            if row["top_wallet_positive_contribution_share_30s"] is not None
            else 999.0,
        )
    )
    return candidates


def run_manual_seed_copy_ready(
    seed_csv: str | Path = DEFAULT_SEED_CSV,
    rerank_csv: str | Path = DEFAULT_LONG_BETTING_RERANK_CSV,
    pnl_csv: str | Path = DEFAULT_MANUAL_SEED_PNL_CSV,
    wallet_features_csv: str | Path = DEFAULT_WALLET_FEATURES_CSV,
    pma_wallet_summary_csv: str | Path = DEFAULT_PMA_WALLET_SUMMARY_CSV,
    pma_trade_diagnostics_csv: str | Path = DEFAULT_PMA_TRADE_DIAGNOSTICS_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Build the revised copy-ready watchlist and investor-readiness report."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rerank = _read_keyed(rerank_csv, "wallet_id")
    pnl = _read_keyed(pnl_csv, "wallet_id")
    features = _read_keyed(wallet_features_csv, "wallet_id")

    decisions = [
        _classify_wallet(seed, rerank.get((seed.get("wallet_address") or "").lower()), pnl.get((seed.get("wallet_address") or "").lower()), features.get((seed.get("wallet_address") or "").lower()))
        for seed in _read_rows(seed_csv)
    ]

    action_rank = {"copy_ready": 0, "monitor": 1, "avoid": 2}
    decisions.sort(
        key=lambda decision: (
            action_rank.get(decision.action_bucket, 9),
            -(decision.pma_copy_combined_net_30s or float("-inf")),
            -(decision.combined_realized_plus_mtm_pnl_usdc_est or float("-inf")),
        )
    )

    decision_rows = [_decision_to_row(decision) for decision in decisions]
    all_path = _write_csv(output_root / "manual_seed_copy_ready_monitor_avoid.csv", decision_rows)

    bucket_paths: dict[str, Path] = {}
    for bucket in ("copy_ready", "monitor", "avoid"):
        subset = [row for row in decision_rows if row["action_bucket"] == bucket]
        if not subset:
            continue
        bucket_paths[f"{bucket}_csv"] = _write_csv(output_root / f"{bucket}.csv", subset)
        bucket_paths[f"{bucket}_txt"] = _write_txt(output_root / f"{bucket}.txt", [row["wallet_id"] for row in subset])

    copy_ready_wallets = [decision.wallet_id for decision in decisions if decision.action_bucket == "copy_ready"]
    trade_rows = _read_rows(pma_trade_diagnostics_csv)
    wallet_summary_rows = _read_rows(pma_wallet_summary_csv)
    selected_rows = _select_trade_rows(trade_rows, set(copy_ready_wallets))

    portfolio_rows: list[dict[str, Any]] = []
    if selected_rows:
        portfolio_rows.append(_portfolio_row(selected_rows, "full_history", ""))
        for fraction in RATIO_SPLITS:
            train, test, cutoff = _ratio_split(selected_rows, fraction)
            portfolio_rows.append(_portfolio_row(train, f"train_{int(fraction*100)}_{int((1-fraction)*100)}", cutoff.isoformat()))
            portfolio_rows.append(_portfolio_row(test, f"test_{int(fraction*100)}_{int((1-fraction)*100)}", cutoff.isoformat()))

    portfolio_path = None
    if portfolio_rows:
        portfolio_path = _write_csv(output_root / "copy_ready_portfolio_splits.csv", portfolio_rows)

    subset_search_rows = _search_best_subset(trade_rows, wallet_summary_rows, copy_ready_wallets) if copy_ready_wallets else []
    subset_search_path = None
    if subset_search_rows:
        subset_search_path = _write_csv(output_root / "copy_ready_subset_search.csv", subset_search_rows)

    investor_ready = False
    investor_ready_reasons: list[str] = []
    if portfolio_rows:
        tests = [row for row in portfolio_rows if row["split"].startswith("test_")]
        positive_tests_30s = sum(1 for row in tests if (row.get("combined_net_total_usdc_30s") or 0.0) > 0)
        full_history_30s = next((row for row in portfolio_rows if row["split"] == "full_history"), None)
        concentration, concentration_name = _subset_concentration(wallet_summary_rows, set(copy_ready_wallets))
        if (
            full_history_30s is not None
            and (full_history_30s.get("combined_net_total_usdc_30s") or 0.0) > 0
            and positive_tests_30s >= 2
            and concentration is not None
            and concentration < 0.70
            and len(copy_ready_wallets) >= 2
        ):
            investor_ready = True
        else:
            if full_history_30s is None or (full_history_30s.get("combined_net_total_usdc_30s") or 0.0) <= 0:
                investor_ready_reasons.append("full_history_copy_ready_portfolio_not_positive_at_30s")
            if positive_tests_30s < 2:
                investor_ready_reasons.append("copy_ready_portfolio_not_positive_in_enough_oos_splits")
            if concentration is None or concentration >= 0.70:
                investor_ready_reasons.append(
                    f"copy_ready_portfolio_too_concentrated_in_{concentration_name or 'one_wallet'}"
                )
            if len(copy_ready_wallets) < 2:
                investor_ready_reasons.append("copy_ready_universe_too_small")
    else:
        investor_ready_reasons.append("no_copy_ready_trade_rows_available")

    summary_lines = [
        "# Manual Seed Copy-Ready Watchlist\n",
        "\n",
        "This watchlist uses the corrected standard: a wallet does not need to be a long-hold model. It only needs to appear copyable for a follower and not look HFT / execution-only.\n",
        "\n",
        "## Counts\n",
        f"- `copy_ready`: `{sum(1 for d in decisions if d.action_bucket == 'copy_ready')}`\n",
        f"- `monitor`: `{sum(1 for d in decisions if d.action_bucket == 'monitor')}`\n",
        f"- `avoid`: `{sum(1 for d in decisions if d.action_bucket == 'avoid')}`\n",
        "\n",
    ]
    for bucket, title in (
        ("copy_ready", "Copy Ready"),
        ("monitor", "Monitor"),
        ("avoid", "Avoid"),
    ):
        summary_lines.append(f"## {title}\n")
        for row in decision_rows:
            if row["action_bucket"] != bucket:
                continue
            summary_lines.append(
                f"- `{row['display_name']}` `{row['wallet_id']}` | rationale `{row['action_rationale']}` | "
                f"PMA copy30 `{row['pma_copy_combined_net_30s']}` | local combined `{row['combined_realized_plus_mtm_pnl_usdc_est']}` | "
                f"fast-exit30 `{row['fast_exit_share_30s']}`\n"
            )
        summary_lines.append("\n")

    summary_lines.append("## Investor Readiness\n")
    summary_lines.append(f"- `investor_ready`: `{investor_ready}`\n")
    if investor_ready:
        summary_lines.append("- The current copy-ready subset passed the full-history and OOS split checks used in this report.\n")
    else:
        summary_lines.append("- Reasons it is **not** investor-ready yet:\n")
        for reason in investor_ready_reasons:
            summary_lines.append(f"  - `{reason}`\n")

    if subset_search_rows:
        best = subset_search_rows[0]
        summary_lines.extend(
            [
                "\n",
                "## Exploratory Best Subset Search\n",
                "- This is exploratory only and should not be treated as clean validation.\n",
                f"- Best subset by OOS count: `{best['wallet_ids']}`\n",
                f"- Positive OOS splits: `{best['positive_oos_splits']}`\n",
                f"- Full-history 30s PnL: `{best['full_history_combined_net_30s']}`\n",
                f"- Positive-contribution concentration: `{best['top_wallet_positive_contribution_share_30s']}`\n",
            ]
        )

    summary_path = output_root / "copy_ready_summary.md"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")

    return {
        "decisions": decision_rows,
        "all_path": all_path,
        "bucket_paths": bucket_paths,
        "portfolio_rows": portfolio_rows,
        "portfolio_path": portfolio_path,
        "subset_search_path": subset_search_path,
        "summary_path": summary_path,
        "investor_ready": investor_ready,
        "investor_ready_reasons": investor_ready_reasons,
    }
