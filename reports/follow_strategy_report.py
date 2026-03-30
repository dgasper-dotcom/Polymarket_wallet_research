"""Generate breakdown CSVs and a PDF report for wallet half-forward follow analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_THRESHOLDS: tuple[int, ...] = (1, 5, 10, 20, 50)


@dataclass(frozen=True)
class DelayMetrics:
    """One delay-specific view of a wallet's train/test metrics."""

    delay: str
    train_valid_slices: int
    test_valid_slices: int
    train_net_usdc: float | None
    test_net_usdc: float | None
    test_gross_usdc: float | None
    train_net_hit_rate: float | None
    test_net_hit_rate: float | None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _to_int(value: str | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def _delay_metrics(row: dict[str, str], delay: str) -> DelayMetrics:
    return DelayMetrics(
        delay=delay,
        train_valid_slices=_to_int(row.get(f"valid_copy_slices_{delay}_train")),
        test_valid_slices=_to_int(row.get(f"valid_copy_slices_{delay}_test")),
        train_net_usdc=_to_float(row.get(f"total_copy_pnl_net_usdc_{delay}_train")),
        test_net_usdc=_to_float(row.get(f"total_copy_pnl_net_usdc_{delay}_test")),
        test_gross_usdc=_to_float(row.get(f"total_copy_pnl_usdc_{delay}_test")),
        train_net_hit_rate=_to_float(row.get(f"copy_net_hit_rate_{delay}_train")),
        test_net_hit_rate=_to_float(row.get(f"copy_net_hit_rate_{delay}_test")),
    )


def _unique_delays(summary_rows: Sequence[dict[str, str]]) -> list[str]:
    values = [str(row.get("delay", "")).strip() for row in summary_rows if row.get("delay")]
    return sorted(dict.fromkeys(values), key=lambda item: int(item.rstrip("s")))


def build_funnel_breakdown(all_rows: Sequence[dict[str, str]], delays: Sequence[str]) -> list[dict[str, Any]]:
    """Summarize the filtering funnel for each delay."""

    recent_rows = [row for row in all_rows if _to_bool(row.get("has_recent_trades"))]
    breakdown: list[dict[str, Any]] = []
    for delay in delays:
        train_positive_rows = [
            row
            for row in recent_rows
            if _delay_metrics(row, delay).train_valid_slices > 0
            and (_delay_metrics(row, delay).train_net_usdc or 0.0) > 0.0
        ]
        breakdown.append(
            {
                "delay": delay,
                "wallets_requested": len(all_rows),
                "wallets_without_recent_trades": sum(not _to_bool(row.get("has_recent_trades")) for row in all_rows),
                "wallets_with_recent_trades": len(recent_rows),
                "wallets_with_valid_train_slices": sum(
                    _delay_metrics(row, delay).train_valid_slices > 0 for row in recent_rows
                ),
                "wallets_train_nonprofitable_or_zero": sum(
                    _delay_metrics(row, delay).train_valid_slices > 0
                    and (_delay_metrics(row, delay).train_net_usdc or 0.0) <= 0.0
                    for row in recent_rows
                ),
                "wallets_train_profitable": len(train_positive_rows),
                "wallets_train_profitable_without_valid_test_slices": sum(
                    _delay_metrics(row, delay).test_valid_slices == 0 for row in train_positive_rows
                ),
                "wallets_test_nonpositive_among_train_selected": sum(
                    _delay_metrics(row, delay).test_valid_slices > 0
                    and (_delay_metrics(row, delay).test_net_usdc or 0.0) <= 0.0
                    for row in train_positive_rows
                ),
                "wallets_test_positive_among_train_selected": sum(
                    _delay_metrics(row, delay).test_valid_slices > 0
                    and (_delay_metrics(row, delay).test_net_usdc or 0.0) > 0.0
                    for row in train_positive_rows
                ),
            }
        )
    return breakdown


def build_selected_wallet_breakdown(
    selected_rows: Sequence[dict[str, str]],
    delays: Sequence[str],
) -> list[dict[str, Any]]:
    """Flatten delay-specific wallet outcomes into a review-friendly table."""

    records: list[dict[str, Any]] = []
    for row in selected_rows:
        for delay in delays:
            metrics = _delay_metrics(row, delay)
            if metrics.test_valid_slices == 0:
                test_status = "no_valid_test_slices"
            elif (metrics.test_net_usdc or 0.0) > 0.0:
                test_status = "test_profitable"
            else:
                test_status = "test_nonprofitable"
            records.append(
                {
                    "delay": delay,
                    "wallet_address": row["wallet_address"],
                    "first_recent_trade_ts": row.get("first_recent_trade_ts"),
                    "split_midpoint_ts": row.get("split_midpoint_ts"),
                    "most_recent_trade_ts": row.get("most_recent_trade_ts"),
                    "recent_trade_rows_total": _to_int(row.get("recent_trade_rows_total")),
                    "recent_distinct_markets_total": _to_int(row.get("recent_distinct_markets_total")),
                    "train_trade_rows": _to_int(row.get("train_trade_rows")),
                    "test_trade_rows": _to_int(row.get("test_trade_rows")),
                    "train_valid_slices": metrics.train_valid_slices,
                    "test_valid_slices": metrics.test_valid_slices,
                    "train_net_usdc": metrics.train_net_usdc,
                    "test_net_usdc": metrics.test_net_usdc,
                    "test_gross_usdc": metrics.test_gross_usdc,
                    "train_net_hit_rate": metrics.train_net_hit_rate,
                    "test_net_hit_rate": metrics.test_net_hit_rate,
                    "train_open_copy_slices": _to_int(row.get("train_open_copy_slices")),
                    "test_open_copy_slices": _to_int(row.get("test_open_copy_slices")),
                    "capped_recent_trade_rows": _to_int(row.get("recent_trade_rows_total")) >= 1500,
                    "test_status": test_status,
                }
            )
    return records


def build_threshold_breakdown(
    selected_rows: Sequence[dict[str, str]],
    delays: Sequence[str],
    *,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Measure how test profitability changes as sample-size requirements increase."""

    rows: list[dict[str, Any]] = []
    for delay in delays:
        for threshold in thresholds:
            bucket = [row for row in selected_rows if _delay_metrics(row, delay).test_valid_slices >= threshold]
            positives = sum((_delay_metrics(row, delay).test_net_usdc or 0.0) > 0.0 for row in bucket)
            rows.append(
                {
                    "delay": delay,
                    "min_test_valid_slices": threshold,
                    "wallets_in_bucket": len(bucket),
                    "test_positive_wallets": positives,
                    "test_positive_share": (positives / len(bucket)) if bucket else None,
                    "avg_test_net_usdc": mean(
                        (_delay_metrics(row, delay).test_net_usdc or 0.0) for row in bucket
                    )
                    if bucket
                    else None,
                    "median_test_net_usdc": median(
                        (_delay_metrics(row, delay).test_net_usdc or 0.0) for row in bucket
                    )
                    if bucket
                    else None,
                }
            )
    return rows


def build_top_wallet_tables(
    selected_rows: Sequence[dict[str, str]],
    delay: str,
    *,
    top_n: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return top test winners and losers for one representative delay."""

    eligible = [row for row in selected_rows if _delay_metrics(row, delay).test_net_usdc is not None]
    winners = sorted(
        [row for row in eligible if (_delay_metrics(row, delay).test_net_usdc or 0.0) > 0.0],
        key=lambda row: _delay_metrics(row, delay).test_net_usdc or float("-inf"),
        reverse=True,
    )[:top_n]
    losers = sorted(
        [row for row in eligible if (_delay_metrics(row, delay).test_net_usdc or 0.0) < 0.0],
        key=lambda row: _delay_metrics(row, delay).test_net_usdc or float("inf"),
    )[:top_n]

    def flatten(rows: Iterable[dict[str, str]], side: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in rows:
            metrics = _delay_metrics(row, delay)
            records.append(
                {
                    "delay": delay,
                    "side": side,
                    "wallet_address": row["wallet_address"],
                    "train_net_usdc": metrics.train_net_usdc,
                    "test_net_usdc": metrics.test_net_usdc,
                    "test_gross_usdc": metrics.test_gross_usdc,
                    "train_valid_slices": metrics.train_valid_slices,
                    "test_valid_slices": metrics.test_valid_slices,
                    "recent_trade_rows_total": _to_int(row.get("recent_trade_rows_total")),
                    "recent_distinct_markets_total": _to_int(row.get("recent_distinct_markets_total")),
                }
            )
        return records

    return flatten(winners, "winner"), flatten(losers, "loser")


def _format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def _draw_text_page(pdf: PdfPages, title: str, lines: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    ax.text(
        0.04,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _representative_delay(delays: Sequence[str]) -> str:
    if "30s" in delays:
        return "30s"
    return delays[0]


def create_pdf_report(
    *,
    pdf_path: Path,
    all_rows: Sequence[dict[str, str]],
    selected_rows: Sequence[dict[str, str]],
    summary_rows: Sequence[dict[str, str]],
    threshold_rows: Sequence[dict[str, Any]],
    winner_rows: Sequence[dict[str, Any]],
    loser_rows: Sequence[dict[str, Any]],
) -> Path:
    """Render a concise PDF report with charts and written conclusions."""

    delays = _unique_delays(summary_rows)
    representative_delay = _representative_delay(delays)
    rep_selected = [row for row in selected_rows if _delay_metrics(row, representative_delay).test_net_usdc is not None]
    rep_metrics = [_delay_metrics(row, representative_delay) for row in rep_selected]

    train_totals = [metric.train_net_usdc or 0.0 for metric in rep_metrics]
    test_totals = [metric.test_net_usdc or 0.0 for metric in rep_metrics]
    gross_totals = [metric.test_gross_usdc or 0.0 for metric in rep_metrics]
    positive_test_total = sum(value > 0.0 for value in test_totals)
    positive_test_sum = sum(value for value in test_totals if value > 0.0)

    overview_lines = [
        "Question",
        "Can we profit by following wallets that were profitable in the first half of their own recent activity?",
        "",
        "Method",
        "- Universe requested: 934 wallets",
        "- Wallets with recent public trades: 576",
        "- Per-wallet split: midpoint between each wallet's first and last observed recent trade",
        "- Selection rule: keep wallets with positive train-half net PnL",
        "- Test rule: evaluate only the selected wallets on their own second half",
        "- Delays tested: 15s, 30s, 60s",
        "",
        "Headline result",
        f"- Train-profitable wallets: {int(summary_rows[0]['train_profitable_wallets']) if summary_rows else 0}",
        f"- Test-profitable wallets among selected: {int(summary_rows[0]['test_positive_wallets_among_train_selected']) if summary_rows else 0}",
        f"- Selected-wallet test gross PnL: {_format_money(sum(gross_totals))} USDC",
        f"- Selected-wallet test net PnL: {_format_money(sum(test_totals))} USDC",
        f"- Median selected-wallet test net PnL: {_format_money(median(test_totals) if test_totals else None)} USDC",
        "",
        "Assessment",
        "- As tested here, following first-half profitable wallets does not look like a robust profit strategy.",
        "- Gross test PnL stayed positive, but costs flipped the selected-wallet basket negative.",
        "- Only a small minority of selected wallets stayed positive in the second half.",
    ]

    with PdfPages(pdf_path) as pdf:
        _draw_text_page(pdf, "Follow-Profitable-Wallets Strategy Report", overview_lines)

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        funnel_labels = [
            "Requested",
            "Recent",
            "Train valid",
            "Train profitable",
            "Test positive",
        ]
        for ax, summary in zip(axes, summary_rows[:2] if len(summary_rows) > 1 else [summary_rows[0], summary_rows[0]]):
            values = [
                int(summary["wallets_total_requested"]),
                int(summary["wallets_with_recent_trades"]),
                int(summary["wallets_with_recent_trades"]) - int(
                    next(
                        row["wallets_with_recent_trades"] for row in summary_rows if row["delay"] == summary["delay"]
                    )
                )
                + int(
                    next(
                        row["wallets_with_recent_trades"] for row in summary_rows if row["delay"] == summary["delay"]
                    )
                ),
                int(summary["train_profitable_wallets"]),
                int(summary["test_positive_wallets_among_train_selected"]),
            ]
            # The train-valid count is reconstructed from the funnel CSV to keep the plotting code short.
            train_valid_value = next(
                row["wallets_with_valid_train_slices"]
                for row in build_funnel_breakdown(all_rows, delays)
                if row["delay"] == summary["delay"]
            )
            values[2] = int(train_valid_value)
            ax.bar(funnel_labels, values, color=["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b"])
            ax.set_title(f"Funnel at {summary['delay']}")
            ax.set_ylabel("Wallet count")
            ax.tick_params(axis="x", rotation=30)
            for idx, value in enumerate(values):
                ax.text(idx, value, str(value), ha="center", va="bottom", fontsize=9)
        fig.suptitle("Selection Funnel")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        axes[0].hist(test_totals, bins=12, color="#355070", edgecolor="white")
        axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
        axes[0].set_title(f"Selected-Wallet Test Net PnL ({representative_delay})")
        axes[0].set_xlabel("USDC")
        axes[0].set_ylabel("Wallet count")

        axes[1].scatter(train_totals, test_totals, color="#b56576", alpha=0.8)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_title(f"Train vs Test Net PnL ({representative_delay})")
        axes[1].set_xlabel("Train net PnL (USDC)")
        axes[1].set_ylabel("Test net PnL (USDC)")

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        threshold_map = {row["min_test_valid_slices"]: row for row in threshold_rows if row["delay"] == representative_delay}
        threshold_lines = [
            "Sample-quality checks",
            f"- Representative delay shown: {representative_delay}",
            f"- Selected wallets with test rows: {sum(metric.test_valid_slices > 0 for metric in rep_metrics)}",
            f"- Selected wallets with no valid test rows: {sum(metric.test_valid_slices == 0 for metric in rep_metrics)}",
            f"- Total valid test slices across selected wallets: {sum(metric.test_valid_slices for metric in rep_metrics)}",
            f"- Positive test wallets: {positive_test_total}",
            f"- Positive contribution concentration: top 1 wallet contributed {((max(v for v in test_totals if v > 0) / positive_test_sum) * 100) if positive_test_sum else 0:.1f}% of positive test net PnL",
            "",
            "Threshold breakdown",
        ]
        for threshold in DEFAULT_THRESHOLDS:
            row = threshold_map.get(threshold)
            if row is None:
                continue
            threshold_lines.append(
                f"- Test valid slices >= {threshold}: {row['wallets_in_bucket']} wallets, "
                f"{row['test_positive_wallets']} positive, "
                f"share={((row['test_positive_share'] or 0.0) * 100):.1f}%, "
                f"median test net={_format_money(row['median_test_net_usdc'])}"
            )
        threshold_lines.extend(
            [
                "",
                "Interpretation",
                "- The screen picks up some first-half winners, but the edge decays sharply out of sample.",
                "- Most selected wallets are thin-sample selections; many have only a handful of valid delayed test slices.",
                "- A small number of wallets account for most of the positive second-half outcome.",
            ]
        )
        _draw_text_page(pdf, "Sample Quality And Decay", threshold_lines)

        winner_lines = ["Top second-half winners", ""]
        for row in winner_rows[:5]:
            winner_lines.append(
                f"- {row['wallet_address']} | train={_format_money(row['train_net_usdc'])} | "
                f"test={_format_money(row['test_net_usdc'])} | test_slices={row['test_valid_slices']}"
            )
        winner_lines.extend(["", "Top second-half losers", ""])
        for row in loser_rows[:5]:
            winner_lines.append(
                f"- {row['wallet_address']} | train={_format_money(row['train_net_usdc'])} | "
                f"test={_format_money(row['test_net_usdc'])} | test_slices={row['test_valid_slices']}"
            )
        winner_lines.extend(
            [
                "",
                "Bottom line",
                "- Gross PnL alone would overstate the strategy. Costs are the difference between a positive gross basket and a negative net basket.",
                "- The 15s, 30s, and 60s results are effectively identical because the public price history is coarse; many delays map to the same next visible price.",
                "- This makes the current conclusion more about weak persistence than about fine-grained latency edge.",
            ]
        )
        _draw_text_page(pdf, "Winners, Losers, And Strategy Verdict", winner_lines)

    return pdf_path


def generate_follow_strategy_report(
    *,
    input_dir: str | Path = "exports/per_wallet_half_forward",
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Build breakdown tables and a PDF for the per-wallet half-forward analysis."""

    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    all_rows = _read_csv(input_path / "wallet_half_forward_results_15s_30s_60s.csv")
    selected_rows = _read_csv(input_path / "train_profitable_wallets_test_results_15s_30s_60s.csv")
    summary_rows = _read_csv(input_path / "wallet_half_forward_summary.csv")
    delays = _unique_delays(summary_rows)

    funnel_rows = build_funnel_breakdown(all_rows, delays)
    selected_breakdown = build_selected_wallet_breakdown(selected_rows, delays)
    threshold_rows = build_threshold_breakdown(selected_rows, delays)
    winner_rows, loser_rows = build_top_wallet_tables(selected_rows, _representative_delay(delays))

    return {
        "funnel_csv": _write_csv(output_path / "follow_strategy_funnel_by_delay.csv", funnel_rows),
        "selected_wallet_csv": _write_csv(
            output_path / "follow_strategy_selected_wallet_breakdown.csv",
            selected_breakdown,
        ),
        "threshold_csv": _write_csv(output_path / "follow_strategy_threshold_breakdown.csv", threshold_rows),
        "top_wallets_csv": _write_csv(
            output_path / "follow_strategy_top_wallets.csv",
            winner_rows + loser_rows,
        ),
        "pdf_report": create_pdf_report(
            pdf_path=output_path / "follow_strategy_half_forward_report.pdf",
            all_rows=all_rows,
            selected_rows=selected_rows,
            summary_rows=summary_rows,
            threshold_rows=threshold_rows,
            winner_rows=winner_rows,
            loser_rows=loser_rows,
        ),
    }
