"""Compare unified house-book backtests under total concurrent capital caps."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.paper_tracking_model import run_paper_tracking_model
from research.paper_tracking_performance import run_paper_tracking_performance


def _safe_sum(frame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(frame[column].fillna(0).sum())


def _read_curve(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_summary_md(path: Path, rows: list[dict[str, object]], chart_path: Path) -> Path:
    lines = [
        "# House Book Total-Cap Comparison",
        "",
        f"- Per-position cap: `{float(rows[0]['max_position_notional_usdc']):,.2f} USDC`" if rows else "",
        "",
        "| Total Open Cap | Realized PnL | Open MTM | Combined PnL | Entry Volume | Gross Turnover | Peak Concurrent Notional | Open Positions | Marked Open Positions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `${int(float(row['max_total_open_notional_usdc'])):,}` | "
            f"{float(row['realized_net_pnl_usdc']):,.2f} | "
            f"{float(row['open_mtm_net_pnl_usdc']):,.2f} | "
            f"{float(row['combined_net_pnl_usdc']):,.2f} | "
            f"{float(row['entry_volume_usdc']):,.2f} | "
            f"{float(row['gross_turnover_usdc']):,.2f} | "
            f"{float(row['peak_concurrent_notional_usdc']):,.2f} | "
            f"{int(row['open_positions'])} | "
            f"{int(row['marked_open_positions'])} |"
        )
    lines.extend(
        [
            "",
            f"- Chart: `{chart_path}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _plot_curves(curve_specs: list[tuple[str, list[dict[str, str]]]], chart_path: Path) -> Path:
    plt.style.use("default")
    figure, axis = plt.subplots(figsize=(16, 9))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#EAEAF2")

    all_labels: list[str] = []
    max_len = 0
    for label, rows in curve_specs:
        x = list(range(len(rows)))
        y = [float(row["combined_equity_net_usdc"]) for row in rows]
        axis.plot(x, y, linewidth=2.8, label=label)
        if len(rows) > max_len:
            max_len = len(rows)
            all_labels = [row["date"] for row in rows]

    axis.set_title("House Book PnL by Total Concurrent Capital Cap", fontsize=24, pad=12)
    axis.set_ylabel("Net PnL (USDC)", fontsize=20)
    axis.set_xlabel("Date", fontsize=20)
    axis.tick_params(axis="y", labelsize=16)
    axis.set_xticks(list(range(max_len)))
    axis.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=7)
    axis.legend(loc="upper left", fontsize=18, framealpha=0.85)
    axis.grid(axis="y", alpha=0.35, linewidth=1.0)
    axis.grid(axis="x", alpha=0.8, linewidth=0.8, color="white")
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    return chart_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="exports/house_book_total_cap_compare",
        help="Directory for compare outputs.",
    )
    parser.add_argument(
        "--wallet-csv",
        default="configs/forward_paper_v1_wallets.csv",
        help="Frozen wallet CSV.",
    )
    parser.add_argument(
        "--mapped-trades-csv",
        default="exports/manual_seed_pma_full_history_backtest_no_lucky/manual_seed_pma_mapped_trades_full_history.csv",
        help="Mapped PMA trades CSV.",
    )
    parser.add_argument(
        "--caps",
        default="10000,50000",
        help="Comma-separated total concurrent open-notional caps.",
    )
    parser.add_argument(
        "--max-position-notional-usdc",
        type=float,
        default=100.0,
        help="Per-position notional cap to enforce alongside the total concurrent cap.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    caps = [float(part.strip()) for part in args.caps.split(",") if part.strip()]

    summary_rows: list[dict[str, object]] = []
    curve_specs: list[tuple[str, list[dict[str, str]]]] = []

    for cap in caps:
        label = f"book_cap_{int(cap)}"
        run_root = output_root / label
        tracking = run_paper_tracking_model(
            wallet_csv=args.wallet_csv,
            mapped_trades_csv=args.mapped_trades_csv,
            output_dir=run_root,
            cluster_window_hours=24,
            action_bucket=None,
            max_position_notional_usdc=args.max_position_notional_usdc,
            max_total_open_notional_usdc=cap,
        )
        performance = run_paper_tracking_performance(
            consolidated_dir=run_root / "consolidated",
            output_dir=run_root / "performance",
            max_position_notional_usdc=args.max_position_notional_usdc,
            max_total_open_notional_usdc=cap,
        )

        curve_path = Path(performance["curve_path"])
        curve_specs.append((f"${int(cap):,} cap", _read_curve(curve_path)))
        summary_rows.append(
            {
                "max_total_open_notional_usdc": cap,
                "max_position_notional_usdc": args.max_position_notional_usdc,
                "realized_net_pnl_usdc": _safe_sum(performance["closed_positions"], "realized_pnl_net_usdc"),
                "open_mtm_net_pnl_usdc": _safe_sum(performance["open_positions"], "mtm_pnl_net_usdc"),
                "combined_net_pnl_usdc": _safe_sum(performance["closed_positions"], "realized_pnl_net_usdc")
                + _safe_sum(performance["open_positions"], "mtm_pnl_net_usdc"),
                "entry_volume_usdc": _safe_sum(performance["closed_positions"], "entry_notional_usdc")
                + _safe_sum(performance["open_positions"], "entry_notional_usdc"),
                "gross_turnover_usdc": _safe_sum(performance["closed_positions"], "entry_notional_usdc")
                + _safe_sum(performance["open_positions"], "entry_notional_usdc")
                + _safe_sum(performance["closed_positions"], "exit_cost_total_usdc"),
                "peak_concurrent_notional_usdc": max(
                    (
                        float(row["combined_equity_net_usdc"]) * 0
                        for row in []
                    ),
                    default=0.0,
                ),
                "open_positions": len(performance["open_positions"]),
                "marked_open_positions": int(performance["open_positions"].get("mtm_pnl_net_usdc").notna().sum()),
                "performance_summary_path": str(performance["summary_path"]),
                "tracking_summary_path": str(tracking["summary_path"]),
            }
        )

    for row, (_, rows) in zip(summary_rows, curve_specs):
        peak_notional = 0.0
        current = 0.0
        events: list[tuple[str, float]] = []
        # Use the written performance CSVs to infer capital usage from position notional.
        label = f"book_cap_{int(float(row['max_total_open_notional_usdc']))}"
        performance_dir = output_root / label / "performance"
        closed_csv = performance_dir / "house_closed_position_performance.csv"
        open_csv = performance_dir / "house_open_position_performance.csv"
        for path, close_key in ((closed_csv, "closed_at"), (open_csv, None)):
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for position in reader:
                    opened_at = position["opened_at"]
                    events.append((opened_at, float(position.get("entry_notional_usdc") or 0.0)))
                    close_ts = position.get(close_key) if close_key else position.get("analysis_cutoff")
                    if close_ts:
                        events.append((close_ts, -float(position.get("entry_notional_usdc") or 0.0)))
        for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
            current += delta
            peak_notional = max(peak_notional, current)
        row["peak_concurrent_notional_usdc"] = peak_notional

    summary_rows.sort(key=lambda item: float(item["max_total_open_notional_usdc"]))
    chart_path = _plot_curves(curve_specs, output_root / "house_book_total_cap_equity_curve.png")
    csv_path = _write_summary_csv(output_root / "house_book_total_cap_summary.csv", summary_rows)
    md_path = _write_summary_md(output_root / "house_book_total_cap_summary.md", summary_rows, chart_path)

    print(csv_path)
    print(md_path)
    print(chart_path)


if __name__ == "__main__":
    main()
