"""Run a small portfolio-construction search for the unified house book."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.house_book_oos import run_house_book_oos
from research.paper_tracking_model import run_paper_tracking_model
from research.paper_tracking_performance import run_paper_tracking_performance
from research.house_portfolio_rules import positive_contribution_shares


def _parse_caps(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item in {"none", "null", "na"}:
            values.append(None)
        else:
            values.append(float(item))
    return values


def _label(value: float | None) -> str:
    return "none" if value is None else str(int(value) if float(value).is_integer() else value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="exports/manual_seed_portfolio_search",
        help="Directory for search outputs.",
    )
    parser.add_argument(
        "--wallet-csv",
        default="configs/forward_paper_v1_wallets.csv",
        help="Wallet CSV to freeze during the search.",
    )
    parser.add_argument(
        "--mapped-trades-csv",
        default="exports/manual_seed_pma_full_history_backtest_no_lucky/manual_seed_pma_mapped_trades_full_history.csv",
        help="Mapped PMA trade CSV.",
    )
    parser.add_argument(
        "--position-caps",
        default="100",
        help="Comma-separated position caps. Use 'none' for uncapped.",
    )
    parser.add_argument(
        "--event-caps",
        default="none,200,300,400",
        help="Comma-separated event caps. Use 'none' for uncapped.",
    )
    parser.add_argument(
        "--wallet-caps",
        default="none,5000",
        help="Comma-separated wallet open-notional caps. Use 'none' for uncapped.",
    )
    parser.add_argument(
        "--date-cutoff",
        default="2026-01-01T00:00:00+00:00",
        help="Chronological OOS cutoff timestamp.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    position_caps = _parse_caps(args.position_caps)
    event_caps = _parse_caps(args.event_caps)
    wallet_caps = _parse_caps(args.wallet_caps)

    summary_rows: list[dict[str, object]] = []
    for position_cap in position_caps:
        for event_cap in event_caps:
            for wallet_cap in wallet_caps:
                label = f"pos_{_label(position_cap)}__event_{_label(event_cap)}__wallet_{_label(wallet_cap)}"
                run_root = output_root / label
                tracking = run_paper_tracking_model(
                    wallet_csv=args.wallet_csv,
                    mapped_trades_csv=args.mapped_trades_csv,
                    output_dir=run_root,
                    cluster_window_hours=24,
                    action_bucket=None,
                    max_position_notional_usdc=position_cap,
                    max_event_notional_usdc=event_cap,
                    max_wallet_open_notional_usdc=wallet_cap,
                )
                performance = run_paper_tracking_performance(
                    consolidated_dir=run_root / "consolidated",
                    output_dir=run_root / "performance",
                    max_position_notional_usdc=position_cap,
                    max_event_notional_usdc=event_cap,
                    max_wallet_open_notional_usdc=wallet_cap,
                )
                oos = run_house_book_oos(
                    closed_csv=run_root / "performance" / "house_closed_position_performance.csv",
                    open_csv=run_root / "performance" / "house_open_position_performance.csv",
                    output_dir=run_root / "performance" / "oos",
                    ratios=(),
                    date_cutoff=args.date_cutoff,
                )
                test_row = next(row for row in oos["summary_rows"] if row["split"].startswith("test_from_"))
                wallet_metrics = positive_contribution_shares(
                    performance["wallet_contribution"].to_dict(orient="records")
                )
                event_metrics = positive_contribution_shares(
                    performance["event_contribution"].to_dict(orient="records")
                )
                oos_return_on_peak = (
                    float(test_row["combined_net_pnl_usdc"]) / float(test_row["peak_concurrent_notional_usdc"])
                    if float(test_row["peak_concurrent_notional_usdc"] or 0.0) > 0
                    else 0.0
                )
                summary_rows.append(
                    {
                        "label": label,
                        "position_cap_usdc": position_cap,
                        "event_cap_usdc": event_cap,
                        "wallet_cap_usdc": wallet_cap,
                        "full_combined_net_pnl_usdc": float(
                            performance["closed_positions"]["realized_pnl_net_usdc"].fillna(0).sum()
                        )
                        + float(performance["open_positions"]["mtm_pnl_net_usdc"].fillna(0).sum()),
                        "full_peak_positive_wallet_share": wallet_metrics["top1_positive_share"],
                        "full_peak_positive_event_share": event_metrics["top1_positive_share"],
                        "oos_combined_net_pnl_usdc": float(test_row["combined_net_pnl_usdc"]),
                        "oos_peak_concurrent_notional_usdc": float(test_row["peak_concurrent_notional_usdc"]),
                        "oos_return_on_peak_notional": oos_return_on_peak,
                        "oos_wallet_top1_positive_share": wallet_metrics["top1_positive_share"],
                        "oos_event_top1_positive_share": event_metrics["top1_positive_share"],
                        "oos_wallet_top5_positive_share": wallet_metrics["top5_positive_share"],
                        "oos_event_top5_positive_share": event_metrics["top5_positive_share"],
                    }
                )

    summary_rows.sort(key=lambda row: row["oos_return_on_peak_notional"], reverse=True)
    csv_path = output_root / "portfolio_search_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# Portfolio Construction Search",
        "",
        f"- Wallet file: `{args.wallet_csv}`",
        f"- Date cutoff: `{args.date_cutoff}`",
        "",
        "| Label | Pos Cap | Event Cap | Wallet Cap | OOS PnL | OOS Peak Notional | OOS Return on Peak | Wallet Top1 Share | Event Top1 Share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['label']}` | "
            f"{row['position_cap_usdc'] if row['position_cap_usdc'] is not None else 'none'} | "
            f"{row['event_cap_usdc'] if row['event_cap_usdc'] is not None else 'none'} | "
            f"{row['wallet_cap_usdc'] if row['wallet_cap_usdc'] is not None else 'none'} | "
            f"{float(row['oos_combined_net_pnl_usdc']):,.2f} | "
            f"{float(row['oos_peak_concurrent_notional_usdc']):,.2f} | "
            f"{100*float(row['oos_return_on_peak_notional']):.2f}% | "
            f"{100*float(row['oos_wallet_top1_positive_share']):.1f}% | "
            f"{100*float(row['oos_event_top1_positive_share']):.1f}% |"
        )
    md_path = output_root / "portfolio_search_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
