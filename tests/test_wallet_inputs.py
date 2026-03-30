"""Tests for wallet file parsing and dry-run backfill behavior."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx

from clients.profile_client import ProfileClient
from ingestion.wallets import parse_wallet_lines
from scripts.run_backfill import run_backfill_workflow


def test_parse_wallet_lines_supports_comments_blank_lines_and_deduping() -> None:
    """Wallet parsing should ignore comments and blank lines and dedupe valid addresses."""

    result = parse_wallet_lines(
        [
            "# comment",
            "",
            "0x56687bf447db6ffa42ffe2204a05edaa20f55839",
            "0x56687bf447db6ffa42ffe2204a05edaa20f55839",
        ],
        source_label="wallets.txt",
    )
    assert result.wallets == ["0x56687bf447db6ffa42ffe2204a05edaa20f55839"]
    assert result.valid_count == 1
    assert result.ignored_count == 2
    assert result.invalid_entries == []


def test_run_backfill_dry_run_does_not_initialize_database(tmp_path: Path) -> None:
    """Dry-run mode should fetch preview data but never initialize or write a database."""

    wallet_file = tmp_path / "wallets.txt"
    wallet_file.write_text(
        "# sample\n\n0x56687bf447db6ffa42ffe2204a05edaa20f55839\n",
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        if offset == 0:
            return httpx.Response(
                200,
                json=[
                    {
                        "proxyWallet": "0x56687bf447db6ffa42ffe2204a05edaa20f55839",
                        "side": "BUY",
                        "asset": "token-1",
                        "conditionId": "0xabc",
                        "size": 10,
                        "price": 0.5,
                        "timestamp": 1731489409,
                        "transactionHash": "0x1",
                    }
                ],
            )
        return httpx.Response(200, json=[])

    init_called = {"value": False}
    session_called = {"value": False}

    def fail_init_db() -> None:
        init_called["value"] = True

    def fail_session_factory():
        session_called["value"] = True
        raise AssertionError("dry-run should not request a database session")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        args = argparse.Namespace(wallets=[str(wallet_file)], price_fidelity=1, dry_run=True)
        async with ProfileClient(base_url="https://data-api.polymarket.com", transport=transport) as client:
            preview = await run_backfill_workflow(
                args,
                profile_client=client,
                init_db_fn=fail_init_db,
                session_factory=fail_session_factory,
                exports_dir=tmp_path / "exports",
            )
        assert preview["wallets"][0]["n_trades"] == 1

    asyncio.run(run())

    assert init_called["value"] is False
    assert session_called["value"] is False
    assert (tmp_path / "exports" / "dry_run_wallets.csv").exists()
    assert (tmp_path / "exports" / "dry_run_market_targets.csv").exists()
    assert (tmp_path / "exports" / "dry_run_token_targets.csv").exists()
