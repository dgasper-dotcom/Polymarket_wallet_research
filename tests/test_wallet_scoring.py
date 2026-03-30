"""Tests for wallet scoring thresholds and confidence labels."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.base import Base
from db.models import WalletTradeEnriched
from research.wallet_scoring import (
    classify_score_confidence,
    recommend_mode,
    score_wallets,
)


def _session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, future=True)()


def test_recommend_mode_uses_stricter_thresholds() -> None:
    """Recommendation should require positive scores and a meaningful gap."""

    assert recommend_mode(copy_score=0.80, fade_score=0.10, n_trades=10) == "copy"
    assert recommend_mode(copy_score=0.20, fade_score=0.00, n_trades=10) == "ignore"
    assert recommend_mode(copy_score=-0.10, fade_score=0.70, n_trades=10) == "fade"
    assert recommend_mode(copy_score=0.90, fade_score=0.00, n_trades=3) == "ignore"


def test_classify_score_confidence_labels() -> None:
    """Confidence labels should respond to breadth, sample size, and concentration."""

    assert classify_score_confidence(n_trades=30, n_markets=6, fraction_top_market=0.30) == "high"
    assert classify_score_confidence(n_trades=12, n_markets=3, fraction_top_market=0.50) == "medium"
    assert classify_score_confidence(n_trades=6, n_markets=1, fraction_top_market=0.90) == "low"


def test_wallet_scoring_integration_exposes_confidence_labels() -> None:
    """Full scoring should produce recommended modes and confidence labels together."""

    session = _session()
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    rows: list[WalletTradeEnriched] = []
    for index in range(12):
        rows.append(
            WalletTradeEnriched(
                trade_id=f"copy-{index}",
                wallet_address="0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                market_id=f"m{index % 4}",
                token_id=f"t{index}",
                timestamp=start + timedelta(minutes=index),
                side="BUY",
                price=0.40,
                size=10,
                best_bid_at_trade=None,
                best_ask_at_trade=None,
                spread_at_trade=0.01,
                mid_at_trade=0.40,
                mid_1m=0.41,
                mid_5m=0.45,
                mid_30m=0.50,
                ret_1m=0.01,
                ret_5m=0.05,
                ret_30m=0.10,
                copy_pnl_1m=0.005,
                copy_pnl_5m=0.04,
                copy_pnl_30m=0.09,
                fade_pnl_1m=-0.02,
                fade_pnl_5m=-0.04,
                fade_pnl_30m=-0.09,
                slippage_bps_assumed=8,
                fees_bps_assumed=0,
                liquidity_bucket="medium",
                raw_json="{}",
            )
        )

    for index in range(30):
        rows.append(
            WalletTradeEnriched(
                trade_id=f"fade-{index}",
                wallet_address="0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                market_id=f"n{index % 6}",
                token_id=f"u{index}",
                timestamp=start + timedelta(minutes=100 + index),
                side="SELL",
                price=0.60,
                size=10,
                best_bid_at_trade=None,
                best_ask_at_trade=None,
                spread_at_trade=0.01,
                mid_at_trade=0.60,
                mid_1m=0.58,
                mid_5m=0.56,
                mid_30m=0.52,
                ret_1m=0.02,
                ret_5m=0.04,
                ret_30m=0.08,
                copy_pnl_1m=-0.01,
                copy_pnl_5m=-0.03,
                copy_pnl_30m=-0.06,
                fade_pnl_1m=0.005,
                fade_pnl_5m=0.03,
                fade_pnl_30m=0.07,
                slippage_bps_assumed=8,
                fees_bps_assumed=0,
                liquidity_bucket="medium",
                raw_json="{}",
            )
        )

    for index in range(4):
        rows.append(
            WalletTradeEnriched(
                trade_id=f"ignore-{index}",
                wallet_address="0xcccccccccccccccccccccccccccccccccccccccc",
                market_id="z",
                token_id=f"z{index}",
                timestamp=start + timedelta(minutes=200 + index),
                side="BUY",
                price=0.5,
                size=5,
                best_bid_at_trade=None,
                best_ask_at_trade=None,
                spread_at_trade=0.01,
                mid_at_trade=0.5,
                mid_1m=0.5,
                mid_5m=0.5,
                mid_30m=0.5,
                ret_1m=0.0,
                ret_5m=0.0,
                ret_30m=0.0,
                copy_pnl_1m=0.0,
                copy_pnl_5m=0.0,
                copy_pnl_30m=0.0,
                fade_pnl_1m=0.0,
                fade_pnl_5m=0.0,
                fade_pnl_30m=0.0,
                slippage_bps_assumed=12,
                fees_bps_assumed=0,
                liquidity_bucket="small",
                raw_json="{}",
            )
        )

    session.add_all(rows)
    session.commit()

    scores = score_wallets(session).set_index("wallet_address")
    assert scores.loc["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "recommended_mode"] == "copy"
    assert scores.loc["0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "recommended_mode"] == "fade"
    assert scores.loc["0xcccccccccccccccccccccccccccccccccccccccc", "recommended_mode"] == "ignore"
    assert scores.loc["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "score_confidence"] == "medium"
    assert scores.loc["0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "score_confidence"] == "high"
    assert scores.loc["0xcccccccccccccccccccccccccccccccccccccccc", "score_confidence"] == "low"
