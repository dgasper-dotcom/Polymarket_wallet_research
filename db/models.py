"""ORM models for raw and derived research datasets."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, PrimaryKeyConstraint, Text
from sqlalchemy.orm import Mapped, mapped_column

from db.base import Base


class Wallet(Base):
    """Tracked public Polymarket wallet."""

    __tablename__ = "wallets"

    wallet_address: Mapped[str] = mapped_column(Text, primary_key=True)
    label: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class Market(Base):
    """Public market metadata sourced from the Gamma API."""

    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    slug: Mapped[str | None] = mapped_column(Text, nullable=True)
    condition_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    closed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    archived: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    enable_order_book: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_json: Mapped[str] = mapped_column(Text, nullable=False)


class Token(Base):
    """Outcome token metadata derived from market metadata."""

    __tablename__ = "tokens"

    token_id: Mapped[str] = mapped_column(Text, primary_key=True)
    market_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_json: Mapped[str] = mapped_column(Text, nullable=False)


class WalletTradeRaw(Base):
    """Raw public trade records by wallet."""

    __tablename__ = "wallet_trades_raw"
    __table_args__ = (
        Index("ix_wallet_trades_raw_wallet_ts", "wallet_address", "timestamp"),
        Index("ix_wallet_trades_raw_token_ts", "token_id", "timestamp"),
    )

    trade_id: Mapped[str] = mapped_column(Text, primary_key=True)
    wallet_address: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    market_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    token_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    side: Mapped[str | None] = mapped_column(Text, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    size: Mapped[float | None] = mapped_column(Float, nullable=True)
    usdc_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    tx_hash: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    raw_json: Mapped[str] = mapped_column(Text, nullable=False)


class MarketTradeRaw(Base):
    """Raw public trade records discovered by scanning markets rather than wallets."""

    __tablename__ = "market_trades_raw"
    __table_args__ = (
        Index("ix_market_trades_raw_wallet_ts", "wallet_address", "timestamp"),
        Index("ix_market_trades_raw_market_ts", "market_id", "timestamp"),
    )

    trade_id: Mapped[str] = mapped_column(Text, primary_key=True)
    wallet_address: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    market_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    token_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    side: Mapped[str | None] = mapped_column(Text, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    size: Mapped[float | None] = mapped_column(Float, nullable=True)
    usdc_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    tx_hash: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    raw_json: Mapped[str] = mapped_column(Text, nullable=False)


class MarketTradeScanProgress(Base):
    """Checkpoint state for resumable market-trade discovery scans."""

    __tablename__ = "market_trade_scan_progress"

    condition_id: Mapped[str] = mapped_column(Text, primary_key=True)
    gamma_market_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    next_offset: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    truncated_by_api_limit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    pages_scanned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trades_scanned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PriceHistory(Base):
    """Historical token price series used for event studies."""

    __tablename__ = "price_history"
    __table_args__ = (
        PrimaryKeyConstraint("token_id", "ts", name="pk_price_history"),
        Index("ix_price_history_token_ts", "token_id", "ts"),
    )

    token_id: Mapped[str] = mapped_column(Text, nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)


class WalletTradeEnriched(Base):
    """Derived trade dataset with returns, costs, and strategy PnL estimates."""

    __tablename__ = "wallet_trades_enriched"
    __table_args__ = (
        Index("ix_wallet_trades_enriched_wallet_ts", "wallet_address", "timestamp"),
        Index("ix_wallet_trades_enriched_token_ts", "token_id", "timestamp"),
    )

    trade_id: Mapped[str] = mapped_column(Text, primary_key=True)
    wallet_address: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    market_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    token_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    size: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_bid_at_trade: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_ask_at_trade: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_at_trade: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid_at_trade: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid_30m: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret_30m: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_30m: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_30m: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_5m_delay_5s: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_5m_delay_15s: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_5m_delay_30s: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_5m_delay_60s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_5m_delay_5s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_5m_delay_15s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_5m_delay_30s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_5m_delay_60s: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_net_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_net_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_net_5m_delay_5s: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_net_5m_delay_15s: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_net_5m_delay_30s: Mapped[float | None] = mapped_column(Float, nullable=True)
    copy_pnl_net_5m_delay_60s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_net_5m_delay_5s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_net_5m_delay_15s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_net_5m_delay_30s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fade_pnl_net_5m_delay_60s: Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage_bps_assumed: Mapped[float | None] = mapped_column(Float, nullable=True)
    fees_bps_assumed: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity_bucket: Mapped[str | None] = mapped_column(Text, nullable=True)
    missing_price_history: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    missing_market_metadata: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    used_fallback_midpoint: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    trade_price_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    midpoint_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    book_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    enrichment_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    missing_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class TradeFeature(Base):
    """Derived trade-level feature table for behavior research."""

    __tablename__ = "trade_features"
    __table_args__ = (
        Index("ix_trade_features_wallet_ts", "wallet_address", "timestamp"),
        Index("ix_trade_features_market_ts", "market_id", "timestamp"),
        Index("ix_trade_features_cluster", "trade_type_cluster"),
    )

    trade_id: Mapped[str] = mapped_column(Text, primary_key=True)
    wallet_address: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    market_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    token_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    trade_notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    standardized_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    size_to_liquidity_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_distance_from_mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    recent_momentum_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    recent_momentum_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    short_term_volatility_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_to_resolution_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    trade_cluster_density_5m: Mapped[int | None] = mapped_column(Integer, nullable=True)
    size_bucket: Mapped[str | None] = mapped_column(Text, nullable=True)
    price_zone: Mapped[str | None] = mapped_column(Text, nullable=True)
    pre_trade_trend_state: Mapped[str | None] = mapped_column(Text, nullable=True)
    market_phase: Mapped[str | None] = mapped_column(Text, nullable=True)
    trend_alignment: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    trade_type_cluster: Mapped[str | None] = mapped_column(Text, nullable=True)
