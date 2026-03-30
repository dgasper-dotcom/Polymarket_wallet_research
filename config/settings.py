"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)


class Settings(BaseModel):
    """Runtime settings for the research toolkit."""

    model_config = ConfigDict(frozen=True)

    poly_gamma_base: str = Field(default="https://gamma-api.polymarket.com")
    poly_data_base: str = Field(default="https://data-api.polymarket.com")
    poly_clob_base: str = Field(default="https://clob.polymarket.com")
    poly_ws_market: str = Field(
        default="wss://ws-subscriptions-clob.polymarket.com/ws/market"
    )
    db_url: str = Field(default="sqlite:///./polymarket_wallets.db")
    request_timeout: float = Field(default=20.0, ge=1.0)
    max_concurrency: int = Field(default=8, ge=1)
    log_level: str = Field(default="INFO")
    oos_train_fraction: float = Field(default=0.70, gt=0.0, lt=1.0)
    oos_select_top_n: int = Field(default=10, ge=1)
    oos_train_min_trades: int = Field(default=10, ge=1)
    oos_train_min_markets: int = Field(default=3, ge=1)
    oos_train_max_top_market_fraction: float = Field(default=0.70, ge=0.0, le=1.0)
    oos_recent_activity_days: int = Field(default=90, ge=1)
    oos_test_min_trades: int = Field(default=5, ge=1)
    oos_test_min_markets: int = Field(default=2, ge=1)
    multi_oos_n_splits: int = Field(default=10, ge=1)
    multi_oos_random_seed: int = Field(default=42)
    multi_oos_min_observed_splits: int = Field(default=3, ge=1)
    multi_oos_min_selected_splits: int = Field(default=2, ge=1)
    multi_oos_robust_selection_frequency: float = Field(default=0.50, ge=0.0, le=1.0)
    multi_oos_robust_positive_test_frequency: float = Field(default=0.60, ge=0.0, le=1.0)
    multi_oos_mode_consistency_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    multi_oos_inconsistent_mode_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    polymarket_fee_k: float = Field(default=0.0625, ge=0.0)
    flat_fee_bps: float | None = Field(default=None, ge=0.0)
    cost_scenario: str = Field(default="base")
    extra_cost_penalty: float = Field(default=0.010, ge=0.0)

    @field_validator("log_level")
    @classmethod
    def uppercase_log_level(cls, value: str) -> str:
        """Normalize log level values for the logging module."""

        return value.upper()

    @field_validator("cost_scenario")
    @classmethod
    def normalize_cost_scenario(cls, value: str) -> str:
        """Normalize the configured cost scenario name."""

        normalized = value.strip().lower()
        if normalized not in {"optimistic", "base", "conservative"}:
            raise ValueError("cost_scenario must be optimistic, base, or conservative")
        return normalized

    @property
    def sqlite_path(self) -> Path | None:
        """Return the local sqlite file path when using sqlite."""

        prefix = "sqlite:///"
        if not self.db_url.startswith(prefix):
            return None
        raw_path = self.db_url.removeprefix(prefix)
        return (PROJECT_ROOT / raw_path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and reuse them across modules."""

    flat_fee_raw = os.getenv("FLAT_FEE_BPS")
    return Settings(
        poly_gamma_base=os.getenv("POLY_GAMMA_BASE", "https://gamma-api.polymarket.com"),
        poly_data_base=os.getenv("POLY_DATA_BASE", "https://data-api.polymarket.com"),
        poly_clob_base=os.getenv("POLY_CLOB_BASE", "https://clob.polymarket.com"),
        poly_ws_market=os.getenv(
            "POLY_WS_MARKET", "wss://ws-subscriptions-clob.polymarket.com/ws/market"
        ),
        db_url=os.getenv("DB_URL", "sqlite:///./polymarket_wallets.db"),
        request_timeout=float(os.getenv("REQUEST_TIMEOUT", "20")),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", "8")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        oos_train_fraction=float(os.getenv("OOS_TRAIN_FRACTION", "0.70")),
        oos_select_top_n=int(os.getenv("OOS_SELECT_TOP_N", "10")),
        oos_train_min_trades=int(os.getenv("OOS_TRAIN_MIN_TRADES", "10")),
        oos_train_min_markets=int(os.getenv("OOS_TRAIN_MIN_MARKETS", "3")),
        oos_train_max_top_market_fraction=float(
            os.getenv("OOS_TRAIN_MAX_TOP_MARKET_FRACTION", "0.70")
        ),
        oos_recent_activity_days=int(os.getenv("OOS_RECENT_ACTIVITY_DAYS", "90")),
        oos_test_min_trades=int(os.getenv("OOS_TEST_MIN_TRADES", "5")),
        oos_test_min_markets=int(os.getenv("OOS_TEST_MIN_MARKETS", "2")),
        multi_oos_n_splits=int(os.getenv("MULTI_OOS_N_SPLITS", "10")),
        multi_oos_random_seed=int(os.getenv("MULTI_OOS_RANDOM_SEED", "42")),
        multi_oos_min_observed_splits=int(os.getenv("MULTI_OOS_MIN_OBSERVED_SPLITS", "3")),
        multi_oos_min_selected_splits=int(os.getenv("MULTI_OOS_MIN_SELECTED_SPLITS", "2")),
        multi_oos_robust_selection_frequency=float(
            os.getenv("MULTI_OOS_ROBUST_SELECTION_FREQUENCY", "0.50")
        ),
        multi_oos_robust_positive_test_frequency=float(
            os.getenv("MULTI_OOS_ROBUST_POSITIVE_TEST_FREQUENCY", "0.60")
        ),
        multi_oos_mode_consistency_threshold=float(
            os.getenv("MULTI_OOS_MODE_CONSISTENCY_THRESHOLD", "0.75")
        ),
        multi_oos_inconsistent_mode_threshold=float(
            os.getenv("MULTI_OOS_INCONSISTENT_MODE_THRESHOLD", "0.60")
        ),
        polymarket_fee_k=float(os.getenv("POLYMARKET_FEE_K", "0.0625")),
        flat_fee_bps=float(flat_fee_raw) if flat_fee_raw not in {None, ""} else None,
        cost_scenario=os.getenv("COST_SCENARIO", "base"),
        extra_cost_penalty=float(os.getenv("EXTRA_COST_PENALTY", "0.010")),
    )
