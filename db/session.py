"""Database engine and session helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import PROJECT_ROOT, get_settings
from db.base import Base


settings = get_settings()

if settings.sqlite_path is not None:
    Path(settings.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

engine = create_engine(settings.db_url, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


SQLITE_ADDITIVE_MIGRATIONS: dict[str, dict[str, str]] = {
    "wallet_trades_enriched": {
        "missing_price_history": "BOOLEAN",
        "missing_market_metadata": "BOOLEAN",
        "used_fallback_midpoint": "BOOLEAN",
        "trade_price_source": "TEXT",
        "midpoint_source": "TEXT",
        "book_source": "TEXT",
        "enrichment_status": "TEXT",
        "missing_reason": "TEXT",
        "copy_pnl_5m_delay_5s": "REAL",
        "copy_pnl_5m_delay_15s": "REAL",
        "copy_pnl_5m_delay_30s": "REAL",
        "copy_pnl_5m_delay_60s": "REAL",
        "fade_pnl_5m_delay_5s": "REAL",
        "fade_pnl_5m_delay_15s": "REAL",
        "fade_pnl_5m_delay_30s": "REAL",
        "fade_pnl_5m_delay_60s": "REAL",
        "copy_pnl_net_5m": "REAL",
        "fade_pnl_net_5m": "REAL",
        "copy_pnl_net_5m_delay_5s": "REAL",
        "copy_pnl_net_5m_delay_15s": "REAL",
        "copy_pnl_net_5m_delay_30s": "REAL",
        "copy_pnl_net_5m_delay_60s": "REAL",
        "fade_pnl_net_5m_delay_5s": "REAL",
        "fade_pnl_net_5m_delay_15s": "REAL",
        "fade_pnl_net_5m_delay_30s": "REAL",
        "fade_pnl_net_5m_delay_60s": "REAL",
    }
}


def _apply_sqlite_additive_migrations() -> None:
    """Apply lightweight additive sqlite migrations for newly added nullable columns."""

    if settings.sqlite_path is None:
        return

    with engine.begin() as connection:
        existing_tables = {
            row[0]
            for row in connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        for table_name, columns in SQLITE_ADDITIVE_MIGRATIONS.items():
            if table_name not in existing_tables:
                continue
            existing_columns = {
                row[1]
                for row in connection.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
            }
            for column_name, column_type in columns.items():
                if column_name in existing_columns:
                    continue
                connection.execute(
                    text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                )
                existing_columns.add(column_name)


def init_db() -> None:
    """Create all project tables."""

    Base.metadata.create_all(bind=engine)
    _apply_sqlite_additive_migrations()


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a managed SQLAlchemy session."""

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
