"""Wallet validation and storage helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Wallet


WALLET_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")


@dataclass(frozen=True)
class WalletLoadResult:
    """Validated wallet input payload plus parsing metadata."""

    wallets: list[str]
    valid_count: int
    ignored_count: int
    invalid_entries: list[str]
    source_label: str


def is_valid_wallet_address(wallet: str) -> bool:
    """Return True when the address matches the public Polymarket wallet format."""

    return bool(WALLET_PATTERN.fullmatch(wallet.strip()))


def normalize_wallet_address(wallet: str) -> str:
    """Normalize a wallet address to lowercase and validate it."""

    normalized = wallet.strip().lower()
    if not is_valid_wallet_address(normalized):
        raise ValueError(f"Invalid wallet address: {wallet}")
    return normalized


def dedupe_wallets(wallets: Sequence[str]) -> list[str]:
    """Normalize, validate, and deduplicate wallet inputs while preserving order."""

    seen: set[str] = set()
    deduped: list[str] = []
    for wallet in wallets:
        normalized = normalize_wallet_address(wallet)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def parse_wallet_lines(lines: Sequence[str], source_label: str) -> WalletLoadResult:
    """Parse wallet lines, ignoring blanks and `#` comments."""

    ignored_count = 0
    invalid_entries: list[str] = []
    normalized_wallets: list[str] = []
    seen: set[str] = set()

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            ignored_count += 1
            continue

        try:
            normalized = normalize_wallet_address(stripped)
        except ValueError:
            invalid_entries.append(stripped)
            continue

        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_wallets.append(normalized)

    return WalletLoadResult(
        wallets=normalized_wallets,
        valid_count=len(normalized_wallets),
        ignored_count=ignored_count,
        invalid_entries=invalid_entries,
        source_label=source_label,
    )


def load_wallet_inputs(values: Sequence[str]) -> WalletLoadResult:
    """Load wallets from a file or directly from CLI arguments."""

    if len(values) == 1:
        candidate = Path(values[0]).expanduser()
        if candidate.exists() and candidate.is_file():
            return parse_wallet_lines(
                candidate.read_text(encoding="utf-8").splitlines(),
                source_label=str(candidate),
            )
    return parse_wallet_lines(values, source_label="cli")


def store_wallets(session: Session, wallets: Sequence[str], source: str | None = "cli") -> list[str]:
    """Insert new wallets and refresh `last_seen` for existing ones."""

    normalized_wallets = dedupe_wallets(wallets)
    if not normalized_wallets:
        return []

    existing = {
        wallet.wallet_address: wallet
        for wallet in session.scalars(
            select(Wallet).where(Wallet.wallet_address.in_(normalized_wallets))
        )
    }
    now = datetime.now(tz=timezone.utc)

    for wallet_address in normalized_wallets:
        wallet = existing.get(wallet_address)
        if wallet is None:
            session.add(
                Wallet(
                    wallet_address=wallet_address,
                    source=source,
                    first_seen=now,
                    last_seen=now,
                )
            )
            continue

        wallet.last_seen = now
        if source and not wallet.source:
            wallet.source = source

    session.flush()
    return normalized_wallets
