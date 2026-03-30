# Project Summary

## Overview

This repository studies whether profitable public Polymarket wallets can be translated into a **copyable betting strategy**.

The core research question is not whether a wallet itself makes money. The real question is:

> If a follower observes a wallet's trade after it happens, pays realistic costs, and enters with some delay, can the follower still earn positive expected value?

This distinction drives the entire project.

The codebase builds a research pipeline around:
- public wallet trade history
- public market metadata
- public price history
- delay-aware copy and fade backtests
- behavior-level analysis
- wallet-universe discovery
- unified paper tracking for a consolidated "house" portfolio

The project is strictly research-only. It does **not** place live orders or include production trading logic.

## What The Project Does

The repository supports five major workflows.

### 1. Wallet Discovery

It scans public Polymarket market activity and reconstructs large wallet universes from observed trades.

This is used to identify:
- active wallets
- high-volume wallets
- manually curated seed wallets
- vertical specialists such as sports or politics-focused accounts

### 2. Enrichment And Price Linking

The pipeline links raw wallet trades to:
- token metadata
- market metadata
- event slugs / categories
- public price history

This makes it possible to analyze not just raw trades, but the economic context around them.

### 3. Copyability Research

The main research layer asks whether a follower can actually capture the wallet's edge.

It models:
- entry delays such as `0s`, `5s`, `15s`, `30s`, `60s`
- spread / slippage / fee assumptions
- wallet-exit replication
- hold-to-expiry logic where appropriate
- realized PnL
- unrealized mark-to-market PnL
- out-of-sample tests

This is the layer that separates:
- wallets that are profitable for themselves
from
- wallets that are realistically copyable by a follower

### 4. Behavior Analysis

The codebase also studies whether the edge comes from the wallet as a whole, or from specific classes of trades.

Examples include:
- early positioning
- aggressive momentum chasing
- passive accumulation
- late entry behavior

This helps explain why some wallets are copyable and others are not.

### 5. Unified Paper Tracking

A later stage of the project consolidates multiple copy-ready wallets into a single paper-tracked "house book."

This prevents:
- buying the same position multiple times across different source wallets
- taking opposite sides of the same market
- over-concentrating exposure due to duplicate signals

The unified house model tracks:
- open positions
- closed positions
- realized PnL
- open-position MTM
- daily equity curve

## Main Findings

### 1. Most profitable wallets are not automatically copyable

A recurring result in this project is that many wallets with strong self-PnL do **not** transfer that edge cleanly to a follower.

Typical reasons:
- the wallet's edge is execution-sensitive
- the wallet trades too frequently
- the wallet scales in and out in ways a follower cannot reproduce
- costs and timing erode the follower's edge

This means "top wallet" and "good wallet to copy" are not equivalent.

### 2. Truly copyable wallets are rare

Once delay, fees, and realistic public-price execution assumptions are imposed, the candidate set shrinks materially.

The project repeatedly finds that only a small minority of public wallets look:
- non-HFT
- non-YOLO
- still active
- economically positive for a lagged follower

That scarcity is not a bug. It is one of the main findings.

### 3. A betting framework fits better than a short-term trading framework

Several wallets should not be judged solely by short-window realized PnL.

For long-duration or medium-duration bettors, the right lens is:
- realized copy PnL
- plus open-position MTM
- plus holding behavior
- plus follower accessibility of the edge

This project therefore moved away from a purely "smooth trading strategy" interpretation and toward a "copyable betting model" interpretation.

### 4. A unified portfolio is more useful than isolated wallet streams

The project eventually moved from:
- "which single wallets look good?"
to
- "what happens if we merge their signals into a coherent paper portfolio?"

This matters because multiple wallets often:
- reinforce the same position
- overlap in the same event
- or create conflicting exposure across outcomes

The unified paper-tracking layer is therefore closer to how a real allocator would want to manage copied signals.

## Current Research State

The repository is in a strong research state, but not in a production execution state.

It is currently suitable for:
- academic review
- strategy evaluation
- wallet selection research
- paper-trading style signal tracking
- future collaboration on deeper validation

It is **not** currently intended for:
- live trading
- unattended order execution
- production portfolio management

## What Is Included In Git

This GitHub-prepared version includes:
- code
- tests
- configuration
- lightweight data files
- a small `results/` bundle with key summaries and charts

Large local artifacts are intentionally excluded:
- SQLite databases
- full raw exports
- large intermediate backtest tables

This keeps the repository usable and reviewable on GitHub.

## Recommended Entry Points

If someone new is reviewing the project, the best starting points are:

1. [`README.md`](/Users/davidgasper/Downloads/polymarket_wallet_research/README.md)
2. [`results/README.md`](/Users/davidgasper/Downloads/polymarket_wallet_research/results/README.md)
3. [`results/copy_ready/copy_ready_summary.md`](/Users/davidgasper/Downloads/polymarket_wallet_research/results/copy_ready/copy_ready_summary.md)
4. [`results/paper_tracking/house_portfolio_performance_summary.md`](/Users/davidgasper/Downloads/polymarket_wallet_research/results/paper_tracking/house_portfolio_performance_summary.md)
5. [`results/investor_note/investor_research_note_manual_seed_copy_strategy.md`](/Users/davidgasper/Downloads/polymarket_wallet_research/results/investor_note/investor_research_note_manual_seed_copy_strategy.md)

## Bottom Line

This project does **not** claim that blindly copying profitable Polymarket wallets is a robust strategy.

Instead, it shows something more useful:

- most profitable wallets are not copyable
- some public wallets appear economically copyable
- copyability must be tested directly
- portfolio construction matters
- and public-wallet research can be turned into a disciplined paper-tracking process

That makes the repository a credible base for further academic work, deeper out-of-sample testing, and more formal strategy evaluation.
