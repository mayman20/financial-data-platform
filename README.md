# Real-Time Financial Data Platform

A multi-service platform for ingesting live financial market data, validating research hypotheses against it, and running always-on paper-trading services — built and operated solo, 2025–present.

> **Why this repo is a mirror.** The production repository is private: it contains API credentials, broker account state, and live trading journals. This mirror documents the architecture, shares the core validation library, and provides sanitized configuration templates. Nothing here can be pointed at an account and run — that's deliberate.

## What it does

- **Ingests live data 24/7** from six external APIs — IBKR, Alpaca, ccxt-connected crypto exchanges, Chainlink price feeds over WebSocket, NOAA weather, and market-data REST endpoints — across services deployed on GCP and AWS Lightsail.
- **Persists everything durably**: JSONL event journals for every trade decision, Parquet caches for research data, SQLite/PostgreSQL stores per service.
- **Validates research before anything runs**: every strategy passes an offline pipeline of out-of-sample testing, permutation tests, deflated Sharpe ratios, and negative controls across hundreds of experiment runs. Most candidates fail and are killed — the framework is designed to make that cheap.
- **Runs unattended**: concurrent systemd services with structured logging, Telegram alerting, per-service HTTP status endpoints, circuit breakers, and journal-based recovery. Services resume cleanly after network drops, API failures, and state mismatches.

## Architecture

```
            ┌────────────────────────────────────────────────┐
            │                DATA INGESTION                  │
            │  IBKR · Alpaca · ccxt · Chainlink WS · REST    │
            └───────────────────────┬────────────────────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
   ┌──────────────────┐   ┌─────────────────┐   ┌────────────────────┐
   │  RESEARCH LAYER  │   │  LIVE SERVICES  │   │      STORAGE       │
   │  backtests, OOS  │   │ systemd daemons │   │  JSONL journals    │
   │  validation,     │   │ paper execution │   │  Parquet caches    │
   │  negative        │   │ status servers  │   │  SQLite/Postgres   │
   │  controls        │   │ circuit breaker │   │                    │
   └────────┬─────────┘   └────────┬────────┘   └────────────────────┘
            │                      │
            ▼                      ▼
   ┌──────────────────┐   ┌─────────────────┐
   │ promotion gates  │   │   MONITORING    │
   │ (kill by default)│   │ Telegram alerts │
   │                  │   │ health checks   │
   └──────────────────┘   └─────────────────┘
```

Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The validation discipline

The most important part of the platform is the part that says **no**. Every strategy candidate must survive:

| Gate | Tool |
|---|---|
| Out-of-sample split | walk-forward windows with purge + embargo |
| Data-snooping correction | deflated Sharpe ratio (Bailey & López de Prado) |
| Edge vs. luck | sign-permutation tests (p < 0.05) |
| Tail risk | block-bootstrap equity confidence intervals |
| Execution realism | adverse slippage injection, fill-latency stress |
| Negative controls | random-side, delayed-fill, and shuffled-return controls run alongside every backtest |

The negative controls have killed more strategies than any other gate — including several that looked excellent on paper. Methodology in [docs/VALIDATION.md](docs/VALIDATION.md); the actual library is in [shared/backtest_validation.py](shared/backtest_validation.py).

## Selected validated results (paper trading)

| Strategy | Out-of-sample result | Status |
|---|---|---|
| Liquidity Sweep Reversal (QQQ, 5-min) | Sharpe 2.66, PF 1.53, 803 trades, perm p < 0.001 | Live paper |
| Liquidity Sweep Reversal (SPY, 5-min) | Sharpe 1.40, PF 1.28, 769 trades, perm p = 0.015 | Live paper |
| All-Weather 50/40/10 rebalancer | CAGR 19.9%, Sharpe 1.06 | Paper, retired |
| Donchian trend (BTC, 20/10) | Sharpe 0.87, DSR 0.98 | Shadow tracking |

The much longer list of strategies that **failed** validation — and the negative-control evidence that killed them — is the platform's real output. A research idea that dies in a $0 backtest instead of a funded account is the system working.

## Stack

Python · pandas / NumPy / SciPy · PostgreSQL / SQLite / Parquet · systemd · cron · GCP · AWS Lightsail · Telegram Bot API · pytest
