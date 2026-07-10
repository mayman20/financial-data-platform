# Architecture

## Service model

Every long-running component is an independent systemd unit on a Linux VM (GCP e2 instances historically; AWS Lightsail currently). There is no orchestration layer — services are deliberately small, single-purpose Python daemons that share a common operational contract:

1. **A local HTTP status endpoint.** Each service exposes `/status` on its own port, returning JSON: uptime, last event processed, open positions, error counts. A single runbook can curl every service on a box and know the fleet's health in seconds.
2. **A journal.** Every decision — signal, order, fill, skip, error — is appended to a JSONL journal before it takes effect. On restart, a service replays its journal to rebuild in-memory state, so crashes and reboots don't produce duplicate orders or orphaned positions.
3. **A circuit breaker.** Consecutive-error and drawdown thresholds trip the breaker, which flattens exposure where applicable and alerts, rather than letting a degraded service keep trading.
4. **Telegram alerting.** All services route notifications through a shared alerting module with per-channel routing (trade fills, system health, research digests are separate channels), so a phone is the pager.

## Data ingestion

Ingestion services are per-source adapters that normalize into a common event shape before persistence:

- **IBKR** — historical bars and live account/news data via the native API
- **Alpaca** — equities/ETF market data and paper-trade execution
- **ccxt** — crypto exchange data (funding rates, perpetuals OHLCV)
- **Chainlink over WebSocket** — sub-second on-chain price feed ticks
- **REST pollers** — weather (NOAA METAR/GEFS), macro calendars, news RSS

Adapters are rate-limit aware and reconnect with backoff; a dropped WebSocket or a 429 degrades a service to stale-data mode (flagged in `/status`) instead of crashing it.

## Storage

| Data | Store | Why |
|---|---|---|
| Decision/event journals | JSONL (append-only) | crash-safe replay, greppable forensics |
| Research/backtest datasets | Parquet | columnar scans over years of bars |
| Service state, small datasets | SQLite / PostgreSQL | transactional reads/writes per service |
| Fleet status | YAML-frontmatter markdown | human- and dashboard-readable at once |

## Research → production pipeline

Strategies move through fixed stages, each with a gate:

```
idea → offline backtest → validation gates → paper deployment → (rarely) capital
```

- The **validation gates** are statistical (see [VALIDATION.md](VALIDATION.md)) and pre-registered: the kill criteria and decision date are written down before the paper deployment starts, so a marginal strategy can't linger on hope.
- **Paper deployments** run on the same code path as live execution — same adapters, same journals, same breakers — so promotion is a config change, not a rewrite.
- Most candidates die at the gates. That is the intended behavior: the platform's job is to make killing a bad idea cost one config file, not a funded account.

## Deployment & ops

- Plain `systemd` units + cron, deployed over SSH with small scripts. No containers — single-owner VMs made simple tooling the right trade.
- Health checks run on cron and alert on dead services, stale journals, disk pressure, and clock drift.
- Every VM has an identical runbook: status endpoints, journal tails, and unit states, so any box can be diagnosed in one pass.
