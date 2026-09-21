# Architecture

These notes describe historical design patterns from the broader private platform. The public repository contains the validation module and documentation, not the services themselves. Deployment topology and operational status are not asserted here.

## Service boundaries

The system used small Python services on Linux VMs, including GCP and AWS Lightsail deployments at different stages. Ingestion, research, and paper execution were separate responsibilities.

Useful operational patterns included:

- HTTP status endpoints for uptime, last processed events, and errors.
- Append-only JSONL decision journals for diagnosis and restart reconciliation.
- Error thresholds and circuit breakers to limit continued operation during failures.
- Telegram alerts for service health and research events.

These are architectural patterns, not a guarantee that every historical service implemented every protection. Journaling alone does not guarantee exactly-once execution; broker state and local state still require reconciliation.

## Data sources

Historical adapters covered IBKR, Alpaca, ccxt-connected exchanges, Chainlink feeds, and REST sources such as weather and news. They are not shipped in this repository. Rate limits, reconnect behavior, data freshness, and source licensing must be handled by each integration.

## Storage choices

| Data | Store | Purpose |
|---|---|---|
| Decision and event journals | JSONL | Append-only records for inspection and reconciliation |
| Research datasets | Parquet | Columnar scans and cached historical data |
| Structured service state | SQLite / PostgreSQL | Transactional records |
| Operational summaries | Markdown / YAML | Human-readable status and configuration |

## Research workflow

```text
idea → offline experiment → statistical checks → forward paper validation
```

Offline results can reject weak candidates; passing a statistical check does not establish executable performance. Forward validation must separately address fills, costs, stale data, and state recovery. Historical strategy backtests are not presented here as current paper-trading outcomes.

See [Validation Methodology](VALIDATION.md) for the functions actually included in the public sample and their limitations. Production deployment and any use of capital are outside this repository.
