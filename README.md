# Financial Data Platform

An engineering case study in market-data ingestion, research validation, and paper-trading infrastructure. This public excerpt contains a Python validation module, architecture notes, and sanitized configuration.

The broader system evolved across several private projects. This repository documents selected work and historical design patterns; it is not a currently deployed trading service or a live performance dashboard.

## Start here

- [Validation module](shared/backtest_validation.py): sign-randomization tests, bootstrap equity distributions, deflated Sharpe calculations, rolling splits, and regime-labeling helpers.
- [Architecture](docs/ARCHITECTURE.md): adapters, storage choices, journals, and service monitoring.
- [Methodology and limitations](docs/VALIDATION.md): what the included code implements and what requires an external research harness.
- [Configuration example](config/config.example.yaml): illustrative configuration, without credentials.

## Run the offline example

From the repository root, with Python 3:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scipy
python shared/backtest_validation.py
```

The module's built-in example generates synthetic return series and prints statistical summaries. It makes no network requests or trades. A printed `[FAIL]` is a synthetic candidate failing the statistical gates, not a program error. This is a smoke test of the example, not evidence of strategy profitability or a comprehensive test suite.

## Engineering decisions

- Separate ingestion, research, and execution so a research experiment does not require broker access.
- Keep append-only decision journals for diagnosis and reconciliation after interruptions.
- Evaluate apparent results against randomized controls and account for multiple trials.
- Keep the public sample small enough to inspect without exposing account-connected operations.

## Scope

Live adapters, broker execution, service definitions, raw research datasets, and full experiment records are not included. Historical backtests are not forward paper results; the public sample does not establish present service uptime or independently reproducible strategy performance. The documentation describes these boundaries explicitly.

Stack represented here: Python, NumPy, pandas, SciPy, and YAML. The historical architecture also used Linux/systemd, cloud VMs, JSONL, Parquet, SQLite, and PostgreSQL.
