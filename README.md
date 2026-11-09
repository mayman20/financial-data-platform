# Financial Data Platform

[![CI](https://github.com/mayman20/financial-data-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/mayman20/financial-data-platform/actions/workflows/ci.yml)

This is a Python research pipeline. It ingests market-shaped data, stores it in SQLite and runs research flows. Every result must clear statistical validation gates before it reaches the report. The gates are the point: they reject strategies that only look good by chance.

**[Open the demo report](https://mayman20.github.io/demos/financial-data-platform/)**. It uses synthetic data and is rebuilt offline by `fdp demo`.

## What the demo shows

| Case | Result |
|---|---|
| Pure noise (no edge) | Passed 0 of 20 runs, a 0% false-positive rate |
| 5 bp planted edge | Detected in 0 of 20 runs |
| 10 bp planted edge | Detected in 12 of 20 runs |
| 20 bp planted edge | Detected in 20 of 20 runs |
| 40 bp planted edge | Detected in 20 of 20 runs |
| 50/40/10 rebalancer | Stays within 2.43 percentage points of target weights |

Every row uses the same 20 predeclared seeds, 5,200 synthetic sessions, 60 bp of daily noise and a 5 bp cost per exposure change. A run counts as detected only if it passes all four gates. The data is synthetic, so this measures the software and the statistics, not a real strategy.

## Quickstart

You need Python 3.11 or newer.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/fdp demo
```

The demo takes about 7 seconds, runs offline and writes `site/index.html`.

## How it works

```
ingestion → SQLite → research flows → validation gates → static report
```

- **Ingestion.** An RSS news parser handles normalization and deterministic deduplication. Live fetching is a separate command you have to opt into: `fdp fetch-news SPY --allow-network`. The demo and tests never use it.
- **Storage.** SQLite snapshots are written atomically and read back before research runs.
- **Research flows.**
  - A monthly 50/40/10 rebalancer with fractional holdings and transaction costs.
  - A sentiment flow: a lexicon score drives next-session long/cash exposure, with a loss circuit breaker.
- **Validation gates.** A strategy must pass all four:
  - positive total return,
  - a sign-randomization test (p < 0.05),
  - a deflated Sharpe ratio test that accounts for the number of trials,
  - a bootstrap check that the 5th-percentile terminal equity stays above 0.80.
- **Report.** Static HTML with PNG charts. No JavaScript or CDN.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) has the module diagram and contracts. [docs/PROVENANCE.md](docs/PROVENANCE.md) lists which modules were adapted from my private research code and what was removed for this public version. To regenerate the detection-rate table, run `.venv/bin/python scripts/detection_power.py`.

## Tests

```sh
.venv/bin/ruff check .
.venv/bin/pytest -q
```

There are 85 offline, deterministic tests. They cover:
- parsing failures and deduplication,
- atomic storage,
- rebalancer math and fees,
- signal timing and circuit-breaker transitions,
- the statistical controls,
- the CLI and report output.

CI runs lint, tests and the demo on Python 3.11 and 3.12. A deterministic scan, `scripts/sanitize.py`, checks the tree and the commit history for secrets and personal data.

## ML experiments

[`ml-experiments/`](ml-experiments/) holds two small NumPy text-classification experiments. A sentence-sentiment classifier reached 87.9% test accuracy on Financial PhraseBank. Predicting a later price move from headlines scored 45.2%, below its 48.4% baseline, and that result is reported as is.

## Scope

- There is no broker connection, live capital or private performance data here.
- Costs, fills and calendars are simplified.
- The optional RSS source keeps its provider's terms.

## License and AI use

The code is [MIT licensed](LICENSE). I used Claude and ChatGPT to help with implementation, review and documentation.
