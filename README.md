# Financial Data Platform

An installable Python research pipeline: ingest local market-shaped data, store
it in SQLite, run two research flows, apply statistical gates, and build a static
report. The default demo is deterministic and entirely offline, using bundled
synthetic prices and fictional headlines. It has no broker or account integration.

## Quickstart

Python 3.11+; from a clone of this repository, run these three commands:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/fdp demo
```

After installation, **`fdp demo` is the single command** that runs the full pipeline
when the environment is active. Open `site/index.html` directly in a browser.
Use `fdp demo --output another-folder` to select a different output directory.

## Demo output

- Bundled inputs: 780 price rows over three fictional assets and 260 headlines,
  generated independently with seed 42. No downloads or trained weights.
- SQLite input snapshot, read back before research runs.
- Monthly 50/40/10 rebalancer with fractional holdings and funded transaction costs.
- Lexicon sentiment → delayed long/cash exposure, with a session loss circuit breaker.
- Four validation gates per flow: positive return, sign randomization, deflated
  Sharpe and bootstrap downside. **FAIL is expected research output**, not a
  pipeline error or a claim about real markets.
- `site/index.html`, `equity.png`, `exposure.png`, `summary.json`,
  `rebalancer.csv`, `sentiment.csv`, and `research.sqlite`. Output is ignored by Git.

The [Pages workflow](.github/workflows/pages.yml) builds the same synthetic report
on pushes to `main`. The coordinator must enable Pages with GitHub Actions before
it can deploy. No live deployment is asserted by this README.

## Architecture

`ingestion → SQLite → rebalancer + sentiment/price → validation → HTML/PNG`

The package lives in [`src/fdp/`](src/fdp/). The CLI orchestrates small modules;
network access is confined to a separate public no-key RSS command with explicit
opt-in (`fdp fetch-news SPY --allow-network`). That command is never part of the
demo or a live test and its content is not automatically published.

See [architecture](docs/ARCHITECTURE.md) for the module diagram and contracts, and
[provenance](docs/PROVENANCE.md) for each sanitized source adaptation.
The original [validation module](shared/backtest_validation.py) and
[methodology notes](docs/VALIDATION.md) keep their existing paths.

## Checks

```sh
.venv/bin/ruff check .
.venv/bin/pytest -q
```

Tests are offline and deterministic, including existing ML regression suites in
isolated subprocesses. They cover parsing failures, deduplication, atomic storage,
rebalancer math, fees, signal timing, breaker transitions, statistical controls,
fixture reproduction, the CLI and report artifacts. Ruff covers the new package,
tests and scripts; the preserved legacy modules retain their original formatting.
CI repeats lint, tests and the demo on Python 3.11 and 3.12.

## Scope and limits

Synthetic output demonstrates code behavior, not strategy validity or profitability.
Costs, fractional fills and calendar handling are simplified; there is no capital,
broker execution, deployment monitoring or private performance data here. The
unchanged [ML experiments](ml-experiments/) have their own honest metrics,
reproduction commands and limitations; they do not power this demo. Their datasets
are separate downloads with separate terms. Optional public RSS availability and
reuse rights belong to its provider. Code is [MIT licensed](LICENSE).

## AI assistance

Claude and ChatGPT assisted with implementation, review and documentation,
including the separate ML experiments.
