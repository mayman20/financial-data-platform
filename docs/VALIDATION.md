# Validation Methodology

The public [validation module](../shared/backtest_validation.py) contains statistical helpers. It is an excerpt from a broader research workflow, not an end-to-end backtester, execution simulator, or automatic deployment gate.

## Included functions

| Function | Current implementation |
|---|---|
| `deflated_sharpe()` | Adjusts an observed Sharpe estimate for trial selection and return moments |
| `permutation_test()` | Randomly flips return signs and compares mean returns |
| `bootstrap_equity()` | Resamples short contiguous blocks to estimate terminal-equity and drawdown distributions |
| `walk_forward_splits()` | Generates rolling train/test index ranges with a purge gap |
| `add_entry_noise()` | Applies positively skewed price perturbations |
| `regime_label()` | Labels a price series using efficiency ratio, Hurst-style estimation, or an ADX-style approximation |
| `validate_returns()` | Combines descriptive statistics, deflated Sharpe, sign randomization, and bootstrap output in a report |

## Important implementation limits

- `walk_forward_splits()` accepts `embargo_bars`, but the current implementation does not apply it. Do not treat these splits as embargo-protected. Purging alone does not establish freedom from leakage.
- `permutation_test()` currently compares means regardless of its `metric` argument. Its older docstring describes sequence shuffling; the actual implementation uses random sign flips.
- `add_entry_noise()` has no position-side input. Its upward price bias is not automatically adverse for short entries.
- The ADX-style regime option uses close prices only; it is not a full OHLC implementation.
- The report uses a fixed annualization convention. Sampling frequency, trial count, dependence, and appropriate input ranges need review for each research use.
- The built-in synthetic example is not a comprehensive regression suite or independent statistical validation.

## Practices outside this module

The broader research process considered out-of-sample selection, execution costs, delayed fills, random-side controls, shuffled returns, and prewritten decision criteria. Those practices require datasets and strategy-specific harnesses that are not included here. This module does not enforce that every candidate follows them.

Historical QQQ/SPY liquidity-sweep experiments still required forward paper validation; the old infrastructure's status does not demonstrate current performance. This repository therefore emphasizes inspectable methods and implementation boundaries rather than a table of headline returns.

## Offline example

Follow the setup in the [README](../README.md), then run:

```sh
python shared/backtest_validation.py
```

The example generates synthetic data locally. Its output demonstrates the report format and code path, not a trading result.

## Packaged demo gates

The installable pipeline uses a separate, documented subset at
[`src/fdp/statistics.py`](../src/fdp/statistics.py), wrapped by
[`src/fdp/validation.py`](../src/fdp/validation.py). It applies all four configured
gates and rejects invalid inputs, instead of accepting missing checks. Its sign
randomization uses a finite-sample +1 correction, and bootstrap drawdown includes
initial equity. These corrections do not change the legacy file described above.
The demo does not use the legacy walk-forward, regime or slippage helpers.

`fdp demo` applies these gates to the planted and noise sentiment cases only.
The rebalancer reports allocation diagnostics instead. The fixed planted signal
need not pass on every seed: the report discloses its effect, costs, primary-seed
failure and empirical detection frequency across predeclared seeds 100–119.

`fdp demo` prints and publishes synthetic gate outcomes; a failed gate is a
research rejection, not a build failure. Seeded positive/negative controls test
gate behavior, separately from any claim that a strategy is valid. See
[architecture](ARCHITECTURE.md) and [provenance](PROVENANCE.md) for assumptions.
