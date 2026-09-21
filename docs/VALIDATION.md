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
