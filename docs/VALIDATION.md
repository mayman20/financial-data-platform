# Validation Methodology

Backtests lie. The whole point of this platform's research layer is to catch the specific ways they lie before any strategy touches even a paper account. This document describes the gauntlet every candidate runs, implemented in [`shared/backtest_validation.py`](../shared/backtest_validation.py).

## 1. Out-of-sample discipline

Every backtest splits data into in-sample (parameter selection) and out-of-sample (verdict) windows before any tuning happens. For walk-forward tests, `walk_forward_splits()` generates rolling train/test windows with **purge and embargo** gaps between them, so information can't leak across the boundary through overlapping bars or serial correlation.

A strategy's reported numbers are always the out-of-sample numbers. In-sample results are treated as an upper bound on hope, not evidence.

## 2. Multiple-trials correction — the Deflated Sharpe Ratio

Testing 300 parameter combinations and reporting the best one is data snooping, even when each individual test is honest. `deflated_sharpe()` implements Bailey & López de Prado's DSR: it estimates the Sharpe you'd expect the *best of N random trials* to show, then asks whether the observed Sharpe clears that bar with statistical significance, accounting for the return distribution's skew and kurtosis.

Every research run logs its trial count, and the DSR is computed against the full count — including the trials that failed.

## 3. Permutation tests

`permutation_test()` asks the simplest question: if the strategy's trade directions were random, how often would results this good appear? Trade signs are permuted thousands of times to build a null distribution; the strategy's edge must land in the tail (p < 0.05).

## 4. Bootstrap confidence intervals

A single equity curve is one draw from a distribution. `bootstrap_equity()` block-bootstraps trade returns (preserving local autocorrelation) to produce confidence intervals on terminal equity and maximum drawdown. A strategy whose 5th-percentile path is ruinous fails, regardless of its mean.

## 5. Execution realism

- `add_entry_noise()` injects adverse slippage into every fill and re-runs the backtest; edges that die under 1-2 ticks of slippage were never real.
- Fill-latency stress: signals are re-evaluated with delayed entry prices to simulate real order placement, which has single-handedly killed strategies whose "edge" was actually just being first in a queue no retail order reaches.

## 6. Negative controls

The most productive gate in the framework. Alongside every backtest, the same harness runs:

| Control | What it isolates |
|---|---|
| **Random-side** | same entries, coin-flip direction — did direction actually matter? |
| **Delayed-fill** | same signals, entry N bars later — is the edge just latency? |
| **Shuffled-return** | same trade sizes on shuffled market returns — is it just exposure in a trending market? |

If a control performs comparably to the strategy, the strategy's performance is an artifact, not an edge. One paper-traded strategy with a 77.8% live win rate was killed this way — the win rate was a favorite-selection bias the random-side control reproduced almost exactly.

## 7. Regime awareness

`regime_label()` classifies bars into trending / ranging / mixed regimes (efficiency ratio, Hurst exponent, ADX). Strategies report performance per regime; an edge that exists only in one regime gets a regime filter and a re-test, not a pass.

## 8. Pre-registration

Before any paper deployment, the kill criteria are written down: minimum trade count, decision date, and the metric thresholds that constitute failure. The deployment cannot be extended past its decision date, and the criteria can't be revised after seeing results. This is borrowed directly from clinical-trial design, and it exists because "just a few more weeks of data" is how dead strategies stay deployed forever.
