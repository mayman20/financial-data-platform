"""
Shared Backtest Validation Module
=================================
Generic statistical tools for validating trading strategy backtests.
Used by ALL APE DEV strategies — crypto, equities, prediction markets.

Tools:
  - deflated_sharpe()       — Corrects Sharpe for multiple trials (Bailey & Lopez de Prado)
  - permutation_test()      — Tests if edge is real vs random (p-value)
  - bootstrap_equity()      — Confidence intervals on equity curve
  - walk_forward_splits()   — Rolling train/test windows with purge/embargo
  - add_entry_noise()       — Adverse slippage injection for stress testing
  - regime_label()          — Classify bars into trending/ranging/volatile regimes

References:
  - Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (SSRN 2460551)
  - Lopez de Prado, "Advances in Financial Machine Learning" (2018)
  - White, "A Reality Check for Data Snooping" (Econometrica, 2000)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from scipy import stats


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (DSR)
# ---------------------------------------------------------------------------

def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_returns: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    annualization_factor: float = 252.0,
) -> dict:
    """
    Compute the Deflated Sharpe Ratio — corrects observed Sharpe for the
    number of strategy variations tested (data snooping bias).

    Parameters
    ----------
    observed_sharpe : float
        The Sharpe ratio of the best strategy found.
    n_trials : int
        Total number of strategy variations tested (e.g., 292 param combos).
    n_returns : int
        Number of return observations in the backtest.
    skewness : float
        Skewness of the return series (0 = normal).
    kurtosis : float
        Kurtosis of the return series (3 = normal).

    Returns
    -------
    dict with keys:
        dsr: float — the deflated Sharpe ratio (probability the true Sharpe > 0)
        expected_max_sharpe: float — expected maximum Sharpe under null hypothesis
        is_significant: bool — True if DSR > 0.95 (5% significance)
    """
    # All Sharpe math below is in PER-OBSERVATION units (Bailey & Lopez de Prado 2014).
    # Callers pass annualized Sharpe; convert first. (Fixed 2026-07-03: the original
    # version compared an annualized Sharpe against the raw order-statistic z-score,
    # which fails almost any real strategy — a pure units bug.)
    sr_obs = observed_sharpe / np.sqrt(annualization_factor)

    # Expected maximum per-observation Sharpe among n_trials null strategies:
    # E[max SR] = sqrt(V[SR_est]) * z_n, with V[SR_est] ~ 1/n under the null
    euler_gamma = 0.5772156649
    z = (1 - euler_gamma) * stats.norm.ppf(1 - 1 / n_trials) + \
        euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    se_null = np.sqrt(1.0 / n_returns)
    expected_max_sr = se_null * z

    # Standard error of the OBSERVED per-observation Sharpe (non-normality corrected)
    se_sharpe = np.sqrt(
        (1 + 0.5 * sr_obs**2 - skewness * sr_obs +
         (kurtosis - 3) / 4 * sr_obs**2) / n_returns
    )

    if se_sharpe == 0:
        return {"dsr": 0.0, "expected_max_sharpe": expected_max_sr, "is_significant": False}

    # DSR = Prob(true Sharpe > 0) given that we selected the best of n_trials
    test_stat = (sr_obs - expected_max_sr) / se_sharpe
    dsr = stats.norm.cdf(test_stat)

    return {
        "dsr": round(dsr, 4),
        "expected_max_sharpe": round(expected_max_sr, 6),
        "expected_max_sharpe_annualized": round(expected_max_sr * np.sqrt(annualization_factor), 4),
        "is_significant": dsr > 0.95,
        "observed_sharpe": round(observed_sharpe, 4),
        "observed_sr_per_obs": round(sr_obs, 6),
        "n_trials": n_trials,
        "se_sharpe": round(se_sharpe, 6),
    }


# ---------------------------------------------------------------------------
# Permutation Test
# ---------------------------------------------------------------------------

def permutation_test(
    returns: np.ndarray,
    n_perms: int = 1000,
    metric: str = "sharpe",
    seed: int = 42,
) -> dict:
    """
    Test if a strategy's performance is distinguishable from random.

    Shuffles the return sequence n_perms times, recomputes the metric each time,
    and reports the p-value (fraction of shuffled results that beat the real one).

    Parameters
    ----------
    returns : array-like
        Per-trade or per-bar return series.
    n_perms : int
        Number of random permutations (1000 minimum, 5000 for publication).
    metric : str
        "sharpe", "mean", or "total" — which metric to compare.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        p_value: float — fraction of permutations that beat the real metric
        observed: float — the real metric value
        perm_mean: float — average metric across permutations
        perm_std: float — std of metric across permutations
        is_significant: bool — True if p_value < 0.05
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)

    # For trade returns, the null hypothesis is "mean return = 0".
    # We test this by randomly flipping the sign of each return (sign permutation),
    # which preserves the magnitude distribution but destroys directional edge.
    observed_mean = np.mean(returns)
    perm_results = np.zeros(n_perms)
    for i in range(n_perms):
        signs = rng.choice([-1, 1], size=len(returns))
        perm_results[i] = np.mean(returns * signs)

    p_value = np.mean(perm_results >= observed_mean)

    return {
        "p_value": round(float(p_value), 4),
        "observed_mean": round(float(observed_mean), 6),
        "perm_mean": round(float(np.mean(perm_results)), 6),
        "perm_std": round(float(np.std(perm_results)), 6),
        "is_significant": p_value < 0.05,
        "n_perms": n_perms,
    }


# ---------------------------------------------------------------------------
# Bootstrap Equity Curve
# ---------------------------------------------------------------------------

def bootstrap_equity(
    trade_returns: np.ndarray,
    n_boots: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap resample trade returns to produce confidence intervals
    on terminal equity and maximum drawdown.

    Uses block bootstrap (block_size=5) to preserve some autocorrelation.

    Parameters
    ----------
    trade_returns : array-like
        Per-trade return series (e.g., [0.03, -0.02, 0.01, ...]).
    n_boots : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict with percentile terminal equities and drawdowns.
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(trade_returns, dtype=float)
    n = len(returns)
    if n < 5:
        return {"error": "Need at least 5 trades for bootstrap"}

    block_size = min(5, n // 2)
    terminal_equities = []
    max_drawdowns = []

    for _ in range(n_boots):
        # Block bootstrap
        sampled = []
        while len(sampled) < n:
            start = rng.integers(0, n - block_size + 1)
            sampled.extend(returns[start:start + block_size])
        sampled = np.array(sampled[:n])

        # Compute equity curve
        equity = np.cumprod(1 + sampled)
        terminal_equities.append(equity[-1])

        # Compute max drawdown
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        max_drawdowns.append(np.max(dd))

    te = np.array(terminal_equities)
    mdd = np.array(max_drawdowns)

    return {
        "terminal_equity": {
            "p5": round(float(np.percentile(te, 5)), 4),
            "p25": round(float(np.percentile(te, 25)), 4),
            "p50": round(float(np.percentile(te, 50)), 4),
            "p75": round(float(np.percentile(te, 75)), 4),
            "p95": round(float(np.percentile(te, 95)), 4),
        },
        "max_drawdown": {
            "p5": round(float(np.percentile(mdd, 5)), 4),
            "p50": round(float(np.percentile(mdd, 50)), 4),
            "p95": round(float(np.percentile(mdd, 95)), 4),
        },
        "bust_probability": round(float(np.mean(te < 0.5)), 4),  # prob of losing >50%
        "n_boots": n_boots,
        "n_trades": n,
    }


# ---------------------------------------------------------------------------
# Walk-Forward Splits
# ---------------------------------------------------------------------------

def walk_forward_splits(
    dates: pd.DatetimeIndex,
    n_windows: int = 10,
    train_ratio: float = 0.7,
    purge_bars: int = 20,
    embargo_bars: int = 5,
) -> list[dict]:
    """
    Generate rolling walk-forward train/test splits with purge and embargo.

    Parameters
    ----------
    dates : DatetimeIndex
        Index of the full dataset.
    n_windows : int
        Number of rolling test windows.
    train_ratio : float
        Fraction of each window used for training.
    purge_bars : int
        Number of bars to remove between train and test (prevents leakage).
    embargo_bars : int
        Number of bars to skip after test set before next train window starts.

    Returns
    -------
    List of dicts, each with:
        train_start, train_end, test_start, test_end (as timestamps)
        train_idx, test_idx (as integer index ranges)
    """
    n = len(dates)
    window_size = n // n_windows
    if window_size < 50:
        raise ValueError(f"Too few bars ({n}) for {n_windows} windows (need 50+ per window)")

    test_size = int(window_size * (1 - train_ratio))
    train_size = window_size - test_size

    splits = []
    for i in range(n_windows):
        test_start_idx = i * test_size + train_size
        test_end_idx = min(test_start_idx + test_size, n)
        train_start_idx = max(0, test_start_idx - train_size - purge_bars)
        train_end_idx = test_start_idx - purge_bars

        if train_end_idx <= train_start_idx or test_end_idx <= test_start_idx:
            continue
        if test_start_idx >= n:
            break

        splits.append({
            "window": i + 1,
            "train_start": dates[train_start_idx],
            "train_end": dates[min(train_end_idx, n - 1)],
            "test_start": dates[min(test_start_idx, n - 1)],
            "test_end": dates[min(test_end_idx - 1, n - 1)],
            "train_idx": (train_start_idx, train_end_idx),
            "test_idx": (test_start_idx, test_end_idx),
        })

    return splits


# ---------------------------------------------------------------------------
# Entry Price Noise Injection
# ---------------------------------------------------------------------------

def add_entry_noise(
    prices: np.ndarray,
    adverse_bps: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Add random adverse slippage to entry prices for stress testing.

    Skews toward unfavorable fills: uniform(-adverse_bps/4, +adverse_bps) in basis points.
    If a strategy collapses from 10bps of noise, the edge is too thin.

    Parameters
    ----------
    prices : array-like
        Entry prices to add noise to.
    adverse_bps : float
        Maximum adverse slippage in basis points (10 = 0.10%).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray of noised prices.
    """
    rng = np.random.default_rng(seed)
    prices = np.asarray(prices, dtype=float)
    # Skew toward adverse fills: mostly positive (against you)
    noise = rng.uniform(-adverse_bps / 4, adverse_bps, size=len(prices)) / 10000
    return prices * (1 + noise)


# ---------------------------------------------------------------------------
# Regime Classification
# ---------------------------------------------------------------------------

def regime_label(
    close: pd.Series,
    method: str = "efficiency_ratio",
    window: int = 20,
) -> pd.Series:
    """
    Label each bar with a market regime.

    Methods:
      - "efficiency_ratio": Kaufman's ER = displacement / path_length
        ER > 0.3 = trending, ER < 0.15 = ranging, else = mixed
      - "hurst": Rolling Hurst exponent
        H > 0.55 = trending, H < 0.45 = mean-reverting, else = mixed
      - "adx": ADX-based
        ADX > 25 = trending, ADX < 20 = ranging, else = mixed

    Parameters
    ----------
    close : pd.Series
        Close prices.
    method : str
        Classification method.
    window : int
        Rolling window for computation.

    Returns
    -------
    pd.Series of labels: "trending", "ranging", or "mixed"
    """
    labels = pd.Series("mixed", index=close.index)

    if method == "efficiency_ratio":
        displacement = abs(close - close.shift(window))
        path_length = close.diff().abs().rolling(window).sum()
        er = displacement / path_length.replace(0, np.nan)
        labels[er > 0.3] = "trending"
        labels[er < 0.15] = "ranging"

    elif method == "hurst":
        def _rolling_hurst(series, w):
            result = pd.Series(np.nan, index=series.index)
            log_returns = np.log(series / series.shift(1)).dropna()
            for i in range(w, len(log_returns)):
                window_data = log_returns.iloc[i - w:i].values
                if len(window_data) < w or np.std(window_data) == 0:
                    continue
                lags = range(2, min(w // 2, 20))
                tau = []
                for lag in lags:
                    chunks = [window_data[j:j + lag] for j in range(0, len(window_data) - lag, lag)]
                    if len(chunks) < 2:
                        continue
                    tau.append(np.std([np.sum(c) for c in chunks]))
                if len(tau) < 3:
                    continue
                log_lags = np.log(list(lags[:len(tau)]))
                log_tau = np.log(tau)
                if len(log_lags) > 1:
                    slope = np.polyfit(log_lags, log_tau, 1)[0]
                    result.iloc[i + 1] = slope  # +1 because log_returns is shifted
            return result

        hurst = _rolling_hurst(close, window)
        labels[hurst > 0.55] = "trending"
        labels[hurst < 0.45] = "ranging"

    elif method == "adx":
        high = close  # Simplified — ideally pass OHLC
        low = close
        tr = close.diff().abs()
        atr = tr.rolling(window).mean()
        dm_plus = close.diff().clip(lower=0)
        dm_minus = (-close.diff()).clip(lower=0)
        di_plus = 100 * dm_plus.rolling(window).mean() / atr.replace(0, np.nan)
        di_minus = 100 * dm_minus.rolling(window).mean() / atr.replace(0, np.nan)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
        adx = dx.rolling(window).mean()
        labels[adx > 25] = "trending"
        labels[adx < 20] = "ranging"

    return labels


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Standardized output for strategy validation."""
    strategy_name: str
    variant: str
    asset: str
    period: str
    n_trades: int
    win_rate: float
    total_return_pct: float
    sharpe: float
    dsr: Optional[dict] = None
    permutation: Optional[dict] = None
    bootstrap: Optional[dict] = None
    walk_forward_efficiency: Optional[float] = None
    regime_breakdown: Optional[dict] = None

    @property
    def passes_all_gates(self) -> bool:
        """Check if this result passes all validation gates."""
        gates = [
            self.total_return_pct > 0,                                    # OOS positive
            self.permutation is None or self.permutation["p_value"] < 0.05,  # Edge is real
            self.walk_forward_efficiency is None or self.walk_forward_efficiency > 0.3,  # Not overfit
            self.bootstrap is None or self.bootstrap["terminal_equity"]["p5"] > 0.8,     # Tail risk ok
            self.dsr is None or self.dsr["is_significant"],               # Survives DSR
        ]
        return all(gates)

    def summary(self) -> str:
        """One-line summary for comparison tables."""
        passed = "PASS" if self.passes_all_gates else "FAIL"
        perm_p = f"p={self.permutation['p_value']:.3f}" if self.permutation else "N/A"
        wfe = f"WFE={self.walk_forward_efficiency:.2f}" if self.walk_forward_efficiency else "N/A"
        return (
            f"[{passed}] {self.strategy_name}/{self.variant}/{self.asset} "
            f"| {self.period} | {self.n_trades}T WR={self.win_rate:.0%} "
            f"PnL={self.total_return_pct:+.1f}% Sharpe={self.sharpe:.2f} "
            f"| {perm_p} {wfe}"
        )


# ---------------------------------------------------------------------------
# Quick validation runner
# ---------------------------------------------------------------------------

def validate_returns(
    returns: np.ndarray,
    strategy_name: str = "strategy",
    variant: str = "default",
    asset: str = "ALL",
    period: str = "OOS",
    n_trials: int = 1,
    n_perms: int = 1000,
    n_boots: int = 1000,
) -> ValidationReport:
    """
    Run the full validation suite on a return series.

    Parameters
    ----------
    returns : array-like
        Per-trade return series.
    strategy_name, variant, asset, period : str
        Labels for the report.
    n_trials : int
        Number of strategy variations tested (for DSR).
    n_perms, n_boots : int
        Permutation and bootstrap sample counts.

    Returns
    -------
    ValidationReport with all validation results populated.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 5:
        return ValidationReport(
            strategy_name=strategy_name, variant=variant, asset=asset,
            period=period, n_trades=n, win_rate=0, total_return_pct=0, sharpe=0,
        )

    win_rate = np.mean(returns > 0)
    total_return = np.sum(returns) * 100
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    dsr_result = deflated_sharpe(
        observed_sharpe=sharpe,
        n_trials=n_trials,
        n_returns=n,
        skewness=float(pd.Series(returns).skew()),
        kurtosis=float(pd.Series(returns).kurtosis() + 3),
    )

    perm_result = permutation_test(returns, n_perms=n_perms)
    boot_result = bootstrap_equity(returns, n_boots=n_boots)

    return ValidationReport(
        strategy_name=strategy_name,
        variant=variant,
        asset=asset,
        period=period,
        n_trades=n,
        win_rate=win_rate,
        total_return_pct=total_return,
        sharpe=sharpe,
        dsr=dsr_result,
        permutation=perm_result,
        bootstrap=boot_result,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Backtest Validation Module — Self Test ===\n")

    # Synthetic winning strategy
    rng = np.random.default_rng(42)
    winning_returns = rng.normal(0.003, 0.02, 200)  # slight positive mean

    report = validate_returns(
        winning_returns,
        strategy_name="test_strategy",
        variant="baseline",
        n_trials=10,
    )
    print(report.summary())
    print(f"  DSR: {report.dsr}")
    print(f"  Permutation: {report.permutation}")
    print(f"  Bootstrap 5th%: {report.bootstrap['terminal_equity']['p5']:.2f}")
    print()

    # Synthetic random strategy (should fail)
    random_returns = rng.normal(0.0, 0.02, 200)  # zero mean
    report2 = validate_returns(
        random_returns,
        strategy_name="random_strategy",
        variant="noise",
        n_trials=100,
    )
    print(report2.summary())
    print(f"  DSR: {report2.dsr}")
    print(f"  Permutation: {report2.permutation}")
    print()
    print("Self-test complete.")
