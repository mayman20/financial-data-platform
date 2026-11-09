"""Statistical helpers adapted from the public validation module."""

import numpy as np
from scipy import stats


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
    sr_obs = observed_sharpe / np.sqrt(annualization_factor)
    euler_gamma = 0.5772156649
    z = (1 - euler_gamma) * stats.norm.ppf(1 - 1 / n_trials) + euler_gamma * stats.norm.ppf(
        1 - 1 / (n_trials * np.e)
    )
    se_null = np.sqrt(1.0 / n_returns)
    expected_max_sr = se_null * z
    se_sharpe = np.sqrt(
        (1 + 0.5 * sr_obs**2 - skewness * sr_obs + (kurtosis - 3) / 4 * sr_obs**2) / n_returns
    )
    if se_sharpe == 0:
        return {"dsr": 0.0, "expected_max_sharpe": expected_max_sr, "is_significant": False}
    test_stat = (sr_obs - expected_max_sr) / se_sharpe
    dsr = stats.norm.cdf(test_stat)
    return {
        "dsr": round(dsr, 4),
        "expected_max_sharpe": round(expected_max_sr, 6),
        "expected_max_sharpe_annualized": round(expected_max_sr * np.sqrt(annualization_factor), 4),
        "is_significant": bool(dsr > 0.95),
        "observed_sharpe": round(observed_sharpe, 4),
        "observed_sr_per_obs": round(sr_obs, 6),
        "n_trials": n_trials,
        "se_sharpe": round(se_sharpe, 6),
    }


def permutation_test(
    returns: np.ndarray, n_perms: int = 1000, metric: str = "sharpe", seed: int = 42
) -> dict:
    """One-sided mean sign-randomization test under a symmetric zero-mean null.

    Seeded sign flips preserve magnitudes. The +1 correction prevents zero
    Monte Carlo p-values. `metric` is retained only for source compatibility;
    mean is the sole supported metric in this packaged subset.
    """
    if metric not in {"mean", "sharpe"}:
        raise ValueError("Only the mean sign-randomization test is implemented")

    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    observed_mean = np.mean(returns)
    perm_results = np.zeros(n_perms)
    for i in range(n_perms):
        signs = rng.choice([-1, 1], size=len(returns))
        perm_results[i] = np.mean(returns * signs)
    p_value = (1 + np.count_nonzero(perm_results >= observed_mean)) / (n_perms + 1)
    return {
        "p_value": round(float(p_value), 4),
        "observed_mean": round(float(observed_mean), 6),
        "perm_mean": round(float(np.mean(perm_results)), 6),
        "perm_std": round(float(np.std(perm_results)), 6),
        "is_significant": bool(p_value < 0.05),
        "n_perms": n_perms,
    }


def bootstrap_equity(trade_returns: np.ndarray, n_boots: int = 1000, seed: int = 42) -> dict:
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
        sampled = []
        while len(sampled) < n:
            start = rng.integers(0, n - block_size + 1)
            sampled.extend(returns[start : start + block_size])
        sampled = np.array(sampled[:n])
        equity = np.cumprod(1 + sampled)
        terminal_equities.append(equity[-1])
        peak = np.maximum.accumulate(np.r_[1.0, equity])[1:]
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
        "bust_probability": round(float(np.mean(te < 0.5)), 4),
        "n_boots": n_boots,
        "n_trades": n,
    }
