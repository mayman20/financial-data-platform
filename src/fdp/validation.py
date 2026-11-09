"""Fail-closed educational validation gates; not strategy certification."""

import numpy as np
from scipy import stats

from .statistics import bootstrap_equity, deflated_sharpe, permutation_test


def validate_returns(returns, *, n_trials: int = 2, seed: int = 42) -> dict:
    values = np.asarray(returns, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 30
        or not np.isfinite(values).all()
        or (values <= -1).any()
        or np.std(values, ddof=1) <= 1e-12
    ):
        raise ValueError("Need at least 30 finite, nonconstant simple returns above -1")
    if not isinstance(n_trials, int) or n_trials < 2:
        raise ValueError("At least two trials required by the deflation approximation")
    sharpe = float(values.mean() / values.std(ddof=1) * np.sqrt(252))
    dsr = deflated_sharpe(
        sharpe,
        n_trials,
        len(values),
        skewness=float(stats.skew(values)),
        kurtosis=float(stats.kurtosis(values, fisher=False)),
    )
    permutation = permutation_test(values, n_perms=499, seed=seed)
    bootstrap = bootstrap_equity(values, n_boots=299, seed=seed)
    total_return = float(np.prod(1 + values) - 1)
    gates = {
        "positive_return": total_return > 0,
        "sign_randomization_p_below_005": permutation["p_value"] < 0.05,
        "deflated_sharpe_above_095": bool(dsr["is_significant"]),
        "bootstrap_p5_above_080": bootstrap["terminal_equity"]["p5"] > 0.8,
    }
    return {
        "observations": len(values),
        "trials_assumed": n_trials,
        "sharpe": sharpe,
        "total_return": total_return,
        "gates": gates,
        "passes_all_gates": all(gates.values()),
        "dsr": dsr,
        "permutation": permutation,
        "bootstrap": bootstrap,
    }
