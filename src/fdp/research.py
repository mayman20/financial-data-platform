"""Two causal research flows on close prices, with explicit simplified costs."""

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from .circuit_breaker import CircuitBreaker
from .sentiment import score_text


@dataclass
class RebalanceTrade:
    symbol: str
    shares_delta: float
    price: float
    value_delta: float
    cost: float


def compute_rebalance(
    shares: dict[str, float],
    cash: float,
    prices: dict[str, float],
    weights: dict[str, float],
    cost_rate: float = 0.0005,
) -> list[RebalanceTrade]:
    """Target-minus-current dollar trades, allowing fractional shares.

    A scalar solve funds proportional transaction costs from portfolio value,
    preventing negative cash on an initially fully invested allocation.
    """
    if (
        set(prices) != set(weights)
        or set(shares) - set(weights)
        or not np.isclose(sum(weights.values()), 1.0)
        or any(not np.isfinite(w) or w < 0 for w in weights.values())
        or not np.isfinite(cost_rate)
        or not 0 <= cost_rate < 1
        or not np.isfinite(cash)
        or cash < -1e-8
        or any(not np.isfinite(p) or p <= 0 for p in prices.values())
        or any(not np.isfinite(s) or s < 0 for s in shares.values())
    ):
        raise ValueError("Require long-only holdings, complete prices and normalized weights")
    current = {s: shares.get(s, 0.0) * prices[s] for s in weights}
    value = cash + sum(current.values())
    if value <= 0:
        raise ValueError("Portfolio value must be positive")
    low, high = 0.0, value
    for _ in range(70):
        investable = (low + high) / 2
        cost = sum(abs(investable * weights[s] - current[s]) * cost_rate for s in weights)
        if investable + cost > value:
            high = investable
        else:
            low = investable
    investable = low
    return [
        RebalanceTrade(
            s,
            (investable * weights[s] - current[s]) / prices[s],
            prices[s],
            investable * weights[s] - current[s],
            abs(investable * weights[s] - current[s]) * cost_rate,
        )
        for s in weights
    ]


def rebalancer_backtest(prices: pd.DataFrame, cost_rate: float = 0.0005) -> pd.DataFrame:
    """Trade at first observed close each month; hold into the next bar.

    Targets fixed in advance. The first row includes initial allocation costs.
    No financing, taxes, dividends or integer-share constraints are modeled.
    """
    wide = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    weights = {"EQ": 0.5, "BOND": 0.4, "GOLD": 0.1}
    if set(wide.columns) != set(weights):
        raise ValueError("Demo rebalancer expects EQ, BOND, GOLD")
    shares = dict.fromkeys(weights, 0.0)
    cash, previous, month = 1.0, 1.0, None
    rows = []
    for date, row in wide.iterrows():
        turnover, cost = 0.0, 0.0
        if date.to_period("M") != month:
            trades = compute_rebalance(shares, cash, row.to_dict(), weights, cost_rate)
            for trade in trades:
                shares[trade.symbol] += trade.shares_delta
                cash -= trade.value_delta + trade.cost
                cost += trade.cost
                turnover += abs(trade.value_delta)
            month = date.to_period("M")
        value = cash + sum(shares[s] * row[s] for s in weights)
        rows.append((date, value / previous - 1, value, turnover, cost))
        previous = value
    return pd.DataFrame(rows, columns=["date", "return", "equity", "turnover", "cost"])


def sentiment_price_flow(
    prices: pd.DataFrame, news: pd.DataFrame, cost_rate: float = 0.0005
) -> pd.DataFrame:
    """Prior-day UTC news determines the position at today's close.

    A full calendar-day lag conservatively excludes the trading day's headlines.
    Loss breaker persists until explicit reset (end of this run).
    """
    if not np.isfinite(cost_rate) or not 0 <= cost_rate < 1:
        raise ValueError("Invalid cost rate")
    bars = prices[prices["symbol"] == "EQ"].sort_values("date")
    scores = news[news["symbol"] == "EQ"].copy()
    scores["date"] = (
        pd.to_datetime(scores["published_at"], utc=True).dt.tz_localize(None).dt.normalize()
    )
    scores["score"] = scores["headline"].map(lambda text: score_text(text)[0])
    daily = scores.groupby("date")["score"].mean()
    score_by_date = {stamp.date(): score for stamp, score in daily.items()}
    breaker = CircuitBreaker()
    last_price, position, equity, pending_cost = None, 0.0, 1.0, 0.0
    rows = []
    for date, close in bars[["date", "close"]].itertuples(index=False, name=None):
        lagged = score_by_date.get(pd.Timestamp(date).date() - timedelta(days=1), 0.0)
        held = position
        gross = held * (close / last_price - 1) if last_price is not None else 0.0
        # Realize the held interval before choosing exposure for the next one.
        # The guard counts net PnL including the previous close's trading cost.
        breaker.record_result(equity * gross - pending_cost)
        target = float(lagged > 0 and breaker.allow_trade())
        cost = abs(target - held) * cost_rate
        ret = (1 + gross) * (1 - cost) - 1
        pending_cost = equity * (1 + gross) * cost
        equity *= 1 + ret
        rows.append((date, ret, equity, target, held, lagged, cost, not breaker.allow_trade()))
        position, last_price = target, close
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "return",
            "equity",
            "position",
            "held_position",
            "lagged_score",
            "cost",
            "paused",
        ],
    )
