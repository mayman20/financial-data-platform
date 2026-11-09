import numpy as np
import pandas as pd
import pytest

from fdp.circuit_breaker import CircuitBreaker
from fdp.research import compute_rebalance, rebalancer_backtest, sentiment_price_flow
from fdp.sentiment import score_text


# Adapted source rebalancer cases: exact targets, drift direction, cost and shares math.
def test_balanced_portfolio_has_no_trades():
    trades = compute_rebalance(
        {"EQ": 5, "BOND": 8, "GOLD": 1},
        0,
        {"EQ": 10, "BOND": 5, "GOLD": 10},
        {"EQ": 0.5, "BOND": 0.4, "GOLD": 0.1},
    )
    assert sum(abs(t.value_delta) for t in trades) == pytest.approx(0, abs=1e-10)
    assert sum(t.cost for t in trades) == pytest.approx(0, abs=1e-10)


def test_drifted_portfolio_direction_shares_and_costs():
    trades = compute_rebalance(
        {"EQ": 6, "BOND": 6, "GOLD": 1},
        0,
        {"EQ": 10, "BOND": 5, "GOLD": 10},
        {"EQ": 0.5, "BOND": 0.4, "GOLD": 0.1},
        cost_rate=0,
    )
    assert [t.value_delta for t in trades] == pytest.approx([-10, 10, 0])
    for t in trades:
        assert t.shares_delta == pytest.approx(t.value_delta / t.price)


def test_fees_funded_and_targets_match():
    weights = {"EQ": 0.5, "BOND": 0.5}
    trades = compute_rebalance({}, 100, {"EQ": 10, "BOND": 20}, weights, 0.01)
    assert sum(t.value_delta + t.cost for t in trades) == pytest.approx(100)
    assert sum(t.cost for t in trades) == pytest.approx(100 / 1.01 * 0.01)
    for t in trades:
        assert t.cost == pytest.approx(abs(t.value_delta) * 0.01)
        assert t.value_delta == pytest.approx(100 / 1.01 * weights[t.symbol])


@pytest.mark.parametrize(
    "prices,weights",
    [
        ({"EQ": 0}, {"EQ": 1}),
        ({"EQ": 10}, {"EQ": 0.5}),
        ({"EQ": 10}, {"BOND": 1}),
        ({"EQ": np.nan}, {"EQ": 1}),
        ({"EQ": 10}, {"EQ": -1}),
    ],
)
def test_invalid_rebalance_rejected(prices, weights):
    with pytest.raises(ValueError):
        compute_rebalance({}, 100, prices, weights)


def test_rebalancer_hand_calculated_path():
    prices = pd.DataFrame(
        [
            (d, s, p)
            for d, eq in [("2024-01-02", 100), ("2024-01-03", 110)]
            for s, p in [("EQ", eq), ("BOND", 100), ("GOLD", 100)]
        ],
        columns=["date", "symbol", "close"],
    )
    prices["date"] = pd.to_datetime(prices["date"])
    result = rebalancer_backtest(prices, cost_rate=0)
    assert result.equity.tolist() == pytest.approx([1, 1.05])
    assert result.turnover.tolist() == pytest.approx([1, 0])


def test_sentiment_no_same_bar_leakage():
    dates = pd.date_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"date": dates, "symbol": "EQ", "close": [100, 110, 121, 133.1]})
    news = pd.DataFrame(
        {
            "published_at": ["2024-01-02T23:59:00+00:00"],
            "symbol": ["EQ"],
            "headline": ["beats estimates"],
        }
    )
    result = sentiment_price_flow(prices, news, cost_rate=0)
    assert result.position.tolist() == [0, 1, 0, 0]
    assert result.held_position.tolist() == [0, 0, 1, 0]
    assert result["return"].tolist() == pytest.approx([0, 0, 0.1, 0])
    changed = news.copy()
    changed.loc[0, "published_at"] = "2024-01-05T23:59:00+00:00"
    assert sentiment_price_flow(prices, changed, cost_rate=0)["return"].eq(0).all()


@pytest.mark.parametrize(
    "text,direction",
    [
        ("beats estimates", 1),
        ("cuts guidance", -1),
        ("not beats estimates", -1),
        ("annual meeting", 0),
    ],
)
def test_lexicon_direction(text, direction):
    score, confidence = score_text(text)
    assert np.sign(score) == direction
    assert -1 <= score <= 1 and 0 <= confidence <= 1


# Adapted from source circuit-breaker tests; deterministic, no clocks or persistence.
def test_loss_streak_triggers_and_stays_paused():
    breaker = CircuitBreaker(max_session_loss=1)
    for _ in range(2):
        breaker.record_result(-0.01)
    assert breaker.allow_trade()
    breaker.record_result(-0.01)
    assert not breaker.allow_trade()
    assert breaker.paused_reason == "max_consecutive_losses"
    breaker.record_result(0.5)
    assert not breaker.allow_trade()
    breaker.reset()
    assert breaker.allow_trade() and breaker.session_pnl == 0


def test_win_resets_loss_counter():
    breaker = CircuitBreaker()
    for pnl in [-0.001, -0.001, 0.01, -0.001, -0.001]:
        breaker.record_result(pnl)
    assert breaker.allow_trade()
    assert breaker.consecutive_losses == 2


def test_session_loss_triggers():
    breaker = CircuitBreaker(max_consecutive_losses=10, max_session_loss=0.05)
    breaker.record_result(-0.03)
    breaker.record_result(-0.03)
    assert breaker.paused_reason == "max_session_loss"


def test_nonfinite_result_rejected():
    with pytest.raises(ValueError):
        CircuitBreaker().record_result(np.nan)


def test_sentiment_breaker_flattens_next_interval():
    dates = pd.date_range("2024-01-02", periods=7)
    prices = pd.DataFrame({"date": dates, "symbol": "EQ", "close": [100, 100, 99, 98, 97, 96, 95]})
    news = pd.DataFrame(
        {
            "published_at": [date.isoformat() + "+00:00" for date in dates],
            "symbol": "EQ",
            "headline": "beats estimates",
        }
    )
    result = sentiment_price_flow(prices, news, cost_rate=0)
    assert result.held_position.tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert result.paused.tolist() == [False, False, False, False, True, True, True]
    assert result["return"].iloc[5:].eq(0).all()


def test_sentiment_charges_entry_and_exit_costs():
    dates = pd.date_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"date": dates, "symbol": "EQ", "close": [100, 110, 121, 133.1]})
    news = pd.DataFrame(
        {
            "published_at": ["2024-01-02T23:59:00+00:00"],
            "symbol": ["EQ"],
            "headline": ["beats estimates"],
        }
    )
    result = sentiment_price_flow(prices, news, cost_rate=0.001)
    assert result.equity.tolist() == pytest.approx(
        [1, 0.999, 0.999 * 1.1 * 0.999, 0.999 * 1.1 * 0.999]
    )
    assert result.cost.tolist() == pytest.approx([0, 0.001, 0.001, 0])
