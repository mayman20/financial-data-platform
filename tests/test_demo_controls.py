import importlib.util
import json
from datetime import timedelta
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fdp.ingestion import parse_prices, parse_rss
from fdp.report import allocation_summary, detection_rates, pause_spans
from fdp.research import rebalancer_backtest, sentiment_price_flow
from fdp.synthetic import planted_inputs
from fdp.validation import validate_returns


def test_planted_timing_and_effect_size_are_exact():
    csv, rss, manifest = planted_inputs(periods=200)
    prices = parse_prices(csv)
    dates = pd.bdate_range("2004-01-02", periods=200)
    rng = np.random.default_rng(43)
    signs = rng.choice([-1, 1], size=200)
    innovations = rng.normal(0, 0.006, 200)
    by_day = {d.date(): sign for d, sign in zip(dates, signs, strict=True)}
    decisions = [by_day.get(d.date() - timedelta(days=1), 0) for d in dates]
    expected = innovations[1:] + np.array(decisions[:-1]) * 0.001
    np.testing.assert_allclose(prices.close.pct_change().iloc[1:], expected, atol=1e-14)
    assert manifest["conditional_effect_bps"] == 10
    assert manifest["innovation_sd_bps"] == 60
    assert len(parse_rss(rss, "EQ")) == 200


def test_illustrative_primary_uses_predeclared_20bp():
    csv, rss, manifest = planted_inputs(effect_bps=20)
    flow = sentiment_price_flow(parse_prices(csv), parse_rss(rss, "EQ"), reset_every=5)
    result = validate_returns(flow["return"])
    assert manifest["seed"] == 43 and manifest["conditional_effect_bps"] == 20
    assert isinstance(result["passes_all_gates"], bool)
    assert flow.guard_reset.sum() == 1039


def test_power_ledger_has_both_gate_directions_and_reproduces_first_seed():
    ledger = json.loads(files("fdp").joinpath("fixtures/detection_power.json").read_text())
    effect = ledger["effects"][2]
    assert [row["seed"] for row in effect["runs"]] == list(range(100, 120))
    assert 0 < effect["detections"] < effect["trials"] == 20
    assert effect["detections"] == sum(all(row["gates"].values()) for row in effect["runs"])
    spec = importlib.util.spec_from_file_location(
        "power", Path(__file__).resolve().parents[1] / "scripts/detection_power.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.evaluate([100])["runs"] == effect["runs"][:1]


def test_demo_reset_does_not_change_library_default():
    dates = pd.date_range("2024-01-02", periods=12)
    prices = pd.DataFrame({"date": dates, "symbol": "EQ", "close": 100 * 0.99 ** np.arange(12)})
    news = pd.DataFrame(
        {
            "published_at": [d.isoformat() + "+00:00" for d in dates],
            "symbol": "EQ",
            "headline": "beats estimates",
        }
    )
    persistent = sentiment_price_flow(prices, news, cost_rate=0)
    resetting = sentiment_price_flow(prices, news, cost_rate=0, reset_every=5)
    assert persistent.paused.iloc[4:].all()
    assert not persistent.guard_reset.any()
    assert resetting.guard_reset.iloc[5] and not resetting.paused.iloc[5]
    assert resetting.held_position.iloc[6] == 1
    assert pause_spans(persistent) == [(dates[4], dates[-1])]


def test_allocation_diagnostics_are_measured_from_holdings():
    root = files("fdp").joinpath("fixtures")
    result = rebalancer_backtest(parse_prices(root.joinpath("prices.csv").read_text()))
    weights = result[["weight_EQ", "weight_BOND", "weight_GOLD"]].to_numpy()
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
    expected = np.abs(weights - [0.5, 0.4, 0.1]).max(axis=1)
    np.testing.assert_allclose(result.max_drift, expected)
    summary = allocation_summary(result)
    assert summary["max_drift_pct_points"] == pytest.approx(expected.max() * 100)
    for _, group in result.groupby(result.date.dt.to_period("M")):
        np.testing.assert_allclose(
            group.iloc[0][["weight_EQ", "weight_BOND", "weight_GOLD"]].astype(float),
            [0.5, 0.4, 0.1],
        )


def test_detection_table_counts_all_runs_and_labels_zero_effect():
    ledger = json.loads(files("fdp").joinpath("fixtures/detection_power.json").read_text())
    rates = detection_rates(ledger)
    assert [row["effect_bps"] for row in rates] == [0, 5, 10, 20, 40]
    assert [row["detected"] for row in rates] == [0, 0, 12, 20, 20]
    assert rates[0]["kind"] == "false-positive rate"
    assert all(row["kind"] == "detection rate" for row in rates[1:])
    for rate, effect in zip(rates, ledger["effects"], strict=True):
        assert rate["trials"] == 20
        assert rate["detected"] == sum(all(run["gates"].values()) for run in effect["runs"])


@pytest.mark.parametrize("defect", ["seeds", "count", "noise", "effects", "gates", "trials"])
def test_detection_table_rejects_incomparable_or_inconsistent_rows(defect):
    ledger = json.loads(files("fdp").joinpath("fixtures/detection_power.json").read_text())
    row = ledger["effects"][0]
    if defect == "seeds":
        row["runs"][0]["seed"] = 43
    elif defect == "count":
        row["detections"] += 1
    elif defect == "noise":
        row["design"]["innovation_sd_bps"] = 120
    elif defect == "effects":
        ledger["effects"].reverse()
    elif defect == "gates":
        row["runs"][0]["gates"].pop("positive_return")
    else:
        row["trials"] = 19
    with pytest.raises(ValueError):
        detection_rates(ledger)


@pytest.mark.parametrize("effect_bps", [0, 5, 10, 20, 40])
def test_effect_grid_preserves_paired_innovations_and_headlines(effect_bps):
    csv, rss, manifest = planted_inputs(seed=100, periods=200, effect_bps=effect_bps)
    zero_csv, zero_rss, _ = planted_inputs(seed=100, periods=200, effect_bps=0)
    prices, zero = parse_prices(csv), parse_prices(zero_csv)
    assert rss == zero_rss
    news = parse_rss(rss, "EQ")
    flow = sentiment_price_flow(zero, news, cost_rate=0)
    # Lagged score at the previous close supplies the return's planted sign;
    # use score, not held_position, because the guard must not affect generation.
    signed_effect = np.sign(flow.lagged_score.shift(1).fillna(0)) * effect_bps / 10_000
    actual_difference = prices.close.pct_change() - zero.close.pct_change()
    np.testing.assert_allclose(actual_difference.iloc[1:], signed_effect.iloc[1:], atol=1e-14)
    assert manifest["sessions"] == 200 and manifest["innovation_sd_bps"] == 60
