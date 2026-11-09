import importlib.util
import json
from datetime import timedelta
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fdp.ingestion import parse_prices, parse_rss
from fdp.report import allocation_summary, pause_spans
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


def test_fixed_primary_signal_reports_failure_honestly():
    csv, rss, _ = planted_inputs()
    flow = sentiment_price_flow(parse_prices(csv), parse_rss(rss, "EQ"), reset_every=5)
    result = validate_returns(flow["return"])
    assert not result["passes_all_gates"]
    assert not result["gates"]["sign_randomization_p_below_005"]
    assert flow.guard_reset.sum() == 1039


def test_power_ledger_has_both_gate_directions_and_reproduces_first_seed():
    ledger = json.loads(files("fdp").joinpath("fixtures/detection_power.json").read_text())
    assert [row["seed"] for row in ledger["runs"]] == list(range(100, 120))
    assert 0 < ledger["detections"] < ledger["trials"] == 20
    assert ledger["detections"] == sum(all(row["gates"].values()) for row in ledger["runs"])
    spec = importlib.util.spec_from_file_location(
        "power", Path(__file__).resolve().parents[1] / "scripts/detection_power.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.evaluate([100])["runs"] == ledger["runs"][:1]


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
