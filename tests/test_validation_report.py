import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from fdp.cli import demo, main
from fdp.statistics import bootstrap_equity, permutation_test
from fdp.synthetic import generate_fixtures
from fdp.validation import validate_returns


@pytest.mark.parametrize(
    "values", [[0.01] * 5, [0.01] * 30, [np.nan, 0.01] * 30, [-1, 0.01] * 30, [np.inf, -0.01] * 30]
)
def test_invalid_returns_fail_closed(values):
    with pytest.raises(ValueError):
        validate_returns(values)


def test_positive_control_passes_and_negative_control_fails():
    values = np.random.default_rng(42).normal(0.01, 0.001, 100)
    passed = validate_returns(values)
    assert passed["passes_all_gates"]
    failed = validate_returns(-values)
    assert not failed["passes_all_gates"]
    assert not failed["gates"]["positive_return"]
    assert validate_returns(values) == passed


def test_randomization_finite_sample_correction():
    result = permutation_test(np.ones(50), n_perms=99)
    assert result["p_value"] >= 0.01


def test_bootstrap_counts_initial_drawdown():
    result = bootstrap_equity(np.full(10, -0.01), n_boots=10)
    assert result["max_drawdown"]["p50"] == pytest.approx(1 - 0.99**10, abs=1e-4)


def test_fixtures_reproduce_byte_for_byte(tmp_path):
    generate_fixtures(tmp_path)
    fixture = files("fdp").joinpath("fixtures")
    for name in ["prices.csv", "news.xml", "manifest.json"]:
        assert (tmp_path / name).read_bytes() == fixture.joinpath(name).read_bytes()


def test_end_to_end_report_and_repeated_run(tmp_path):
    checks = demo(tmp_path)
    assert set(checks) == {"Pure noise", "Planted signal"}
    page = (tmp_path / "index.html").read_text()
    assert "All results are synthetic" in page
    assert "<script" not in page
    assert "http" not in page
    for name in ["equity.png", "allocation.png"]:
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert f'src="{name}"' in page
    before = (tmp_path / "summary.json").read_bytes()
    summary = json.loads(before)
    assert summary["rows"] == {"prices": 780, "news": 260}
    assert summary["summary"]["Pure noise"] == "rejected"
    assert "12/20" in summary["summary"]["Planted signal"]
    assert "seed 43 not detected" in page
    assert "Rebalancer" not in summary["validation"]
    assert page.index('aria-label="Demo summary"') < page.index("All results are synthetic")
    demo(tmp_path)
    assert (tmp_path / "summary.json").read_bytes() == before
    for name in [
        "rebalancer.csv",
        "sentiment.csv",
        "planted.csv",
        "research.sqlite",
        "planted.sqlite",
    ]:
        assert (tmp_path / name).stat().st_size > 0


def test_cli_demo(tmp_path, capsys):
    assert main(["demo", "--output", str(tmp_path)]) == 0
    assert "Synthetic report" in capsys.readouterr().out


def test_cli_network_requires_flag():
    with pytest.raises(SystemExit) as exc:
        main(["fetch-news", "EQ"])
    assert exc.value.code == 2


@pytest.mark.parametrize("folder", [".", "sentiment", "news_price", "news_price/iterations"])
def test_existing_ml_regressions_in_isolated_process(folder):
    """Preserve their legacy module names while enforcing offline sockets."""
    root = Path(__file__).resolve().parents[1] / "ml-experiments" / folder
    program = """import socket, unittest

def blocked(*args, **kwargs):
    raise AssertionError("Network forbidden")
socket.socket.connect = blocked
socket.create_connection = blocked
suite = unittest.defaultTestLoader.discover(".", pattern="test_*.py")
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0, result.stdout
