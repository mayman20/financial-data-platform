"""The lexical release gate must catch constructed fixtures without leaking values."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "sanitize", Path(__file__).resolve().parents[1] / "scripts" / "sanitize.py"
)
sanitize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sanitize)


@pytest.mark.parametrize(
    "sample,category",
    [
        ("/Us" + "ers/example", "personal path"),
        ("/ho" + "me/example", "personal path"),
        ("person" + "@" + "example.invalid", "email"),
        ("0" + "x" + "a" * 40, "wallet"),
        ("-----BEGIN " + "PRIVATE KEY-----", "private key"),
        ("sk" + "-" + "x" * 32, "provider key"),
        ("12345678" + ":" + "x" * 35, "telegram credential"),
        ("api_key = " + repr("x" * 32), "assigned secret"),
        ("chat_id = " + "123456789", "account identifier"),
        ("clai" + "re", "private hostname"),
        ("192" + ".0.2.1", "IP address"),
        ("2001" + ":db8::1", "IP address"),
    ],
)
def test_sensitive_constructed_values_are_redacted(sample, category):
    result = sanitize.scan_text("fixture", sample)
    assert result == [f"fixture:1: {category}"]
    assert all(sample not in finding for finding in result)


def test_python_slice_is_not_an_ip_address():
    assert sanitize.scan_text("fixture", "again = train[::-1]") == []


def test_diff_decorators_are_not_emails_but_real_emails_still_flag():
    assert sanitize.scan_diff("fixture", "+" + "@" + "pytest.fixture") == []
    sample = "person" + "@" + "example.invalid"
    assert sanitize.scan_diff("fixture", "+" + sample) == ["fixture:1: email"]
    assert sanitize.scan_diff("fixture", "-" + sample) == ["fixture:1: email"]
