import sqlite3
from importlib.resources import files

import pandas as pd
import pytest

from fdp.ingestion import fetch_public_rss, parse_prices, parse_rss
from fdp.storage import read_snapshot, write_snapshot


@pytest.fixture
def inputs():
    root = files("fdp").joinpath("fixtures")
    return parse_prices(root.joinpath("prices.csv").read_text()), parse_rss(
        root.joinpath("news.xml").read_bytes(), "EQ", "synthetic"
    )


def test_bundled_ingestion(inputs):
    prices, news = inputs
    assert len(prices) == 780
    assert len(news) == 260
    assert prices.groupby("date").size().eq(3).all()
    assert news["id"].is_unique


@pytest.mark.parametrize(
    "data",
    [
        "date,symbol,close\n2024-01-01,EQ,-1\n",
        "date,symbol,close\n2024-01-01,EQ,nan\n",
        "date,symbol,close\n2024-01-01,EQ,inf\n",
        "date,symbol,close\n2024-01-01,EQ,1\n2024-01-01,EQ,1\n",
        "date,symbol,close\n,EQ,1\n",
        "date,symbol,close\n2024-01-01,,1\n",
        "date,symbol,close\n2024-01-01,EQ,1\n2024-01-02,BOND,1\n",
        "date,symbol,wrong\n2024-01-01,EQ,1\n",
    ],
)
def test_invalid_prices_rejected(data):
    with pytest.raises(ValueError):
        parse_prices(data)


def test_rss_deduplication_and_timezone():
    item = "<item><title>Fictional beats estimates!</title><pubDate>Tue, 02 Jan 2024 10:00:00 -0500</pubDate></item>"
    news = parse_rss(f"<rss><channel>{item}{item}</channel></rss>".encode(), "EQ")
    assert len(news) == 1
    assert news.iloc[0].published_at == "2024-01-02T15:00:00+00:00"


@pytest.mark.parametrize(
    "payload",
    [
        b"<rss><channel><item/></channel></rss>",
        b"<html/>",
        b"<!DOCTYPE rss><rss><channel/></rss>",
        b"x" * 1_000_001,
    ],
)
def test_invalid_rss_rejected(payload):
    with pytest.raises(ValueError):
        parse_rss(payload, "EQ")


def test_network_requires_opt_in():
    with pytest.raises(PermissionError):
        fetch_public_rss("EQ")


def test_adapter_opt_in_uses_same_parser(monkeypatch):
    from fdp import ingestion

    payload = b"<rss><channel><item><title>Fixture</title><pubDate>Tue, 02 Jan 2024 00:00:00 GMT</pubDate></item></channel></rss>"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            assert limit == 1_000_001
            return payload

    def fake_open(request, timeout):
        assert request.full_url.startswith("https://feeds.finance.yahoo.com/")
        assert timeout == 15
        return Response()

    monkeypatch.setattr(ingestion.urllib.request, "urlopen", fake_open)
    assert len(fetch_public_rss("EQ", allow_network=True)) == 1


def test_storage_roundtrip_and_idempotent_replace(tmp_path, inputs):
    prices, news = inputs
    path = tmp_path / "input.sqlite"
    write_snapshot(path, prices, news)
    write_snapshot(path, prices, news)
    actual_prices, actual_news = read_snapshot(path)
    pd.testing.assert_frame_equal(actual_prices, prices)
    pd.testing.assert_frame_equal(actual_news, news)


def test_storage_rolls_back_both_tables_on_failure(tmp_path, inputs):
    prices, news = inputs
    path = tmp_path / "input.sqlite"
    write_snapshot(path, prices, news)
    with pytest.raises(sqlite3.IntegrityError):
        write_snapshot(path, prices.iloc[:3], pd.concat([news, news]))
    actual_prices, actual_news = read_snapshot(path)
    pd.testing.assert_frame_equal(actual_prices, prices)
    pd.testing.assert_frame_equal(actual_news, news)
