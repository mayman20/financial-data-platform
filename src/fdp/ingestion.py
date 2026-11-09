"""Strict local ingestion and an explicitly opted-in public RSS adapter."""

from __future__ import annotations

import hashlib
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from io import StringIO
from urllib.parse import quote

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["date", "symbol", "close"]
NEWS_COLUMNS = ["id", "published_at", "symbol", "headline", "source"]


def parse_prices(text: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text), dtype={"symbol": str})
    if list(frame.columns) != PRICE_COLUMNS or frame.empty:
        raise ValueError("Expected nonempty date,symbol,close CSV")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    if not np.isfinite(frame["close"]).all() or (frame["close"] <= 0).any():
        raise ValueError("Prices must be finite and positive")
    if not frame["symbol"].str.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", na=False).all():
        raise ValueError("Invalid symbol")
    if frame["date"].isna().any() or frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Missing dates or duplicate price keys")
    wide = frame.pivot(index="date", columns="symbol", values="close")
    if wide.isna().any().any():
        raise ValueError("All symbols must have prices on every date")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def parse_rss(payload: bytes, symbol: str, source: str = "rss") -> pd.DataFrame:
    """Normalize and deduplicate RSS items, rejecting incomplete or malformed records.

    Deliberately no raw payload, URL, HTML rendering or account-connected fallback.
    """
    if len(payload) > 1_000_000 or b"<!DOCTYPE" in payload.upper():
        raise ValueError("Oversized RSS or prohibited document type")
    root = ET.fromstring(payload)
    if root.tag != "rss" or root.find("channel") is None:
        raise ValueError("Expected RSS channel")
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol):
        raise ValueError("Invalid symbol")
    rows = []
    seen = set()
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        stamp = item.findtext("pubDate")
        if not title or not stamp:
            raise ValueError("RSS item requires title and publication date")
        dt = parsedate_to_datetime(stamp)
        if dt.tzinfo is None:
            raise ValueError("Publication date requires timezone")
        dt = dt.astimezone(timezone.utc)
        normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", title.lower())).strip()
        key = hashlib.sha256(f"{source}|{symbol}|{dt.date()}|{normalized}".encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        rows.append([key, dt.isoformat(), symbol, title, source])
    return (
        pd.DataFrame(rows, columns=NEWS_COLUMNS).sort_values("published_at").reset_index(drop=True)
    )


def fetch_public_rss(symbol: str, *, allow_network: bool = False) -> pd.DataFrame:
    """Fetch Yahoo public headlines only when the caller explicitly opts in.

    Availability and reuse terms belong to the provider; nothing fetched is bundled.
    Network and parse errors propagate; never substitute synthetic rows.
    """
    if not allow_network:
        raise PermissionError("Public network access requires explicit opt-in")
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol):
        raise ValueError("Invalid symbol")
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(symbol)}&region=US&lang=en-US"
    request = urllib.request.Request(url, headers={"User-Agent": "financial-data-platform/0.1"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = response.read(1_000_001)
    return parse_rss(payload, symbol, "yahoo_rss")
