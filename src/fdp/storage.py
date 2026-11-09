"""SQLite snapshot storage: each demo run replaces its own two input tables."""

import sqlite3
from pathlib import Path

import pandas as pd

from .ingestion import NEWS_COLUMNS, PRICE_COLUMNS


def write_snapshot(path: Path, prices: pd.DataFrame, news: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit transaction makes the pair atomic; executemany avoids pandas' commits.
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS prices (date TEXT, symbol TEXT, close REAL, "
            "PRIMARY KEY(date, symbol))"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS news (id TEXT PRIMARY KEY, published_at TEXT, "
            "symbol TEXT, headline TEXT, source TEXT)"
        )
        db.execute("DELETE FROM prices")
        db.execute("DELETE FROM news")
        rows = prices.copy()
        rows["date"] = rows["date"].dt.strftime("%Y-%m-%d")
        db.executemany(
            "INSERT INTO prices VALUES (?, ?, ?)",
            rows[PRICE_COLUMNS].itertuples(index=False, name=None),
        )
        db.executemany(
            "INSERT INTO news VALUES (?, ?, ?, ?, ?)",
            news[NEWS_COLUMNS].itertuples(index=False, name=None),
        )


def read_snapshot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        prices = pd.read_sql_query(
            "SELECT * FROM prices ORDER BY date, symbol", db, parse_dates=["date"]
        )
        news = pd.read_sql_query("SELECT * FROM news ORDER BY published_at, id", db)
    return prices, news
