"""Generate license-clean fictional prices and headlines from a fixed seed."""

import json
import xml.etree.ElementTree as ET
from email.utils import format_datetime
from pathlib import Path

import numpy as np
import pandas as pd


def generate_fixtures(folder: Path, seed: int = 42) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=260)
    rows = []
    for symbol, volatility in [("EQ", 0.012), ("BOND", 0.004), ("GOLD", 0.009)]:
        closes = 100 * np.exp(np.cumsum(rng.normal(0, volatility, len(dates))))
        rows.extend(
            (date.strftime("%Y-%m-%d"), symbol, round(close, 6))
            for date, close in zip(dates, closes, strict=True)
        )
    pd.DataFrame(rows, columns=["date", "symbol", "close"]).to_csv(
        folder / "prices.csv", index=False
    )
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = "Synthetic EQ headlines"
    headlines = [
        "Fictional company beats estimates",
        "Fictional company cuts guidance",
        "Fictional company holds annual meeting",
        "Fictional company raises outlook",
    ]
    for date in dates:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = headlines[int(rng.integers(len(headlines)))]
        ET.SubElement(item, "pubDate").text = format_datetime(
            date.tz_localize("UTC").to_pydatetime()
        )
    ET.indent(rss)
    ET.ElementTree(rss).write(folder / "news.xml", encoding="utf-8", xml_declaration=True)
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "synthetic",
                "seed": seed,
                "observations_per_asset": len(dates),
                "symbols": ["EQ", "BOND", "GOLD"],
                "generator": "fdp.synthetic.generate_fixtures",
                "price_process": "independent zero-drift Gaussian log increments",
                "headline_process": "independent uniform choice; no relationship to prices",
                "rights": "Generated for this repository; MIT license",
            },
            indent=2,
        )
        + "\n"
    )
