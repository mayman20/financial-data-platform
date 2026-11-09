"""Generate license-clean fictional prices and headlines from a fixed seed."""

import json
import xml.etree.ElementTree as ET
from datetime import timedelta
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


def planted_inputs(seed: int = 43, periods: int = 5200) -> tuple[str, bytes, dict]:
    """Fixed positive-control design: conditional ±10 bps, innovation SD 60 bps.

    Parameters fixed before evaluating detection. Twenty synthetic years provide
    power without increasing effect size. No relation to real asset forecasts.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-02", periods=periods)
    signs = rng.choice([-1, 1], size=periods)
    by_day = {date.date(): sign for date, sign in zip(dates, signs, strict=True)}
    # A headline on t-1 determines exposure at close t and affects t -> t+1.
    decision = np.array([by_day.get(date.date() - timedelta(days=1), 0) for date in dates])
    effect = np.r_[0, decision[:-1]] * 0.001
    innovations = rng.normal(0, 0.006, periods)
    closes = 100 * np.cumprod(1 + innovations + effect)
    prices = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "symbol": "EQ", "close": closes})
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")
    for date, sign in zip(dates, signs, strict=True):
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = (
            "Fictional company beats estimates" if sign > 0 else "Fictional company cuts guidance"
        )
        ET.SubElement(item, "pubDate").text = format_datetime(
            date.tz_localize("UTC").to_pydatetime()
        )
    manifest = {
        "kind": "synthetic planted signal",
        "seed": seed,
        "sessions": periods,
        "conditional_effect_bps": 10,
        "innovation_sd_bps": 60,
        "cost_bps_per_exposure_change": 5,
        "demo_guard_reset_sessions": 5,
        "timing": "prior-day headline -> decision close -> following interval",
        "design": "fixed before detection evaluation; no effect-size search",
    }
    return prices.to_csv(index=False), ET.tostring(rss, encoding="utf-8"), manifest
