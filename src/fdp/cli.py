"""Command-line entry point; the demo never invokes a live adapter."""

import argparse
import json
from importlib.resources import files
from pathlib import Path

from .ingestion import fetch_public_rss, parse_prices, parse_rss
from .report import write_report
from .research import rebalancer_backtest, sentiment_price_flow
from .storage import read_snapshot, write_snapshot
from .validation import validate_returns


def demo(output: Path) -> dict:
    fixture = files("fdp").joinpath("fixtures")
    prices = parse_prices(fixture.joinpath("prices.csv").read_text())
    news = parse_rss(fixture.joinpath("news.xml").read_bytes(), "EQ", "synthetic")
    manifest = json.loads(fixture.joinpath("manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    write_snapshot(output / "research.sqlite", prices, news)
    prices, news = read_snapshot(output / "research.sqlite")
    flows = {
        "Rebalancer": rebalancer_backtest(prices),
        "Sentiment / price": sentiment_price_flow(prices, news),
    }
    checks = {name: validate_returns(frame["return"].to_numpy()) for name, frame in flows.items()}
    write_report(output, flows, checks, manifest, {"prices": len(prices), "news": len(news)})
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo_parser = sub.add_parser("demo", help="Build the bundled synthetic report offline")
    demo_parser.add_argument("--output", type=Path, default=Path("site"))
    live = sub.add_parser(
        "fetch-news", help="Separate optional public RSS fetch (never demo input)"
    )
    live.add_argument("symbol")
    live.add_argument(
        "--allow-network", action="store_true", help="Explicitly permit one public request"
    )
    live.add_argument("--output", type=Path, default=Path("site/public-news.csv"))
    args = parser.parse_args(argv)
    if args.command == "demo":
        checks = demo(args.output)
        print(f"Synthetic report: {args.output / 'index.html'}")
        for name, result in checks.items():
            print(f"{name}: {'PASS' if result['passes_all_gates'] else 'FAIL'} research gates")
    else:
        if not args.allow_network:
            parser.error("fetch-news requires --allow-network")
        news = fetch_public_rss(args.symbol, allow_network=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        news.to_csv(args.output, index=False)
        print(f"Wrote {len(news)} public headlines")
    return 0
