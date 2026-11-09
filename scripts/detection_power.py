"""Reproduce the fixed synthetic control experiment; no parameter/seed selection."""

import json
from pathlib import Path

from fdp.ingestion import parse_prices, parse_rss
from fdp.research import sentiment_price_flow
from fdp.synthetic import planted_inputs
from fdp.validation import validate_returns


def evaluate(seeds=range(100, 120)):
    rows = []
    for seed in seeds:
        csv, rss, manifest = planted_inputs(seed)
        prices, news = parse_prices(csv), parse_rss(rss, "EQ", "planted")
        flow = sentiment_price_flow(prices, news, reset_every=5)
        result = validate_returns(flow["return"])
        rows.append(
            {
                "seed": seed,
                "detected": result["passes_all_gates"],
                "gates": result["gates"],
                "p_value": result["permutation"]["p_value"],
            }
        )
    return {
        "design": manifest | {"seed": "each seed listed below"},
        "runs": rows,
        "detections": sum(row["detected"] for row in rows),
        "trials": len(rows),
        "interpretation": "Empirical all-gate detection frequency; 20 runs is an imprecise power estimate. No effect or seed tuning.",
    }


if __name__ == "__main__":
    result = evaluate()
    Path("src/fdp/fixtures/detection_power.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Fixed-design detections: {result['detections']}/{result['trials']}")
