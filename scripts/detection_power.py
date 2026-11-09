"""Reproduce the fixed synthetic control experiment; no parameter/seed selection."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from fdp.ingestion import parse_prices, parse_rss
from fdp.research import sentiment_price_flow
from fdp.synthetic import planted_inputs
from fdp.validation import validate_returns

EFFECT_SIZES = (0, 5, 10, 20, 40)


def evaluate(seeds=range(100, 120), effect_bps=10):
    rows = []
    for seed in seeds:
        csv, rss, manifest = planted_inputs(seed, effect_bps=effect_bps)
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
        "effect_bps": effect_bps,
        "design": manifest | {"seed": "each seed listed below"},
        "runs": rows,
        "detections": sum(row["detected"] for row in rows),
        "trials": len(rows),
        "interpretation": "Empirical all-gate detection frequency; 20 runs is an imprecise power estimate. Predeclared effect grid; no winning-seed selection.",
    }


def evaluate_effect(effect_bps):
    return evaluate(effect_bps=effect_bps)


def evaluate_grid():
    # Finite offline numerical work; three processes bound resource consumption.
    with ProcessPoolExecutor(max_workers=3) as pool:
        effects = list(pool.map(evaluate_effect, EFFECT_SIZES))
    return {
        "effects": effects,
        "seeds": list(range(100, 120)),
        "interpretation": "0 bp estimates false positives; other rows estimate detection power. Twenty trials per row are imprecise. Paired innovations/headlines across effects.",
    }


if __name__ == "__main__":
    result = evaluate_grid()
    Path("src/fdp/fixtures/detection_power.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in result["effects"]:
        print(f"{row['effect_bps']} bp: {row['detections']}/{row['trials']}", flush=True)
