"""Self-contained static HTML and build-time PNG charts (no JavaScript)."""

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_report(output: Path, flows: dict, checks: dict, manifest: dict, counts: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for name, frame in flows.items():
        ax.plot(frame["date"], frame["equity"], label=name)
    ax.set(
        title="Synthetic research flows — not historical performance",
        ylabel="Equity / initial capital",
        xlabel="Synthetic date",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / "equity.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    sentiment = flows["Sentiment / price"]
    ax.step(sentiment["date"], sentiment["position"], where="post", label="Long / cash")
    ax.set(
        title="Synthetic sentiment exposure with session loss breaker",
        ylabel="Exposure",
        xlabel="Synthetic date",
        ylim=(-0.05, 1.05),
    )
    ax.grid(alpha=0.2)
    fig.savefig(output / "exposure.png", dpi=140)
    plt.close(fig)
    rows = []
    for name, result in checks.items():
        for gate, passed in result["gates"].items():
            rows.append(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(gate)}</td>"
                f"<td>{'PASS' if passed else 'FAIL'}</td></tr>"
            )
    summary = {"dataset": manifest, "rows": counts, "validation": checks}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, frame in flows.items():
        slug = "rebalancer" if name == "Rebalancer" else "sentiment"
        frame.to_csv(output / f"{slug}.csv", index=False)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Financial Data Platform — synthetic demo</title>
<style>
body {{max-width:1000px;margin:2rem auto;padding:0 1rem;font:17px/1.6 system-ui;color:#17233b;background:#f7f9fc}}
h1,h2 {{line-height:1.2}} .notice {{background:#fff0c8;padding:1rem;border-left:5px solid #9c6e00}}
img {{width:100%;height:auto;background:white}} table {{border-collapse:collapse;width:100%;background:white}}
th,td {{padding:.6rem;text-align:left;border-bottom:1px solid #ccd3df}} code {{font-size:.9em}}
</style></head><body>
<h1>Financial Data Platform</h1><p>Offline ingestion → SQLite → two research flows → validation → this report.</p>
<p class="notice"><strong>All results are synthetic.</strong> These fictional prices and headlines
were independently generated with seed {manifest["seed"]}. Passing code tests or statistical gates
does not establish strategy validity, executable returns or profitability.</p>
<h2>Inputs and reproducibility</h2><p>{counts["prices"]} price rows, {counts["news"]} deduplicated headlines.
Three fictional assets: EQ, BOND and GOLD. Prices use independent zero-drift log increments;
headlines are unrelated to subsequent returns. Run <code>fdp demo</code> to rebuild offline.</p>
<h2>Research flows</h2><p>The rebalancer targets 50% EQ, 40% BOND and 10% GOLD at the first
observed close of each month. Fractional holdings reserve 5 basis points per traded dollar for costs.
The sentiment baseline scores prior-calendar-day headlines and sets EQ long or cash at today's close for the next
close-to-close interval, charging 5 basis points per unit of exposure change. Its loss guard pauses
after three consecutive losses or a 5% cumulative loss of starting equity; no automatic reset occurs.</p>
<img src="equity.png" alt="Synthetic equity paths for the rebalancer and sentiment flow">
<img src="exposure.png" alt="Synthetic sentiment position, either long or cash">
<h2>Validation gates</h2><p>FAIL is a research rejection, not a pipeline error. Both fixed flows use
an assumed two-trial deflated Sharpe adjustment, 499 seeded sign randomizations and 299 block
bootstrap samples. The one-sided sign test assumes a symmetric zero-mean null. The two-trial
assumption describes this demo only; it does not account for arbitrary prior research.</p>
<table><thead><tr><th>Flow</th><th>Gate</th><th>Result</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Artifacts</h2><p><a href="summary.json">Inputs and validation JSON</a> ·
<a href="rebalancer.csv">Rebalancer daily rows</a> · <a href="sentiment.csv">Sentiment daily rows</a></p>
<h2>Scope and limits</h2><p>No broker, orders, account credentials or operational deployment.
This single synthetic path demonstrates engineering behavior. No train/test optimization or
out-of-sample performance claim is made. Costs are simplified; fills, liquidity, taxes, dividends
and overnight execution timing are not modeled realistically. The optional public RSS adapter is
separate from this demo. No external assets, CDN or JavaScript are required.</p>
</body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
