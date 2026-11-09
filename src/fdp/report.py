"""Static synthetic control results and allocation diagnostics; no JavaScript."""

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def allocation_summary(frame):
    return {
        "target_weights": {"EQ": 0.5, "BOND": 0.4, "GOLD": 0.1},
        "final_weights": {s: float(frame[f"weight_{s}"].iloc[-1]) for s in ["EQ", "BOND", "GOLD"]},
        "max_drift_pct_points": float(frame.max_drift.max() * 100),
        "turnover_starting_capital": float(frame.turnover.sum()),
        "cost_starting_capital": float(frame.cost.sum()),
    }


def pause_spans(frame):
    """Exact observed-close spans, including gaps between sessions."""
    spans, start = [], None
    for i, row in frame.iterrows():
        if row.paused and start is None:
            start = row.date
        if start is not None and (not row.paused or i == len(frame) - 1):
            spans.append((start, row.date))
            start = None
    return spans


def write_report(output: Path, flows: dict, checks: dict, manifest: dict, counts: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    for ax, name in zip(axes, ["Planted signal", "Pure noise"], strict=True):
        frame = flows[name]
        ax.plot(frame.date, frame.equity, label="Synthetic net equity")
        for j, (start, end) in enumerate(pause_spans(frame)):
            ax.axvspan(
                start,
                end,
                color="orange",
                alpha=0.3,
                label="Guard paused (cash)" if j == 0 else None,
            )
        ax.set(title=name, ylabel="Equity / initial capital")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.2)
    fig.savefig(output / "equity.png", dpi=140)
    plt.close(fig)
    allocation = allocation_summary(flows["Rebalancer"])
    frame = flows["Rebalancer"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    for symbol, target in allocation["target_weights"].items():
        (line,) = axes[0].plot(frame.date, frame[f"weight_{symbol}"] * 100, label=symbol)
        axes[0].axhline(target * 100, color=line.get_color(), linestyle="--", alpha=0.6)
    axes[0].set(title="Allocation engine: actual weights and dashed targets", ylabel="Weight (%)")
    axes[0].legend()
    axes[1].plot(frame.date, frame.max_drift * 100)
    axes[1].set(title="Largest absolute deviation from target", ylabel="Percentage points")
    fig.savefig(output / "allocation.png", dpi=140)
    plt.close(fig)
    rows = []
    for name, result in checks.items():
        for gate, passed in result["gates"].items():
            rows.append(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(gate)}</td>"
                f"<td>{'PASS' if passed else 'FAIL'}</td></tr>"
            )
    planted = "detected" if checks["Planted signal"]["passes_all_gates"] else "not detected"
    noise = "rejected" if not checks["Pure noise"]["passes_all_gates"] else "not rejected"
    power = manifest["detection_power"]
    signal_summary = (
        f"detected in {power['detections']}/{power['trials']} runs "
        f"({power['detections'] / power['trials']:.0%} empirical power); seed 43 {planted}"
    )
    headline = {
        "Planted signal": signal_summary,
        "Pure noise": noise,
        "Rebalancer": f"holds 50/40/10 within {allocation['max_drift_pct_points']:.2f}% drift",
    }
    summary = {
        "summary": headline,
        "dataset": manifest,
        "rows": counts,
        "validation": checks,
        "allocation": allocation,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, slug in [
        ("Rebalancer", "rebalancer"),
        ("Pure noise", "sentiment"),
        ("Planted signal", "planted"),
    ]:
        flows[name].to_csv(output / f"{slug}.csv", index=False)
    power = manifest["detection_power"]
    power_rows = "".join(
        f"<tr><td>{row['seed']}</td><td>{'PASS' if row['detected'] else 'FAIL'}</td>"
        f"<td>{row['p_value']:.4f}</td></tr>"
        for row in power["runs"]
    )
    weights_rows = "".join(
        f"<tr><td>{s}</td><td>{target:.0%}</td><td>{allocation['final_weights'][s]:.2%}</td></tr>"
        for s, target in allocation["target_weights"].items()
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Financial Data Platform — synthetic controls</title>
<style>body{{max-width:1000px;margin:2rem auto;padding:0 1rem;font:17px/1.6 system-ui;color:#17233b;background:#f7f9fc}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:.6rem;text-align:left;border-bottom:1px solid #ccd3df}}.notice{{background:#fff0c8;padding:1rem}}</style>
</head><body><h1>Financial Data Platform</h1>
<table aria-label="Demo summary"><thead><tr><th>Case</th><th>Outcome</th></tr></thead><tbody>
<tr><td>Planted signal</td><td>{signal_summary}</td></tr><tr><td>Pure noise</td><td>{noise}</td></tr>
<tr><td>Rebalancer</td><td>{headline["Rebalancer"]}</td></tr></tbody></table>
<p class="notice"><strong>All results are synthetic.</strong> These controls test software and statistical detection, not real strategy profitability.</p>
<h2>Planted signal and detection power</h2>
<p>The fixed design adds <strong>+10 basis points</strong> to the next interval's simple return after a positive actionable headline,
and −10 after a negative one, with independent Gaussian noise of <strong>60 bp standard deviation</strong>.
Headlines precede the decision close by a full UTC calendar day; returns accrue over the following interval.
The long/cash flow pays 5 bp per exposure change. Sample: 5,200 business sessions (about 20 synthetic years), seed 43.
The effect and sample length were fixed before evaluation; no seed selection or effect-size search was performed.</p>
<p><strong>Seed 43: {planted}.</strong> The modest signal does not guarantee passing after costs and guard pauses.
Across predeclared seeds 100–119, {power["detections"]}/{power["trials"]} runs pass all four gates
(empirical detection frequency {power["detections"] / power["trials"]:.0%}; only 20 runs, so this is an imprecise power estimate).
The effect was retained despite the primary seed's failure. The full seed ledger is in the JSON artifact.</p>
<details><summary>All 20 predeclared runs: {power["detections"]} pass, {power["trials"] - power["detections"]} fail</summary>
<table><thead><tr><th>Seed</th><th>All four gates</th><th>Sign-test p</th></tr></thead><tbody>{power_rows}</tbody></table></details>
<h2>Noise control and guard behavior</h2><p>The original independent, zero-drift prices and unrelated headlines remain unchanged:
{counts["prices"]} price rows and {counts["news"]} headlines, seed 42. They are {noise} by the all-gate rule.
Both sentiment cases use the same loss guard: three consecutive losses or 5% session loss.
The demo explicitly resets it every five observed sessions; the library still defaults to explicit reset only.
Orange spans label paused cash holdings. Flat segments are intentional; every pause and reset is exported in CSV.</p>
<img src="equity.png" alt="Separate synthetic equity charts; orange spans indicate guard-paused cash exposure">
<h2>Allocation engine</h2><p>Monthly rebalancing targets 50/40/10 across EQ/BOND/GOLD, with fractional shares and funded 5 bp costs.
Maximum observed drift: {allocation["max_drift_pct_points"]:.2f} percentage points (not relative percent).
Turnover: {allocation["turnover_starting_capital"]:.4f} × starting capital, including initial allocation;
costs: {allocation["cost_starting_capital"]:.4%} of starting capital. No profitability gates apply.</p>
<table><thead><tr><th>Asset</th><th>Target</th><th>Final actual</th></tr></thead><tbody>{weights_rows}</tbody></table>
<img src="allocation.png" alt="Actual versus target allocation weights and observed drift">
<h2>Statistical gates</h2><p>Four gates apply only to the two sentiment cases: positive compounded return,
one-sided sign-randomization p below .05, deflated Sharpe probability above .95, bootstrap terminal-equity fifth percentile above .8.
Both use two assumed trials, 499 randomizations and 299 block bootstraps, seed 42.
The sign test assumes a symmetric zero-mean null; short bootstrap blocks and guard dependence limit interpretation.</p>
<table><thead><tr><th>Case</th><th>Gate</th><th>Result</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Artifacts and reproduction</h2><p>Run <code>fdp demo</code> offline.
<a href="summary.json">Summary and full detection ledger</a> · <a href="planted.csv">Planted rows</a> ·
<a href="sentiment.csv">Noise rows</a> · <a href="rebalancer.csv">Allocation rows</a>.</p>
<p>Calendar and costs are simplified; holdings are marked to market without terminal liquidation.
No broker, external assets, JavaScript or live requests are involved.</p></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
