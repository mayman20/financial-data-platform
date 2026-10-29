"""Headline text -> next observed price session open->close bucket (educational only).

Task: for each AA-associated FNSPID headline with an absolute publication DATE,
take the first daily price row dated STRICTLY AFTER that date (gap <= 7 calendar
days) and label its open->close return DOWN (< -0.25%), UP (> +0.25%) or FLAT
(inclusive band). Headlines sharing that entry date form ONE sample; features
are TF-IDF of headline text only. This is not sentiment, not causality and not
a trading or profit claim.

IMPORTANT: alignment is conservative and DATE-based. Headlines carry only a
calendar date (no time, timezone or ingestion time), so the publication day is
skipped entirely. This is NOT proven point-in-time availability. "Next observed
price session" means the next row of the price file, not a verified
exchange-calendar event.

Data: owner-downloaded pinned FNSPID samples, read locally (no network). Only
aa_headlines.csv and aa_prices.csv are used; aapl_* files are coverage-audited
only, never pooled or trained. Conflicting upstream license statements: local
educational use only. TF-IDF, MLP maths and persistence come unchanged from
sentiment_core.py; labels, metrics and reporting are defined here.
"""

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

import sentiment_core as core

LABELS = ["DOWN", "FLAT", "UP"]  # numeric 0 / 1 / 2
BAND = 0.0025
MAX_GAP_DAYS = 7
MIN_GROUPS, MIN_EVAL_GROUPS, MAX_ERROR_ROWS = 100, 10, 50
NEWS_FILE, PRICE_FILE = "aa_headlines.csv", "aa_prices.csv"
NEWS_FIELDS = ("Date", "Headline", "URL")
PRICE_FIELDS = ("Date", "Open", "High", "Low", "Close", "Adj Close", "Volume")
SOURCE_REPO = "Zdong104/FNSPID_Financial_News_Dataset"
SOURCE_REVISION = "4054842ec476953b30ee874d4b7e8eea786a21fa"
CONFIG = {
    "labels": LABELS, "neutral_band": BAND, "return": "(Close - Open) / Open of the entry row",
    "alignment": "first price date strictly greater than headline date",
    "max_gap_calendar_days": MAX_GAP_DAYS, "grouping": "one sample per entry date",
    "dedup": "case/whitespace-normalised exact headline; earliest (date, file row) kept globally",
    "split": "chronological by unique entry date; train floor(70%), val floor(15%), test rest",
    "max_features": 800, "min_df": 2, "hidden": 16, "lr": 0.003, "l2": 1e-4,
    "batch_size": 64, "max_epochs": 30, "patience": 5, "seed": core.SEED, "nb_alpha": 1.0,
    "selection": "MLP epoch with lowest validation CE; test evaluated once afterwards",
    "min_groups": MIN_GROUPS, "min_val_test_groups": MIN_EVAL_GROUPS,
}
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
ABS_RE = re.compile(r"([A-Za-z]{3}) (\d{1,2}), (\d{4})")
REL_RE = re.compile(r"\d+\s+(SECOND|MINUTE|HOUR|DAY|WEEK|MONTH|YEAR)S?\s+AGO|YESTERDAY|TODAY|JUST NOW",
                    re.IGNORECASE)
ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
LIMITATIONS = [
    "Date-only alignment: no publication time, timezone or ingestion time, so availability "
    "before the entry session open is assumed from the calendar date, not proven.",
    "'Next observed price session' is the next row of the price file, not a verified exchange calendar.",
    "AA-associated publisher sample: historical issuer/ticker continuity and per-headline company "
    "relevance are not verified; many headlines may not be about the company.",
    "Only exact (case/whitespace-normalised) duplicates are removed; near-duplicates and syndicated "
    "copies can still leak across splits.",
    "Single chronological split and single seed on a small sample; a near-chance result is expected "
    "and differences between models may be noise. No causality, trading or profit claim.",
    "Neutral band and all hyperparameters fixed before running; nothing was tuned on test.",
    "Upstream license statements conflict: local educational use only, no commercial deployment.",
    "AAPL samples are coverage-audited only (few dated groups); not pooled and not modelled.",
]


# ----------------------------------------------------------------- data

def parse_news_date(raw):
    """Return ('absolute', date) for exact 'MON D, YYYY'; else ('relative'|'unknown', None)."""
    s = raw.strip()
    m = ABS_RE.fullmatch(s)
    if m and m.group(1).upper() in MONTHS:
        try:
            return "absolute", dt.date(int(m.group(3)), MONTHS[m.group(1).upper()], int(m.group(2)))
        except ValueError:
            return "unknown", None
    return ("relative" if REL_RE.fullmatch(s) else "unknown"), None


def read_csv(path, required):
    """UTF-8 (BOM tolerated) CSV rows as (record_number, dict); fails clearly on schema problems."""
    name = Path(path).name
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in required if c not in header]
        if missing or len(set(header)) != len(header):
            raise ValueError(f"{name}: bad header (missing {missing}, {len(header)} columns, "
                             f"duplicates {len(header) != len(set(header))}); required {list(required)}")
        rows = []
        for n, row in enumerate(reader, start=1):
            if None in row or any(row[c] is None for c in required):
                raise ValueError(f"{name}: data record {n} has the wrong number of fields")
            rows.append((n, row))
    return rows


def load_headlines(path):
    counts = {"rows": 0, "kept_absolute_dated": 0, "relative_date_rejected": 0,
              "unknown_date_rejected": 0, "empty_headline_rejected": 0}
    records = []
    for n, row in read_csv(path, NEWS_FIELDS):
        counts["rows"] += 1
        kind, day = parse_news_date(row["Date"])
        if kind != "absolute":
            counts[kind + "_date_rejected"] += 1
            continue
        text = row["Headline"].strip()
        if not text:
            counts["empty_headline_rejected"] += 1
            continue
        counts["kept_absolute_dated"] += 1
        records.append({"record": n, "date": day, "text": text,
                        "id": core.record_id(core.normalize_sentence(text))})
    if records:
        counts["absolute_date_range"] = [min(r["date"] for r in records).isoformat(),
                                         max(r["date"] for r in records).isoformat()]
    return records, counts


def load_prices(path):
    """{date: (open, close)}; finite positive floats, ISO dates, conflicting duplicates fail."""
    name, prices, identical_dups = Path(path).name, {}, 0
    rows = read_csv(path, PRICE_FIELDS)
    for n, row in rows:
        raw = row["Date"].strip()
        try:
            day = dt.date.fromisoformat(raw) if ISO_RE.fullmatch(raw) else None
        except ValueError:
            day = None
        if day is None:
            raise ValueError(f"{name}: data record {n} Date is not a valid ISO YYYY-MM-DD date")
        vals = []
        for col in ("Open", "Close"):
            try:
                v = float(row[col])
            except ValueError:
                v = float("nan")
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name}: data record {n} {col} is not a finite positive number")
            vals.append(v)
        if day in prices:
            if prices[day] != tuple(vals):
                raise ValueError(f"{name}: conflicting duplicate rows for {day.isoformat()}")
            identical_dups += 1
            continue
        prices[day] = tuple(vals)
    if not prices:
        raise ValueError(f"{name}: no price rows")
    return prices, {"rows": len(rows), "unique_dates": len(prices),
                    "identical_duplicate_rows_dropped": identical_dups,
                    "date_range": [min(prices).isoformat(), max(prices).isoformat()]}


def session_return(open_, close):
    return (close - open_) / open_


def bucket(ret):
    return 0 if ret < -BAND else 2 if ret > BAND else 1  # band edges are FLAT


def dedup_earliest(records):
    seen, kept = set(), []
    for r in sorted(records, key=lambda r: (r["date"], r["record"])):
        if r["id"] not in seen:
            seen.add(r["id"])
            kept.append(r)
    return kept, len(records) - len(kept)


def align_and_group(records, prices, max_gap_days=MAX_GAP_DAYS):
    """One sample per entry date; label from that row only, text features only."""
    days = sorted(prices)
    groups = {}
    counts = {"aligned_headlines": 0, "no_future_price_excluded": 0, "gap_too_large_excluded": 0}
    for r in records:
        j = bisect.bisect_right(days, r["date"])  # strictly after: same-day row never chosen
        if j == len(days):
            counts["no_future_price_excluded"] += 1
        elif (days[j] - r["date"]).days > max_gap_days:
            counts["gap_too_large_excluded"] += 1
        else:
            counts["aligned_headlines"] += 1
            groups.setdefault(days[j], []).append(r)
    samples = []
    for entry in sorted(groups):
        members = sorted(groups[entry], key=lambda r: (r["date"], r["record"]))
        max_news = max(r["date"] for r in members)
        if not max_news < entry:
            raise ValueError("leakage check failed: feature date not before entry date")
        ret = session_return(*prices[entry])
        ids = [r["id"] for r in members]
        samples.append({
            "sample_id": hashlib.sha256((entry.isoformat() + "|" + ",".join(sorted(ids))).encode()).hexdigest()[:16],
            "entry_date": entry.isoformat(), "exit_date": entry.isoformat(),
            "min_news_date": members[0]["date"].isoformat(), "max_news_date": max_news.isoformat(),
            "n_headlines": len(members), "headline_ids": ids,
            "open_to_close_return": ret, "y": bucket(ret), "label": LABELS[bucket(ret)],
            "text": " ".join(r["text"] for r in members),
        })
    counts["entry_date_groups"] = len(samples)
    return samples, counts


def check_split(samples, split):
    entries = [s["entry_date"] for s in samples]
    if entries != sorted(set(entries)):
        raise ValueError("samples must be sorted by unique entry date")
    if any(not s["max_news_date"] < s["entry_date"] for s in samples):
        raise ValueError("leakage check failed: max news date not before entry date")
    for a, b in (("train", "val"), ("val", "test")):
        if split[a] and split[b] and not (max(samples[i]["exit_date"] for i in split[a])
                                          < min(samples[i]["entry_date"] for i in split[b])):
            raise ValueError(f"leakage check failed: {a} label dates reach into {b}")
    owner = {}
    for name, idx in split.items():
        for i in idx:
            for h in samples[i]["headline_ids"]:
                if owner.setdefault(h, name) != name:
                    raise ValueError("leakage check failed: headline identity in two splits")


def chronological_split(samples):
    n = len(samples)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    split = {"train": list(range(n_tr)), "val": list(range(n_tr, n_tr + n_va)),
             "test": list(range(n_tr + n_va, n))}
    check_split(samples, split)
    return split


# -------------------------------------------------------------- metrics

def classification_metrics(y_true, y_pred):
    """3-class metrics; a class with no support or no predictions gets precision/recall/F1 0."""
    k = len(LABELS)
    conf = np.zeros((k, k), dtype=int)  # rows true, cols predicted
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    per_class, f1s = {}, []
    for c, name in enumerate(LABELS):
        tp, pred_n, support = conf[c, c], conf[:, c].sum(), conf[c, :].sum()
        prec = tp / pred_n if pred_n else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[name] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                           "support": int(support)}
        f1s.append(f1)
    return {"accuracy": float(np.trace(conf) / max(len(y_true), 1)), "macro_f1": float(np.mean(f1s)),
            "per_class": per_class, "confusion_rows_true_cols_pred": conf.tolist(),
            "label_order": LABELS, "classes_without_support": [n for n in LABELS if not per_class[n]["support"]]}


def label_counts(y):
    c = np.bincount(np.asarray(y, dtype=int), minlength=len(LABELS))
    return {name: int(c[i]) for i, name in enumerate(LABELS)}


# ------------------------------------------------------------------ run

def provenance(data_dir, used):
    prov_path = data_dir / "DATA_PROVENANCE.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else None
    blob = json.dumps(prov).lower() if prov is not None else ""
    files = {}
    for p in used:
        h = core.file_sha256(p)
        files[p.name] = {"sha256": h, "sha256_listed_in_DATA_PROVENANCE": (h in blob) if prov is not None else None}
    return {"source_repository": SOURCE_REPO, "source_revision": SOURCE_REVISION, "files_used": files,
            "sentiment_core_sha256": core.file_sha256(core.__file__),
            "price_experiment_sha256": core.file_sha256(__file__),
            "DATA_PROVENANCE.json": prov if prov is not None else "not found"}


def audit_aapl(data_dir):
    news, prices = data_dir / "aapl_headlines.csv", data_dir / "aapl_prices.csv"
    if not news.exists():
        return {"status": f"{news.name} not found"}
    try:
        records, counts = load_headlines(news)
        out = {"status": "coverage audit only; NOT pooled, NOT trained", **counts,
               "distinct_absolute_dates": len({r["date"] for r in records})}
        if prices.exists():
            out["aligned_entry_date_groups"] = len(align_and_group(dedup_earliest(records)[0],
                                                                   load_prices(prices)[0])[0])
        return out
    except ValueError as e:
        return {"status": f"audit failed: {e}"}


def split_summary(samples, split):
    def part(idx):
        s = [samples[i] for i in idx]
        return {"groups": len(s), "headlines": sum(x["n_headlines"] for x in s),
                "entry_dates": [s[0]["entry_date"], s[-1]["entry_date"]] if s else None,
                "news_dates": [min(x["min_news_date"] for x in s), max(x["max_news_date"] for x in s)] if s else None,
                "labels": label_counts([x["y"] for x in s])}
    return {k: part(v) for k, v in split.items()}


def run(data_dir, output_dir):
    data_dir, out = Path(data_dir), Path(output_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing path: {out} (choose a new --output)")
    news_path, price_path = data_dir / NEWS_FILE, data_dir / PRICE_FILE
    records, news_counts = load_headlines(news_path)
    prices, price_counts = load_prices(price_path)
    unique, dups = dedup_earliest(records)
    samples, align_counts = align_and_group(unique, prices)
    split = chronological_split(samples)
    y = np.array([s["y"] for s in samples], dtype=int)
    tr, va, te = (np.array(split[k], dtype=int) for k in ("train", "val", "test"))
    report = {"task": "headline text -> next observed price session open->close bucket",
              "IMPORTANT": LIMITATIONS[0], "config": CONFIG, "provenance": provenance(data_dir, [news_path, price_path]),
              "counters": {"headlines": news_counts, "duplicate_headlines_removed": dups,
                           "alignment": align_counts, "prices": price_counts},
              "split": split_summary(samples, split), "aapl_audit": audit_aapl(data_dir),
              "limitations": LIMITATIONS}

    problems = []
    if len(samples) < MIN_GROUPS:
        problems.append(f"only {len(samples)} entry-date groups (< {MIN_GROUPS})")
    if min(len(va), len(te)) < MIN_EVAL_GROUPS:
        problems.append(f"val/test groups {len(va)}/{len(te)} (< {MIN_EVAL_GROUPS} each)")
    missing = [n for n, c in label_counts(y[tr]).items() if c == 0]
    if missing:
        problems.append(f"train split lacks class(es) {missing}; refusing to train")
    if not problems:
        texts = [s["text"] for s in samples]
        vec = core.TfidfVectorizer().fit([texts[i] for i in tr], max_features=CONFIG["max_features"],
                                         min_df=CONFIG["min_df"])  # train text only
        if not vec.vocab:
            problems.append("empty training vocabulary")
    out.mkdir(parents=True, exist_ok=False)
    if problems:
        core.write_json(out / "diagnostic.json", {"status": "refused before training", "problems": problems, **report})
        print("refused before training: " + "; ".join(problems))
        return 2

    (Xtr, st_tr), (Xva, st_va), (Xte, st_te) = (vec.transform([texts[i] for i in idx]) for idx in (tr, va, te))
    ytr, yva, yte = y[tr], y[va], y[te]
    maj = core.fit_majority(ytr)
    nb = core.fit_naive_bayes(Xtr, ytr, alpha=CONFIG["nb_alpha"])
    mlp, history, best_epoch = core.train_mlp(
        Xtr, ytr, Xva, yva, hidden=CONFIG["hidden"], lr=CONFIG["lr"], max_epochs=CONFIG["max_epochs"],
        patience=CONFIG["patience"], batch_size=CONFIG["batch_size"], l2=CONFIG["l2"], seed=CONFIG["seed"])
    evaluated = {}
    for split_name, X, yy in (("validation", Xva, yva), ("test", Xte, yte)):  # test once, after selection
        evaluated[split_name] = {}
        for name, probs in (("majority_train", core.predict_majority(maj, len(yy))),
                            ("naive_bayes", core.predict_nb_proba(nb, X)), ("mlp", core.mlp_proba(mlp, X))):
            core.check_finite(f"{name} {split_name} probs", probs)
            evaluated[split_name][name] = {**classification_metrics(yy, probs.argmax(1)),
                                           "cross_entropy": core.prob_cross_entropy(probs, yy), "_probs": probs}
    test_probs = evaluated["test"]["mlp"]["_probs"]
    for block in evaluated.values():
        for m in block.values():
            del m["_probs"]

    report.update({"status": "completed",
                   "features": {"vocab_size": len(vec.vocab),
                                "unknown_tokens": {"train": st_tr, "val": st_va, "test": st_te}},
                   "model_selection": {"best_epoch": best_epoch, "epochs_run": len(history),
                                       "best_val_ce": history[best_epoch - 1]["val_ce"]},
                   **evaluated})
    core.save_model(mlp, out)
    core.write_json(out / "vectorizer.json", {"vocab": vec.vocab, "idf": [float(v) for v in vec.idf],
                                              "tokenizer": core.TOKEN_RE.pattern, "labels": LABELS})
    core.write_json(out / "config.json", CONFIG)
    core.write_json(out / "results.json", report)
    core.write_json(out / "samples.json", {"note": "metadata only; no headline text", "samples": [
        {**{k: v for k, v in s.items() if k != "text"}, "split": name}
        for name, idx in split.items() for s in (samples[i] for i in idx)]})
    core.write_history_csv(out / "history.csv", history)
    core.write_loss_svg(out / "loss_curve.svg", history, best_epoch)
    pred = test_probs.argmax(1)
    wrong = [j for j in range(len(yte)) if pred[j] != yte[j]]
    core.write_json(out / "error_analysis.json", {
        "model": "mlp", "total_test_errors": len(wrong), "shown": min(len(wrong), MAX_ERROR_ROWS),
        "errors": [{"sample_id": samples[te[j]]["sample_id"], "entry_date": samples[te[j]]["entry_date"],
                    "max_news_date": samples[te[j]]["max_news_date"], "n_headlines": samples[te[j]]["n_headlines"],
                    "true": LABELS[yte[j]], "predicted": LABELS[pred[j]],
                    "probabilities": {n: float(test_probs[j, i]) for i, n in enumerate(LABELS)}}
                   for j in wrong[:MAX_ERROR_ROWS]]})

    print(f"groups {len(samples)}  train/val/test {len(tr)}/{len(va)}/{len(te)}  best epoch {best_epoch}")
    for name, r in evaluated["test"].items():
        print(f"test {name:15s} acc {r['accuracy']:.3f}  macro_f1 {r['macro_f1']:.3f}  ce {r['cross_entropy']:.4f}")
    print(f"artifacts written to {out}")
    return 0


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(here))
    ap.add_argument("--output", default=str(here / "price_run"), help="must not exist")
    ap.add_argument("--predict", metavar="TEXT", help="bucket one headline text with a trained run")
    ap.add_argument("--model-dir", help="run directory with model.npz and vectorizer.json")
    args = ap.parse_args(argv)
    if args.predict is not None:
        if not args.model_dir:
            ap.error("--predict requires --model-dir")
        vec_dict = json.loads((Path(args.model_dir) / "vectorizer.json").read_text(encoding="utf-8"))
        if vec_dict.get("labels") != LABELS:
            ap.error("vectorizer.json is not from this experiment")
        X, _ = core.TfidfVectorizer.from_dict(vec_dict).transform([args.predict])
        p = core.mlp_proba(core.load_model(args.model_dir), X)[0]
        print(json.dumps({"label": LABELS[int(p.argmax())],
                          "probabilities": {n: float(p[i]) for i, n in enumerate(LABELS)}}, indent=2))
        return 0
    try:
        return run(args.data_dir, args.output)
    except (ValueError, FloatingPointError) as e:
        print(f"diagnostic abort: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
