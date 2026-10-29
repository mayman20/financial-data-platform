"""Focused tests for price_experiment.py on tiny synthetic fixtures (no real data).

Model maths / gradients are covered by sentiment_core's existing tests and not repeated.
Run: python -m unittest test_price_experiment
"""

import csv
import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import price_experiment as pe  # noqa: E402

D = dt.date
MONTHS = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()


def news_date(day):
    return f"{MONTHS[day.month - 1]} {day.day:02d}, {day.year}"


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # writes a BOM, like the samples
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def price_rows(prices, high=None, volume=1000):
    return [[d.isoformat(), o, high if high is not None else max(o, c), min(o, c), c, c, volume]
            for d, (o, c) in sorted(prices.items())]


def rec(n, day, text):
    return {"record": n, "date": day, "text": text,
            "id": pe.core.record_id(pe.core.normalize_sentence(text))}


def weekdays(start, count):
    out, d = [], start
    while len(out) < count:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


class PriceExperimentTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_bom_absolute_dates_relative_rejection_and_schema(self):
        p = self.tmp / "news.csv"
        write_csv(p, ["Date", "Headline", "URL"], [
            ["JAN 17, 2024", "Alpha one", "u1"], ["Jan 5, 2024", "Alpha two", "u2"],
            ["14 HOURS AGO", "Alpha three", "u3"], ["2 days ago", "Alpha four", "u4"],
            ["FEB 30, 2024", "Alpha five", "u5"], ["2024-01-17", "Alpha six", "u6"],
            ["JAN 18, 2024", "   ", "u7"],
        ])
        self.assertTrue(p.read_bytes().startswith(b"\xef\xbb\xbf"))
        records, counts = pe.load_headlines(p)
        self.assertEqual([r["date"] for r in records], [D(2024, 1, 17), D(2024, 1, 5)])
        self.assertEqual((counts["rows"], counts["kept_absolute_dated"], counts["relative_date_rejected"],
                          counts["unknown_date_rejected"], counts["empty_headline_rejected"]), (7, 2, 2, 2, 1))
        bad = self.tmp / "bad.csv"
        write_csv(bad, ["Date", "Title", "URL"], [["JAN 17, 2024", "x", "u"]])
        with self.assertRaisesRegex(ValueError, "missing"):
            pe.load_headlines(bad)
        write_csv(bad, ["Date", "Headline", "URL"], [["JAN 17, 2024", "x"]])
        with self.assertRaisesRegex(ValueError, "wrong number of fields"):
            pe.load_headlines(bad)

    def test_strict_next_session_weekend_gap_and_price_validation(self):
        prices = {D(2024, 1, 4): (10.0, 11.0), D(2024, 1, 5): (10.0, 10.0),
                  D(2024, 1, 8): (10.0, 9.0), D(2024, 1, 22): (10.0, 10.5)}
        records = [rec(1, D(2024, 1, 5), "fri"), rec(2, D(2024, 1, 6), "sat"), rec(3, D(2024, 1, 7), "sun"),
                   rec(4, D(2024, 1, 8), "gap fourteen"), rec(5, D(2024, 1, 14), "gap eight"),
                   rec(6, D(2024, 1, 15), "gap seven"), rec(7, D(2024, 1, 22), "last day")]
        samples, counts = pe.align_and_group(records, prices)
        self.assertEqual([s["entry_date"] for s in samples], ["2024-01-08", "2024-01-22"])
        self.assertEqual(samples[0]["n_headlines"], 3)  # Friday news is NOT joined to Friday's row
        self.assertEqual(samples[0]["max_news_date"], "2024-01-07")
        self.assertEqual(samples[0]["label"], "DOWN")
        self.assertEqual((counts["aligned_headlines"], counts["gap_too_large_excluded"],
                          counts["no_future_price_excluded"]), (4, 2, 1))

        p = self.tmp / "prices.csv"
        rows = price_rows(prices)
        write_csv(p, list(pe.PRICE_FIELDS), rows + [rows[0]])  # identical duplicate is tolerated
        self.assertEqual(pe.load_prices(p)[0], prices)
        for bad_rows, msg in ((rows + [[rows[0][0], 10, 12, 9, 12, 12, 5]], "conflicting"),
                              ([rows[0][:1] + ["0"] + rows[0][2:]], "Open"),
                              ([rows[0][:4] + ["nan"] + rows[0][5:]], "Close"),
                              ([rows[0][:4] + ["null"] + rows[0][5:]], "Close"),
                              ([["01/04/2024"] + rows[0][1:]], "ISO")):
            write_csv(p, list(pe.PRICE_FIELDS), bad_rows)
            with self.assertRaisesRegex(ValueError, msg):
                pe.load_prices(p)

    def test_return_label_band_edges(self):
        label = lambda o, c: pe.LABELS[pe.bucket(pe.session_return(o, c))]  # noqa: E731
        self.assertEqual(label(100.0, 100.25), "FLAT")  # exactly +0.25% is inside the band
        self.assertEqual(label(100.0, 99.75), "FLAT")   # exactly -0.25% is inside the band
        self.assertEqual(label(100.0, 100.0), "FLAT")
        self.assertEqual(label(100.0, 100.26), "UP")
        self.assertEqual(label(100.0, 99.74), "DOWN")
        self.assertEqual((pe.bucket(0.0025 + 1e-12), pe.bucket(-0.0025 - 1e-12)), (2, 0))

    def test_grouping_dedup_and_chronological_split_without_overlap(self):
        days = weekdays(D(2024, 1, 1), 40)
        prices = {d: (10.0, 10.0 + (i % 3 - 1) * 0.5) for i, d in enumerate(days)}
        records = [rec(0, days[30], "STORY 0   alpha")]  # later-dated variant appears first in file
        for k, d in enumerate(days[1:], start=1):
            records.append(rec(k, d - dt.timedelta(days=1), f"story {k - 1} alpha"))
            if d.weekday() == 0:
                records.append(rec(100 + k, d - dt.timedelta(days=2), f"weekend {k} beta"))
        unique, dups = pe.dedup_earliest(records)
        self.assertEqual(dups, 1)
        kept = [r for r in unique if r["id"] == records[0]["id"]]
        self.assertEqual([r["date"] for r in kept], [days[0]])  # earliest publication kept

        samples, _ = pe.align_and_group(unique, prices)
        entries = [s["entry_date"] for s in samples]
        self.assertEqual(entries, sorted(set(entries)))
        self.assertTrue(all(s["max_news_date"] < s["entry_date"] for s in samples))
        self.assertTrue(any(s["n_headlines"] == 2 for s in samples))  # Sat + Sun -> Monday group
        split = pe.chronological_split(samples)
        self.assertEqual(sum(map(len, split.values())), len(samples))
        for a, b in (("train", "val"), ("val", "test")):
            self.assertLess(max(samples[i]["exit_date"] for i in split[a]),
                            min(samples[i]["entry_date"] for i in split[b]))
        ids = {k: {h for i in v for h in samples[i]["headline_ids"]} for k, v in split.items()}
        self.assertFalse(ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])

        leaked = [dict(s) for s in samples]
        leaked[split["val"][0]]["headline_ids"] = leaked[split["train"][0]]["headline_ids"]
        with self.assertRaisesRegex(ValueError, "two splits"):
            pe.check_split(leaked, split)
        leaked = [dict(s) for s in samples]
        leaked[0]["max_news_date"] = leaked[0]["entry_date"]
        with self.assertRaisesRegex(ValueError, "max news date"):
            pe.check_split(leaked, split)

    def test_label_alignment_around_split_and_no_future_ohlc_in_features(self):
        days = weekdays(D(2024, 3, 4), 30)
        prices = {d: (10.0, 11.0 if i % 2 else 9.0) for i, d in enumerate(days)}  # alternate UP/DOWN
        records = [rec(k, d, f"note {k} metals") for k, d in enumerate(days[:-1])]
        samples, _ = pe.align_and_group(records, prices)
        split = pe.chronological_split(samples)
        for i in (split["train"][-1], split["val"][0], split["val"][-1], split["test"][0]):
            s = samples[i]
            entry = D.fromisoformat(s["entry_date"])
            self.assertGreater(entry, D.fromisoformat(s["max_news_date"]))
            self.assertEqual(s["y"], pe.bucket(pe.session_return(*prices[entry])))
            # the publication-day row has the opposite direction, so it was not used for the label
            self.assertNotEqual(s["y"], pe.bucket(pe.session_return(*prices[D.fromisoformat(s["max_news_date"])])))

        # Changing every OHLC/volume value changes labels only; text features are identical.
        shifted = {d: (o * 3.0, o * 3.0) for d, (o, _) in prices.items()}
        samples2, _ = pe.align_and_group(records, shifted)
        self.assertEqual([s["text"] for s in samples], [s["text"] for s in samples2])
        self.assertEqual({s["label"] for s in samples2}, {"FLAT"})
        tr = split["train"]
        vec = pe.core.TfidfVectorizer().fit([samples[i]["text"] for i in tr], max_features=800, min_df=2)
        X1, _ = vec.transform([s["text"] for s in samples])
        X2, _ = vec.transform([s["text"] for s in samples2])
        np.testing.assert_array_equal(X1, X2)
        p1, p2 = self.tmp / "p1.csv", self.tmp / "p2.csv"
        write_csv(p1, list(pe.PRICE_FIELDS), price_rows(prices))
        write_csv(p2, list(pe.PRICE_FIELDS), price_rows(prices, high=999.0, volume=7))
        self.assertEqual(pe.load_prices(p1)[0], pe.load_prices(p2)[0])  # High/Low/Adj/Volume unused

    def test_run_refuses_missing_train_class_and_existing_output(self):
        days = weekdays(D(2023, 1, 2), 131)
        write_csv(self.tmp / pe.PRICE_FILE, list(pe.PRICE_FIELDS), price_rows({d: (10.0, 10.0) for d in days}))
        write_csv(self.tmp / pe.NEWS_FILE, list(pe.NEWS_FIELDS),
                  [[news_date(d - dt.timedelta(days=1)), f"secret headline {k} metals", f"u{k}"]
                   for k, d in enumerate(days[1:])])
        out = self.tmp / "run"
        self.assertEqual(pe.run(self.tmp, out), 2)
        diag_text = (out / "diagnostic.json").read_text(encoding="utf-8")
        diag = json.loads(diag_text)
        self.assertEqual(diag["problems"], ["train split lacks class(es) ['DOWN', 'UP']; refusing to train"])
        self.assertEqual(diag["counters"]["alignment"]["entry_date_groups"], 130)
        self.assertNotIn("secret headline", diag_text)
        self.assertFalse((out / "model.npz").exists())
        with self.assertRaises(SystemExit):
            pe.run(self.tmp, out)


if __name__ == "__main__":
    unittest.main()
