"""Focused toy-data tests for iterate.py (no real data). Run: python -m unittest test_iterate -v"""

import datetime as dt
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import iterate as it
import price_experiment as pe
import sentiment_core as core

HELD_OUT = "zzheldout"  # marker word that appears only in test-period toy headlines


def numeric_grad(f, params, key, eps=1e-6):
    g = np.zeros_like(params[key])
    for idx in np.ndindex(g.shape):
        old = params[key][idx]
        params[key][idx] = old + eps
        up = f()
        params[key][idx] = old - eps
        down = f()
        params[key][idx] = old
        g[idx] = (up - down) / (2 * eps)
    return g


def toy_samples():
    """40 grouped toy samples built with the original pipeline functions (28/6/6 chronological split)."""
    phrases = {0: "shares fell after weak demand", 1: "board meeting scheduled", 2: "profit rose on strong sales"}
    closes = {0: 98.0, 1: 100.1, 2: 102.0}
    records, prices = [], {}
    for g in range(40):
        news_day = dt.date(2020, 1, 1) + dt.timedelta(days=2 * g)
        prices[news_day + dt.timedelta(days=1)] = (100.0, closes[g % 3])
        extra = HELD_OUT if g >= 34 else f"item{g % 5}"
        for text in (f"{phrases[g % 3]} {extra}", f"update {g} {extra} note"):
            records.append({"record": len(records) + 1, "date": news_day, "text": text,
                            "id": core.record_id(core.normalize_sentence(text))})
    unique, _ = pe.dedup_earliest(records)
    samples, _ = pe.align_and_group(unique, prices)
    split = pe.chronological_split(samples)
    return samples, split, {r["id"]: r["text"] for r in unique}


def toy_parts():
    parts, _, _ = it.train_val_only(*toy_samples())
    return parts


class GradientTests(unittest.TestCase):
    def test_weighted_mlp_and_linear_match_finite_differences(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(7, 5))
        y = np.array([0, 1, 2, 1, 0, 2, 2])
        w = np.array([0.5, 3.0, 1.0, 3.0, 0.5, 1.2, 1.2])
        l2 = 0.05
        cases = [(core.init_mlp(5, 4, seed=1), it.weighted_mlp_loss_and_grads, core.mlp_logits),
                 (it.init_linear(5), it.linear_loss_and_grads, it.linear_logits)]
        for params, fn, logits_fn in cases:
            params = {k: p + rng.normal(scale=0.3, size=p.shape) for k, p in params.items()}  # non-zero biases
            loss, grads = fn(params, X, y, w, l2)
            for key in params:
                num = numeric_grad(lambda: fn(params, X, y, w, l2)[0], params, key)
                np.testing.assert_allclose(grads[key], num, rtol=1e-5, atol=1e-8, err_msg=key)
            # loss is sum(w_i CE_i) / sum(w_i) + L2 on weights only, and invariant to rescaling all weights
            ce = -core.log_softmax(logits_fn(params, X))[np.arange(len(y)), y]
            penalty = 0.5 * l2 * sum(np.sum(p ** 2) for k, p in params.items() if k.startswith("W"))
            self.assertAlmostEqual(loss, float(np.sum(w * ce) / np.sum(w) + penalty), places=12)
            loss10, grads10 = fn(params, X, y, 10 * w, l2)
            self.assertAlmostEqual(loss, loss10, places=12)
            for key in params:
                np.testing.assert_allclose(grads[key], grads10[key], rtol=1e-12, atol=1e-15)


class ClassWeightTests(unittest.TestCase):
    def test_train_only_weights_and_equal_weights_match_core(self):
        ytr = np.array([0, 0, 0, 0, 1, 2, 2, 2])
        w = it.class_weights(ytr)
        np.testing.assert_allclose(w, [8 / 12, 8 / 3, 8 / 9])
        np.testing.assert_allclose(np.bincount(ytr, weights=w[ytr]), [8 / 3] * 3)  # equal total per class
        self.assertEqual(list(inspect.signature(it.class_weights).parameters), ["ytr"])
        with self.assertRaises(ValueError):
            it.class_weights(np.array([0, 2, 2]))

        rng = np.random.default_rng(3)
        Xtr, y20 = rng.random((20, 6)), rng.integers(0, 3, 20)
        Xva, yva = rng.random((8, 6)), rng.integers(0, 3, 8)
        p = core.init_mlp(6, 4, seed=42)
        loss_core, g_core = core.mlp_loss_and_grads(p, Xtr, y20, l2=1e-3)
        loss_w, g_w = it.weighted_mlp_loss_and_grads(p, Xtr, y20, np.ones(20), 1e-3)
        self.assertAlmostEqual(loss_core, loss_w, places=12)
        for k in p:
            np.testing.assert_allclose(g_core[k], g_w[k], rtol=1e-12, atol=1e-15)

        settings = dict(lr=0.003, max_epochs=8, patience=5, batch_size=6, l2=1e-4, seed=42)
        ref, hist_ref, best_ref = core.train_mlp(Xtr, y20, Xva, yva, hidden=4, **settings)
        got, hist, best = it.train_adam(core.init_mlp(6, 4, 42), it.weighted_mlp_loss_and_grads, core.mlp_logits,
                                        Xtr, y20, Xva, yva, sample_weight=np.ones(20), **settings)
        self.assertEqual((best, len(hist)), (best_ref, len(hist_ref)))
        for a, b in zip(hist, hist_ref):
            for key in b:
                self.assertAlmostEqual(a[key], b[key], places=10)
            self.assertAlmostEqual(a["train_weighted_ce"], a["train_ce"], places=12)  # weighted column kept separate
        for k in ref:
            np.testing.assert_allclose(got[k], ref[k], rtol=1e-10, atol=1e-13)


class BigramTests(unittest.TestCase):
    def test_bigrams_keep_order_stay_in_headline_and_vocab_is_train_only(self):
        self.assertEqual(it.ngram_features("Profit rose 5%"),
                         (["profit", "rose", "<num>", "%"], ["profit rose", "rose <num>", "<num> %"]))
        self.assertEqual(it.ngram_features(["profit", "rose"]), (["profit", "rose"], []))  # no cross-headline pair

        train = [["profit rose", "shares fell"], ["profit rose again"], ["shares fell"], ["rose profit"]]
        vec = it.NgramTfidfVectorizer().fit(train, max_features=800, min_df=2)
        self.assertEqual(vec.vocab, ["fell", "profit", "profit rose", "rose", "shares", "shares fell"])
        self.assertEqual(it.feature_counts(vec), {"unigrams": 4, "bigrams": 2})
        self.assertTrue(all(f.count(it.BIGRAM_SEP) <= 1 for f in vec.vocab))

        X, stats = vec.transform(["profit rose", "rose profit", ["profit", "rose"], "zzval words"])
        j = vec.index["profit rose"]
        self.assertGreater(X[0, j], 0)
        self.assertEqual((X[1, j], X[2, j]), (0.0, 0.0))
        self.assertFalse(np.allclose(X[0], X[1]))  # word order changes the vector
        np.testing.assert_allclose(X[1], X[2])      # same unigrams, no order signal
        self.assertFalse(X[3].any())
        self.assertEqual(stats["unknown_tokens"], 2)

        # vocabulary comes from training docs only, is order-independent, and the cap is shared
        val = [["zzval words"], ["zzval words"]]
        self.assertFalse(any("zzval" in f for f in vec.vocab))
        self.assertTrue(any("zzval" in f for f in it.NgramTfidfVectorizer().fit(train + val, 800, 2).vocab))
        again = it.NgramTfidfVectorizer().fit(train[::-1], max_features=800, min_df=2)
        self.assertEqual(again.vocab, vec.vocab)
        np.testing.assert_array_equal(again.idf, vec.idf)
        capped = it.NgramTfidfVectorizer().fit(train, max_features=4, min_df=1)
        self.assertEqual(capped.vocab, ["fell", "profit", "profit rose", "rose"])
        self.assertEqual(it.feature_counts(capped), {"unigrams": 3, "bigrams": 1})


class PipelineTests(unittest.TestCase):
    def test_only_train_and_val_reach_candidate_fit_and_eval(self):
        samples, split, texts = toy_samples()
        self.assertTrue(all(HELD_OUT in samples[i]["text"] for i in split["test"]))  # marker is real
        parts, summary, held_out = it.train_val_only(samples, split, texts)
        self.assertEqual((set(parts), set(summary)), ({"train", "val"}, {"train", "val"}))
        self.assertEqual(held_out["groups"], len(split["test"]))
        self.assertNotIn(HELD_OUT, json.dumps([parts, summary, held_out]))
        self.assertEqual(list(inspect.signature(it.fit_candidate).parameters), ["name", "train", "val"])
        self.assertEqual(list(inspect.signature(it.evaluate_candidate).parameters), ["fitted", "val"])

        seen, fit_calls, eval_calls = [], [], []

        def spy(orig):
            def wrapper(self, docs, *args, **kwargs):
                seen.append(docs)
                return orig(self, docs, *args, **kwargs)
            return wrapper

        def recording_fit(*args, **kwargs):
            fit_calls.append((args, kwargs))
            return it.fit_candidate(*args, **kwargs)

        def recording_eval(*args, **kwargs):
            eval_calls.append((args, kwargs))
            return it.evaluate_candidate(*args, **kwargs)

        baselines = it.load_context()["baseline_validation"]
        patches = [mock.patch.object(cls, meth, spy(getattr(cls, meth)))
                   for cls in (core.TfidfVectorizer, it.NgramTfidfVectorizer) for meth in ("fit", "transform")]
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.print"):
            for p in patches:
                p.start()
            try:
                results = it.run_candidates(parts["train"], parts["val"], Path(tmp), baselines,
                                            fit=recording_fit, evaluate=recording_eval)
            finally:
                for p in patches:
                    p.stop()
            saved = sorted(p.name for p in Path(tmp).iterdir())

        self.assertEqual(results["test_evaluations"], 0)
        self.assertEqual(saved, sorted(it.ORDER))
        self.assertEqual([a[0] for a, _ in fit_calls], it.ORDER)
        for args, kwargs in fit_calls:
            self.assertEqual((len(args), kwargs), (3, {}))
            self.assertIs(args[1], parts["train"])
            self.assertIs(args[2], parts["val"])
        for args, kwargs in eval_calls:
            self.assertEqual((len(args), kwargs), (2, {}))
            self.assertIs(args[1], parts["val"])
        allowed = [parts[p][k] for p in ("train", "val") for k in ("texts", "segments")]
        self.assertTrue(seen)
        for docs in seen:
            self.assertIn(docs, allowed)
        self.assertNotIn(HELD_OUT, json.dumps(seen))

        ytr = np.array(parts["train"]["y"])
        expected_w = dict(zip(it.LABELS, it.class_weights(ytr).tolist()))
        self.assertEqual(results["candidates"]["balanced_mlp"]["class_weights"], expected_w)
        self.assertIsNone(results["candidates"]["linear_softmax"]["class_weights"])
        val_metrics = {n: results["candidates"][n]["validation"] for n in it.ORDER}
        self.assertEqual(results["selection"]["selected"], it.select_candidate(val_metrics)[0])

    def test_selection_rule_order(self):
        def m(f1, acc, ce):
            return {"macro_f1": f1, "accuracy": acc, "cross_entropy": ce}
        self.assertEqual(it.select_candidate({"linear_softmax": m(.40, .1, 2.0), "bigram_mlp": m(.3, .9, .1),
                                              "balanced_mlp": m(.3, .9, .1)})[0], "linear_softmax")
        self.assertEqual(it.select_candidate({"linear_softmax": m(.3, .5, .9), "bigram_mlp": m(.3, .6, .95),
                                              "balanced_mlp": m(.3, .6, .9)})[0], "balanced_mlp")
        tie = m(.3, .5, .9)
        self.assertEqual(it.select_candidate({"balanced_mlp": tie, "bigram_mlp": tie, "linear_softmax": tie})[1],
                         it.ORDER)


class PersistenceTests(unittest.TestCase):
    def test_save_reload_gives_identical_probabilities(self):
        parts = toy_parts()
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.print"):
            for spec in it.CANDIDATES:
                fitted = it.fit_candidate(spec["name"], parts["train"], parts["val"])
                d = Path(tmp) / spec["name"]
                it.save_candidate(fitted, d)
                meta, vec, params = it.load_candidate(d)
                self.assertEqual((meta["feature_mode"], meta["model"], meta["labels"]),
                                 (spec["features"], spec["model"], pe.LABELS))
                self.assertEqual(vec.vocab, fitted["vectorizer"].vocab)
                docs = it.docs_for(spec, parts["val"])
                p0 = it.predict_proba(spec["model"], fitted["params"], fitted["vectorizer"].transform(docs)[0])
                p1 = it.predict_proba(meta["model"], params, vec.transform(docs)[0])
                np.testing.assert_array_equal(p0, p1)
                pred = it.predict_texts(["profit rose on strong sales"], d)[0]
                self.assertIn(pred["label"], pe.LABELS)
                self.assertAlmostEqual(sum(pred["probabilities"].values()), 1.0, places=12)
                with self.assertRaises(FileExistsError):
                    it.save_candidate(fitted, d)  # never overwrite


if __name__ == "__main__":
    unittest.main()
