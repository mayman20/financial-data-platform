"""Focused unit tests for train.py (toy data only; no claims about real metrics)."""

import tempfile
import unittest

import numpy as np

import train


def tiny_problem(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, 5))
    y = np.array([0, 1, 2, 0, 1, 2])
    params = {
        "W1": rng.normal(size=(5, 4)),
        "b1": rng.normal(size=4),
        "W2": rng.normal(size=(4, 3)),
        "b2": rng.normal(size=3),
    }
    return X, y, params


def away_from_relu_kink(margin=0.05):
    for seed in range(200):
        X, y, params = tiny_problem(seed)
        Z = X @ params["W1"] + params["b1"]
        if np.abs(Z).min() > margin:
            return X, y, params
    raise AssertionError("no kink-free toy problem found")


class GradientTests(unittest.TestCase):
    def test_finite_difference_all_parameter_families(self):
        X, y, params = away_from_relu_kink()
        l2 = 1e-2  # large enough that the penalty gradient is actually exercised
        _, grads = train.mlp_loss_and_grads(params, X, y, l2)
        eps = 1e-6
        for key in ("W1", "b1", "W2", "b2"):
            numeric = np.zeros_like(params[key])
            it = np.nditer(params[key], flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                orig = params[key][idx]
                params[key][idx] = orig + eps
                up, _ = train.mlp_loss_and_grads(params, X, y, l2)
                params[key][idx] = orig - eps
                down, _ = train.mlp_loss_and_grads(params, X, y, l2)
                params[key][idx] = orig
                numeric[idx] = (up - down) / (2 * eps)
            np.testing.assert_allclose(grads[key], numeric, rtol=1e-5, atol=1e-8, err_msg=key)


class LossTests(unittest.TestCase):
    def test_extreme_logits_are_finite_and_correct(self):
        logits = np.array([[1000.0, -1000.0, 0.0], [1000.0, -1000.0, 0.0]])
        y = np.array([0, 1])
        ce = train.cross_entropy(logits, y)
        self.assertTrue(np.isfinite(ce))
        self.assertAlmostEqual(ce, (0.0 + 2000.0) / 2, places=6)
        self.assertTrue(np.all(np.isfinite(train.softmax(logits))))

    def test_loss_decreases_on_toy_data(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, size=90)
        X = np.eye(3)[y] + 0.1 * rng.normal(size=(90, 3))
        _, history, best_epoch = train.train_mlp(X, y, X, y, hidden=8, lr=0.01,
                                                 max_epochs=15, patience=15, batch_size=16)
        self.assertLess(history[-1]["train_ce"], history[0]["train_ce"])
        self.assertGreaterEqual(best_epoch, 1)


class DataTests(unittest.TestCase):
    def test_duplicates_and_conflicts(self):
        lines = ["Profit  rose@positive", "profit rose@positive", "Sales fell@negative",
                 "Mixed one@positive", "mixed one@negative", "Plain@neutral"]
        records, stats = train.parse_lines(lines)
        self.assertEqual(stats["raw_rows"], 6)
        self.assertEqual(stats["exact_duplicate_rows_removed"], 1)
        self.assertEqual(stats["conflicting_sentences_rejected"], 1)
        self.assertEqual(stats["conflicting_rows_rejected"], 2)
        self.assertEqual(len(records), 3)
        self.assertEqual(len({r["id"] for r in records}), 3)

    def test_split_deterministic_disjoint_complete(self):
        labels = [i % 3 for i in range(300)]
        ids = [train.record_id(str(i)) for i in range(300)]
        a = train.stratified_split(ids, labels)
        b = train.stratified_split(ids, labels)
        self.assertEqual(a, b)
        sets = [set(a[k]) for k in ("train", "val", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set().union(*sets), set(range(300)))
        self.assertEqual(len(a["train"]), 210)
        for k in a:  # stratified: equal class counts here
            counts = np.bincount([labels[i] for i in a[k]])
            self.assertEqual(len(set(counts)), 1)

    def test_vectorizer_fit_on_train_only(self):
        train_texts = ["profit rose", "profit fell", "sales rose", "sales fell"]
        vec = train.TfidfVectorizer().fit(train_texts)
        self.assertEqual(vec.vocab, sorted(vec.vocab))
        self.assertNotIn("dividend", vec.vocab)
        X, stats = vec.transform(["dividend profit"])
        self.assertEqual(stats, {"tokens": 2, "unknown_tokens": 1})
        self.assertAlmostEqual(float(np.linalg.norm(X[0])), 1.0)

    def test_min_df_and_max_features(self):
        texts = ["alpha beta", "alpha gamma", "alpha beta delta"]
        vec = train.TfidfVectorizer().fit(texts)  # min_df=2
        self.assertEqual(vec.vocab, ["alpha", "beta"])
        vec = train.TfidfVectorizer().fit(texts, max_features=1)
        self.assertEqual(vec.vocab, ["alpha"])


class MetricTests(unittest.TestCase):
    def test_perfect_and_wrong(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        perfect = train.classification_metrics(y, y)
        self.assertEqual(perfect["accuracy"], 1.0)
        self.assertEqual(perfect["macro_f1"], 1.0)
        wrong = train.classification_metrics(y, (y + 1) % 3)
        self.assertEqual(wrong["accuracy"], 0.0)
        self.assertEqual(wrong["macro_f1"], 0.0)

    def test_class_order_and_missing_class(self):
        self.assertEqual(train.LABELS, ["negative", "neutral", "positive"])
        y_true = np.array([0, 0, 1, 2])
        y_pred = np.array([1, 1, 1, 1])  # never predicts negative or positive
        m = train.classification_metrics(y_true, y_pred)
        self.assertEqual(list(m["per_class"]), train.LABELS)
        self.assertEqual(m["confusion_rows_true_cols_pred"],
                         [[0, 2, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(m["per_class"]["negative"]["support"], 2)
        self.assertAlmostEqual(m["per_class"]["neutral"]["precision"], 0.25)
        self.assertAlmostEqual(m["per_class"]["neutral"]["recall"], 1.0)
        self.assertAlmostEqual(m["macro_f1"], (0.0 + 0.4 + 0.0) / 3)


class PersistenceTests(unittest.TestCase):
    def test_save_reload_prediction_parity(self):
        texts = ["profit rose sharply", "sales fell sharply", "meeting held today",
                 "profit rose today", "sales fell today", "meeting held sharply"]
        y = np.array([2, 0, 1, 2, 0, 1])
        vec = train.TfidfVectorizer().fit(texts)
        X, _ = vec.transform(texts)
        model, _, _ = train.train_mlp(X, y, X, y, hidden=6, lr=0.01, max_epochs=5, patience=5)
        before = train.predict_texts(texts + ["unseen words only"], model, vec)
        with tempfile.TemporaryDirectory() as d:
            train.save_model(model, d)
            train.save_vectorizer(vec, d)
            after = train.predict_texts(texts + ["unseen words only"],
                                        train.load_model(d), train.load_vectorizer(d))
        self.assertEqual(before, after)
        self.assertEqual(set(before[0]["probabilities"]), set(train.LABELS))


if __name__ == "__main__":
    unittest.main()
