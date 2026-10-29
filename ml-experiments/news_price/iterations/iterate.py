"""Three pre-registered model iterations on the AA headline task, scored on VALIDATION ONLY.

Candidates (fixed in CANDIDATES before any fit; one seeded run each, 3 fits total):
  1. linear_softmax : original unigram TF-IDF (800, min_df 2) -> 3 logits, no hidden layer
  2. bigram_mlp     : unigrams + adjacent within-headline token pairs sharing the 800 cap -> 16 ReLU -> 3
  3. balanced_mlp   : original unigram features and 16-unit MLP, class-weighted training CE

Weighted objective (balanced_mlp; linear_softmax uses w_i = 1, i.e. the plain mean):
  loss    = sum_i w_i CE_i / sum_i w_i + (l2 / 2) * sum(W^2)   per minibatch, biases unpenalised
  dlogits = w_i (P_i - onehot(y_i)) / sum_i w_i ; the rest of backprop is sentiment_core's MLP
            (or gW = X^T dlogits + l2 W, gb = sum_rows(dlogits) for the linear model)
  w_i     = N_train / (3 * count_train(y_i)), computed from training labels only
Every candidate is checkpointed on the UNWEIGHTED validation CE (patience 5, max 30), as in the first
run. Final pick: highest validation macro-F1, then accuracy, then lower validation CE, then CANDIDATES order.

TEST POLICY: the first run's test split has already been inspected. The original grouping/split is
rebuilt only to reproduce split boundaries and counts; train_val_only() drops test samples before any
vectorizer is fit or applied, and no test labels/features are used. results.json records
test_evaluations: 0. Validation gains are exploratory (validation also chose epochs and the final
candidate), not proof of future generalisation.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import price_experiment as pe
import sentiment_core as core

HERE = Path(__file__).resolve().parent
CONTEXT_PATH = HERE / "VALIDATION_CONTEXT.json"
LABELS = pe.LABELS  # names for core's numeric slots 0/1/2; core.LABELS is never used for reporting or mutated
K = len(LABELS)
SETTINGS = {k: pe.CONFIG[k] for k in ("max_features", "min_df", "hidden", "lr", "l2", "batch_size",
                                      "max_epochs", "patience", "seed")}
BIGRAM_SEP = " "  # TOKEN_RE tokens never contain whitespace, so "a b" cannot collide with a unigram or another pair
HISTORY_FIELDS = ["epoch", "train_ce", "val_ce", "train_acc", "val_acc", "train_weighted_ce"]
MODEL_KEYS = {"linear_softmax": ("W", "b"), "mlp": ("W1", "b1", "W2", "b2")}

CANDIDATES = [
    {"name": "linear_softmax", "features": "unigram", "model": "linear_softmax", "hidden": None,
     "training_loss": "unweighted_ce", "init": "zeros (convex objective)", "trainer": "iterate.train_adam",
     "question": "Does lower capacity (no hidden layer, 800*3+3 = 2403 parameters) generalise better "
                 "than the original 16-unit MLP?"},
    {"name": "bigram_mlp", "features": "unigram+bigram", "model": "mlp", "hidden": 16,
     "training_loss": "unweighted_ce", "init": "sentiment_core.init_mlp(seed)", "trainer": "sentiment_core.train_mlp",
     "question": "Does local word order (adjacent token pairs inside one headline) help?"},
    {"name": "balanced_mlp", "features": "unigram", "model": "mlp", "hidden": 16,
     "training_loss": "class_weighted_ce", "init": "sentiment_core.init_mlp(seed)",
     "trainer": "iterate.train_adam (class-weighted)",
     "question": "Does paying more attention to the rare FLAT class improve per-class recall and macro-F1?"},
]
CANDIDATE_BY_NAME = {c["name"]: c for c in CANDIDATES}
ORDER = [c["name"] for c in CANDIDATES]
CHECKPOINT_RULE = ("per candidate: epoch with the lowest UNWEIGHTED validation cross-entropy "
                   "(patience 5, max 30 epochs), as in the first run")
SELECTION_RULE = ("highest validation macro-F1; tie -> higher validation accuracy; tie -> lower validation "
                  "cross-entropy; tie -> CANDIDATES order (values rounded to 12 decimals)")
CLASS_WEIGHT_RULE = "w_c = N_train / (3 * count_train(c)); minibatch CE = sum(w_i CE_i) / sum(w_i)"
TEST_POLICY = ("Original grouping/split rebuilt only to reproduce boundaries and counts; test samples are "
               "dropped before any vectorizer fit/transform; no test features, labels or predictions are used.")
LIMITATIONS = pe.LIMITATIONS + [
    "Validation (154 groups, only 5 FLAT) chose each candidate's epoch AND the final candidate, so the "
    "selected score is optimistically biased; any gain is exploratory, not evidence of future generalisation.",
    "One validation group moves accuracy by 1/154 ~ 0.0065; FLAT recall/F1 rest on 5 validation groups.",
    "No test evaluation was run for these candidates; the old test split was already seen and cannot serve "
    "as an untouched check, so a fresh later period would be needed to confirm anything.",
    "One seed per candidate and no seed/setting search; differences may be seed noise.",
    "bigram_mlp shares the 800-feature cap, so bigrams displace some unigrams; pairs never span headlines.",
    "Saved weights are for local inspection/inference only; not a deployable or trading model.",
]


# ------------------------------------------------------------ data split

def build_samples(data_dir):
    """Original pipeline steps, unchanged (price_experiment functions)."""
    news_path, price_path = data_dir / pe.NEWS_FILE, data_dir / pe.PRICE_FILE
    records, news_counts = pe.load_headlines(news_path)
    prices, price_counts = pe.load_prices(price_path)
    unique, dups = pe.dedup_earliest(records)
    samples, align_counts = pe.align_and_group(unique, prices)
    split = pe.chronological_split(samples)
    counters = {"headlines": news_counts, "duplicate_headlines_removed": dups,
                "alignment": align_counts, "prices": price_counts}
    return samples, split, {r["id"]: r["text"] for r in unique}, counters, [news_path, price_path]


def ids_sha256(ids):
    return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()


def train_val_only(samples, split, texts_by_id):
    """Keep ONLY train/val samples (grouped text, per-headline segments, label).

    Test samples contribute nothing but their count and a sample-ID hash (entry date + headline IDs,
    no labels or text) used to confirm the original split boundary.
    """
    parts = {}
    for name in ("train", "val"):
        rows = [samples[i] for i in split[name]]
        segments = [[texts_by_id[h] for h in s["headline_ids"]] for s in rows]
        for s, seg in zip(rows, segments):
            if " ".join(seg) != s["text"]:
                raise ValueError("headline segments do not rebuild the grouped sample text")
        parts[name] = {"sample_ids": [s["sample_id"] for s in rows], "texts": [s["text"] for s in rows],
                       "segments": segments, "y": [int(s["y"]) for s in rows]}
    summary = pe.split_summary(samples, {"train": split["train"], "val": split["val"]})
    held_out = {"groups": len(split["test"]),
                "sample_ids_sha256": ids_sha256([samples[i]["sample_id"] for i in split["test"]]),
                "use": "count and ID hash only; no features, labels or predictions"}
    return parts, summary, held_out


def load_context():
    return json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))


def check_settings(context):
    if pe.CONFIG != context["settings"]:
        raise ValueError("price_experiment.CONFIG differs from VALIDATION_CONTEXT.json settings")


def verify_continuity(summary, held_out, counters, context):
    problems = [f"{k} split summary differs" for k in ("train", "val") if summary[k] != context["split"][k]]
    if held_out["groups"] != context["split"]["test"]["groups"]:
        problems.append("held-out group count differs")
    problems += [f"counter '{k}' differs" for k in context["counters"] if counters.get(k) != context["counters"][k]]
    if problems:
        raise ValueError("continuity check against VALIDATION_CONTEXT.json failed: " + "; ".join(problems))


def verify_prior_samples(path, samples, split):
    """Compare (sample_id, split) order with the first run's samples.json; no other field is read."""
    prior = json.loads(Path(path).read_text(encoding="utf-8"))["samples"]
    got = [(p["sample_id"], p["split"]) for p in prior]
    want = [(samples[i]["sample_id"], name) for name, idx in split.items() for i in idx]
    if got != want:
        raise ValueError("sample IDs / split membership differ from the first run's samples.json")
    return {"checked": True, "file": Path(path).name, "samples": len(got)}


# ------------------------------------------------------------ features

def ngram_features(doc):
    """(unigrams, bigrams) of a doc (string or list of headline strings); pairs never span headlines."""
    unigrams, bigrams = [], []
    for seg in [doc] if isinstance(doc, str) else doc:
        toks = core.tokenize(seg)
        unigrams += toks
        bigrams += [a + BIGRAM_SEP + b for a, b in zip(toks, toks[1:])]
    return unigrams, bigrams


class NgramTfidfVectorizer:
    """Unigram + bigram TF-IDF sharing one cap; same df/idf/L2 rules as core.TfidfVectorizer."""

    def __init__(self, vocab=None, idf=None):
        self.vocab = list(vocab or [])
        self.idf = np.asarray(idf if idf is not None else [], dtype=np.float64)
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.eligible = None

    def fit(self, docs, max_features, min_df):
        df = {}
        for doc in docs:
            uni, bi = ngram_features(doc)
            for f in set(uni + bi):
                df[f] = df.get(f, 0) + 1
        eligible = [(f, c) for f, c in df.items() if c >= min_df]
        eligible.sort(key=lambda fc: (-fc[1], fc[0]))
        kept = sorted(eligible[:max_features], key=lambda fc: fc[0])  # deterministic alphabetical vocabulary
        n = len(docs)
        self.vocab = [f for f, _ in kept]
        self.idf = np.array([np.log((1 + n) / (1 + c)) + 1.0 for _, c in kept])
        self.index = {t: i for i, t in enumerate(self.vocab)}
        n_bi = sum(BIGRAM_SEP in f for f, _ in eligible)
        self.eligible = {"unigrams": len(eligible) - n_bi, "bigrams": n_bi}
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)), dtype=np.float64)
        stats = {"tokens": 0, "unknown_tokens": 0, "bigrams": 0, "unknown_bigrams": 0}
        for r, doc in enumerate(docs):
            for kind, feats in zip(("tokens", "bigrams"), ngram_features(doc)):
                for f in feats:
                    stats[kind] += 1
                    j = self.index.get(f)
                    if j is None:
                        stats["unknown_" + kind] += 1
                    else:
                        X[r, j] += 1.0
        X *= self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X /= np.where(norms > 0, norms, 1.0)
        return X, stats


def make_vectorizer(spec):
    return NgramTfidfVectorizer() if spec["features"] == "unigram+bigram" else core.TfidfVectorizer()


def docs_for(spec, part):
    """Bigram features need headline boundaries; unigram features use the original grouped text."""
    return part["segments"] if spec["features"] == "unigram+bigram" else part["texts"]


def feature_counts(vec):
    n_bi = sum(BIGRAM_SEP in f for f in vec.vocab)
    return {"unigrams": len(vec.vocab) - n_bi, "bigrams": n_bi}


def coverage(X, stats):
    out = dict(stats)
    out["token_coverage"] = 1 - stats["unknown_tokens"] / stats["tokens"] if stats["tokens"] else None
    out["zero_feature_rows"] = int((~X.any(axis=1)).sum())
    return out


# --------------------------------------------------------------- models

def weighted_ce_and_dlogits(logits, y, w):
    """sum(w_i CE_i) / sum(w_i) and its logits gradient; w = ones reproduces core's mean CE exactly."""
    n = len(y)
    logp = core.log_softmax(logits)
    loss = float(np.sum(w * -logp[np.arange(n), y]) / np.sum(w))
    dlogits = np.exp(logp)
    dlogits[np.arange(n), y] -= 1.0
    dlogits *= w[:, None]
    dlogits /= np.sum(w)
    return loss, dlogits


def weighted_mlp_loss_and_grads(params, X, y, w, l2):
    """core.mlp_loss_and_grads with the 1/n row factor replaced by w_i / sum(w); L2 unchanged."""
    Z = X @ params["W1"] + params["b1"]
    H = np.maximum(Z, 0.0)
    loss, dlogits = weighted_ce_and_dlogits(H @ params["W2"] + params["b2"], y, w)
    loss += 0.5 * l2 * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    dZ = (dlogits @ params["W2"].T) * (Z > 0)
    return float(loss), {"W1": X.T @ dZ + l2 * params["W1"], "b1": dZ.sum(axis=0),
                         "W2": H.T @ dlogits + l2 * params["W2"], "b2": dlogits.sum(axis=0)}


def init_linear(d):
    return {"W": np.zeros((d, K)), "b": np.zeros(K)}


def linear_logits(params, X):
    return X @ params["W"] + params["b"]


def linear_loss_and_grads(params, X, y, w, l2):
    """Loss = sum(w_i CE_i) / sum(w_i) + l2/2 * |W|^2 (bias unpenalised)."""
    loss, dlogits = weighted_ce_and_dlogits(linear_logits(params, X), y, w)
    loss += 0.5 * l2 * np.sum(params["W"] ** 2)
    return float(loss), {"W": X.T @ dlogits + l2 * params["W"], "b": dlogits.sum(axis=0)}


def predict_proba(model, params, X):
    return core.softmax(linear_logits(params, X)) if model == "linear_softmax" else core.mlp_proba(params, X)


def class_weights(ytr):
    """N / (3 * count_c) from TRAINING labels only; fails if a class is missing."""
    counts = np.bincount(np.asarray(ytr, dtype=int), minlength=K)
    if len(counts) != K or np.any(counts == 0):
        raise ValueError(f"training labels must cover exactly {K} classes, got counts {counts.tolist()}")
    return len(ytr) / (K * counts.astype(np.float64))


def train_adam(params, loss_and_grads, logits_fn, Xtr, ytr, Xva, yva, sample_weight=None,
               lr=core.LEARNING_RATE, max_epochs=core.MAX_EPOCHS, patience=core.PATIENCE,
               batch_size=core.BATCH_SIZE, l2=core.L2, seed=core.SEED, log=None):
    """core.train_mlp's shuffling / Adam / early stopping for any (loss_and_grads, logits_fn) pair.

    sample_weight (one per training row) only changes the minibatch objective. Checkpointing always uses
    the UNWEIGHTED validation CE; history adds train_weighted_ce (no L2) when weights are given.
    Returns (best_params, history, best_epoch).
    """
    w_all = np.ones(len(ytr)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if w_all.shape != (len(ytr),) or not np.all(np.isfinite(w_all) & (w_all > 0)):
        raise ValueError("sample_weight must be finite, positive and one per training row")
    params = {k: p.copy() for k, p in params.items()}
    m = {k: np.zeros_like(p) for k, p in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed)
    step = 0
    best = None
    best_val, best_epoch, bad = np.inf, 0, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(ytr))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            loss, grads = loss_and_grads(params, Xtr[idx], ytr[idx], w_all[idx], l2)
            core.check_finite("minibatch loss", np.array(loss))
            step += 1
            for k in params:
                core.check_finite("gradient " + k, grads[k])
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1 - beta1 ** step)
                v_hat = v[k] / (1 - beta2 ** step)
                params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        ltr, lva = logits_fn(params, Xtr), logits_fn(params, Xva)
        row = {
            "epoch": epoch,
            "train_ce": core.cross_entropy(ltr, ytr),
            "val_ce": core.cross_entropy(lva, yva),
            "train_acc": float((ltr.argmax(1) == ytr).mean()),
            "val_acc": float((lva.argmax(1) == yva).mean()),
        }
        if sample_weight is not None:
            row["train_weighted_ce"] = weighted_ce_and_dlogits(ltr, ytr, w_all)[0]
        for key in ("train_ce", "val_ce", "train_weighted_ce"):
            if key in row:
                core.check_finite(key, np.array(row[key]))
        history.append(row)
        if log:
            log(row)
        if row["val_ce"] < best_val:
            best_val, best_epoch, bad = row["val_ce"], epoch, 0
            best = {k: p.copy() for k, p in params.items()}
        else:
            bad += 1
            if bad >= patience:
                break
    return best, history, best_epoch


# ----------------------------------------------------------- candidates

def fit_candidate(name, train, val):
    """Fit one pre-registered candidate. Receives ONLY the training and validation parts."""
    spec, s = CANDIDATE_BY_NAME[name], SETTINGS
    vec = make_vectorizer(spec).fit(docs_for(spec, train), max_features=s["max_features"],
                                    min_df=s["min_df"])  # train text only
    if not vec.vocab:
        raise ValueError(f"{name}: empty training vocabulary")
    (Xtr, st_tr), (Xva, st_va) = (vec.transform(docs_for(spec, p)) for p in (train, val))
    ytr, yva = np.asarray(train["y"], dtype=int), np.asarray(val["y"], dtype=int)
    for label, X in (("Xtr", Xtr), ("Xva", Xva)):
        core.check_finite(f"{name} {label}", X)

    def log(row):
        extra = f"  train_weighted_ce {row['train_weighted_ce']:.4f}" if "train_weighted_ce" in row else ""
        print(f"{name:14s} epoch {row['epoch']:2d}  train_ce {row['train_ce']:.4f}  val_ce {row['val_ce']:.4f}  "
              f"train_acc {row['train_acc']:.3f}  val_acc {row['val_acc']:.3f}{extra}")

    common = {k: s[k] for k in ("lr", "max_epochs", "patience", "batch_size", "l2", "seed")}
    weights = None
    if spec["model"] == "linear_softmax":
        params, history, best_epoch = train_adam(init_linear(Xtr.shape[1]), linear_loss_and_grads, linear_logits,
                                                 Xtr, ytr, Xva, yva, log=log, **common)
    elif spec["training_loss"] == "class_weighted_ce":
        weights = class_weights(ytr)
        params, history, best_epoch = train_adam(core.init_mlp(Xtr.shape[1], s["hidden"], s["seed"]),
                                                 weighted_mlp_loss_and_grads, core.mlp_logits,
                                                 Xtr, ytr, Xva, yva, sample_weight=weights[ytr], log=log, **common)
    else:
        params, history, best_epoch = core.train_mlp(Xtr, ytr, Xva, yva, hidden=s["hidden"], log=log, **common)
    for k, p in params.items():
        core.check_finite(f"{name} weights {k}", p)

    features = {"mode": spec["features"], "vocab_size": len(vec.vocab), "feature_counts": feature_counts(vec),
                "idf_range": [float(vec.idf.min()), float(vec.idf.max())],
                "coverage": {"train": coverage(Xtr, st_tr), "val": coverage(Xva, st_va)}}
    if getattr(vec, "eligible", None):
        features["eligible_before_cap"] = vec.eligible
    return {"name": name, "spec": spec, "vectorizer": vec, "params": params, "history": history,
            "best_epoch": best_epoch, "features": features,
            "class_weights": None if weights is None else {n: float(weights[i]) for i, n in enumerate(LABELS)}}


def evaluate_candidate(fitted, val):
    """Validation metrics only (semantic DOWN/FLAT/UP names via price_experiment)."""
    X, _ = fitted["vectorizer"].transform(docs_for(fitted["spec"], val))
    probs = predict_proba(fitted["spec"]["model"], fitted["params"], X)
    core.check_finite(fitted["name"] + " validation probs", probs)
    yva = np.asarray(val["y"], dtype=int)
    return {**pe.classification_metrics(yva, probs.argmax(1)), "cross_entropy": core.prob_cross_entropy(probs, yva)}


def candidate_report(fitted, metrics):
    best = fitted["history"][fitted["best_epoch"] - 1]
    return {"question": fitted["spec"]["question"], "spec": fitted["spec"], "validation": metrics,
            "parameter_count": int(sum(p.size for p in fitted["params"].values())),
            "best_epoch": fitted["best_epoch"], "epochs_run": len(fitted["history"]),
            "checkpoint": {"val_ce_unweighted": best["val_ce"], "train_ce_unweighted": best["train_ce"],
                           "train_weighted_ce": best.get("train_weighted_ce"),
                           "note": "CE values exclude the L2 penalty; checkpoint chosen by val_ce_unweighted"},
            "class_weights": fitted["class_weights"], "features": fitted["features"]}


def select_candidate(val_metrics):
    """Frozen rule (SELECTION_RULE). Returns (selected_name, ranking)."""
    def key(name):
        m = val_metrics[name]
        return (-round(m["macro_f1"], 12), -round(m["accuracy"], 12), round(m["cross_entropy"], 12),
                ORDER.index(name))
    ranking = sorted(val_metrics, key=key)
    return ranking[0], ranking


def compare(val_metrics, baselines):
    out = {}
    for name, m in val_metrics.items():
        out[name] = {f"vs_{b}": {"delta_macro_f1": m["macro_f1"] - bm["macro_f1"],
                                 "delta_accuracy": m["accuracy"] - bm["accuracy"],
                                 "delta_cross_entropy": m["cross_entropy"] - bm["cross_entropy"]}
                     for b, bm in baselines.items()}
        out[name]["improves_macro_f1_over_original_mlp"] = m["macro_f1"] > baselines["mlp"]["macro_f1"]
    return out


def findings(val_metrics, comparison, baselines, selected):
    ref = baselines["mlp"]["macro_f1"]
    sel = val_metrics[selected]
    n_val = sum(c["support"] for c in sel["per_class"].values())
    improved = [n for n in ORDER if comparison[n]["improves_macro_f1_over_original_mlp"]]
    lines = [f"Frozen validation rule selects {selected}: macro-F1 {sel['macro_f1']:.5f}, accuracy "
             f"{sel['accuracy']:.5f}, CE {sel['cross_entropy']:.5f}."]
    if improved:
        lines.append(f"Validation macro-F1 above the original MLP ({ref:.5f}): {', '.join(improved)}. Exploratory "
                     f"only: validation also chose epochs and the candidate; one group moves accuracy by "
                     f"{1 / max(n_val, 1):.4f}.")
    else:
        lines.append(f"No candidate improved validation macro-F1 over the original MLP ({ref:.5f}); reported as a "
                     "negative result. No further seeds or settings were searched.")
    for n in ORDER:
        flat = val_metrics[n]["per_class"]["FLAT"]
        lines.append(f"{n}: FLAT recall {flat['recall']:.3f} / F1 {flat['f1']:.3f} on {flat['support']} "
                     "validation groups.")
    lines.append("No test evaluation was performed (test_evaluations = 0).")
    return lines


def save_candidate(fitted, directory):
    """Inference-only artefacts; refuses to overwrite an existing directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez(directory / "model.npz", **fitted["params"])
    core.write_json(directory / "vectorizer.json", vectorizer_dict(fitted))
    write_history_csv(directory / "history.csv", fitted["history"])
    core.write_loss_svg(directory / "loss_curve.svg", fitted["history"], fitted["best_epoch"])


def vectorizer_dict(fitted):
    spec, vec = fitted["spec"], fitted["vectorizer"]
    d = {"candidate": fitted["name"], "feature_mode": spec["features"], "model": spec["model"],
         "vocab": vec.vocab, "idf": [float(v) for v in vec.idf], "tokenizer": core.TOKEN_RE.pattern,
         "tokenization": "lowercase, TOKEN_RE matches, tokens starting with a digit -> '<num>'",
         "weighting": "count * idf, idf = ln((1 + n_train) / (1 + df)) + 1, rows L2-normalised; fit on train only",
         "labels": LABELS, "feature_counts": feature_counts(vec)}
    if spec["features"] == "unigram+bigram":
        d["bigram_separator"] = BIGRAM_SEP
        d["bigram_scope"] = "adjacent tokens within one headline; a list input = one string per headline"
    return d


def load_candidate(directory):
    """Returns (vectorizer_json, vectorizer, params) for a saved candidate directory."""
    directory = Path(directory)
    d = json.loads((directory / "vectorizer.json").read_text(encoding="utf-8"))
    if d.get("labels") != LABELS or d.get("tokenizer") != core.TOKEN_RE.pattern or d.get("model") not in MODEL_KEYS:
        raise ValueError("vectorizer.json is not from this experiment")
    if d.get("feature_mode") == "unigram":
        vec = core.TfidfVectorizer.from_dict(d)
    elif d.get("feature_mode") == "unigram+bigram" and d.get("bigram_separator") == BIGRAM_SEP:
        vec = NgramTfidfVectorizer(d["vocab"], d["idf"])
    else:
        raise ValueError(f"unsupported feature_mode {d.get('feature_mode')!r}")
    keys = MODEL_KEYS[d["model"]]
    with np.load(directory / "model.npz", allow_pickle=False) as data:
        if sorted(data.files) != sorted(keys):
            raise ValueError(f"model.npz keys {sorted(data.files)} do not match model {d['model']}")
        params = {k: data[k] for k in keys}
    for k, p in params.items():
        core.check_finite("weights " + k, p)
    return d, vec, params


def predict_texts(texts, directory):
    d, vec, params = load_candidate(directory)
    X, _ = vec.transform(list(texts))
    probs = predict_proba(d["model"], params, X)
    return [{"label": LABELS[int(p.argmax())], "probabilities": {n: float(p[i]) for i, n in enumerate(LABELS)}}
            for p in probs]


def write_history_csv(path, history):
    """train_ce / val_ce are UNWEIGHTED CE (no L2); train_weighted_ce is blank for unweighted candidates."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, restval="")
        w.writeheader()
        w.writerows(history)


def run_candidates(train, val, out_dir, baselines, fit=fit_candidate, evaluate=evaluate_candidate):
    """One bounded fit per candidate on train/val only; selection by the frozen validation rule."""
    reports, val_metrics = {}, {}
    for spec in CANDIDATES:
        fitted = fit(spec["name"], train, val)
        val_metrics[spec["name"]] = evaluate(fitted, val)
        save_candidate(fitted, Path(out_dir) / spec["name"])
        reports[spec["name"]] = candidate_report(fitted, val_metrics[spec["name"]])
    selected, ranking = select_candidate(val_metrics)
    comparison = compare(val_metrics, baselines)
    return {"test_evaluations": 0, "fits": len(CANDIDATES), "candidates": reports,
            "baselines_validation_from_context": baselines, "comparison": comparison,
            "selection": {"rule": SELECTION_RULE, "selected": selected, "ranking": ranking,
                          "status": "chosen on validation only; not a generalisation claim"},
            "findings": findings(val_metrics, comparison, baselines, selected)}


def build_config():
    return {"purpose": "three pre-registered iterations, validation-only comparison (educational)",
            "candidates": CANDIDATES, "shared_settings": SETTINGS, "checkpoint_rule": CHECKPOINT_RULE,
            "selection_rule": SELECTION_RULE, "class_weight_rule": CLASS_WEIGHT_RULE,
            "bigram_separator": BIGRAM_SEP, "labels": LABELS,
            "label_slots": "numeric 0/1/2 = DOWN/FLAT/UP (sentiment_core.LABELS names are unused slot names)",
            "baselines": "original majority_train / naive_bayes / mlp validation metrics copied from "
                         "VALIDATION_CONTEXT.json (not refit)",
            "fits_planned": len(CANDIDATES), "test_evaluations": 0, "test_policy": TEST_POLICY}


def run(data_dir, output_dir, prior_samples=None):
    data_dir, out = Path(data_dir), Path(output_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing path: {out} (choose a new --output)")
    context = load_context()
    check_settings(context)
    samples, split, texts_by_id, counters, used = build_samples(data_dir)
    prior = verify_prior_samples(prior_samples, samples, split) if prior_samples else {"checked": False}
    parts, summary, held_out = train_val_only(samples, split, texts_by_id)
    del samples, texts_by_id  # nothing from the test period is kept past this point
    verify_continuity(summary, held_out, counters, context)
    bindings = {**pe.provenance(data_dir, used), "iterate_sha256": core.file_sha256(__file__),
                "validation_context_sha256": core.file_sha256(CONTEXT_PATH)}

    config = build_config()
    out.mkdir(parents=True, exist_ok=False)
    core.write_json(out / "config.json", config)  # written before any fit
    try:
        results = run_candidates(parts["train"], parts["val"], out, context["baseline_validation"])
    except (ValueError, FloatingPointError) as e:
        core.write_json(out / "diagnostic.json", {"status": "aborted during candidate fitting", "error": str(e),
                                                  "test_evaluations": 0})
        raise
    report = {"status": "completed", "task": "headline text -> next observed price session open->close bucket",
              "IMPORTANT": pe.LIMITATIONS[0], "config": config, "bindings": bindings,
              "data": {"counters": counters, "split_train_val": summary, "held_out_test": held_out,
                       "sample_ids_sha256": {k: ids_sha256(parts[k]["sample_ids"]) for k in ("train", "val")},
                       "continuity": "counters, train/val summaries and test group count match VALIDATION_CONTEXT.json",
                       "prior_samples_check": prior},
              **results, "limitations": LIMITATIONS}
    core.write_json(out / "results.json", report)

    for name in ORDER:
        m = results["candidates"][name]["validation"]
        print(f"validation {name:14s} acc {m['accuracy']:.3f}  macro_f1 {m['macro_f1']:.3f}  ce {m['cross_entropy']:.4f}")
    for line in results["findings"]:
        print(line)
    print(f"artifacts written to {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(HERE))
    ap.add_argument("--output", default=str(HERE / "iterate_run"), help="must not exist; never overwritten")
    ap.add_argument("--prior-samples", help="optional first-run samples.json; only sample_id/split are compared")
    ap.add_argument("--predict", metavar="TEXT", help="bucket one headline with a saved candidate (inference only)")
    ap.add_argument("--model-dir", help="candidate directory with model.npz and vectorizer.json (for --predict)")
    args = ap.parse_args(argv)
    try:
        if args.predict is not None:
            if not args.model_dir:
                ap.error("--predict requires --model-dir")
            print(json.dumps(predict_texts([args.predict], args.model_dir)[0], indent=2))
            return 0
        return run(args.data_dir, args.output, args.prior_samples)
    except (ValueError, FloatingPointError) as e:
        print(f"diagnostic abort: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
