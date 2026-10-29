"""Educational financial-sentiment experiment: TF-IDF + three small models.

Data: Financial PhraseBank "AllAgree" (lines of ``sentence@label``, ISO-8859-1),
CC BY-NC-SA 3.0, educational use only. Nothing is downloaded by this script.

Models (all trained from numbers, no hand-written word rules):
  1. majority class
  2. multinomial Naive Bayes on TF-IDF (Laplace smoothing, alpha=1)
  3. one-hidden-layer MLP:  D -> 32 (ReLU) -> 3 logits

MLP forward / backward / update
  forward : Z = X W1 + b1 ; H = relu(Z) ; logits = H W2 + b2 ; P = softmax(logits)
  loss    : mean_i CE(P_i, y_i) + (l2 / 2) * (sum W1^2 + sum W2^2)   (biases unpenalised)
  backward: dL = (P - onehot(y)) / n
            gW2 = H^T dL + l2 W2        gb2 = sum_rows(dL)
            dH  = dL W2^T ; dZ = dH * (Z > 0)
            gW1 = X^T dZ + l2 W1        gb1 = sum_rows(dZ)
  update  : Adam (lr .003, betas .9/.999) on minibatches of 64; the epoch with the
            lowest validation cross-entropy is restored (patience 5, max 30 epochs).

Limitations: the split is a random sentence-level split. The data has no dates
and no trustworthy story grouping, so near-paraphrases from the same story may
straddle train/test; no claim about generalisation over time or to new
companies is made. Settings are fixed before the test set is touched, and the
test set is evaluated once, after the checkpoint is chosen.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

LABELS = ["negative", "neutral", "positive"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}
SEED = 42
MAX_FEATURES = 1200
MIN_DF = 2
HIDDEN = 32
LEARNING_RATE = 0.003
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 64
L2 = 1e-4
MAX_ERROR_ROWS = 50

# AI-authored synthetic sentences for illustration only; NOT evaluation data.
PROBES = [
    ("Quarterly profit rose sharply and exceeded expectations.", "positive"),
    ("The company raised its full-year revenue outlook.", "positive"),
    ("Operating profit increased compared with the previous year.", "positive"),
    ("Net sales fell and the company reported an operating loss.", "negative"),
    ("The firm will cut jobs after profit declined significantly.", "negative"),
    ("Demand weakened and the outlook was lowered.", "negative"),
    ("The board meeting will be held on Tuesday at the head office.", "neutral"),
    ("The company has offices in three countries.", "neutral"),
    ("The report was published on the company website.", "neutral"),
]


# ----------------------------------------------------------------- data

def normalize_sentence(text):
    return " ".join(text.lower().split())


def record_id(normalized):
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def parse_lines(lines):
    """Parse ``text@label`` lines; drop exact duplicates, reject conflicting ones.

    Returns (records, stats). Records keep first-occurrence file order.
    """
    by_norm = {}
    order = []
    raw_rows = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        text, sep, label = line.rpartition("@")
        label = label.strip().lower()
        if not sep or label not in LABEL_TO_INDEX:
            raise ValueError("malformed line (bad label)")
        raw_rows += 1
        norm = normalize_sentence(text)
        if norm not in by_norm:
            by_norm[norm] = {"text": text.strip(), "labels": [label]}
            order.append(norm)
        else:
            by_norm[norm]["labels"].append(label)

    records = []
    exact_dups_removed = 0
    conflicting_sentences = 0
    conflicting_rows = 0
    for norm in order:
        entry = by_norm[norm]
        labels = entry["labels"]
        if len(set(labels)) > 1:
            conflicting_sentences += 1
            conflicting_rows += len(labels)
            continue
        exact_dups_removed += len(labels) - 1
        records.append({
            "id": record_id(norm),
            "text": entry["text"],
            "label": labels[0],
            "y": LABEL_TO_INDEX[labels[0]],
        })
    stats = {
        "raw_rows": raw_rows,
        "exact_duplicate_rows_removed": exact_dups_removed,
        "conflicting_sentences_rejected": conflicting_sentences,
        "conflicting_rows_rejected": conflicting_rows,
        "kept_records": len(records),
    }
    return records, stats


def load_records(path):
    with open(path, "r", encoding="iso-8859-1") as f:
        return parse_lines(f.read().splitlines())


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_split(ids, labels, seed=SEED, fractions=(0.70, 0.15, 0.15)):
    """Deterministic stratified train/val/test index lists (independent of row order)."""
    rng = random.Random(seed)
    parts = {"train": [], "val": [], "test": []}
    for cls in sorted(set(labels)):
        members = sorted((i for i, y in enumerate(labels) if y == cls), key=lambda i: ids[i])
        rng.shuffle(members)
        n = len(members)
        n_train = round(fractions[0] * n)
        n_val = round(fractions[1] * n)
        parts["train"] += members[:n_train]
        parts["val"] += members[n_train:n_train + n_val]
        parts["test"] += members[n_train + n_val:]
    return {k: sorted(v) for k, v in parts.items()}


# ----------------------------------------------------------- vectorizer

TOKEN_RE = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*|%")


def tokenize(text):
    tokens = TOKEN_RE.findall(text.lower())
    return ["<num>" if t[0].isdigit() else t for t in tokens]


class TfidfVectorizer:
    """Unigram TF-IDF, fit on training text only, rows L2-normalised."""

    def __init__(self, vocab=None, idf=None):
        self.vocab = vocab or []
        self.idf = np.asarray(idf if idf is not None else [], dtype=np.float64)
        self.index = {t: i for i, t in enumerate(self.vocab)}

    def fit(self, texts, max_features=MAX_FEATURES, min_df=MIN_DF):
        df = {}
        for text in texts:
            for tok in set(tokenize(text)):
                df[tok] = df.get(tok, 0) + 1
        eligible = [(t, c) for t, c in df.items() if c >= min_df]
        eligible.sort(key=lambda tc: (-tc[1], tc[0]))
        kept = eligible[:max_features]
        kept.sort(key=lambda tc: tc[0])  # deterministic alphabetical vocabulary
        n = len(texts)
        self.vocab = [t for t, _ in kept]
        self.idf = np.array([np.log((1 + n) / (1 + c)) + 1.0 for _, c in kept])
        self.index = {t: i for i, t in enumerate(self.vocab)}
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        total = unknown = 0
        for r, text in enumerate(texts):
            for tok in tokenize(text):
                total += 1
                j = self.index.get(tok)
                if j is None:
                    unknown += 1
                else:
                    X[r, j] += 1.0
        X *= self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X /= np.where(norms > 0, norms, 1.0)
        return X, {"tokens": total, "unknown_tokens": unknown}

    def to_dict(self):
        return {"vocab": self.vocab, "idf": [float(v) for v in self.idf],
                "tokenizer": TOKEN_RE.pattern, "labels": LABELS}

    @classmethod
    def from_dict(cls, d):
        return cls(d["vocab"], d["idf"])


# -------------------------------------------------------------- metrics

def check_finite(name, arr):
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"non-finite values in {name}; aborting")


def log_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax(logits):
    return np.exp(log_softmax(logits))


def cross_entropy(logits, y):
    """Mean cross-entropy from logits, stable for extreme values."""
    return float(-log_softmax(logits)[np.arange(len(y)), y].mean())


def prob_cross_entropy(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    k = len(LABELS)
    conf = np.zeros((k, k), dtype=int)  # rows true, cols predicted
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    per_class = {}
    f1s = []
    for c, name in enumerate(LABELS):
        tp = conf[c, c]
        pred_n = conf[:, c].sum()
        support = conf[c, :].sum()
        prec = tp / pred_n if pred_n else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[name] = {"precision": float(prec), "recall": float(rec),
                           "f1": float(f1), "support": int(support)}
        f1s.append(f1)
    return {
        "accuracy": float(np.trace(conf) / max(len(y_true), 1)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_rows_true_cols_pred": conf.tolist(),
        "label_order": LABELS,
    }


# --------------------------------------------------------------- models

def fit_majority(y_train):
    counts = np.bincount(y_train, minlength=len(LABELS)).astype(np.float64)
    return {"prior": counts / counts.sum(), "label": int(counts.argmax())}


def predict_majority(model, n):
    return np.tile(model["prior"], (n, 1))  # probabilities = train prior


def fit_naive_bayes(X, y, alpha=1.0):
    k, d = len(LABELS), X.shape[1]
    feat = np.zeros((k, d))
    for c in range(k):
        feat[c] = X[y == c].sum(axis=0) + alpha
    log_like = np.log(feat) - np.log(feat.sum(axis=1, keepdims=True))
    prior = np.bincount(y, minlength=k) / len(y)
    return {"log_like": log_like, "log_prior": np.log(prior)}


def predict_nb_proba(model, X):
    return softmax(X @ model["log_like"].T + model["log_prior"])


def init_mlp(d, hidden=HIDDEN, seed=SEED):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / d), size=(d, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0.0, np.sqrt(1.0 / hidden), size=(hidden, len(LABELS))),
        "b2": np.zeros(len(LABELS)),
    }


def mlp_logits(params, X):
    return np.maximum(X @ params["W1"] + params["b1"], 0.0) @ params["W2"] + params["b2"]


def mlp_proba(params, X):
    return softmax(mlp_logits(params, X))


def mlp_loss_and_grads(params, X, y, l2=L2):
    """Loss = mean CE + l2/2 * (|W1|^2 + |W2|^2); returns (loss, grads)."""
    n = len(y)
    Z = X @ params["W1"] + params["b1"]
    H = np.maximum(Z, 0.0)
    logits = H @ params["W2"] + params["b2"]
    logp = log_softmax(logits)
    penalty = 0.5 * l2 * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    loss = float(-logp[np.arange(n), y].mean() + penalty)

    dlogits = np.exp(logp)
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    dH = dlogits @ params["W2"].T
    dZ = dH * (Z > 0)
    grads = {
        "W1": X.T @ dZ + l2 * params["W1"],
        "b1": dZ.sum(axis=0),
        "W2": H.T @ dlogits + l2 * params["W2"],
        "b2": dlogits.sum(axis=0),
    }
    return loss, grads


def train_mlp(Xtr, ytr, Xva, yva, hidden=HIDDEN, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS,
              patience=PATIENCE, batch_size=BATCH_SIZE, l2=L2, seed=SEED, log=None):
    """Adam + early stopping on validation CE. Returns (best_params, history, best_epoch)."""
    params = init_mlp(Xtr.shape[1], hidden, seed)
    m = {k: np.zeros_like(v) for k, v in params.items()}
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
            loss, grads = mlp_loss_and_grads(params, Xtr[idx], ytr[idx], l2)
            check_finite("minibatch loss", np.array(loss))
            step += 1
            for k in params:
                check_finite("gradient " + k, grads[k])
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1 - beta1 ** step)
                v_hat = v[k] / (1 - beta2 ** step)
                params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        ltr, lva = mlp_logits(params, Xtr), mlp_logits(params, Xva)
        row = {
            "epoch": epoch,
            "train_ce": cross_entropy(ltr, ytr),
            "val_ce": cross_entropy(lva, yva),
            "train_acc": float((ltr.argmax(1) == ytr).mean()),
            "val_acc": float((lva.argmax(1) == yva).mean()),
        }
        for key in ("train_ce", "val_ce"):
            check_finite(key, np.array(row[key]))
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


# ------------------------------------------------------ save / inference

def save_model(params, directory):
    np.savez(os.path.join(directory, "model.npz"), **params)


def load_model(directory):
    with np.load(os.path.join(directory, "model.npz"), allow_pickle=False) as data:
        return {k: data[k] for k in ("W1", "b1", "W2", "b2")}


def save_vectorizer(vec, directory):
    with open(os.path.join(directory, "vectorizer.json"), "w", encoding="utf-8") as f:
        json.dump(vec.to_dict(), f, allow_nan=False)


def load_vectorizer(directory):
    with open(os.path.join(directory, "vectorizer.json"), "r", encoding="utf-8") as f:
        return TfidfVectorizer.from_dict(json.load(f))


def predict_texts(texts, model, vectorizer):
    """Predict sentiment for raw strings with a trained MLP and fitted vectorizer."""
    X, _ = vectorizer.transform(list(texts))
    probs = mlp_proba(model, X)
    return [{"label": LABELS[int(p.argmax())],
             "probabilities": {name: float(p[i]) for i, name in enumerate(LABELS)}}
            for p in probs]


# ------------------------------------------------------------ reporting

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def write_history_csv(path, history):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_ce", "val_ce", "train_acc", "val_acc"])
        w.writeheader()
        w.writerows(history)


def write_loss_svg(path, history, best_epoch):
    W, H, left, right, top, bottom = 640, 400, 60, 20, 30, 50
    train = [r["train_ce"] for r in history]
    val = [r["val_ce"] for r in history]
    lo, hi = min(train + val), max(train + val)
    if hi - lo < 1e-12:
        hi = lo + 1.0
    n = len(history)

    def px(i):
        return left + (W - left - right) * (i / max(n - 1, 1))

    def py(v):
        return top + (H - top - bottom) * (1 - (v - lo) / (hi - lo))

    def line(vals, color):
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'

    bx = px(best_epoch - 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{H - bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{H - bottom}" x2="{W - right}" y2="{H - bottom}" stroke="black"/>',
        f'<line x1="{bx:.1f}" y1="{top}" x2="{bx:.1f}" y2="{H - bottom}" stroke="gray" stroke-dasharray="4"/>',
        line(train, "steelblue"),
        line(val, "crimson"),
        f'<text x="{left - 5}" y="{top + 4}" text-anchor="end">{hi:.3f}</text>',
        f'<text x="{left - 5}" y="{H - bottom}" text-anchor="end">{lo:.3f}</text>',
        f'<text x="{left}" y="{H - bottom + 18}">epoch 1</text>',
        f'<text x="{W - right}" y="{H - bottom + 18}" text-anchor="end">epoch {n}</text>',
        f'<text x="{W / 2}" y="{H - 10}" text-anchor="middle">cross-entropy (dashed = best val epoch {best_epoch})</text>',
        f'<text x="{left + 10}" y="{top + 15}" fill="steelblue">train CE</text>',
        f'<text x="{left + 80}" y="{top + 15}" fill="crimson">validation CE</text>',
        "</svg>",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ------------------------------------------------------------------ run

def run(data_path, output_dir):
    out = Path(output_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing path: {out} (choose a new --output)")

    records, data_stats = load_records(data_path)
    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    y_all = np.array([r["y"] for r in records])
    split = stratified_split(ids, y_all.tolist())
    tr, va, te = (np.array(split[k]) for k in ("train", "val", "test"))

    vec = TfidfVectorizer().fit([texts[i] for i in tr])  # train only
    Xtr, st_tr = vec.transform([texts[i] for i in tr])
    Xva, st_va = vec.transform([texts[i] for i in va])
    Xte, st_te = vec.transform([texts[i] for i in te])
    ytr, yva, yte = y_all[tr], y_all[va], y_all[te]
    for name, arr in (("Xtr", Xtr), ("Xva", Xva), ("Xte", Xte)):
        check_finite(name, arr)

    out.mkdir(parents=True, exist_ok=False)

    def log(row):
        print(f"epoch {row['epoch']:2d}  train_ce {row['train_ce']:.4f}  val_ce {row['val_ce']:.4f}  "
              f"train_acc {row['train_acc']:.3f}  val_acc {row['val_acc']:.3f}")

    maj = fit_majority(ytr)
    nb = fit_naive_bayes(Xtr, ytr)
    mlp, history, best_epoch = train_mlp(Xtr, ytr, Xva, yva, log=log)
    for k, p in mlp.items():
        check_finite("weights " + k, p)

    # Test set is evaluated exactly once, after the checkpoint is chosen.
    test_probs = {
        "majority": predict_majority(maj, len(yte)),
        "naive_bayes": predict_nb_proba(nb, Xte),
        "mlp": mlp_proba(mlp, Xte),
    }
    val_probs = {
        "majority": predict_majority(maj, len(yva)),
        "naive_bayes": predict_nb_proba(nb, Xva),
        "mlp": mlp_proba(mlp, Xva),
    }
    test_results, val_results = {}, {}
    for name in test_probs:
        check_finite(name + " test probs", test_probs[name])
        test_results[name] = classification_metrics(yte, test_probs[name].argmax(1))
        test_results[name]["cross_entropy"] = prob_cross_entropy(test_probs[name], yte)
        val_results[name] = classification_metrics(yva, val_probs[name].argmax(1))
        val_results[name]["cross_entropy"] = prob_cross_entropy(val_probs[name], yva)

    def class_counts(y):
        c = np.bincount(y, minlength=len(LABELS))
        return {n: {"count": int(c[i]), "fraction": float(c[i] / len(y))} for i, n in enumerate(LABELS)}

    results = {
        "data": {**data_stats, "raw_sha256": file_sha256(data_path),
                 "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
                 "unknown_tokens": {"train": st_tr, "val": st_va, "test": st_te},
                 "vocab_size": len(vec.vocab)},
        "class_imbalance": {"all": class_counts(y_all), "train": class_counts(ytr),
                            "val": class_counts(yva), "test": class_counts(yte)},
        "settings": {"seed": SEED, "max_features": MAX_FEATURES, "min_df": MIN_DF,
                     "hidden": HIDDEN, "lr": LEARNING_RATE, "max_epochs": MAX_EPOCHS,
                     "patience": PATIENCE, "batch_size": BATCH_SIZE, "l2": L2,
                     "nb_alpha": 1.0},
        "model_selection": {"criterion": "lowest validation cross-entropy (MLP epoch)",
                            "best_epoch": best_epoch, "epochs_run": len(history),
                            "best_val_ce": history[best_epoch - 1]["val_ce"]},
        "validation": val_results,
        "test": test_results,
        "limitations": [
            "Random sentence-level split; no dates or story groups, so no time-generalisation claim.",
            "Dataset license CC BY-NC-SA 3.0, educational use only.",
            "Single seed, single split; differences between models may be within noise.",
        ],
    }
    prov = Path(data_path).with_name("DATA_PROVENANCE.json")
    if prov.exists():
        with open(prov, "r", encoding="utf-8") as f:
            results["data_provenance"] = json.load(f)

    save_model(mlp, out)
    save_vectorizer(vec, out)
    write_json(out / "results.json", results)
    write_json(out / "split.json", {
        "seed": SEED,
        "counts": results["data"]["split_sizes"],
        "ids": {k: [ids[i] for i in split[k]] for k in split},
        "indices": split,
        "note": "indices refer to the de-duplicated records in first-occurrence file order",
    })
    write_history_csv(out / "history.csv", history)

    pred = test_probs["mlp"].argmax(1)
    wrong = [j for j in range(len(yte)) if pred[j] != yte[j]]
    write_json(out / "error_analysis.json", {
        "model": "mlp",
        "total_test_errors": len(wrong),
        "shown": min(len(wrong), MAX_ERROR_ROWS),
        "errors": [{"id": ids[te[j]], "true": LABELS[yte[j]], "predicted": LABELS[pred[j]],
                    "probabilities": {n: float(test_probs["mlp"][j, i]) for i, n in enumerate(LABELS)}}
                   for j in wrong[:MAX_ERROR_ROWS]],
    })

    preds = predict_texts([p[0] for p in PROBES], mlp, vec)
    write_json(out / "probes.json", {
        "note": "AI-authored synthetic illustrative probes; NOT test metrics and not counted in evaluation.",
        "probes": [{"text": t, "expected_label": lab, **p} for (t, lab), p in zip(PROBES, preds)],
    })
    write_loss_svg(out / "loss_curve.svg", history, best_epoch)

    print(f"best epoch {best_epoch} (val_ce {history[best_epoch - 1]['val_ce']:.4f})")
    for name, r in test_results.items():
        print(f"test {name:12s} acc {r['accuracy']:.3f}  macro_f1 {r['macro_f1']:.3f}  ce {r['cross_entropy']:.4f}")
    print(f"artifacts written to {out}")


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(here / "sentences_allagree.txt"))
    ap.add_argument("--output", default=str(here / "sentiment_run"),
                    help="must not exist; existing outputs are never overwritten")
    ap.add_argument("--predict", metavar="TEXT", help="predict one sentence instead of training")
    ap.add_argument("--model-dir", help="directory with model.npz and vectorizer.json (for --predict)")
    args = ap.parse_args(argv)

    if args.predict is not None:
        if not args.model_dir:
            ap.error("--predict requires --model-dir")
        result = predict_texts([args.predict], load_model(args.model_dir),
                               load_vectorizer(args.model_dir))[0]
        print(json.dumps(result, indent=2))
        return 0
    try:
        run(args.data, args.output)
    except FloatingPointError as e:
        print(f"diagnostic abort: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
