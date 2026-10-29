# ML experiments

Two small text-classification experiments written in plain NumPy on October 29, 2026, as a learning exercise. They were developed with AI assistance, and some source comments note assistant-written code. Training ran locally with NumPy, not on cloud infrastructure. These are **not** the historical news/sentiment service described elsewhere in this repository, not a deployed model, and not a validated forecast.

Short version: classifying the sentiment of financial sentences worked reasonably well. Classifying a later observed price session's open-to-close direction from dated headlines did not beat a simple baseline.

## 1. Sentence sentiment (`sentiment/`)

- **Data:** Financial PhraseBank, `Sentences_AllAgree` subset (sentences where all human annotators agreed). After deduplication: 2,259 sentences, labeled negative / neutral / positive.
- **Model:** TF-IDF over 1,200 terms → one 32-unit ReLU hidden layer → 3-way softmax.
- **Split:** one random split, 1,581 train / 339 validation / 339 test.
- **Result (test):** 87.9% accuracy, macro-F1 0.824.
- **Limits:** this is a single random split, and sentence dates and story overlap are unknown, so related sentences may fall on both sides of the split. The model labels the tone of a sentence. It does not predict returns.

## 2. Headlines → later price move (`news_price/`)

- **Data:** pinned FNSPID headline and daily price CSVs (see `DATA_SOURCES.json`).
- **Task:** group headlines by date, align each group to the first price row strictly after the headline date (gap of 7 calendar days at most), and label that day's open-to-close return as DOWN, FLAT (within ±0.25%), or UP.
- **Split:** chronological by entry date, 718 train / 154 validation / 155 test groups (1,027 total).
- **Result (test):** small MLP 45.2% accuracy, versus 48.4% for always predicting the most common training class. **The model did worse than the baseline.**
- **Limits:** headline dates come from the source dataset and were not verified as point-in-time, so the alignment is retrospective. Ticker/issuer continuity and whether each headline is actually about the company were not verified. Nothing here supports a causal, trading, or profit claim.

Full settings, split dates, and counters are in [`news_price/iterations/VALIDATION_CONTEXT.json`](news_price/iterations/VALIDATION_CONTEXT.json).

### Follow-up iterations (validation only)

After the first run, the test set had already been looked at, so `news_price/iterations/` compares variants **on the 154 validation groups only**. No new test score was computed.

| Variant | Val accuracy | Val macro-F1 |
|---|---|---|
| Majority class (baseline) | 52.6% | 0.230 |
| Original MLP | 50.6% | 0.341 |
| Linear | 51.9% | 0.348 |
| Bigram features | 52.6% | 0.354 |
| Class-balanced | 44.8% | 0.325 |

The bigram variant had the best macro-F1 but only tied the baseline on accuracy. The class-balanced variant still identified none of the actual FLAT days (only 5 in validation). This was an exploratory selection on a small validation set, not evidence of a working model.

### Longer training

A 100-epoch diagnostic run still found its best validation loss at epoch 8 (cross-entropy 0.856). By epoch 100, validation cross-entropy had risen to 2.166, with 98.6% training accuracy against 46.8% validation accuracy. That is overfitting: more training did not help.

Aggregate numbers for both experiments are in [`RESULTS_SUMMARY.json`](RESULTS_SUMMARY.json).

## Running it

Prerequisites: Python 3 and NumPy. Run commands from this `ml-experiments/` folder. A fresh virtual environment is optional:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
```

Download the data. Nothing is fetched until you run this:

```sh
python3 download_data.py --dataset all        # or: sentiment | news-price
```

The helper uses only the standard library. It downloads the pinned files listed in `DATA_SOURCES.json` (a few megabytes), checks their SHA-256 hashes before writing anything, and reads only the one pinned file from the PhraseBank zip. It never overwrites a file that doesn't match, and it writes a `DATA_PROVENANCE.json` next to the data. Downloaded data stays out of git (see `.gitignore`).

Then run the experiments and their tests:

```sh
python3 sentiment/train.py
python3 news_price/price_experiment.py
python3 news_price/iterations/iterate.py --data-dir news_price

python3 -m unittest discover -s . -p "test_download_data.py"
python3 -m unittest discover -s sentiment -p "test_*.py"
python3 -m unittest discover -s news_price -p "test_*.py"
python3 -m unittest discover -s news_price/iterations -p "test_*.py"
```

## Data and licensing

No datasets or trained weights are included in this repository.

- **Financial PhraseBank:** CC BY-NC-SA 3.0, used here for non-commercial education only. Source: <https://huggingface.co/datasets/takala/financial_phrasebank>
- **FNSPID:** the release and footer license statements differ, so it is used here only for local educational experiments. Source: <https://github.com/Zdong104/FNSPID_Financial_News_Dataset>

Check each dataset's terms before any other use.
