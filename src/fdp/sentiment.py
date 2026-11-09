"""Ported weighted lexicon baseline. Heuristic scores are not probabilities."""

import re

NEGATION_WORDS: set[str] = {
    "not",
    "no",
    "never",
    "without",
    "fails",
    "failed",
    "fail",
    "unable",
    "couldn't",
    "cannot",
    "can't",
    "didn't",
    "doesn't",
    "isn't",
    "wasn't",
    "weren't",
    "wouldn't",
    "shouldn't",
    "won't",
    "hardly",
    "barely",
    "scarcely",
}

_NEGATION_WINDOW = 3

POSITIVE_PHRASES: dict[str, float] = {
    "beat estimates": 1.3,
    "beats estimates": 1.3,
    "beat expectations": 1.3,
    "beats expectations": 1.3,
    "beat consensus": 1.2,
    "beats consensus": 1.2,
    "revenue beat": 1.2,
    "earnings beat": 1.3,
    "top-line beat": 1.2,
    "bottom-line beat": 1.2,
    "eps beat": 1.3,
    "profit beat": 1.2,
    "raises guidance": 1.3,
    "raised guidance": 1.3,
    "raises outlook": 1.2,
    "raised outlook": 1.2,
    "raises forecast": 1.2,
    "raised forecast": 1.2,
    "raises full-year": 1.1,
    "raises full year": 1.1,
    "price target raised": 1.1,
    "price target increase": 1.1,
    "raises price target": 1.1,
    "raised price target": 1.1,
    "analyst upgrade": 1.0,
    "upgraded to buy": 1.2,
    "upgraded to overweight": 1.1,
    "upgraded to outperform": 1.1,
    "initiated with buy": 1.0,
    "initiated at buy": 1.0,
    "initiated at overweight": 1.0,
    "deal approved": 0.9,
    "merger approved": 0.9,
    "acquisition complete": 0.8,
    "filed 10-k": 0.2,
    "filed 10-q": 0.15,
    "record revenue": 1.2,
    "record earnings": 1.2,
    "record profit": 1.2,
    "all-time high": 0.9,
    "market share gain": 0.8,
    "strong demand": 0.7,
    "better than expected": 1.1,
    "ahead of expectations": 1.1,
    "ahead of estimates": 1.1,
    "above estimates": 1.0,
    "above consensus": 1.0,
    "dividend increase": 0.8,
    "dividend raised": 0.8,
    "buyback": 0.7,
    "share repurchase": 0.7,
    "stock repurchase": 0.7,
    "beats by": 1.4,
    "beat by": 1.4,
    "topped estimates by": 1.4,
    "exceeded estimates by": 1.4,
    "blew past estimates": 1.5,
    "massive earnings beat": 1.5,
    "blowout quarter": 1.5,
    "blowout earnings": 1.5,
    "upgraded from sell": 1.5,
    "upgraded from underperform": 1.4,
    "upgraded from underweight": 1.4,
    "upgrade from sell to buy": 1.5,
    "upgrade to strong buy": 1.4,
    "raises price target to": 1.2,
    "fda approval": 1.5,
    "fda approved": 1.5,
    "fda approves": 1.5,
    "fda grants approval": 1.5,
    "breakthrough therapy designation": 1.3,
    "priority review designation": 1.2,
    "accelerated approval": 1.2,
    "antitrust approval": 1.1,
    "regulatory approval": 1.1,
    "acquisition announced": 1.3,
    "merger agreement": 1.2,
    "buyout offer": 1.2,
    "takeover bid": 1.1,
    "strategic acquisition": 1.1,
    "all-cash offer": 1.2,
    "premium to current price": 1.1,
    "strategic review": 0.8,
    "fed rate cut": 1.3,
    "rate cut": 1.1,
    "interest rate cut": 1.1,
    "fed pivot": 1.2,
    "fed cuts rates": 1.3,
    "inflation cools": 1.1,
    "inflation falls": 1.1,
    "jobs beat": 1.0,
    "employment beats": 1.0,
    "tariff exemption": 1.0,
    "tariff relief": 1.0,
    "sanctions lifted": 1.0,
    "export ban lifted": 1.0,
    "trade deal signed": 1.1,
    "buidl": 1.3,
    "tokenized fund": 1.2,
    "tokenized treasury": 1.2,
    "tokenized treasuries": 1.2,
    "tokenized money market": 1.2,
    "tokenized us treasury": 1.3,
    "blackrock tokenized": 1.3,
    "blackrock buidl": 1.4,
    "real world asset": 1.1,
    "real-world asset": 1.1,
    "rwa tokenization": 1.3,
    "tokenized real world": 1.2,
    "tokenized stocks": 1.2,
    "tokenized equities": 1.2,
    "tokenized bonds": 1.1,
    "on-chain treasury": 1.1,
    "ondo finance": 1.2,
    "ousg token": 1.1,
    "usdy token": 1.1,
    "defi trading": 1.0,
    "institutional defi": 1.1,
    "on-chain settlement": 1.0,
    "tokenization of assets": 1.2,
    "asset tokenization": 1.2,
    "tokenization launch": 1.2,
    "tokenized collateral": 1.1,
    "chainlink integration": 1.1,
    "chainlink oracle": 1.0,
    "clarity act": 1.3,
    "crypto market structure": 1.2,
    "digital asset regulation": 1.0,
    "stablecoin bill": 1.2,
    "stablecoin legislation": 1.2,
    "crypto regulatory clarity": 1.2,
}

POSITIVE_WORDS: dict[str, float] = {
    "beat": 1.1,
    "surge": 1.2,
    "growth": 0.7,
    "profit": 0.8,
    "upgrade": 1.0,
    "upgraded": 1.0,
    "strong": 0.6,
    "record": 0.7,
    "approved": 0.7,
    "expands": 0.6,
    "expansion": 0.6,
    "outperform": 0.8,
    "outperformed": 0.8,
    "overweight": 0.8,
    "outperformer": 0.8,
    "rallied": 0.8,
    "rally": 0.7,
    "boosted": 0.7,
    "soared": 1.0,
    "soars": 1.0,
    "blowout": 1.1,
    "upbeat": 0.7,
    "climbed": 0.6,
    "jumped": 0.8,
    "gains": 0.6,
    "profitable": 0.8,
    "resilient": 0.5,
    "rebound": 0.6,
    "recovery": 0.5,
    "accelerates": 0.6,
    "accelerating": 0.6,
    "acceleration": 0.6,
}

NEGATIVE_PHRASES: dict[str, float] = {
    "missed estimates": -1.3,
    "misses estimates": -1.3,
    "missed expectations": -1.3,
    "misses expectations": -1.3,
    "missed consensus": -1.2,
    "misses consensus": -1.2,
    "revenue miss": -1.2,
    "earnings miss": -1.3,
    "eps miss": -1.3,
    "below estimates": -1.1,
    "below expectations": -1.1,
    "below consensus": -1.1,
    "worse than expected": -1.1,
    "disappointing results": -1.1,
    "disappointing earnings": -1.2,
    "disappointing revenue": -1.1,
    "cuts guidance": -1.3,
    "cut guidance": -1.3,
    "lowered guidance": -1.2,
    "lowers guidance": -1.2,
    "cuts outlook": -1.3,
    "cut outlook": -1.3,
    "lowered outlook": -1.2,
    "lowers outlook": -1.2,
    "lowers forecast": -1.2,
    "lowered forecast": -1.2,
    "reduces forecast": -1.1,
    "lowered full-year": -1.1,
    "lowered full year": -1.1,
    "price target cut": -1.1,
    "price target reduced": -1.1,
    "cuts price target": -1.1,
    "cut price target": -1.1,
    "reduced price target": -1.0,
    "analyst downgrade": -1.0,
    "downgraded to sell": -1.2,
    "downgraded to underperform": -1.1,
    "downgraded to underweight": -1.1,
    "sec investigation": -1.2,
    "regulatory probe": -1.1,
    "class action": -1.1,
    "filed lawsuit": -1.0,
    "initiated with sell": -1.0,
    "initiated at sell": -1.0,
    "initiated with underweight": -1.0,
    "initiated at underweight": -1.0,
    "initiated with underperform": -0.9,
    "initiated at underperform": -0.9,
    "filed 8-k": -0.1,
    "liquidity concerns": -1.1,
    "going concern": -1.2,
    "margin pressure": -0.8,
    "cost overruns": -0.8,
    "layoffs": -0.7,
    "job cuts": -0.7,
    "restructuring charges": -0.7,
    "impairment charge": -0.8,
    "goodwill impairment": -0.9,
    "misses by": -1.4,
    "missed by": -1.4,
    "missed estimates by": -1.4,
    "trailed estimates by": -1.3,
    "fell short by": -1.3,
    "massive earnings miss": -1.5,
    "downgraded to sell from": -1.5,
    "downgraded from buy to sell": -1.5,
    "downgrade to sell": -1.4,
    "cuts to sell": -1.4,
    "cuts to underperform": -1.3,
    "cuts to underweight": -1.3,
    "cuts price target to": -1.2,
    "fda rejection": -1.5,
    "fda rejects": -1.5,
    "fda rejected": -1.5,
    "complete response letter": -1.4,
    "fda refuse to file": -1.4,
    "sec charges": -1.3,
    "sec enforcement": -1.3,
    "doj investigation": -1.2,
    "antitrust blocked": -1.3,
    "merger blocked": -1.3,
    "deal blocked": -1.2,
    "regulatory rejection": -1.2,
    "merger rejected": -1.1,
    "hostile takeover": -0.9,
    "acquisition falls through": -1.1,
    "deal terminated": -1.2,
    "deal collapsed": -1.2,
    "deal breaks": -1.1,
    "fed rate hike": -1.2,
    "rate hike": -1.0,
    "interest rate hike": -1.0,
    "inflation surprise": -1.1,
    "inflation accelerates": -1.1,
    "jobs miss": -1.0,
    "employment miss": -1.0,
    "recession fears": -1.0,
    "recession warning": -1.1,
    "tariff announcement": -1.0,
    "tariffs imposed": -1.1,
    "new tariffs": -1.0,
    "sanctions imposed": -1.1,
    "export ban": -1.1,
    "trade war": -1.0,
    "supply chain disruption": -0.9,
    "crypto ban": -1.3,
    "tokenization banned": -1.3,
    "defi crackdown": -1.2,
    "sec crypto enforcement": -1.3,
}

NEGATIVE_WORDS: dict[str, float] = {
    "miss": -1.1,
    "drop": -0.9,
    "decline": -0.8,
    "downgrade": -1.0,
    "downgraded": -1.0,
    "underweight": -0.9,
    "underperform": -0.9,
    "underperformer": -0.9,
    "lawsuit": -1.0,
    "investigation": -0.9,
    "weak": -0.7,
    "loss": -0.7,
    "recall": -1.0,
    "plunged": -1.0,
    "plunge": -1.0,
    "slides": -0.8,
    "sliding": -0.7,
    "slump": -0.8,
    "slumped": -0.8,
    "tumbled": -0.9,
    "tumble": -0.9,
    "fell": -0.6,
    "dropped": -0.8,
    "disappointing": -0.9,
    "warning": -0.7,
    "cautious": -0.5,
    "concern": -0.5,
    "headwinds": -0.6,
    "impairment": -0.8,
    "restructuring": -0.6,
    "layoff": -0.7,
    "bankrupt": -1.3,
    "bankruptcy": -1.3,
    "default": -1.1,
    "fraud": -1.2,
    "disappoints": -1.0,
    "selloff": -0.8,
    "sell-off": -0.8,
    "sinks": -0.9,
    "sank": -0.8,
}


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser (lowercase)."""
    return re.findall("[a-z']+", text.lower())


def _check_negated(
    tokens: list[str], phrase_start_idx: int, window: int = _NEGATION_WINDOW
) -> bool:
    """Return True if any negation word appears within *window* tokens before phrase_start_idx."""
    start = max(0, phrase_start_idx - window)
    return any((t in NEGATION_WORDS for t in tokens[start:phrase_start_idx]))


_PCT_BEAT_RE = re.compile(
    "beats?\\s+(?:estimates?|expectations?|consensus)\\s+by\\s+(\\d+(?:\\.\\d+)?)\\s*%"
)

_PCT_MISS_RE = re.compile(
    "miss(?:es|ed)?\\s+(?:estimates?|expectations?|consensus)\\s+by\\s+(\\d+(?:\\.\\d+)?)\\s*%"
)

_REVENUE_GROWTH_RE = re.compile(
    "revenue\\s+(?:growth|up|grew|grew\\s+by)\\s+(\\d+(?:\\.\\d+)?)\\s*%"
)

_DEAL_BILLIONS_RE = re.compile(
    "\\$(\\d+(?:\\.\\d+)?)\\s*(?:b|bn|billion)\\s+(?:deal|acquisition|merger|offer|buyout)"
)

_TARGET_RAISE_RE = re.compile(
    "(?:raises?|raised|increase[sd]?)\\s+price\\s+target\\s+(?:to\\s+)?\\$(\\d+(?:\\.\\d+)?)"
)

_PCT_HIGH_THRESHOLD = 10.0

_PCT_MEDIUM_THRESHOLD = 5.0


def _magnitude_boost(lowered: str) -> float:
    """Compute an additive score bonus (0.0–0.6) based on numeric magnitude cues.

    Returns a positive float; the caller decides the direction (add if bullish,
    subtract if bearish).  Never returns negative — direction is determined by
    the existing phrase score.
    """
    boost = 0.0
    for m in _PCT_BEAT_RE.finditer(lowered):
        pct = float(m.group(1))
        if pct >= _PCT_HIGH_THRESHOLD:
            boost = max(boost, 0.6)
        elif pct >= _PCT_MEDIUM_THRESHOLD:
            boost = max(boost, 0.3)
    for m in _PCT_MISS_RE.finditer(lowered):
        pct = float(m.group(1))
        if pct >= _PCT_HIGH_THRESHOLD:
            boost = max(boost, 0.6)
        elif pct >= _PCT_MEDIUM_THRESHOLD:
            boost = max(boost, 0.3)
    for m in _REVENUE_GROWTH_RE.finditer(lowered):
        pct = float(m.group(1))
        if pct >= 20.0:
            boost = max(boost, 0.4)
        elif pct >= 10.0:
            boost = max(boost, 0.2)
    for m in _DEAL_BILLIONS_RE.finditer(lowered):
        billions = float(m.group(1))
        if billions >= 10.0:
            boost = max(boost, 0.5)
        elif billions >= 1.0:
            boost = max(boost, 0.3)
    return min(boost, 0.6)


def score_text(text: str) -> tuple[float, float]:
    """Score *text* using the lexicon with negation handling.

    Returns (normalised_score, confidence) where score ∈ [-1, 1].
    Confidence rises with number of phrase hits.
    """
    lowered = text.lower()
    tokens = _tokenize(lowered)
    score = 0.0
    hits = 0
    for phrase, value in POSITIVE_PHRASES.items():
        idx = lowered.find(phrase)
        while idx != -1:
            token_pos = len(_tokenize(lowered[:idx]))
            if _check_negated(tokens, token_pos):
                score -= value
            else:
                score += value
            hits += 1
            idx = lowered.find(phrase, idx + len(phrase))
    for phrase, value in NEGATIVE_PHRASES.items():
        idx = lowered.find(phrase)
        while idx != -1:
            token_pos = len(_tokenize(lowered[:idx]))
            if _check_negated(tokens, token_pos):
                score -= value
            else:
                score += value
            hits += 1
            idx = lowered.find(phrase, idx + len(phrase))
    for phrase, value in POSITIVE_WORDS.items():
        for mp in POSITIVE_PHRASES:
            if phrase in mp and mp in lowered:
                break
        else:
            for m in re.finditer("\\b" + re.escape(phrase) + "\\b", lowered):
                token_pos = len(_tokenize(lowered[: m.start()]))
                if _check_negated(tokens, token_pos):
                    score -= value
                else:
                    score += value
                hits += 1
    for phrase, value in NEGATIVE_WORDS.items():
        for mp in NEGATIVE_PHRASES:
            if phrase in mp and mp in lowered:
                break
        else:
            for m in re.finditer("\\b" + re.escape(phrase) + "\\b", lowered):
                token_pos = len(_tokenize(lowered[: m.start()]))
                if _check_negated(tokens, token_pos):
                    score -= value
                else:
                    score += value
                hits += 1
    mag_boost = _magnitude_boost(lowered)
    if mag_boost != 0.0:
        if score > 0:
            score += mag_boost
        elif score < 0:
            score -= mag_boost
    normalized = max(-1.0, min(1.0, score / 4.0))
    confidence = min(1.0, 0.15 + hits * 0.15)
    return (normalized, confidence)
