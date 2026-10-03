"""BLEU and ROUGE implemented from first principles.

BLEU (Papineni et al., 2002):
    p_n  = clipped n-gram matches / candidate n-grams        (modified precision, n = 1..4)
    BP   = 1 if c > r else exp(1 - r / c)                     (brevity penalty)
    BLEU = BP * exp( sum_n w_n * log p_n ),  w_n = 1/N
Smoothing "add-one for n > 1" (Lin & Och, 2004; NLTK method2) avoids log(0) on short texts.

The results are cross-checked against NLTK and sacreBLEU in evaluate.py and in the unit tests.
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    """Lower-case word/punctuation tokenizer (keeps contractions such as don't)."""
    return TOKEN_RE.findall(text.lower().replace("’", "'"))


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def modified_precision(candidate: list[str], references: list[list[str]], n: int) -> tuple[int, int]:
    """Clipped n-gram matches and total candidate n-grams."""
    cand = ngrams(candidate, n)
    max_ref: Counter = Counter()
    for ref in references:
        for g, c in ngrams(ref, n).items():
            max_ref[g] = max(max_ref[g], c)
    matches = sum(min(c, max_ref[g]) for g, c in cand.items())
    return matches, max(0, len(candidate) - n + 1)


def bleu(candidate: str | list[str], references: list[str] | list[list[str]], max_n: int = 4,
         smooth: bool = True) -> dict:
    """Sentence/document BLEU with per-order precisions and the brevity penalty."""
    cand = tokenize(candidate) if isinstance(candidate, str) else candidate
    refs = [tokenize(r) if isinstance(r, str) else r for r in references]
    if not cand:
        return {"bleu": 0.0, "precisions": [0.0] * max_n, "bp": 0.0, "c": 0, "r": 0}
    c = len(cand)
    # Effective reference length: the reference closest in length to the candidate.
    r = min((abs(len(ref) - c), len(ref)) for ref in refs)[1]
    precisions, log_sum = [], 0.0
    for n in range(1, max_n + 1):
        m, total = modified_precision(cand, refs, n)
        if smooth and n > 1:
            m, total = m + 1, total + 1
        p = m / total if total else 0.0
        precisions.append(p)
        log_sum += math.log(p) / max_n if p > 0 else -math.inf
    bp = 1.0 if c > r else math.exp(1 - r / c)
    score = bp * math.exp(log_sum) if log_sum > -math.inf else 0.0
    return {"bleu": score, "precisions": precisions, "bp": bp, "c": c, "r": r}


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge(candidate: str, reference: str) -> dict:
    """ROUGE-1, ROUGE-2 and ROUGE-L F1 (recall-oriented counterpart to BLEU)."""
    c, r = tokenize(candidate), tokenize(reference)
    c = [t for t in c if t.isalnum()]
    r = [t for t in r if t.isalnum()]
    out = {}
    for n in (1, 2):
        cg, rg = ngrams(c, n), ngrams(r, n)
        overlap = sum((cg & rg).values())
        p = overlap / max(1, sum(cg.values()))
        rec = overlap / max(1, sum(rg.values()))
        out[f"rouge{n}"] = 2 * p * rec / (p + rec) if p + rec else 0.0
    lcs = _lcs(c, r)
    p, rec = lcs / max(1, len(c)), lcs / max(1, len(r))
    out["rougeL"] = 2 * p * rec / (p + rec) if p + rec else 0.0
    return out
