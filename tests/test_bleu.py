"""BLEU / ROUGE: own implementation vs. known values and NLTK."""

import math

import pytest
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from recap.bleu import bleu, modified_precision, rouge, tokenize


def test_identical_text_scores_one():
    text = "Al hides the body and drives the car to Los Angeles."
    assert bleu(text, [text])["bleu"] == pytest.approx(1.0)
    assert rouge(text, text)["rougeL"] == pytest.approx(1.0)


def test_clipped_precision_textbook_example():
    # Papineni et al.: "the the the the the the the" vs "the cat is on the mat" -> 2/7
    m, total = modified_precision(tokenize("the the the the the the the"), [tokenize("the cat is on the mat")], 1)
    assert (m, total) == (2, 7)


def test_brevity_penalty():
    res = bleu("the cat", ["the cat is on the mat"])
    assert res["bp"] == pytest.approx(math.exp(1 - 6 / 2))


@pytest.mark.parametrize("cand,ref", [
    ("Al picks up a hitchhiker named Vera who blackmails him.", "Al picks up a hitchhiker named Vera. She blackmails Al."),
    ("The police car picks him up on the highway.", "A police car picks him up moments later, and they drive off."),
    ("Haskell falls from the car and dies after hitting a rock.", "Haskell tumbles out, striking his head on a rock. He is dead."),
])
def test_matches_nltk(cand, ref):
    ours = bleu(cand, [ref])["bleu"]
    theirs = corpus_bleu([[tokenize(ref)]], [tokenize(cand)], smoothing_function=SmoothingFunction().method2)
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_unrelated_text_scores_low():
    assert bleu("completely different words here", ["Al hitchhikes to California"])["bleu"] < 0.1
