"""eval_mrr and eval_faithfulness helper tests."""
from __future__ import annotations

from eval.eval_faithfulness import score_answer, split_sentences
from eval.eval_mrr import reciprocal_rank


# --- MRR --------------------------------------------------------------

def test_rr_at_1_when_relevant_is_first():
    assert reciprocal_rank(["A", "B"], ["A"]) == 1.0


def test_rr_at_3_is_one_third():
    assert reciprocal_rank(["X", "Y", "A"], ["A"]) == 1 / 3


def test_rr_zero_when_relevant_not_in_list():
    assert reciprocal_rank(["X", "Y", "Z"], ["A"]) == 0.0


def test_rr_zero_when_no_relevant_supplied():
    assert reciprocal_rank(["A"], []) == 0.0


# --- faithfulness scorer ---------------------------------------------

def test_split_sentences_basic():
    assert split_sentences("Hi. There. World!") == ["Hi.", "There.", "World!"]


def test_score_full_citation_density():
    answer = "Sentence one [doc_1]. Sentence two [doc_2]."
    s = score_answer(answer=answer, n_docs=2, expected_refusal=False)
    assert s["citation_density"] == 1.0
    assert s["citation_validity"] == 1.0
    assert s["refused"] is False


def test_score_invalid_tag_is_caught():
    answer = "Real [doc_1]. Made up [doc_42]."
    s = score_answer(answer=answer, n_docs=1, expected_refusal=False)
    assert s["citation_validity"] < 1.0
    assert 42 in s["invalid_tags"]


def test_score_refusal_honesty_when_expected():
    answer = "The supplied documents do not answer this question."
    s = score_answer(answer=answer, n_docs=2, expected_refusal=True)
    assert s["refused"] is True
    assert s["refusal_honesty"] == 1.0


def test_score_refusal_dishonest_when_unexpected():
    answer = "The supplied documents do not answer this question."
    s = score_answer(answer=answer, n_docs=2, expected_refusal=False)
    assert s["refusal_honesty"] == 0.0
