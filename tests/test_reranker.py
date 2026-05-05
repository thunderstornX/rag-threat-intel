"""MMR reranker invariants."""
from __future__ import annotations

import pytest

from retrieval.reranker import _cosine, mmr_rerank
from retrieval.store import RetrievedChunk


def _chunk(idx: int) -> RetrievedChunk:
    return RetrievedChunk(text=f"t{idx}", score=0.0,
                            parent_source="x", parent_source_id=str(idx),
                            chunk_index=0)


def test_cosine_orthogonal_is_zero():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_identical_is_one():
    assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0, rel=1e-9)


def test_mmr_picks_diverse_top_n():
    """Three near-duplicate candidates and one dissimilar one. With
    lambda=0.5, MMR should not stack all duplicates at the top."""
    cands = [_chunk(i) for i in range(4)]
    # near-duplicate vectors a/b/c, distinctive d
    vecs = [
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.98, 0.02, 0.0],
        [0.0, 0.0, 1.0],
    ]
    q = [1.0, 0.0, 0.0]
    # lambda=0.3 weights diversity more heavily than relevance, so
    # the second pick must be the dissimilar candidate (d), not one
    # of a's near-duplicates.
    out = mmr_rerank(query_vector=q, candidates=cands,
                       candidate_vectors=vecs, top_n=2, lambda_=0.3)
    assert len(out) == 2
    # The first pick is the most relevant (a). The second pick should
    # NOT be one of a's near-duplicates, because diversity penalises
    # them. So the second pick has to be the distinctive one.
    chosen_ids = {o.parent_source_id for o in out}
    assert "0" in chosen_ids
    assert "3" in chosen_ids


def test_mmr_with_top_n_zero_returns_empty():
    assert mmr_rerank(query_vector=[1.0], candidates=[_chunk(0)],
                        candidate_vectors=[[1.0]], top_n=0) == []


def test_mmr_length_mismatch_raises():
    with pytest.raises(ValueError):
        mmr_rerank(query_vector=[1.0],
                    candidates=[_chunk(0), _chunk(1)],
                    candidate_vectors=[[1.0]], top_n=2)
