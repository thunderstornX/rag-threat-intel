"""MMR (Maximal Marginal Relevance) diversity reranking.

The original MMR formulation balances similarity to the query against
similarity to documents already in the result set:

    MMR(d) = lambda * sim(q, d) - (1 - lambda) * max sim(d, d') for d' in selected

Higher lambda -> more relevance, lower diversity. Lambda=0.5 is the
common default. We rerank a top-k retrieval set down to a smaller
top-n where the *n* are spread across distinct sources rather than
all crowded around the same paragraph."""
from __future__ import annotations

import math
from typing import Sequence

from retrieval.store import RetrievedChunk


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


def mmr_rerank(
    *,
    query_vector: Sequence[float],
    candidates: list[RetrievedChunk],
    candidate_vectors: list[list[float]],
    top_n: int,
    lambda_: float = 0.5,
) -> list[RetrievedChunk]:
    """Pick ``top_n`` candidates greedily by MMR.

    The caller supplies pre-computed ``candidate_vectors`` because
    fetching them from pgvector during reranking would be a
    round-trip we can avoid (we already paid for them at retrieval
    time)."""
    if not candidates or top_n <= 0:
        return []
    if len(candidates) != len(candidate_vectors):
        raise ValueError("candidates / candidate_vectors length mismatch")

    # Pre-compute query similarities (these come straight from the
    # store but we recompute against the supplied vectors so the
    # function works in tests too).
    sim_to_q = [_cosine(query_vector, v) for v in candidate_vectors]
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while len(selected) < top_n and remaining:
        if not selected:
            best = max(remaining, key=lambda i: sim_to_q[i])
        else:
            best = None
            best_score = -float("inf")
            for i in remaining:
                max_sim_to_selected = max(
                    _cosine(candidate_vectors[i], candidate_vectors[j])
                    for j in selected
                )
                score = (lambda_ * sim_to_q[i]
                          - (1 - lambda_) * max_sim_to_selected)
                if score > best_score:
                    best_score, best = score, i
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]
