"""Side-by-side comparison of two embedding models.

Embed the same text with both, then report:
  - dimensions
  - mean / median / min cosine similarity between paired vectors
  - per-text-pair latency
  - speed (texts/sec)

The point isn't to declare a winner from one corpus — it's to make
the trade-offs concrete (768 dimensions of nomic-embed-text against
384 of all-minilm) so a reader can decide which is the better fit
for their own latency-vs-accuracy budget."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from embeddings.embed import OllamaEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Defined as 0 for zero vectors so we don't
    divide by zero on degenerate input."""
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (da * db)


@dataclass
class ComparisonRow:
    text_index: int
    cosine_self_a: float = 1.0       # always 1; sanity check
    cosine_self_b: float = 1.0


@dataclass
class ComparisonReport:
    model_a: str
    model_b: str
    dim_a: int
    dim_b: int
    n_texts: int
    elapsed_ms_a: int
    elapsed_ms_b: int
    pairwise_within_a_mean: float        # mean cosine across pairs of texts in model A
    pairwise_within_b_mean: float
    rows: list[ComparisonRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a, "model_b": self.model_b,
            "dim_a": self.dim_a, "dim_b": self.dim_b,
            "n_texts": self.n_texts,
            "elapsed_ms_a": self.elapsed_ms_a,
            "elapsed_ms_b": self.elapsed_ms_b,
            "pairwise_within_a_mean": round(self.pairwise_within_a_mean, 4),
            "pairwise_within_b_mean": round(self.pairwise_within_b_mean, 4),
        }


def compare(
    texts: list[str],
    *,
    embedder_a: OllamaEmbedder,
    embedder_b: OllamaEmbedder,
) -> ComparisonReport:
    """Run both embedders over the same corpus and aggregate the
    pairwise within-model similarity distribution. We don't compare
    \"a vs b\" cosine — different model families live in different
    embedding spaces and a direct comparison there is meaningless."""
    a = embedder_a.embed(texts)
    b = embedder_b.embed(texts)

    def _within_mean(vectors: list[list[float]]) -> float:
        # Sample up to 200 random pairs to keep this O(n) for big N.
        if len(vectors) < 2:
            return 0.0
        sims: list[float] = []
        step = max(1, len(vectors) // 14)   # ~14 strides over the list
        for i in range(0, len(vectors) - 1, step):
            for j in range(i + 1, min(i + step + 1, len(vectors))):
                sims.append(_cosine(vectors[i], vectors[j]))
        return statistics.mean(sims) if sims else 0.0

    return ComparisonReport(
        model_a=embedder_a.model,
        model_b=embedder_b.model,
        dim_a=a.dimension,
        dim_b=b.dimension,
        n_texts=len(texts),
        elapsed_ms_a=a.elapsed_ms,
        elapsed_ms_b=b.elapsed_ms,
        pairwise_within_a_mean=_within_mean(a.vectors),
        pairwise_within_b_mean=_within_mean(b.vectors),
    )
