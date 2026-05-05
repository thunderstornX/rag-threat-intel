#!/usr/bin/env python3
"""Answer-faithfulness evaluation.

For each query in the test set we run the full RAG loop once
(retrieve -> rerank -> generate) and score the answer along three
axes:

  1. **Citation density** — fraction of sentences in the answer that
     end with a [doc_N] tag. A high-quality answer should cite at
     least most of its sentences; the prompt asks for it explicitly.

  2. **Citation validity** — every cited [doc_N] tag must point to a
     valid index in the retrieved set. Invalid tags count as
     fabrications.

  3. **Refusal honesty** — for queries marked
     ``expected_refusal: true``, the answer must contain the canonical
     refusal sentence. For queries NOT so marked, refusing is a
     failure.

We *don't* use an LLM-as-judge to label the prose; the protocol is
deterministic by design. A more rigorous offline study would add a
human pass over a sample of the answers; the script reports the
sample IDs to make that easy."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from config import get_settings  # noqa: E402
from embeddings.embed import OllamaEmbedder  # noqa: E402
from generation.generator import OllamaGenerator  # noqa: E402
from ingest.chunker import ChunkStrategy  # noqa: E402
from retrieval.reranker import mmr_rerank  # noqa: E402
from retrieval.store import PgVectorStore  # noqa: E402


_CITATION_RE = re.compile(r"\[doc_\d+\]")
_REFUSAL_PHRASE = "do not answer this question"


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
              if s.strip()]
    return parts


def score_answer(*, answer: str, n_docs: int, expected_refusal: bool):
    refused = _REFUSAL_PHRASE in answer.lower()
    sentences = split_sentences(answer)
    cited_sentences = sum(1 for s in sentences if _CITATION_RE.search(s))

    citation_tags = [int(m.group(1)) for m in
                      re.finditer(r"\[doc_(\d+)\]", answer)]
    invalid_tags = [t for t in citation_tags if not (1 <= t <= n_docs)]

    citation_density = (cited_sentences / len(sentences)) if sentences else 0.0
    citation_validity = (
        1.0 if not citation_tags else
        1.0 - (len(invalid_tags) / len(citation_tags))
    )

    if expected_refusal:
        refusal_honesty = 1.0 if refused else 0.0
    else:
        refusal_honesty = 0.0 if refused else 1.0

    return {
        "n_sentences": len(sentences),
        "cited_sentences": cited_sentences,
        "citation_density": round(citation_density, 3),
        "citation_validity": round(citation_validity, 3),
        "refused": refused,
        "refusal_honesty": refusal_honesty,
        "invalid_tags": invalid_tags,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", default=str(_REPO / "eval" / "test_queries.json"))
    p.add_argument("--strategy", default="semantic",
                    choices=[s.value for s in ChunkStrategy])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--top-n", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-csv", default=str(_REPO / "results" / "eval_faithfulness.csv"))
    p.add_argument("--out-json", default=str(_REPO / "results" / "eval_faithfulness_summary.json"))
    args = p.parse_args(argv)

    settings = get_settings()
    queries = json.loads(Path(args.queries).read_text())["queries"]
    if args.limit:
        queries = queries[:args.limit]

    embedder = OllamaEmbedder(base_url=settings.ollama_base_url,
                               model=settings.ollama_embedding_model)
    generator = OllamaGenerator(base_url=settings.ollama_base_url,
                                  model=settings.ollama_generation_model)
    store = PgVectorStore(settings.pg_dsn)
    strategy = ChunkStrategy(args.strategy)

    rows: list[dict] = []
    for q in queries:
        t0 = time.monotonic()
        try:
            qvec = embedder.embed_one(q["question"])
            cands = store.search(strategy=strategy, query_vector=qvec,
                                   k=args.top_k)
            if not cands:
                rows.append({
                    "qid": q["id"], "category": q.get("category", ""),
                    "expected_refusal": bool(q.get("expected_refusal")),
                    "n_docs": 0, "answer": "",
                    "citation_density": 0.0, "citation_validity": 1.0,
                    "refused": True,
                    "refusal_honesty": 1.0 if q.get("expected_refusal") else 0.0,
                    "elapsed_ms": int((time.monotonic() - t0) * 1000),
                    "error": "",
                })
                continue
            cand_vecs = embedder.embed([c.text for c in cands]).vectors
            reranked = mmr_rerank(query_vector=qvec, candidates=cands,
                                    candidate_vectors=cand_vecs,
                                    top_n=args.top_n)
            result = generator.generate(question=q["question"],
                                          documents=reranked)
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "qid": q["id"], "category": q.get("category", ""),
                "expected_refusal": bool(q.get("expected_refusal")),
                "n_docs": 0, "answer": "",
                "citation_density": 0.0, "citation_validity": 0.0,
                "refused": False, "refusal_honesty": 0.0,
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
                "error": str(exc),
            })
            continue
        scored = score_answer(answer=result.answer,
                                n_docs=len(reranked),
                                expected_refusal=bool(q.get("expected_refusal")))
        rows.append({
            "qid": q["id"],
            "category": q.get("category", ""),
            "expected_refusal": bool(q.get("expected_refusal")),
            "n_docs": len(reranked),
            "answer": result.answer.replace("\n", " ")[:600],
            "citation_density": scored["citation_density"],
            "citation_validity": scored["citation_validity"],
            "refused": scored["refused"],
            "refusal_honesty": scored["refusal_honesty"],
            "elapsed_ms": int((time.monotonic() - t0) * 1000),
            "error": "",
        })
    embedder.close()
    generator.close()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "qid", "category", "expected_refusal", "n_docs", "answer",
        "citation_density", "citation_validity", "refused",
        "refusal_honesty", "elapsed_ms", "error",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if rows:
        summary = {
            "n_queries": len(rows),
            "strategy": args.strategy,
            "mean_citation_density": round(
                sum(r["citation_density"] for r in rows) / len(rows), 3),
            "mean_citation_validity": round(
                sum(r["citation_validity"] for r in rows) / len(rows), 3),
            "mean_refusal_honesty": round(
                sum(r["refusal_honesty"] for r in rows) / len(rows), 3),
            "refusal_count": sum(1 for r in rows if r["refused"]),
            "expected_refusal_count": sum(1 for r in rows if r["expected_refusal"]),
            "mean_elapsed_ms": round(
                sum(r["elapsed_ms"] for r in rows) / len(rows), 1),
            "errors": sum(1 for r in rows if r["error"]),
        }
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
