#!/usr/bin/env python3
"""Mean Reciprocal Rank (MRR@k) evaluation across chunking strategies.

Reads ``eval/test_queries.json``, runs each query through the
retrieval pipeline once per chunking strategy, and reports MRR@5 and
MRR@10 plus mean retrieval latency per (strategy, query) pair.

MRR is the average of the reciprocal of the rank position of the
*first* relevant document in the result list. A query whose first
relevant doc is at position 1 contributes 1.0; position 3 contributes
1/3; nothing-relevant contributes 0. We use the
``expected_relevant_source_ids`` list from the test queries to decide
relevance.

This script does not call the LLM — only the retriever. Generation
faithfulness is in ``eval_faithfulness.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from config import get_settings  # noqa: E402
from embeddings.embed import OllamaEmbedder  # noqa: E402
from ingest.chunker import ChunkStrategy  # noqa: E402
from retrieval.store import PgVectorStore  # noqa: E402


def reciprocal_rank(retrieved_source_ids: list[str],
                    relevant_source_ids: list[str]) -> float:
    if not relevant_source_ids:
        return 0.0
    relevant_set = set(relevant_source_ids)
    for i, sid in enumerate(retrieved_source_ids, start=1):
        if sid in relevant_set:
            return 1.0 / i
    return 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", default=str(_REPO / "eval" / "test_queries.json"))
    p.add_argument("--out-csv", default=str(_REPO / "results" / "eval_mrr.csv"))
    p.add_argument("--out-json", default=str(_REPO / "results" / "eval_mrr_summary.json"))
    p.add_argument("--limit", type=int, default=0,
                    help="evaluate at most N queries (0 = all)")
    args = p.parse_args(argv)

    settings = get_settings()
    queries = json.loads(Path(args.queries).read_text())["queries"]
    if args.limit:
        queries = queries[:args.limit]
    queries_with_relevance = [q for q in queries
                                if q.get("expected_relevant_source_ids")]

    embedder = OllamaEmbedder(base_url=settings.ollama_base_url,
                               model=settings.ollama_embedding_model)
    store = PgVectorStore(settings.pg_dsn)

    rows: list[dict] = []
    for strategy in ChunkStrategy:
        for q in queries_with_relevance:
            t0 = time.monotonic()
            try:
                vec = embedder.embed_one(q["question"])
                hits = store.search(strategy=strategy,
                                      query_vector=vec, k=10)
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "strategy": strategy.value,
                    "qid": q["id"],
                    "elapsed_ms": int((time.monotonic() - t0) * 1000),
                    "rr_at_5": 0.0, "rr_at_10": 0.0,
                    "error": str(exc),
                })
                continue
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            ids = [h.parent_source_id for h in hits]
            rows.append({
                "strategy": strategy.value,
                "qid": q["id"],
                "elapsed_ms": elapsed_ms,
                "rr_at_5":  reciprocal_rank(ids[:5], q["expected_relevant_source_ids"]),
                "rr_at_10": reciprocal_rank(ids[:10], q["expected_relevant_source_ids"]),
                "error": "",
            })
    embedder.close()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())
                            if rows else
                            ["strategy", "qid", "elapsed_ms", "rr_at_5",
                             "rr_at_10", "error"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary: dict = {"by_strategy": {}}
    for strategy in ChunkStrategy:
        srows = [r for r in rows if r["strategy"] == strategy.value]
        if not srows:
            continue
        summary["by_strategy"][strategy.value] = {
            "n_queries": len(srows),
            "mrr_at_5":  round(statistics.mean(r["rr_at_5"] for r in srows), 4),
            "mrr_at_10": round(statistics.mean(r["rr_at_10"] for r in srows), 4),
            "mean_latency_ms": round(statistics.mean(r["elapsed_ms"] for r in srows), 1),
            "errors": sum(1 for r in srows if r["error"]),
        }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
