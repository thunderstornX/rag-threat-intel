"""FastAPI service exposing the RAG pipeline.

The endpoint is intentionally small:

    POST /query   { question, strategy, top_k, top_n }
              ->  { answer, refused, sources, ... }

Heavy lifting (chunk store, embedder, generator) lives in their own
modules. The API just composes them — that keeps the wire schema
versionable independent of the internal layout."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from api.models import (
    HealthResponse, QueryRequest, QueryResponse, SourceCitation,
)
from config import Settings, get_settings
from embeddings.embed import OllamaEmbedder
from generation.generator import OllamaGenerator, GenerationError
from ingest.chunker import ChunkStrategy
from retrieval.reranker import mmr_rerank
from retrieval.store import PgVectorStore


_log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log = logging.getLogger("api.boot")
    log.info("rag-threat-intel API starting")
    yield
    log.info("rag-threat-intel API stopping")


app = FastAPI(
    title="rag-threat-intel",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Cheap liveness check — does not touch Ollama or pgvector. The
    detailed component check is `/health/ready`."""
    return HealthResponse(status="ok", components={"api": True})


@app.get("/health/ready", response_model=HealthResponse)
async def ready(settings: Settings = Depends(get_settings)) -> HealthResponse:
    components: dict[str, bool] = {}
    # pgvector
    try:
        store = PgVectorStore(settings.pg_dsn)
        store.count(ChunkStrategy.SEMANTIC)
        components["pgvector"] = True
    except Exception as exc:  # noqa: BLE001 - probe is allowed to fail
        _log.warning("pgvector probe failed: %s", exc)
        components["pgvector"] = False
    # ollama
    try:
        emb = OllamaEmbedder(base_url=settings.ollama_base_url,
                              model=settings.ollama_embedding_model)
        emb.embed_one("ping")
        emb.close()
        components["ollama"] = True
    except Exception as exc:  # noqa: BLE001
        _log.warning("ollama probe failed: %s", exc)
        components["ollama"] = False
    overall = "ok" if all(components.values()) else "degraded"
    return HealthResponse(status=overall, components=components)


@app.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    strategy = ChunkStrategy(payload.strategy)
    embedder = OllamaEmbedder(base_url=settings.ollama_base_url,
                               model=settings.ollama_embedding_model)
    store = PgVectorStore(settings.pg_dsn)
    generator = OllamaGenerator(base_url=settings.ollama_base_url,
                                  model=settings.ollama_generation_model)

    try:
        # ---- retrieve ------------------------------------------------
        t0 = time.monotonic()
        q_vec = embedder.embed_one(payload.question)
        candidates = store.search(strategy=strategy,
                                    query_vector=q_vec,
                                    k=payload.top_k)
        retrieval_ms = int((time.monotonic() - t0) * 1000)

        if not candidates:
            return QueryResponse(
                answer="The supplied documents do not answer this question.",
                refused=True,
                strategy=payload.strategy,
                cited_doc_ids=[],
                sources=[],
                elapsed_ms_retrieval=retrieval_ms,
                elapsed_ms_generation=0,
            )

        # ---- rerank with MMR for diversity ---------------------------
        # We re-embed only the candidates' text to get vectors locally;
        # in a hot-path service you'd round-trip them out of pgvector.
        cand_vectors = embedder.embed([c.text for c in candidates]).vectors
        reranked = mmr_rerank(
            query_vector=q_vec,
            candidates=candidates,
            candidate_vectors=cand_vectors,
            top_n=payload.top_n,
        )

        # ---- generate ------------------------------------------------
        result = generator.generate(question=payload.question,
                                      documents=reranked)
    except GenerationError as exc:
        raise HTTPException(502, detail={
            "code": "generation.upstream",
            "message": str(exc),
        })
    finally:
        embedder.close()
        generator.close()

    sources = [SourceCitation(
        doc_id=i + 1,
        parent_source=c.parent_source,
        parent_source_id=c.parent_source_id,
        chunk_index=c.chunk_index,
        score=round(c.score, 4),
        snippet=(c.text[:240] + "…") if len(c.text) > 240 else c.text,
    ) for i, c in enumerate(reranked)]

    return QueryResponse(
        answer=result.answer,
        refused=result.refused,
        strategy=payload.strategy,
        cited_doc_ids=result.cited_doc_ids,
        sources=sources,
        elapsed_ms_retrieval=retrieval_ms,
        elapsed_ms_generation=result.elapsed_ms,
    )
