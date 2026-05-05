"""pgvector-backed chunk store.

One table per chunking strategy so the retrieval-time decision is
just "which table do I look in?". That keeps the schema simple at
the cost of a small write-time duplication; for the eval scope of
this repo (a few thousand CVEs + a handful of PDFs) the trade is
clearly correct.

We use the cosine-distance operator (`<=>`) and a single-column
index (HNSW) per table. HNSW is the default index that ships with
recent pgvector releases and is the right choice when you don't
know your query distribution in advance.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import psycopg
from psycopg.rows import dict_row

from ingest.chunker import Chunk, ChunkStrategy


_log = logging.getLogger("retrieval.store")


@dataclass
class RetrievedChunk:
    text: str
    score: float                 # cosine similarity in [0, 1]; 1 = identical
    parent_source_id: str
    parent_source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _table_for(strategy: ChunkStrategy) -> str:
    return f"chunks_{strategy.value}"


def _safe_table(strategy: ChunkStrategy) -> str:
    """Validate the table name is one we control before splicing it
    into a SQL string. Defence in depth — `_table_for` already
    guarantees the prefix is fixed and the suffix is enum-valued, but
    we re-check at the wire."""
    table = _table_for(strategy)
    allowed = {f"chunks_{s.value}" for s in ChunkStrategy}
    if table not in allowed:
        raise ValueError(f"refusing unknown table {table!r}")
    return table


_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table} (
    id              BIGSERIAL PRIMARY KEY,
    text            TEXT     NOT NULL,
    parent_source   TEXT     NOT NULL,
    parent_source_id TEXT    NOT NULL,
    chunk_index     INTEGER  NOT NULL,
    metadata        JSONB    NOT NULL DEFAULT '{{}}'::jsonb,
    embedding       vector({dim}) NOT NULL,
    UNIQUE (parent_source_id, chunk_index)
);

-- HNSW index on cosine distance. We tune m/ef_construction to the
-- defaults pgvector ships with — they're sane for corpora <100k.
CREATE INDEX IF NOT EXISTS {table}_hnsw
    ON {table}
    USING hnsw (embedding vector_cosine_ops);
"""


class PgVectorStore:
    """Thin wrapper around psycopg + pgvector. Holds a single
    connection for the lifetime of the call site; this matches the
    eval-script use case better than a connection pool, which would
    be overkill."""

    def __init__(self, dsn: str):
        self._dsn = dsn

    # ----------------------------------------- DDL ----------------
    def init_schema(self, *, dim: int) -> None:
        """Create the per-strategy chunk tables. Idempotent."""
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                for strategy in ChunkStrategy:
                    table = _safe_table(strategy)
                    cur.execute(_DDL.format(table=table, dim=dim))
            conn.commit()

    # ----------------------------------------- write -------------
    def upsert(
        self,
        *,
        strategy: ChunkStrategy,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: "
                f"{len(chunks)} vs {len(embeddings)}")
        table = _safe_table(strategy)
        if not chunks:
            return 0
        rows: list[tuple] = []
        for chunk, vec in zip(chunks, embeddings):
            rows.append((
                chunk.text,
                chunk.parent_source,
                chunk.parent_source_id,
                chunk.chunk_index,
                json.dumps(chunk.metadata or {}),
                _vector_literal(vec),
            ))
        sql = (
            f"INSERT INTO {table} "
            "(text, parent_source, parent_source_id, chunk_index, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector) "
            "ON CONFLICT (parent_source_id, chunk_index) DO UPDATE SET "
            "text = EXCLUDED.text, embedding = EXCLUDED.embedding, "
            "metadata = EXCLUDED.metadata"
        )
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()
        return len(rows)

    # ----------------------------------------- read --------------
    def search(
        self,
        *,
        strategy: ChunkStrategy,
        query_vector: Sequence[float],
        k: int = 10,
    ) -> list[RetrievedChunk]:
        table = _safe_table(strategy)
        sql = (
            f"SELECT text, parent_source, parent_source_id, chunk_index, metadata, "
            f"       1 - (embedding <=> %s::vector) AS similarity "
            f"FROM {table} "
            f"ORDER BY embedding <=> %s::vector "
            f"LIMIT %s"
        )
        vec_lit = _vector_literal(query_vector)
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_lit, vec_lit, k))
                rows = cur.fetchall()
        return [RetrievedChunk(
            text=r["text"],
            score=float(r["similarity"]),
            parent_source=r["parent_source"],
            parent_source_id=r["parent_source_id"],
            chunk_index=r["chunk_index"],
            metadata=r["metadata"] or {},
        ) for r in rows]

    # ----------------------------------------- ops --------------
    def count(self, strategy: ChunkStrategy) -> int:
        table = _safe_table(strategy)
        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM {table}")
                row = cur.fetchone()
        return int(row[0]) if row else 0


def _vector_literal(vec: Sequence[float]) -> str:
    """pgvector accepts a `[1,2,3]` text literal cast to ::vector."""
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
