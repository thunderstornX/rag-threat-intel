#!/usr/bin/env python3
"""One-shot ingest: fetch CVEs + load PDFs, embed, write to pgvector.

Usage:

    python -m ingest.bootstrap                  # all three strategies
    python -m ingest.bootstrap --strategy semantic
    python -m ingest.bootstrap --no-cves        # only PDFs
    python -m ingest.bootstrap --cve-limit 100

Idempotent: re-running just upserts new chunks. Each strategy
populates its own pgvector table so a later eval can compare
retrieval quality without re-ingesting.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rich.console import Console  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from config import get_settings  # noqa: E402
from embeddings.embed import OllamaEmbedder  # noqa: E402
from ingest.chunker import ChunkStrategy, chunk_documents  # noqa: E402
from ingest.nvd_fetcher import fetch_recent_cves  # noqa: E402
from ingest.pdf_loader import load_all_pdfs  # noqa: E402
from retrieval.store import PgVectorStore  # noqa: E402


_log = logging.getLogger("ingest.bootstrap")
console = Console()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", choices=[s.value for s in ChunkStrategy] + ["all"],
                    default="all")
    p.add_argument("--cve-limit", type=int, default=None,
                    help="override settings.nvd_initial_cve_count")
    p.add_argument("--no-cves", action="store_true")
    p.add_argument("--no-pdfs", action="store_true")
    args = p.parse_args(argv)

    settings = get_settings()
    cve_limit = args.cve_limit if args.cve_limit is not None else settings.nvd_initial_cve_count

    # ---- gather documents ------------------------------------------
    docs = []
    if not args.no_cves and cve_limit > 0:
        console.rule("[bold cyan]CVE ingest[/]")
        with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}[/]"),
                       BarColumn(), TextColumn("[dim]{task.completed}/{task.total}[/]"),
                       TimeElapsedColumn(), console=console) as p_:
            t = p_.add_task(description="fetching CVEs", total=cve_limit)
            for d in fetch_recent_cves(settings, limit=cve_limit):
                docs.append(d)
                p_.advance(t)
        console.print(f"  fetched [bold]{len(docs)}[/] CVE documents")
    if not args.no_pdfs:
        pdf_docs = load_all_pdfs(_REPO / "corpus" / "pdfs")
        console.print(f"  loaded [bold]{len(pdf_docs)}[/] PDF page documents")
        docs.extend(pdf_docs)
    if not docs:
        console.print("[red]no documents to ingest — exiting[/]")
        return 1

    # ---- embed + write per strategy --------------------------------
    embedder = OllamaEmbedder(base_url=settings.ollama_base_url,
                               model=settings.ollama_embedding_model)
    # Probe one embedding to discover the dimension and init schema.
    sample = embedder.embed_one(docs[0].text[:200])
    dim = len(sample)
    store = PgVectorStore(settings.pg_dsn)
    store.init_schema(dim=dim)
    console.print(f"  pgvector schema initialised at dim={dim}")

    strategies = (list(ChunkStrategy)
                   if args.strategy == "all"
                   else [ChunkStrategy(args.strategy)])

    for strategy in strategies:
        chunks = chunk_documents(docs, strategy)
        console.rule(f"[bold]{strategy.value}[/] · {len(chunks)} chunks")
        # Embed in batches so we get a progress bar.
        with Progress(SpinnerColumn(),
                       TextColumn("[bold]{task.description}[/]"),
                       BarColumn(),
                       TextColumn("[dim]{task.completed}/{task.total}[/]"),
                       TimeElapsedColumn(), console=console) as p_:
            t = p_.add_task(description="embedding", total=len(chunks))
            BATCH = 32
            all_vecs: list[list[float]] = []
            for i in range(0, len(chunks), BATCH):
                batch = [c.text for c in chunks[i:i + BATCH]]
                res = embedder.embed(batch)
                all_vecs.extend(res.vectors)
                p_.update(t, completed=len(all_vecs))
        n_written = store.upsert(strategy=strategy, chunks=chunks,
                                   embeddings=all_vecs)
        console.print(f"  wrote {n_written} chunks to chunks_{strategy.value}")
    embedder.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
