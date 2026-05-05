"""Local PDF loader.

We deliberately keep this dumb: extract per-page text via pypdf,
emit one ``Document`` per page with a stable ``source_id`` of
``<filename>:p<N>``. That granularity gives the chunker something
sensible to subdivide further, and lets retrieval cite exact page
numbers.

PDFs are expected to live under ``corpus/pdfs/``. The repo ships
with a tiny seed set; see ``corpus/pdfs/README.md`` for what to add
and how to keep the licensing clean."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pypdf

from ingest.document import Document, DocSource


_log = logging.getLogger("ingest.pdf")


def load_pdf(path: Path) -> Iterator[Document]:
    """Yield one Document per page of a single PDF."""
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as exc:
        _log.warning("could not open %s: %s", path, exc)
        return

    title = (reader.metadata.title if reader.metadata else None) or path.stem
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if len(text) < 40:
            # too short to be useful; pypdf sometimes returns layout
            # artefacts on cover pages
            continue
        yield Document(
            text=text,
            source=DocSource.PDF,
            source_id=f"{path.name}:p{i}",
            metadata={"title": title, "page": i, "filename": path.name},
        )


def load_all_pdfs(corpus_dir: Path) -> list[Document]:
    """Walk ``corpus_dir`` for *.pdf and concatenate page-Documents."""
    docs: list[Document] = []
    if not corpus_dir.exists():
        _log.warning("corpus dir %s does not exist", corpus_dir)
        return docs
    for path in sorted(corpus_dir.glob("*.pdf")):
        docs.extend(load_pdf(path))
    return docs
