"""Document ingest layer.

Three sub-modules in dependency order:

  nvd_fetcher  → pulls CVE records from the NIST NVD 2.0 API
  pdf_loader   → loads local PDFs into raw text + metadata
  chunker      → applies one of three chunking strategies

Each ingest path produces a list of ``Document`` objects whose shape
is identical regardless of source — that's what lets the same
embedder/retriever stack work over both CVE rows and PDF pages.
"""
from ingest.document import Document, DocSource

__all__ = ["Document", "DocSource"]
