"""Three chunking strategies — the experimental knob this repo evaluates.

  fixed_size      : token-budgeted, fixed length, fixed overlap
  semantic        : split on heading patterns + paragraph breaks
  sentence_window : every sentence is its own chunk; retrieval returns
                     the sentence plus ±N neighbouring sentences

We don't ship a tokeniser dependency; ``token_count_estimate`` is
roughly 1 token ≈ 4 characters of English text, which is good enough
for the budgeting decisions chunkers make. The eval harness uses
exact embed-time tokenisation downstream when it matters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from ingest.document import Document


class ChunkStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SENTENCE_WINDOW = "sentence_window"


@dataclass
class Chunk:
    """A retrievable text unit, traceable back to its parent Document."""
    text: str
    parent_source_id: str
    parent_source: str
    chunk_index: int
    strategy: ChunkStrategy
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# fixed-size, with overlap (the textbook strategy)
# ---------------------------------------------------------------------

def chunk_fixed_size(
    text: str,
    *,
    chunk_size_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[str]:
    """Approximate token-aware split. 1 token ≈ 4 chars."""
    if not text:
        return []
    # convert token budget to char budget; round generously
    chunk_chars = chunk_size_tokens * 4
    overlap_chars = overlap_tokens * 4
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_chars, n)
        # Avoid splitting mid-word; back off to nearest whitespace.
        if end < n:
            j = text.rfind(" ", i + chunk_chars - 60, end)
            if j > i:
                end = j
        out.append(text[i:end].strip())
        if end == n:
            break
        i = max(end - overlap_chars, i + 1)
    return [c for c in out if c]


# ---------------------------------------------------------------------
# semantic chunking — heading- and paragraph-aware
# ---------------------------------------------------------------------

# Headings recognised: markdown ``#`` / ``##``, NIST-style numbered
# section IDs ("3.1.2 Identification"), all-caps short lines.
_HEADING_RE = re.compile(
    r"(?m)^(?:#{1,4} +.+|[0-9]+(?:\.[0-9]+){0,4} +[A-Z][A-Za-z0-9 ,/&-]{2,80}|[A-Z][A-Z0-9 /&-]{6,80})$"
)


def chunk_semantic(text: str, *, max_chars: int = 2000) -> list[str]:
    """Split on headings and double-newline paragraph breaks; coalesce
    very short adjacent chunks up to ``max_chars`` so we don't drown
    the embedder in single-sentence fragments."""
    if not text:
        return []

    # First, find every heading boundary; everything between two
    # boundaries is a "section".
    boundaries = [0]
    for m in _HEADING_RE.finditer(text):
        if m.start() > boundaries[-1]:
            boundaries.append(m.start())
    boundaries.append(len(text))

    # A chunk MUST NOT straddle a heading — the whole point of the
    # semantic strategy is that you can read a chunk back and the
    # heading scopes it. Coalesce short paragraphs WITHIN each
    # section only.
    out: list[str] = []
    for a, b in zip(boundaries, boundaries[1:]):
        section = text[a:b].strip()
        if not section:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", section)
                       if p.strip()]
        buf: list[str] = []
        buflen = 0
        for p in paragraphs:
            if buflen + len(p) + 2 > max_chars and buf:
                out.append("\n\n".join(buf))
                buf, buflen = [p], len(p)
            else:
                buf.append(p)
                buflen += len(p) + 2
        if buf:
            out.append("\n\n".join(buf))
    return out


# ---------------------------------------------------------------------
# sentence-window: each sentence + ±N neighbours
# ---------------------------------------------------------------------

# Conservative sentence splitter — abbreviations like "e.g." cause
# spurious splits with naive `.\s` regex; we look for a period
# followed by whitespace AND a capital letter.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"])")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = [s.strip() for s in _SENTENCE_RE.split(text)]
    return [p for p in parts if p]


def chunk_sentence_window(
    text: str,
    *,
    window: int = 2,
) -> list[str]:
    """Each sentence becomes its own chunk; the chunk text includes
    ``window`` sentences before and after, but the *embedding* could
    use just the centre sentence (caller's choice). Here we return
    the windowed text directly so the chunker stays a pure-string
    transformation."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    out = []
    for i, _ in enumerate(sentences):
        lo = max(0, i - window)
        hi = min(len(sentences), i + window + 1)
        out.append(" ".join(sentences[lo:hi]))
    return out


# ---------------------------------------------------------------------
# unified entry point — what the rest of the pipeline calls
# ---------------------------------------------------------------------

def chunk_documents(
    docs: Iterable[Document],
    strategy: ChunkStrategy,
) -> list[Chunk]:
    """Apply the chosen strategy to a stream of Documents."""
    out: list[Chunk] = []
    for doc in docs:
        if strategy is ChunkStrategy.FIXED_SIZE:
            pieces = chunk_fixed_size(doc.text)
        elif strategy is ChunkStrategy.SEMANTIC:
            pieces = chunk_semantic(doc.text)
        elif strategy is ChunkStrategy.SENTENCE_WINDOW:
            pieces = chunk_sentence_window(doc.text)
        else:  # pragma: no cover
            raise ValueError(f"unknown strategy: {strategy}")
        for i, piece in enumerate(pieces):
            out.append(Chunk(
                text=piece,
                parent_source_id=doc.source_id,
                parent_source=doc.source.value,
                chunk_index=i,
                strategy=strategy,
                metadata=dict(doc.metadata),
            ))
    return out
