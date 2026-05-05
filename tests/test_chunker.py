"""Chunker invariants. The chunkers are pure-Python; everything in
this file is in-memory and deterministic."""
from __future__ import annotations

from ingest.chunker import (
    ChunkStrategy,
    chunk_documents,
    chunk_fixed_size,
    chunk_semantic,
    chunk_sentence_window,
    split_sentences,
)
from ingest.document import Document, DocSource


# ----------------------------- fixed_size ---------------------------

def test_fixed_size_respects_max_chunk_chars():
    text = "word " * 1500   # 7500 chars
    chunks = chunk_fixed_size(text, chunk_size_tokens=128, overlap_tokens=10)
    # token-budget*4 = 512 chars per chunk; allow slight back-off to
    # avoid mid-word splits
    assert all(len(c) <= 600 for c in chunks)
    assert len(chunks) > 1


def test_fixed_size_overlap_means_adjacent_chunks_share_some_text():
    text = "alpha beta gamma delta " * 200   # ~4400 chars
    chunks = chunk_fixed_size(text, chunk_size_tokens=128, overlap_tokens=20)
    if len(chunks) >= 2:
        # the last 60 chars of chunk[0] should overlap with chunk[1]
        tail = chunks[0][-60:]
        head = chunks[1][:160]
        assert any(piece in head for piece in tail.split())


def test_fixed_size_empty_input_is_empty_list():
    assert chunk_fixed_size("") == []


# ----------------------------- semantic -----------------------------

def test_semantic_splits_on_markdown_headings():
    text = "# Section A\n\nfirst paragraph here.\n\n# Section B\n\nsecond paragraph."
    chunks = chunk_semantic(text, max_chars=200)
    assert any("Section A" in c for c in chunks)
    assert any("Section B" in c for c in chunks)


def test_semantic_coalesces_short_paragraphs():
    """Many tiny paragraphs should coalesce up to max_chars rather
    than producing a separate chunk per paragraph."""
    text = "\n\n".join(["short para one.", "short para two.", "short para three."])
    chunks = chunk_semantic(text, max_chars=500)
    assert len(chunks) == 1


def test_semantic_recognises_nist_style_section_ids():
    text = "1.1 Identification\n\nfoo bar.\n\n1.2 Authorisation\n\nbaz quux."
    chunks = chunk_semantic(text, max_chars=200)
    assert len(chunks) == 2


# ----------------------------- sentence_window ----------------------

def test_sentence_window_emits_one_chunk_per_sentence():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_sentence_window(text, window=1)
    assert len(chunks) == 3


def test_sentence_window_includes_neighbours():
    """Centre sentence is at index N; its window includes ±N
    neighbours so the embedder sees more context around the same
    centre."""
    text = "S1 done. S2 done. S3 done. S4 done. S5 done."
    chunks = chunk_sentence_window(text, window=1)
    # second chunk is centred on S2, so it should contain S1 and S3
    assert "S1 done" in chunks[1]
    assert "S2 done" in chunks[1]
    assert "S3 done" in chunks[1]


def test_split_sentences_handles_simple_punctuation():
    s = split_sentences("Hello world. This is fine. Yes? OK!")
    assert len(s) == 4


# ----------------------------- chunk_documents ---------------------

def test_chunk_documents_attaches_parent_metadata():
    doc = Document(text="Alpha. Beta. Gamma.", source=DocSource.NVD,
                    source_id="CVE-9999-0001",
                    metadata={"cvss_score": 7.5})
    chunks = chunk_documents([doc], ChunkStrategy.SENTENCE_WINDOW)
    assert chunks
    assert all(c.parent_source_id == "CVE-9999-0001" for c in chunks)
    assert all(c.parent_source == "nvd" for c in chunks)
    assert chunks[0].metadata.get("cvss_score") == 7.5


def test_chunk_documents_increments_chunk_index():
    doc = Document(text="A. B. C. D.", source=DocSource.PDF,
                    source_id="x.pdf:p1")
    chunks = chunk_documents([doc], ChunkStrategy.SENTENCE_WINDOW)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
