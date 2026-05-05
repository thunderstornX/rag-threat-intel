"""Generator tests — extracts citations + handles refusals."""
from __future__ import annotations

import httpx
import pytest
import respx

from generation.generator import GenerationError, OllamaGenerator
from retrieval.store import RetrievedChunk


def _doc(i):
    return RetrievedChunk(text=f"text {i}", score=0.5,
                            parent_source="nvd", parent_source_id=f"CVE-X-{i}",
                            chunk_index=0)


def _ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={
        "model": "test", "message": {"content": content},
        "prompt_eval_count": 50, "eval_count": 30,
    })


def test_extracts_valid_citations():
    docs = [_doc(1), _doc(2), _doc(3)]
    answer = "First sentence with [doc_1]. Second with [doc_2]. Third [doc_3]."
    with respx.mock(assert_all_called=True) as m:
        m.post("http://localhost:11434/api/chat").mock(
            return_value=_ok(answer))
        with httpx.Client() as c:
            gen = OllamaGenerator(base_url="http://localhost:11434",
                                    model="test", client=c)
            r = gen.generate(question="Q?", documents=docs)
    assert r.cited_doc_ids == [1, 2, 3]
    assert len(r.used_sources) == 3
    assert not r.refused


def test_drops_invalid_citation_tags():
    """Model invents [doc_99]; we filter it out rather than crashing."""
    docs = [_doc(1)]
    answer = "Answer with [doc_1] and [doc_99] (made up)."
    with respx.mock(assert_all_called=True) as m:
        m.post("http://localhost:11434/api/chat").mock(
            return_value=_ok(answer))
        with httpx.Client() as c:
            gen = OllamaGenerator(base_url="http://localhost:11434",
                                    model="test", client=c)
            r = gen.generate(question="Q?", documents=docs)
    assert r.cited_doc_ids == [1]
    assert len(r.used_sources) == 1


def test_refusal_is_detected():
    docs = [_doc(1)]
    answer = "The supplied documents do not answer this question."
    with respx.mock(assert_all_called=True) as m:
        m.post("http://localhost:11434/api/chat").mock(
            return_value=_ok(answer))
        with httpx.Client() as c:
            gen = OllamaGenerator(base_url="http://localhost:11434",
                                    model="test", client=c)
            r = gen.generate(question="Q?", documents=docs)
    assert r.refused is True
    assert r.cited_doc_ids == []


def test_http_error_does_not_echo_body():
    docs = [_doc(1)]
    body = "this is a leaky stack trace"
    with respx.mock(assert_all_called=True) as m:
        m.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(500, text=body))
        with httpx.Client() as c:
            gen = OllamaGenerator(base_url="http://localhost:11434",
                                    model="test", client=c)
            with pytest.raises(GenerationError) as info:
                gen.generate(question="Q?", documents=docs)
    assert "leaky stack trace" not in str(info.value)
