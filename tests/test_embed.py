"""Embedder unit tests — wire-format only, no real Ollama call."""
from __future__ import annotations

import httpx
import pytest
import respx

from embeddings.embed import EmbeddingError, OllamaEmbedder


def test_embed_one_returns_vector():
    payload = {"embedding": [0.1, 0.2, 0.3, 0.4]}
    with respx.mock(assert_all_called=True) as m:
        m.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json=payload))
        with httpx.Client() as c:
            embedder = OllamaEmbedder(base_url="http://localhost:11434",
                                        model="test-model", client=c)
            vec = embedder.embed_one("hello")
    assert vec == [0.1, 0.2, 0.3, 0.4]


def test_embed_records_dimension_from_first_vector():
    payload = {"embedding": [0.0] * 384}
    with respx.mock(assert_all_called=False) as m:
        m.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json=payload))
        with httpx.Client() as c:
            embedder = OllamaEmbedder(base_url="http://localhost:11434",
                                        model="test-model", client=c)
            res = embedder.embed(["a", "b"])
    assert res.dimension == 384
    assert len(res.vectors) == 2


def test_http_error_does_not_echo_response_body():
    leaky_body = "internal-stack-trace-with-secrets"
    with respx.mock(assert_all_called=True) as m:
        m.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(500, text=leaky_body))
        with httpx.Client() as c:
            embedder = OllamaEmbedder(base_url="http://localhost:11434",
                                        model="test-model", client=c)
            with pytest.raises(EmbeddingError) as info:
                embedder.embed_one("hi")
    assert "internal-stack-trace" not in str(info.value)


def test_missing_embedding_field_raises():
    with respx.mock(assert_all_called=True) as m:
        m.post("http://localhost:11434/api/embeddings").mock(
            return_value=httpx.Response(200, json={"foo": "bar"}))
        with httpx.Client() as c:
            embedder = OllamaEmbedder(base_url="http://localhost:11434",
                                        model="test-model", client=c)
            with pytest.raises(EmbeddingError):
                embedder.embed_one("hi")
