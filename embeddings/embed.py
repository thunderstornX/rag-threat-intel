"""Ollama embeddings client.

POST /api/embeddings against the Ollama base URL. We batch by calling
once per text — Ollama's API doesn't take a list parameter, so the
"batch" is a Python loop. We DO pool the connection though, which is
where the wall-clock saving lives."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx


_log = logging.getLogger("embeddings")


class EmbeddingError(RuntimeError):
    """Generic failure; never echoes server response body."""


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    dimension: int
    elapsed_ms: int
    model: str


class OllamaEmbedder:
    """Sync embedder. The pipeline runs ingest as a one-shot script,
    not a service — async would be busywork."""

    def __init__(self, *, base_url: str, model: str,
                 timeout_s: float = 60.0,
                 client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client or httpx.Client(timeout=timeout_s)
        self._owns = client is None

    def close(self) -> None:
        if self._owns:
            self._client.close()

    def embed(self, texts: list[str]) -> EmbedResult:
        if not texts:
            return EmbedResult(vectors=[], dimension=0, elapsed_ms=0,
                                model=self.model)
        url = f"{self.base_url}/api/embeddings"
        started = time.monotonic()
        vectors: list[list[float]] = []
        for t in texts:
            try:
                r = self._client.post(url, json={"model": self.model,
                                                   "prompt": t})
            except httpx.HTTPError as exc:
                raise EmbeddingError(
                    f"ollama embeddings request failed: {exc.__class__.__name__}"
                ) from exc
            if r.status_code >= 400:
                raise EmbeddingError(
                    f"ollama embeddings returned HTTP {r.status_code}")
            try:
                data = r.json()
            except ValueError as exc:
                raise EmbeddingError("ollama embeddings response was not JSON") from exc
            vec = data.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise EmbeddingError("ollama embeddings response had no 'embedding'")
            vectors.append(vec)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return EmbedResult(
            vectors=vectors,
            dimension=len(vectors[0]) if vectors else 0,
            elapsed_ms=elapsed_ms,
            model=self.model,
        )

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text]).vectors[0]
