"""Ollama Llama generator with mandatory-citation discipline.

Calls Ollama's chat-completions endpoint, parses the answer, and
extracts the ``[doc_N]`` citation tags so the API can return both
the prose and the structured set of sources actually used.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

import httpx

from generation.prompts import RAG_SYSTEM, RAG_USER
from retrieval.store import RetrievedChunk


_log = logging.getLogger("generation")
_CITATION_RE = re.compile(r"\[doc_(\d+)\]")


class GenerationError(RuntimeError):
    """Generic adapter error; never echoes the response body."""


@dataclass
class AnswerWithSources:
    answer: str
    cited_doc_ids: list[int]                 # 1-based, deduped
    used_sources: list[RetrievedChunk]       # subset of the retrieved set
    refused: bool                             # True if model said "I cannot"
    elapsed_ms: int
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OllamaGenerator:
    """Sync generator. Sized for the eval harness's serial use."""

    def __init__(self, *, base_url: str, model: str,
                 timeout_s: float = 120.0,
                 client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = client or httpx.Client(timeout=timeout_s)
        self._owns = client is None

    def close(self) -> None:
        if self._owns:
            self._client.close()

    def _format_documents(self, docs: Sequence[RetrievedChunk]) -> str:
        out = []
        for i, d in enumerate(docs, start=1):
            origin = f"{d.parent_source}:{d.parent_source_id}"
            out.append(f"[doc_{i}] (source: {origin})\n{d.text}")
        return "\n\n".join(out) if out else "(no documents retrieved)"

    def generate(
        self,
        *,
        question: str,
        documents: list[RetrievedChunk],
        max_tokens: int = 600,
    ) -> AnswerWithSources:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": RAG_USER.format(
                    question=question.strip(),
                    document_block=self._format_documents(documents),
                )},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }
        started = time.monotonic()
        try:
            r = self._client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise GenerationError(
                f"ollama generation failed: {exc.__class__.__name__}"
            ) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if r.status_code >= 400:
            raise GenerationError(
                f"ollama generation returned HTTP {r.status_code}")
        try:
            data = r.json()
        except ValueError as exc:
            raise GenerationError("ollama generation response was not JSON") from exc

        content = (data.get("message") or {}).get("content") or ""
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("ollama generation produced empty content")

        cited = sorted({int(m.group(1)) for m in _CITATION_RE.finditer(content)})
        # Filter to valid 1-based ids; the prompt tells the model not
        # to invent tags but a robust pipeline still validates.
        valid_cited = [i for i in cited if 1 <= i <= len(documents)]
        used = [documents[i - 1] for i in valid_cited]
        refused = "do not answer this question" in content.lower()

        return AnswerWithSources(
            answer=content.strip(),
            cited_doc_ids=valid_cited,
            used_sources=used,
            refused=refused,
            elapsed_ms=elapsed_ms,
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            completion_tokens=data.get("eval_count", 0) or 0,
        )
