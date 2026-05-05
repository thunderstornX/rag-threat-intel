"""HTTP request/response shapes for the RAG API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    strategy: Literal["fixed_size", "semantic", "sentence_window"] = "semantic"
    top_k: int = Field(default=10, ge=1, le=50,
                        description="initial retrieval depth")
    top_n: int = Field(default=4, ge=1, le=20,
                        description="post-MMR depth fed to the generator")


class SourceCitation(BaseModel):
    doc_id: int                  # 1-based [doc_N] tag
    parent_source: str
    parent_source_id: str
    chunk_index: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    strategy: str
    cited_doc_ids: list[int]
    sources: list[SourceCitation]
    elapsed_ms_retrieval: int
    elapsed_ms_generation: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: dict[str, bool]
