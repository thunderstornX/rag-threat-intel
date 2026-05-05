"""Embeddings: thin Ollama client + a side-by-side comparison harness.

The Ollama embeddings endpoint is OpenAI-style enough that we don't
need a SDK; httpx with a small JSON body is all of it. Two models are
exposed so the eval can show what changes when you swap them:

  - nomic-embed-text (768d, default)
  - all-minilm       (384d, the lightweight baseline)
"""
from embeddings.embed import OllamaEmbedder, EmbeddingError

__all__ = ["OllamaEmbedder", "EmbeddingError"]
