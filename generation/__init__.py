"""Answer generation: prompt + Ollama Llama call + citation extraction."""
from generation.generator import OllamaGenerator, GenerationError, AnswerWithSources
from generation.prompts import RAG_SYSTEM, RAG_USER

__all__ = [
    "OllamaGenerator", "GenerationError", "AnswerWithSources",
    "RAG_SYSTEM", "RAG_USER",
]
