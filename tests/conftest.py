"""Test fixtures.

Sets dummy env vars before any module imports config — that way the
`Settings` validator doesn't blow up when pytest is run without a
populated .env file."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Defaults that satisfy the pydantic Settings validators in tests.
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_GENERATION_MODEL", "llama3.2:3b")
os.environ.setdefault("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
os.environ.setdefault("PG_HOST", "localhost")
os.environ.setdefault("PG_DB", "rti")
os.environ.setdefault("PG_USER", "rti")
os.environ.setdefault("PG_PASSWORD", "test-only")


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Some modules cache settings at import time; nothing to do here
    yet, but the fixture is set up so future per-test isolation has a
    seam to use."""
    yield
