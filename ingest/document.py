"""The single Document shape used everywhere in the pipeline.

Keeping this small and explicit means the rest of the system never
has to ask "is this a CVE or a PDF?" — it just reads ``text``,
``source``, and ``metadata`` and gets on with it."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DocSource(str, Enum):
    NVD = "nvd"
    PDF = "pdf"


@dataclass
class Document:
    text: str
    source: DocSource
    source_id: str             # e.g. "CVE-2024-1234" or "nist-sp-800-53.pdf:p12"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Stable hash used to dedup duplicate ingests."""
        h = hashlib.sha256(
            f"{self.source.value}|{self.source_id}|{self.text}".encode("utf-8"),
            usedforsecurity=False,
        )
        return h.hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source.value,
            "source_id": self.source_id,
            "metadata": self.metadata,
            "fingerprint": self.fingerprint,
        }
