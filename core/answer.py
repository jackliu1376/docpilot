"""Structured answer and citation models."""

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Source:
    id: str
    kind: str  # "local" | "web"
    snippet: str
    filename: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    url: Optional[str] = None
    title: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[Source]
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_citations(answer: str) -> set[str]:
    """Return the unique [S1]-style source ids cited in an answer."""
    return set(re.findall(r"\[(S\d+)\]", answer))
