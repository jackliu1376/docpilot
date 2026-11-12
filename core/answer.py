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


_CITATION_PATTERN = re.compile(
    r"[\[\(（]?\s*(?:来源[:：]\s*)?S(\d+)\s*[\]\)）]?"
)


def parse_citations(answer: str) -> set[str]:
    """Return the unique [S1]-style source ids cited in an answer."""
    return set(re.findall(r"\[(S\d+)\]", answer))


def normalize_citations(text, valid_ids=None):
    """Normalize S1 / [来源：S1] / （S1） into [S1] and drop invalid ids."""
    def replace(match):
        source_id = f"S{match.group(1)}"
        if valid_ids is not None and source_id not in valid_ids:
            return ""
        return f"[{source_id}]"

    return _CITATION_PATTERN.sub(replace, text or "")
