"""Core data types for the Standard Memory Protocol (SMP).

These types define the contract between the evaluation harness and any memory backend.
All adapters translate their native types to/from these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """Classification of memory following the CoALA taxonomy."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass(frozen=True)
class MemoryMetadata:
    """Metadata attached to a memory entry.

    Adapters map provider-specific fields into this structure.
    The `extra` dict captures anything provider-specific that doesn't
    fit the standard fields.
    """

    source: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    timestamp: datetime | None = None
    tags: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **kwargs: Any) -> MemoryMetadata:
        """Return a new MemoryMetadata with the given fields replaced."""
        current = {
            "source": self.source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "extra": self.extra,
        }
        current.update(kwargs)
        return MemoryMetadata(**current)


@dataclass(frozen=True)
class MemoryEntry:
    """A single memory stored in the backend."""

    key: str
    content: str
    memory_type: MemoryType
    metadata: MemoryMetadata
    created_at: datetime
    updated_at: datetime | None = None


@dataclass(frozen=True)
class SearchResult:
    """A single result from a memory search operation."""

    entry: MemoryEntry
    score: float  # 0.0 to 1.0, higher = more relevant
    rank: int  # 1-indexed position in the result list


@dataclass(frozen=True)
class WriteResult:
    """Result of a write/update/consolidate operation."""

    key: str
    success: bool
    latency_ms: float
    tokens_used: int | None = None


@dataclass(frozen=True)
class SearchFilters:
    """Filters applied to search and list operations."""

    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    memory_type: MemoryType | None = None
    tags: tuple[str, ...] | None = None
    after: datetime | None = None
    before: datetime | None = None
