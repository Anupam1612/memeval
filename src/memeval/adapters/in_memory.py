"""In-memory adapter for testing memeval itself.

No external dependencies. Uses substring matching by default,
or sentence-transformers embeddings if installed.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime

import numpy as np

from memeval.protocol.base import MemoryProtocol
from memeval.protocol.types import (
    MemoryEntry,
    MemoryMetadata,
    MemoryType,
    SearchFilters,
    SearchResult,
    WriteResult,
)


class InMemoryAdapter(MemoryProtocol):
    """Dict-based memory backend for testing the framework.

    Supports two search modes:
        - "substring": simple case-insensitive substring matching (default)
        - "embedding": semantic search using sentence-transformers (requires install)

    Usage:
        adapter = InMemoryAdapter()  # substring mode
        adapter = InMemoryAdapter(search_mode="embedding",
                                   embedding_model="all-MiniLM-L6-v2")
    """

    def __init__(
        self,
        search_mode: str = "substring",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        super().__init__()
        self._store: dict[str, MemoryEntry] = {}
        self._search_mode = search_mode
        self._embedder = None
        self._embeddings: dict[str, np.ndarray] = {}

        if search_mode == "embedding":
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(embedding_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding mode. "
                    "Install with: pip install memeval[embeddings]"
                )

    def _generate_key(self) -> str:
        return f"mem_{uuid.uuid4().hex[:12]}"

    def _matches_filters(self, entry: MemoryEntry, filters: SearchFilters) -> bool:
        if filters.user_id and entry.metadata.user_id != filters.user_id:
            return False
        if filters.session_id and entry.metadata.session_id != filters.session_id:
            return False
        if filters.agent_id and entry.metadata.agent_id != filters.agent_id:
            return False
        if filters.memory_type and entry.memory_type != filters.memory_type:
            return False
        if filters.tags and not set(filters.tags).issubset(set(entry.metadata.tags)):
            return False
        if filters.after and entry.created_at < filters.after:
            return False
        if filters.before and entry.created_at > filters.before:
            return False
        return True

    async def write(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: MemoryMetadata | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> WriteResult:
        async with self._track("write") as record:
            assigned_key = key or self._generate_key()
            now = datetime.now()

            entry = MemoryEntry(
                key=assigned_key,
                content=content,
                memory_type=memory_type,
                metadata=metadata or MemoryMetadata(),
                created_at=now,
            )
            self._store[assigned_key] = entry

            if self._embedder is not None:
                self._embeddings[assigned_key] = self._embedder.encode(
                    content, normalize_embeddings=True
                )

        return WriteResult(
            key=assigned_key,
            success=True,
            latency_ms=record["latency_ms"],
        )

    async def read(self, key: str) -> MemoryEntry | None:
        async with self._track("read"):
            return self._store.get(key)

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        async with self._track("search"):
            candidates = list(self._store.values())
            if filters:
                candidates = [e for e in candidates if self._matches_filters(e, filters)]

            if not candidates:
                return []

            if self._search_mode == "embedding" and self._embedder is not None:
                return self._embedding_search(query, candidates, limit)
            else:
                return self._substring_search(query, candidates, limit)

    def _substring_search(
        self, query: str, candidates: list[MemoryEntry], limit: int
    ) -> list[SearchResult]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: list[tuple[MemoryEntry, float]] = []

        for entry in candidates:
            content_lower = entry.content.lower()
            if query_lower in content_lower:
                scored.append((entry, 1.0))
            else:
                content_words = set(content_lower.split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    score = overlap / len(query_words)
                    scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(entry=entry, score=score, rank=i + 1)
            for i, (entry, score) in enumerate(scored[:limit])
        ]

    def _embedding_search(
        self, query: str, candidates: list[MemoryEntry], limit: int
    ) -> list[SearchResult]:
        assert self._embedder is not None
        query_emb = self._embedder.encode(query, normalize_embeddings=True)

        scored: list[tuple[MemoryEntry, float]] = []
        for entry in candidates:
            emb = self._embeddings.get(entry.key)
            if emb is not None:
                similarity = float(np.dot(query_emb, emb))
                scored.append((entry, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(entry=entry, score=max(0.0, score), rank=i + 1)
            for i, (entry, score) in enumerate(scored[:limit])
        ]

    async def update(
        self,
        key: str,
        content: str,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> WriteResult:
        async with self._track("update") as record:
            existing = self._store.get(key)
            if existing is None:
                return WriteResult(key=key, success=False, latency_ms=record["latency_ms"])

            updated_metadata = metadata if metadata is not None else existing.metadata
            updated_entry = MemoryEntry(
                key=key,
                content=content,
                memory_type=existing.memory_type,
                metadata=updated_metadata,
                created_at=existing.created_at,
                updated_at=datetime.now(),
            )
            self._store[key] = updated_entry

            if self._embedder is not None:
                self._embeddings[key] = self._embedder.encode(
                    content, normalize_embeddings=True
                )

        return WriteResult(key=key, success=True, latency_ms=record["latency_ms"])

    async def delete(self, key: str) -> bool:
        async with self._track("delete"):
            if key in self._store:
                del self._store[key]
                self._embeddings.pop(key, None)
                return True
            return False

    async def list_all(
        self,
        *,
        filters: SearchFilters | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        async with self._track("list_all"):
            entries = list(self._store.values())
            if filters:
                entries = [e for e in entries if self._matches_filters(e, filters)]
            entries.sort(key=lambda e: e.created_at, reverse=True)
            return entries[:limit]

    async def consolidate(
        self,
        source_keys: list[str],
        *,
        strategy: str = "merge",
    ) -> WriteResult:
        async with self._track("consolidate") as record:
            entries = []
            for key in source_keys:
                entry = self._store.get(key)
                if entry:
                    entries.append(entry)

            if not entries:
                return WriteResult(key="", success=False, latency_ms=record["latency_ms"])

            if strategy == "merge":
                merged_content = "\n".join(e.content for e in entries)
            elif strategy == "deduplicate":
                seen: set[str] = set()
                unique: list[str] = []
                for e in entries:
                    if e.content not in seen:
                        seen.add(e.content)
                        unique.append(e.content)
                merged_content = "\n".join(unique)
            else:
                merged_content = "\n".join(e.content for e in entries)

            # Delete originals
            for key in source_keys:
                await self.delete(key)

            # Write consolidated
            result = await self.write(
                merged_content,
                memory_type=entries[0].memory_type,
                metadata=entries[0].metadata,
            )

        return result

    async def reset(self) -> None:
        self._store.clear()
        self._embeddings.clear()
        self.clear_operation_log()
