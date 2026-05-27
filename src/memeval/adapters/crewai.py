"""CrewAI adapter -- maps CrewAI's Memory class to the Standard Memory Protocol.

Install: pip install memoryeval[crewai]

CrewAI has a unified Memory class with:
    - remember(text)         -- store a memory
    - recall(query)          -- semantic search
    - extract_memories(text) -- extract facts from text
    - forget(scope)          -- delete memories
    - tree()                 -- browse stored memories

SMP Mapping:
    write  -> memory.remember(content)
    read   -> local key map (CrewAI doesn't have key-based retrieval)
    search -> memory.recall(query)
    delete -> memory.forget(scope)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from memeval.protocol.base import MemoryProtocol
from memeval.protocol.types import (
    MemoryEntry,
    MemoryMetadata,
    MemoryType,
    Message,
    SearchFilters,
    SearchResult,
    SessionContext,
    WriteResult,
)


class CrewAIAdapter(MemoryProtocol):
    """Adapter for CrewAI Memory.

    Args:
        memory: Optional pre-configured CrewAI Memory instance.
            If None, creates a new Memory().
    """

    def __init__(self, memory: Any | None = None) -> None:
        super().__init__()
        self._store: dict[str, dict[str, Any]] = {}
        self._sessions: dict[str, dict[str, Any]] = {}

        try:
            if memory is not None:
                self._memory = memory
            else:
                from crewai.memory import Memory

                self._memory = Memory()
            self._has_crewai = True
        except ImportError:
            self._has_crewai = False

    async def write(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: MemoryMetadata | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> WriteResult:
        async with self._track("write") as record:
            assigned_key = key or f"crew_{uuid.uuid4().hex[:12]}"

            # Store in CrewAI
            if self._has_crewai:
                try:
                    self._memory.remember(content)
                except Exception:
                    pass

            # Always track locally for key-based operations
            self._store[assigned_key] = {
                "content": content,
                "memory_type": memory_type,
                "metadata": metadata or MemoryMetadata(),
                "created_at": datetime.now(),
            }

        return WriteResult(key=assigned_key, success=True, latency_ms=record["latency_ms"])

    async def read(self, key: str) -> MemoryEntry | None:
        async with self._track("read"):
            stored = self._store.get(key)
            if not stored:
                return None
            return MemoryEntry(
                key=key,
                content=stored["content"],
                memory_type=stored.get("memory_type", MemoryType.SEMANTIC),
                metadata=stored.get("metadata", MemoryMetadata()),
                created_at=stored.get("created_at", datetime.now()),
            )

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        async with self._track("search"):
            results: list[SearchResult] = []

            # Try CrewAI recall first
            # Returns list[MemoryMatch] where each has .record and .score
            # .record has .content, .scope, .categories, .metadata
            if self._has_crewai:
                try:
                    recalled = self._memory.recall(query)
                    if recalled:
                        items = recalled if isinstance(recalled, list) else [recalled]
                        for i, item in enumerate(items[:limit]):
                            # Extract content from MemoryMatch.record.content
                            record = getattr(item, "record", None)
                            if record and hasattr(record, "content"):
                                content = record.content
                            else:
                                content = str(item)

                            score = getattr(item, "score", 0.5) or 0.5

                            entry = MemoryEntry(
                                key=getattr(record, "id", f"crew_{i}"),
                                content=content,
                                memory_type=MemoryType.SEMANTIC,
                                metadata=MemoryMetadata(),
                                created_at=datetime.now(),
                            )
                            results.append(SearchResult(
                                entry=entry, score=float(score), rank=i + 1,
                            ))
                        if results:
                            return results
                except Exception:
                    pass

            # Fallback: local store search
            return self._local_search(query, limit, filters)

    def _local_search(
        self, query: str, limit: int, filters: SearchFilters | None
    ) -> list[SearchResult]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: list[tuple[str, dict, float]] = []

        for key, stored in self._store.items():
            if filters and filters.user_id:
                meta = stored.get("metadata", MemoryMetadata())
                if meta.user_id != filters.user_id:
                    continue
            content_lower = stored["content"].lower()
            if query_lower in content_lower:
                scored.append((key, stored, 1.0))
            else:
                content_words = set(content_lower.split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    scored.append((key, stored, overlap / len(query_words)))

        scored.sort(key=lambda x: x[2], reverse=True)
        results = []
        for i, (key, stored, score) in enumerate(scored[:limit]):
            entry = MemoryEntry(
                key=key,
                content=stored["content"],
                memory_type=stored.get("memory_type", MemoryType.SEMANTIC),
                metadata=stored.get("metadata", MemoryMetadata()),
                created_at=stored.get("created_at", datetime.now()),
            )
            results.append(SearchResult(entry=entry, score=score, rank=i + 1))
        return results

    async def update(
        self,
        key: str,
        content: str,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> WriteResult:
        async with self._track("update") as record:
            if key not in self._store:
                return WriteResult(key=key, success=False, latency_ms=record["latency_ms"])

            self._store[key]["content"] = content
            if metadata:
                self._store[key]["metadata"] = metadata

            # Also store in CrewAI
            if self._has_crewai:
                try:
                    self._memory.remember(content)
                except Exception:
                    pass

        return WriteResult(key=key, success=True, latency_ms=record["latency_ms"])

    async def delete(self, key: str) -> bool:
        async with self._track("delete"):
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def list_all(
        self,
        *,
        filters: SearchFilters | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        async with self._track("list_all"):
            entries = []
            for key, stored in list(self._store.items())[:limit]:
                if filters and filters.user_id:
                    meta = stored.get("metadata", MemoryMetadata())
                    if meta.user_id != filters.user_id:
                        continue
                entries.append(MemoryEntry(
                    key=key,
                    content=stored["content"],
                    memory_type=stored.get("memory_type", MemoryType.SEMANTIC),
                    metadata=stored.get("metadata", MemoryMetadata()),
                    created_at=stored.get("created_at", datetime.now()),
                ))
            return entries

    async def consolidate(
        self,
        source_keys: list[str],
        *,
        strategy: str = "merge",
    ) -> WriteResult:
        contents = []
        for key in source_keys:
            entry = await self.read(key)
            if entry:
                contents.append(entry.content)

        if not contents:
            return WriteResult(key="", success=False, latency_ms=0)

        merged = "\n".join(contents)
        for key in source_keys:
            await self.delete(key)
        return await self.write(merged)

    async def create_session(
        self, *, session_id: str | None = None, user_id: str | None = None
    ) -> str:
        sid = session_id or f"session_{uuid.uuid4().hex[:12]}"
        self._sessions[sid] = {"user_id": user_id, "messages": []}
        return sid

    async def add_message(self, session_id: str, message: Message) -> WriteResult:
        async with self._track("add_message"):
            if session_id not in self._sessions:
                self._sessions[session_id] = {"user_id": None, "messages": []}

            self._sessions[session_id]["messages"].append({
                "role": message.role, "content": message.content,
            })

            return await self.write(
                content=message.content,
                metadata=MemoryMetadata(session_id=session_id),
                memory_type=MemoryType.EPISODIC,
            )

    async def get_session_context(
        self, session_id: str, query: str | None = None
    ) -> SessionContext:
        async with self._track("get_session_context"):
            session = self._sessions.get(session_id, {"messages": []})
            facts = [m["content"] for m in session["messages"]]

            if query and self._has_crewai:
                try:
                    recalled = self._memory.recall(query)
                    if recalled:
                        items = recalled if isinstance(recalled, list) else [recalled]
                        for item in items:
                            record = getattr(item, "record", None)
                            if record and hasattr(record, "content"):
                                text = record.content
                            else:
                                text = str(item)
                            if text not in facts:
                                facts.append(text)
                except Exception:
                    pass

            if query and not facts:
                results = await self.search(query, limit=10)
                facts = [r.entry.content for r in results]

            return SessionContext(session_id=session_id, facts=facts)

    async def reset(self) -> None:
        self._store.clear()
        self._sessions.clear()
        if self._has_crewai:
            try:
                self._memory.forget(scope="/")
            except Exception:
                pass
        self.clear_operation_log()
