"""LangGraph adapter -- maps LangGraph's InMemoryStore to the Standard Memory Protocol.

Install: pip install memoryeval[langgraph]

LangGraph has two memory systems:
    - Checkpointers (short-term, per-thread state)
    - Store (long-term, cross-thread key-value with semantic search)

This adapter uses the Store API since it maps cleanly to SMP operations.

SMP Mapping:
    write  -> store.put(namespace, key, value)
    read   -> store.get(namespace, key)
    search -> store.search(namespace, query=query)
    update -> store.put(namespace, key, new_value)  (overwrite)
    delete -> store.delete(namespace, key)
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


class LangGraphAdapter(MemoryProtocol):
    """Adapter for LangGraph's InMemoryStore.

    Args:
        default_namespace: Tuple prefix for memory namespaces.
            Defaults to ("memeval", "memories").
        store: Optional pre-configured store instance.
            If None, creates an InMemoryStore.
    """

    def __init__(
        self,
        default_namespace: tuple[str, ...] = ("memeval", "memories"),
        store: Any | None = None,
    ) -> None:
        super().__init__()
        self._default_namespace = default_namespace
        self._sessions: dict[str, dict[str, Any]] = {}

        try:
            if store is not None:
                self._store = store
            else:
                from langgraph.store.memory import InMemoryStore

                self._store = InMemoryStore()
        except ImportError:
            raise ImportError(
                "langgraph required. Install: pip install memoryeval[langgraph]"
            )

    def _namespace(self, metadata: MemoryMetadata | None = None) -> tuple[str, ...]:
        """Build namespace from metadata."""
        if metadata and metadata.user_id:
            return (metadata.user_id, "memories")
        return self._default_namespace

    async def write(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: MemoryMetadata | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> WriteResult:
        async with self._track("write") as record:
            assigned_key = key or f"mem_{uuid.uuid4().hex[:12]}"
            ns = self._namespace(metadata)

            self._store.put(
                ns,
                assigned_key,
                {
                    "content": content,
                    "memory_type": memory_type.value,
                    "metadata": {
                        "user_id": metadata.user_id if metadata else None,
                        "session_id": metadata.session_id if metadata else None,
                        "tags": list(metadata.tags) if metadata else [],
                    },
                    "created_at": datetime.now().isoformat(),
                },
            )

        return WriteResult(key=assigned_key, success=True, latency_ms=record["latency_ms"])

    async def read(self, key: str) -> MemoryEntry | None:
        async with self._track("read"):
            try:
                item = self._store.get(self._default_namespace, key)
                if item is None:
                    return None
                value = item.value
                return MemoryEntry(
                    key=key,
                    content=value.get("content", ""),
                    memory_type=MemoryType(value.get("memory_type", "semantic")),
                    metadata=MemoryMetadata(
                        user_id=value.get("metadata", {}).get("user_id"),
                        tags=tuple(value.get("metadata", {}).get("tags", [])),
                    ),
                    created_at=datetime.fromisoformat(
                        value.get("created_at", datetime.now().isoformat())
                    ),
                )
            except Exception:
                return None

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        async with self._track("search"):
            ns = self._default_namespace
            if filters and filters.user_id:
                ns = (filters.user_id, "memories")

            try:
                items = self._store.search(ns, query=query, limit=limit)
            except Exception:
                items = self._store.search(ns, limit=limit)

            results = []
            for rank, item in enumerate(items, 1):
                value = item.value
                score = getattr(item, "score", 1.0 - rank * 0.1) or (1.0 - rank * 0.1)
                entry = MemoryEntry(
                    key=item.key,
                    content=value.get("content", ""),
                    memory_type=MemoryType(value.get("memory_type", "semantic")),
                    metadata=MemoryMetadata(
                        user_id=value.get("metadata", {}).get("user_id"),
                    ),
                    created_at=datetime.now(),
                )
                results.append(SearchResult(entry=entry, score=float(score), rank=rank))

            return results

    async def update(
        self,
        key: str,
        content: str,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> WriteResult:
        async with self._track("update") as record:
            existing = self._store.get(self._default_namespace, key)
            if existing is None:
                return WriteResult(key=key, success=False, latency_ms=record["latency_ms"])

            value = existing.value
            value["content"] = content
            value["updated_at"] = datetime.now().isoformat()
            self._store.put(self._default_namespace, key, value)

        return WriteResult(key=key, success=True, latency_ms=record["latency_ms"])

    async def delete(self, key: str) -> bool:
        async with self._track("delete"):
            try:
                self._store.delete(self._default_namespace, key)
                return True
            except Exception:
                return False

    async def list_all(
        self,
        *,
        filters: SearchFilters | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        async with self._track("list_all"):
            ns = self._default_namespace
            if filters and filters.user_id:
                ns = (filters.user_id, "memories")

            items = self._store.search(ns, limit=limit)

            entries = []
            for item in items:
                value = item.value
                entries.append(MemoryEntry(
                    key=item.key,
                    content=value.get("content", ""),
                    memory_type=MemoryType(value.get("memory_type", "semantic")),
                    metadata=MemoryMetadata(
                        user_id=value.get("metadata", {}).get("user_id"),
                    ),
                    created_at=datetime.now(),
                ))

            return entries[:limit]

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

            if query:
                results = await self.search(query, limit=10)
                for r in results:
                    if r.entry.content not in facts:
                        facts.append(r.entry.content)

            return SessionContext(session_id=session_id, facts=facts)

    async def reset(self) -> None:
        try:
            items = self._store.search(self._default_namespace, limit=10000)
            for item in items:
                self._store.delete(self._default_namespace, item.key)
        except Exception:
            pass
        self._sessions.clear()
        self.clear_operation_log()
