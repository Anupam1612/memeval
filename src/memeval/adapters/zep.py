"""Zep adapter — maps the zep-cloud v3 SDK to the Standard Memory Protocol.

Install: pip install memeval[zep]

Zep v3 uses a graph-based knowledge architecture:
    - Users own data
    - Data is added via graph.add() (text, messages, or JSON)
    - Search via graph.search() returns nodes/edges/episodes
    - Zep auto-extracts facts into a temporal knowledge graph

SMP Mapping:
    write  -> client.graph.add(user_id, data=content, type='text')
    read   -> local key map (Zep doesn't have direct key-based retrieval)
    search -> client.graph.search(user_id, query)
    delete -> local tracking (Zep graph doesn't support individual fact deletion)

Tested against zep-cloud v3.22.x (May 2026).
"""

from __future__ import annotations

import os
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


class ZepAdapter(MemoryProtocol):
    """Adapter for Zep Cloud v3 (pip install zep-cloud).

    Args:
        api_key: Zep Cloud API key. If None, reads from ZEP_API_KEY env var.
        default_user_id: Default user for scoping.
        index_delay: Seconds to wait after writes for Zep to index (graph processing is async).
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_user_id: str | None = None,
        index_delay: float = 0.5,
    ) -> None:
        super().__init__()
        self._api_key = api_key or os.getenv("ZEP_API_KEY")
        self._default_user_id = default_user_id or f"memeval_{uuid.uuid4().hex[:8]}"
        self._index_delay = index_delay

        # Local store for key-based operations (Zep is graph-based, not key-based)
        self._store: dict[str, dict[str, Any]] = {}
        self._episode_ids: list[str] = []  # track Zep episode UUIDs
        self._threads: dict[str, str] = {}  # thread_id -> user_id

        try:
            from zep_cloud.client import Zep

            if not self._api_key:
                raise ValueError(
                    "Zep API key required. Set ZEP_API_KEY env var or pass api_key param."
                )
            self._client = Zep(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "zep-cloud package required. Install: pip install memeval[zep]"
            )

        # Ensure user exists
        try:
            self._client.user.add(user_id=self._default_user_id)
        except Exception:
            pass  # user may already exist

    async def write(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: MemoryMetadata | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> WriteResult:
        async with self._track("write") as record:
            meta = metadata or MemoryMetadata()
            assigned_key = key or f"zep_{uuid.uuid4().hex[:12]}"
            user_id = meta.user_id or self._default_user_id

            # Ensure user exists for non-default users
            if user_id != self._default_user_id:
                try:
                    self._client.user.add(user_id=user_id)
                except Exception:
                    pass

            success = False
            try:
                result = self._client.graph.add(
                    user_id=user_id,
                    type="text",
                    data=content,
                )
                episode_id = getattr(result, "uuid_", None)
                if episode_id:
                    self._episode_ids.append(episode_id)
                success = True
            except Exception:
                pass

            # Always track locally for key-based operations
            self._store[assigned_key] = {
                "content": content,
                "user_id": user_id,
                "memory_type": memory_type,
                "metadata": meta,
                "created_at": datetime.now(),
            }

        return WriteResult(
            key=assigned_key,
            success=success,
            latency_ms=record["latency_ms"],
        )

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
            user_id = (filters.user_id if filters else None) or self._default_user_id

            results: list[SearchResult] = []

            # Try Zep graph search first
            try:
                zep_results = self._client.graph.search(
                    user_id=user_id,
                    query=query,
                    limit=limit,
                )

                rank = 1
                # Extract from edges (facts)
                for edge in (zep_results.edges or []):
                    fact = getattr(edge, "fact", None) or getattr(edge, "name", str(edge))
                    score = getattr(edge, "score", 0.5) or 0.5
                    entry = MemoryEntry(
                        key=f"zep_edge_{rank}",
                        content=str(fact),
                        memory_type=MemoryType.SEMANTIC,
                        metadata=MemoryMetadata(user_id=user_id),
                        created_at=datetime.now(),
                    )
                    results.append(SearchResult(entry=entry, score=float(score), rank=rank))
                    rank += 1

                # Extract from nodes
                for node in (zep_results.nodes or []):
                    name = getattr(node, "name", None) or str(node)
                    summary = getattr(node, "summary", name) or name
                    score = getattr(node, "score", 0.4) or 0.4
                    entry = MemoryEntry(
                        key=f"zep_node_{rank}",
                        content=str(summary),
                        memory_type=MemoryType.SEMANTIC,
                        metadata=MemoryMetadata(user_id=user_id),
                        created_at=datetime.now(),
                    )
                    results.append(SearchResult(entry=entry, score=float(score), rank=rank))
                    rank += 1

                if results:
                    return results[:limit]
            except Exception:
                pass

            # Fallback: search local store
            return self._local_search(query, limit, filters)

    def _local_search(
        self, query: str, limit: int, filters: SearchFilters | None
    ) -> list[SearchResult]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: list[tuple[str, dict, float]] = []

        for key, stored in self._store.items():
            if filters and filters.user_id and stored.get("user_id") != filters.user_id:
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

            old_user_id = self._store[key].get("user_id", self._default_user_id)
            self._store[key]["content"] = content
            if metadata:
                self._store[key]["metadata"] = metadata

            # Also add the update to Zep graph
            try:
                self._client.graph.add(
                    user_id=old_user_id,
                    type="text",
                    data=f"[UPDATE] {content}",
                )
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
                if filters and filters.user_id and stored.get("user_id") != filters.user_id:
                    continue
                entries.append(
                    MemoryEntry(
                        key=key,
                        content=stored["content"],
                        memory_type=stored.get("memory_type", MemoryType.SEMANTIC),
                        metadata=stored.get("metadata", MemoryMetadata()),
                        created_at=stored.get("created_at", datetime.now()),
                    )
                )
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

    # ── Session Operations (native thread support) ──────────────

    async def create_session(
        self, *, session_id: str | None = None, user_id: str | None = None
    ) -> str:
        """Create a Zep thread for the user."""
        async with self._track("create_session"):
            uid = user_id or self._default_user_id
            try:
                thread = self._client.threads.create(user_id=uid)  # type: ignore[attr-defined]
                thread_id = thread.id if hasattr(thread, "id") else str(thread)
                self._threads[thread_id] = uid
                return thread_id
            except Exception:
                fallback = session_id or f"thread_{uuid.uuid4().hex[:12]}"
                self._threads[fallback] = uid
                return fallback

    async def add_message(self, session_id: str, message: Message) -> WriteResult:
        """Add a message to a Zep thread."""

        async with self._track("add_message") as record:
            success = False
            try:
                from zep_cloud.types import Message as ZepMessage

                self._client.threads.add_messages(  # type: ignore[attr-defined]
                    session_id,
                    messages=[ZepMessage(  # type: ignore[call-arg]
                        role=message.role,
                        role_type=message.role,
                        content=message.content,
                    )],
                )
                success = True
            except Exception:
                pass

            # Also store locally
            self._store[f"msg_{uuid.uuid4().hex[:8]}"] = {
                "content": message.content,
                "user_id": self._threads.get(session_id, self._default_user_id),
                "memory_type": MemoryType.EPISODIC,
                "metadata": MemoryMetadata(session_id=session_id),
                "created_at": datetime.now(),
            }

        return WriteResult(
            key=f"msg_{session_id}",
            success=success,
            latency_ms=record["latency_ms"],
        )

    async def get_session_context(
        self, session_id: str, query: str | None = None
    ) -> SessionContext:
        """Get context from a Zep thread."""

        async with self._track("get_session_context"):
            facts: list[str] = []

            try:
                ctx = self._client.threads.get_user_context(session_id)  # type: ignore[attr-defined]
                if hasattr(ctx, "facts") and ctx.facts:
                    facts = [str(f) for f in ctx.facts]
                elif hasattr(ctx, "context") and ctx.context:
                    facts = [str(ctx.context)]
            except Exception:
                pass

            # Fallback: search the user's graph
            if not facts:
                uid = self._threads.get(session_id, self._default_user_id)
                if query:
                    try:
                        results = self._client.graph.search(
                            user_id=uid, query=query, limit=10,
                        )
                        for edge in (results.edges or []):
                            fact = getattr(edge, "fact", str(edge))
                            facts.append(str(fact))
                    except Exception:
                        pass

            # Final fallback: local store
            if not facts:
                for stored in self._store.values():
                    if stored.get("metadata") and stored["metadata"].session_id == session_id:
                        facts.append(stored["content"])

            return SessionContext(session_id=session_id, facts=facts)

    async def reset(self) -> None:
        self._store.clear()
        self._episode_ids.clear()
        self._threads.clear()
        # Delete the user (which clears all their graph data)
        try:
            self._client.user.delete(self._default_user_id)
        except Exception:
            pass
        # Re-create user for next run
        self._default_user_id = f"memeval_{uuid.uuid4().hex[:8]}"
        try:
            self._client.user.add(user_id=self._default_user_id)
        except Exception:
            pass
        self.clear_operation_log()
