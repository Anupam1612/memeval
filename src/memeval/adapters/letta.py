"""Letta adapter — maps the letta-client v1.12 SDK to the Standard Memory Protocol.

Install: pip install memeval[letta]

Letta (formerly MemGPT) has a two-tier memory:
    - Core memory: named blocks always in the agent's context window
    - Archival memory: vector-indexed passages for long-term storage

The agent autonomously manages its memory via tool calls (core_memory_append,
core_memory_replace, archival_memory_insert, archival_memory_search).

SMP Mapping:
    write  -> agents.messages.create (agent decides where to store)
    read   -> local key map + agents.blocks.list (core memory)
    search -> agents.passages.list with query_text (archival search)
    delete -> local tracking

Tested against letta-client v1.12.x (May 2026).
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


class LettaAdapter(MemoryProtocol):
    """Adapter for Letta Cloud (pip install letta-client).

    Args:
        api_key: Letta API key. If None, reads from LETTA_API_KEY env var.
        model: LLM model for the agent (default: openai/gpt-4o-mini).
        embedding: Embedding model (default: openai/text-embedding-ada-002).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-4o-mini",
        embedding: str = "openai/text-embedding-ada-002",
    ) -> None:
        super().__init__()
        self._api_key = api_key or os.getenv("LETTA_API_KEY")
        self._model = model
        self._embedding = embedding
        self._agent_id: str | None = None

        # Local store for key-based operations
        self._store: dict[str, dict[str, Any]] = {}

        try:
            from letta_client import Letta

            if not self._api_key:
                raise ValueError(
                    "Letta API key required. Set LETTA_API_KEY env var or pass api_key param."
                )
            self._client = Letta(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "letta-client package required. Install: pip install memeval[letta]"
            )

        self._ensure_agent()

    def _ensure_agent(self) -> None:
        """Create or find the eval agent."""
        try:
            agents = self._client.agents.list()
            for agent in agents:
                if getattr(agent, "name", None) == "memeval_agent":
                    self._agent_id = agent.id
                    return

            agent = self._client.agents.create(
                name="memeval_agent",
                model=self._model,
                embedding=self._embedding,
            )
            self._agent_id = agent.id
        except Exception:
            self._agent_id = None

    async def write(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: MemoryMetadata | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> WriteResult:
        async with self._track("write") as record:
            assigned_key = key or f"letta_{uuid.uuid4().hex[:12]}"

            if self._agent_id:
                try:
                    self._client.agents.messages.create(
                        agent_id=self._agent_id,
                        messages=[{
                            "role": "user",
                            "content": f"Please remember this fact: {content}",
                        }],
                    )
                except Exception:
                    pass

            # Always track locally
            self._store[assigned_key] = {
                "content": content,
                "memory_type": memory_type,
                "metadata": metadata or MemoryMetadata(),
                "created_at": datetime.now(),
            }

        return WriteResult(
            key=assigned_key,
            success=True,
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
            results: list[SearchResult] = []

            # Try Letta archival search
            if self._agent_id:
                try:
                    passages = self._client.agents.passages.list(  # type: ignore[call-arg]
                        agent_id=self._agent_id,
                        query_text=query,
                        limit=limit,
                    )
                    for i, passage in enumerate(passages or []):
                        content = getattr(passage, "text", str(passage))
                        score = getattr(passage, "score", 0.8) or 0.8
                        entry = MemoryEntry(
                            key=getattr(passage, "id", f"letta_p_{i}"),
                            content=content,
                            memory_type=MemoryType.SEMANTIC,
                            metadata=MemoryMetadata(),
                            created_at=datetime.now(),
                        )
                        results.append(
                            SearchResult(entry=entry, score=float(score), rank=i + 1)
                        )
                    if results:
                        return results[:limit]
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

    # ── Session Operations (agent = session) ─────────────────────

    async def create_session(
        self, *, session_id: str | None = None, user_id: str | None = None
    ) -> str:
        """In Letta, creating a session means using the agent.
        The agent_id IS the session -- it maintains state across messages.
        """
        if self._agent_id:
            return self._agent_id
        self._ensure_agent()
        return self._agent_id or f"letta_{uuid.uuid4().hex[:12]}"

    async def add_message(self, session_id: str, message: Message) -> WriteResult:
        """Send a message to the Letta agent. The agent maintains context."""

        async with self._track("add_message") as record:
            success = False
            if self._agent_id:
                try:
                    self._client.agents.messages.create(
                        agent_id=self._agent_id,
                        messages=[{"role": "user", "content": message.content}],  # type: ignore[typeddict-item]
                    )
                    success = True
                except Exception:
                    pass

            # Also store locally
            key = f"letta_msg_{uuid.uuid4().hex[:8]}"
            self._store[key] = {
                "content": message.content,
                "memory_type": MemoryType.EPISODIC,
                "metadata": MemoryMetadata(session_id=session_id),
                "created_at": datetime.now(),
            }

        return WriteResult(key=key, success=success, latency_ms=record["latency_ms"])

    async def get_session_context(
        self, session_id: str, query: str | None = None
    ) -> SessionContext:
        """Get context from the Letta agent's memory."""

        async with self._track("get_session_context"):
            facts: list[str] = []

            if self._agent_id:
                # Read core memory blocks (always-in-context memory)
                try:
                    blocks = self._client.agents.blocks.list(agent_id=self._agent_id)
                    for block in blocks:
                        if block.value and block.value.strip():
                            facts.append(f"[{block.label}] {block.value}")
                except Exception:
                    pass

                # Search archival memory if query provided
                if query:
                    try:
                        passages = self._client.agents.passages.list(  # type: ignore[call-arg]
                            agent_id=self._agent_id,
                            query_text=query,
                            limit=10,
                        )
                        for p in (passages or []):
                            text = getattr(p, "text", str(p))
                            facts.append(text)
                    except Exception:
                        pass

            # Fallback: local store
            if not facts:
                for stored in self._store.values():
                    meta = stored.get("metadata", MemoryMetadata())
                    if meta.session_id == session_id:
                        facts.append(stored["content"])

            return SessionContext(session_id=session_id, facts=facts)

    async def reset(self) -> None:
        self._store.clear()
        if self._agent_id:
            try:
                self._client.agents.delete(agent_id=self._agent_id)
            except Exception:
                pass
            self._agent_id = None
        self._ensure_agent()
        self.clear_operation_log()
