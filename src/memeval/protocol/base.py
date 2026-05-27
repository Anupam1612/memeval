"""Standard Memory Protocol (SMP) — the core abstraction of memeval.

Every memory backend implements this interface via an adapter.
The evaluation harness operates ONLY against this protocol,
making all metrics and scenarios backend-agnostic.

7 Operations:
    write       — store a memory
    read        — retrieve by key
    search      — semantic search
    update      — modify existing memory
    delete      — remove a memory
    list_all    — enumerate memories (for audit/consistency checks)
    consolidate — merge multiple memories (tests memory management)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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


class MemoryProtocol(ABC):
    """Abstract base class defining the Standard Memory Protocol.

    Subclass this to create an adapter for any memory backend.
    The telemetry wrapper automatically tracks latency for every operation.
    """

    def __init__(self) -> None:
        self._operation_log: list[dict] = []

    @asynccontextmanager
    async def _track(self, op_name: str) -> AsyncIterator[dict]:
        """Context manager that tracks operation latency and logs it."""
        start_ms = time.perf_counter() * 1000
        record: dict = {"operation": op_name, "start_ms": start_ms, "latency_ms": 0.0}
        await self.on_operation_start(op_name)
        try:
            yield record
        finally:
            end_ms = time.perf_counter() * 1000
            record["latency_ms"] = end_ms - record["start_ms"]
            self._operation_log.append(record)
            await self.on_operation_end(op_name, record["latency_ms"])

    def get_operation_log(self) -> list[dict]:
        """Return all tracked operations for metric computation."""
        return list(self._operation_log)

    def clear_operation_log(self) -> None:
        self._operation_log.clear()

    # ── Core CRUD ──────────────────────────────────────────────

    @abstractmethod
    async def write(
        self,
        content: str,
        *,
        key: str | None = None,
        metadata: MemoryMetadata | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> WriteResult:
        """Store a memory.

        Args:
            content: The text content to store.
            key: Optional explicit key. If None, the adapter assigns one.
            metadata: Optional metadata (user_id, tags, etc.).
            memory_type: Classification (episodic, semantic, procedural).

        Returns:
            WriteResult with the assigned key, success status, and latency.
        """
        ...

    @abstractmethod
    async def read(self, key: str) -> MemoryEntry | None:
        """Retrieve a specific memory by key.

        Returns None if the key doesn't exist.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """Semantic search over memories.

        Args:
            query: Natural language search query.
            limit: Maximum number of results to return.
            filters: Optional filters (user_id, memory_type, date range, etc.).

        Returns:
            List of SearchResult sorted by relevance (highest first).
        """
        ...

    @abstractmethod
    async def update(
        self,
        key: str,
        content: str,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> WriteResult:
        """Update an existing memory.

        Args:
            key: The key of the memory to update.
            content: The new content.
            metadata: Optional new metadata (merged with existing).

        Returns:
            WriteResult. success=False if the key doesn't exist.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a memory by key.

        Returns True if the memory existed and was deleted.
        """
        ...

    # ── Extended Operations ────────────────────────────────────

    @abstractmethod
    async def list_all(
        self,
        *,
        filters: SearchFilters | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List all memories, optionally filtered.

        Used for audit, consistency checks, and benchmarking.
        """
        ...

    @abstractmethod
    async def consolidate(
        self,
        source_keys: list[str],
        *,
        strategy: str = "merge",
    ) -> WriteResult:
        """Combine multiple memories into one.

        Strategies:
            merge       — concatenate contents
            summarize   — LLM-summarize (if supported)
            deduplicate — remove duplicate information

        Args:
            source_keys: Keys of memories to consolidate.
            strategy: Consolidation strategy.

        Returns:
            WriteResult for the new consolidated memory.
            Original memories are deleted.
        """
        ...

    # ── Session Operations (optional) ─────────────────────────
    #
    # These model multi-turn conversations. Providers map them to
    # their native session/thread concept:
    #   Mem0:  run_id
    #   Zep:   thread
    #   Letta: agent message sequence
    #
    # Default implementations fall back to write/search, so adapters
    # that don't support native sessions still work.

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Create a new conversation session. Returns the session ID.

        Override to use the provider's native session/thread/run concept.
        Default: generates a session ID without provider-side state.
        """
        import uuid

        return session_id or f"session_{uuid.uuid4().hex[:12]}"

    async def add_message(
        self,
        session_id: str,
        message: Message,
    ) -> WriteResult:
        """Add a message to an existing session.

        This is the core multi-turn operation. In production:
        - Mem0:  calls add() with run_id=session_id
        - Zep:   calls threads.add_messages(thread_id)
        - Letta: calls agents.messages.create(agent_id)

        Default: falls back to write() with session_id in metadata.
        """
        return await self.write(
            content=f"[{message.role}] {message.content}",
            metadata=MemoryMetadata(session_id=session_id),
            memory_type=MemoryType.EPISODIC,
        )

    async def get_session_context(
        self,
        session_id: str,
        query: str | None = None,
    ) -> SessionContext:
        """Get the memory context for a session.

        Returns what the memory system "knows" from this session --
        extracted facts, summaries, and relevant memories.

        Override to use the provider's native context retrieval:
        - Mem0:  search() with run_id filter
        - Zep:   threads.get_user_context(thread_id)
        - Letta: read core memory blocks + archival search

        Default: searches all memories filtered by session_id.
        """
        results = await self.search(
            query=query or "",
            limit=20,
            filters=SearchFilters(session_id=session_id),
        )
        facts = [r.entry.content for r in results]
        return SessionContext(session_id=session_id, facts=facts)

    # ── Lifecycle Hooks ────────────────────────────────────────

    async def on_operation_start(self, op_name: str) -> None:
        """Called before each operation. Override for custom telemetry."""

    async def on_operation_end(self, op_name: str, latency_ms: float) -> None:
        """Called after each operation. Override for custom telemetry."""

    # ── Convenience ────────────────────────────────────────────

    async def reset(self) -> None:
        """Clear all stored memories. Used between test scenarios.

        Default implementation lists all and deletes each.
        Override for a more efficient bulk-delete if available.
        """
        entries = await self.list_all(limit=10000)
        for entry in entries:
            await self.delete(entry.key)
        self.clear_operation_log()
