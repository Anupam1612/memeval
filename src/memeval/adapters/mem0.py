"""Mem0 adapter — maps the mem0ai SDK to the Standard Memory Protocol.

Install: pip install memeval[mem0]

Mem0 organizes memories as flat extracted facts with vector embeddings.
Supports user_id, agent_id, run_id scoping via filters dict.

Tested against mem0ai v2.0.x (May 2026).
"""

from __future__ import annotations

import time
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

# Default config that works with OpenAI API key
DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini"},
    },
}


class Mem0Adapter(MemoryProtocol):
    """Adapter for Mem0 (pip install mem0ai).

    Args:
        config: Mem0 configuration dict (vector_store, llm, embedder settings).
                If None, uses DEFAULT_CONFIG (gpt-4o-mini).
        use_client: If True, uses MemoryClient (hosted platform) instead of
                    self-hosted Memory class.
        api_key: API key for Mem0 platform (required if use_client=True).
        default_user_id: Default user_id for scoping memories.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        use_client: bool = False,
        api_key: str | None = None,
        default_user_id: str = "memeval_user",
    ) -> None:
        super().__init__()
        self._default_user_id = default_user_id
        self._key_map: dict[str, str] = {}  # our_key -> mem0_id
        self._reverse_map: dict[str, str] = {}  # mem0_id -> our_key

        try:
            if use_client:
                from mem0 import MemoryClient

                self._client = MemoryClient(api_key=api_key)
                self._is_client = True
            else:
                from mem0 import Memory

                effective_config = config if config is not None else DEFAULT_CONFIG
                self._client = Memory.from_config(effective_config)
                self._is_client = False
        except ImportError:
            raise ImportError(
                "mem0ai package required. Install: pip install memeval[mem0]"
            )

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
            messages = [{"role": "user", "content": content}]

            kwargs: dict[str, Any] = {}
            kwargs["user_id"] = meta.user_id or self._default_user_id
            if meta.agent_id:
                kwargs["agent_id"] = meta.agent_id
            if meta.extra:
                kwargs["metadata"] = meta.extra

            result = self._client.add(messages, **kwargs)

            mem_id = None
            if isinstance(result, dict) and result.get("results"):
                mem_id = result["results"][0].get("id")

            assigned_key = key or mem_id or f"mem0_{time.time_ns()}"
            if mem_id:
                self._key_map[assigned_key] = mem_id
                self._reverse_map[mem_id] = assigned_key

        return WriteResult(
            key=assigned_key,
            success=mem_id is not None,
            latency_ms=record["latency_ms"],
        )

    async def read(self, key: str) -> MemoryEntry | None:
        async with self._track("read"):
            mem0_id = self._key_map.get(key)
            if not mem0_id:
                return None
            try:
                result = self._client.get(mem0_id)
                return MemoryEntry(
                    key=key,
                    content=result.get("memory", ""),
                    memory_type=MemoryType.SEMANTIC,
                    metadata=MemoryMetadata(
                        user_id=result.get("user_id"),
                        tags=tuple(result.get("categories", [])),
                    ),
                    created_at=_parse_datetime(result.get("created_at")),
                    updated_at=_parse_datetime(result.get("updated_at")),
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
            user_id = (filters.user_id if filters else None) or self._default_user_id

            # Mem0 v2 uses filters= dict for search
            result = self._client.search(
                query,
                filters={"user_id": user_id},
                limit=limit,
            )

            results = []
            items = result.get("results", []) if isinstance(result, dict) else []
            for i, item in enumerate(items):
                entry = MemoryEntry(
                    key=item.get("id", ""),
                    content=item.get("memory", ""),
                    memory_type=MemoryType.SEMANTIC,
                    metadata=MemoryMetadata(user_id=item.get("user_id")),
                    created_at=_parse_datetime(item.get("created_at")),
                )
                results.append(
                    SearchResult(entry=entry, score=item.get("score", 0.0), rank=i + 1)
                )
            return results

    async def update(
        self,
        key: str,
        content: str,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> WriteResult:
        async with self._track("update") as record:
            mem0_id = self._key_map.get(key)
            if not mem0_id:
                return WriteResult(key=key, success=False, latency_ms=record["latency_ms"])
            try:
                self._client.update(mem0_id, content)
                return WriteResult(key=key, success=True, latency_ms=record["latency_ms"])
            except Exception:
                return WriteResult(key=key, success=False, latency_ms=record["latency_ms"])

    async def delete(self, key: str) -> bool:
        async with self._track("delete"):
            mem0_id = self._key_map.get(key)
            if not mem0_id:
                return False
            try:
                self._client.delete(mem0_id)
                del self._key_map[key]
                self._reverse_map.pop(mem0_id, None)
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
            user_id = (filters.user_id if filters else None) or self._default_user_id

            # Mem0 v2 uses filters= dict for get_all
            result = self._client.get_all(
                filters={"user_id": user_id},
                limit=limit,
            )
            items = result.get("results", []) if isinstance(result, dict) else []

            entries = []
            for item in items[:limit]:
                mem0_id = item.get("id", "")
                # Reverse-map Mem0 ID back to our key if available
                mapped_key = self._reverse_map.get(mem0_id, mem0_id)
                entries.append(
                    MemoryEntry(
                        key=mapped_key,
                        content=item.get("memory", ""),
                        memory_type=MemoryType.SEMANTIC,
                        metadata=MemoryMetadata(user_id=item.get("user_id")),
                        created_at=_parse_datetime(item.get("created_at")),
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

        merged = " | ".join(contents) if strategy == "merge" else " | ".join(set(contents))

        for key in source_keys:
            await self.delete(key)

        return await self.write(merged)

    # ── Session Operations (native run_id support) ──────────────

    async def create_session(
        self, *, session_id: str | None = None, user_id: str | None = None
    ) -> str:
        import uuid

        return session_id or f"run_{uuid.uuid4().hex[:12]}"

    async def add_message(self, session_id: str, message: Message) -> WriteResult:
        """Add a message using Mem0's native run_id for session scoping."""

        async with self._track("add_message") as record:
            messages = [{"role": message.role, "content": message.content}]
            kwargs: dict[str, Any] = {
                "user_id": self._default_user_id,
                "run_id": session_id,
            }

            result = self._client.add(messages, **kwargs)

            mem_id = None
            if isinstance(result, dict) and result.get("results"):
                mem_id = result["results"][0].get("id")

        return WriteResult(
            key=mem_id or f"mem0_{session_id}",
            success=mem_id is not None,
            latency_ms=record["latency_ms"],
        )

    async def get_session_context(
        self, session_id: str, query: str | None = None
    ) -> SessionContext:
        """Get context scoped to a specific Mem0 run_id."""

        async with self._track("get_session_context"):
            if query:
                result = self._client.search(
                    query,
                    filters={"user_id": self._default_user_id, "run_id": session_id},
                    limit=20,
                )
                items = result.get("results", []) if isinstance(result, dict) else []
                facts = [item.get("memory", "") for item in items]
            else:
                result = self._client.get_all(
                    filters={"user_id": self._default_user_id, "run_id": session_id},
                    limit=50,
                )
                items = result.get("results", []) if isinstance(result, dict) else []
                facts = [item.get("memory", "") for item in items]

            return SessionContext(
                session_id=session_id,
                facts=facts,
                raw={"items": items},
            )

    async def reset(self) -> None:
        for key in list(self._key_map.keys()):
            await self.delete(key)
        try:
            self._client.delete_all(user_id=self._default_user_id)
        except Exception:
            pass
        self._key_map.clear()
        self._reverse_map.clear()
        self.clear_operation_log()


def _parse_datetime(val: str | None) -> datetime:
    if val:
        try:
            return datetime.fromisoformat(val)
        except (ValueError, TypeError):
            pass
    return datetime.now()
