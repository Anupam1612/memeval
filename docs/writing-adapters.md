# Writing Custom Adapters

memeval works with any memory backend through the Standard Memory Protocol (SMP). To test your own memory system, implement the `MemoryProtocol` abstract class.

## The Protocol

Every adapter must implement 7 operations:

```python
from memeval.protocol import MemoryProtocol, MemoryEntry, WriteResult, SearchResult
from memeval.protocol.types import MemoryMetadata, MemoryType, SearchFilters


class MyAdapter(MemoryProtocol):

    async def write(self, content, *, key=None, metadata=None,
                    memory_type=MemoryType.SEMANTIC):
        """Store a memory. Return WriteResult with the assigned key."""
        ...

    async def read(self, key):
        """Retrieve a memory by key. Return MemoryEntry or None."""
        ...

    async def search(self, query, *, limit=10, filters=None):
        """Semantic search. Return list of SearchResult sorted by relevance."""
        ...

    async def update(self, key, content, *, metadata=None):
        """Update an existing memory. Return WriteResult (success=False if not found)."""
        ...

    async def delete(self, key):
        """Delete a memory. Return True if it existed and was deleted."""
        ...

    async def list_all(self, *, filters=None, limit=100):
        """List all memories. Return list of MemoryEntry."""
        ...

    async def consolidate(self, source_keys, *, strategy="merge"):
        """Merge multiple memories into one. Delete originals. Return WriteResult."""
        ...
```

## Data Types

```python
from memeval.protocol.types import (
    MemoryEntry,      # key, content, memory_type, metadata, created_at
    MemoryMetadata,   # user_id, session_id, agent_id, tags, source, timestamp
    MemoryType,       # EPISODIC, SEMANTIC, PROCEDURAL
    SearchResult,     # entry, score (0-1), rank (1-indexed)
    WriteResult,      # key, success, latency_ms, tokens_used
    SearchFilters,    # user_id, session_id, memory_type, tags, after, before
)
```

## Minimal Example

A simple adapter wrapping a Python dict:

```python
from datetime import datetime
from memeval.protocol import MemoryProtocol, MemoryEntry, WriteResult, SearchResult
from memeval.protocol.types import MemoryMetadata, MemoryType, SearchFilters


class DictAdapter(MemoryProtocol):
    def __init__(self):
        super().__init__()
        self._store = {}
        self._counter = 0

    async def write(self, content, *, key=None, metadata=None,
                    memory_type=MemoryType.SEMANTIC):
        async with self._track("write") as record:
            self._counter += 1
            k = key or f"key_{self._counter}"
            self._store[k] = MemoryEntry(
                key=k, content=content, memory_type=memory_type,
                metadata=metadata or MemoryMetadata(),
                created_at=datetime.now(),
            )
        return WriteResult(key=k, success=True, latency_ms=record["latency_ms"])

    async def read(self, key):
        async with self._track("read"):
            return self._store.get(key)

    async def search(self, query, *, limit=10, filters=None):
        async with self._track("search"):
            results = []
            for i, (k, entry) in enumerate(self._store.items()):
                if query.lower() in entry.content.lower():
                    results.append(SearchResult(entry=entry, score=1.0, rank=i+1))
            return results[:limit]

    async def update(self, key, content, *, metadata=None):
        async with self._track("update") as record:
            if key not in self._store:
                return WriteResult(key=key, success=False, latency_ms=record["latency_ms"])
            old = self._store[key]
            self._store[key] = MemoryEntry(
                key=key, content=content, memory_type=old.memory_type,
                metadata=metadata or old.metadata,
                created_at=old.created_at, updated_at=datetime.now(),
            )
        return WriteResult(key=key, success=True, latency_ms=record["latency_ms"])

    async def delete(self, key):
        async with self._track("delete"):
            return self._store.pop(key, None) is not None

    async def list_all(self, *, filters=None, limit=100):
        async with self._track("list_all"):
            return list(self._store.values())[:limit]

    async def consolidate(self, source_keys, *, strategy="merge"):
        contents = [self._store[k].content for k in source_keys if k in self._store]
        for k in source_keys:
            await self.delete(k)
        return await self.write("\n".join(contents))
```

## Using Your Adapter

### With the CLI

Register your adapter in a conftest.py or plugin, then reference it:

```python
# In your test file
import asyncio
from memeval import evaluate
from my_module import DictAdapter

async def main():
    adapter = DictAdapter()
    results = await evaluate(adapter=adapter, scenarios="builtin")
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'} {r.scenario.name}")

asyncio.run(main())
```

### With pytest

```python
# test_my_memory.py
import pytest
from memeval.metrics import RecallAccuracyMetric
from memeval.scenarios.loader import load_builtin_scenarios
from memeval.scenarios.runner import ScenarioRunner
from my_module import DictAdapter

@pytest.mark.asyncio
async def test_my_memory():
    adapter = DictAdapter()
    runner = ScenarioRunner()
    scenarios = load_builtin_scenarios()

    for scenario in scenarios:
        result = await runner.run(scenario, adapter, [RecallAccuracyMetric()])
        assert result.metric_results["recall_accuracy"].score >= 0.5
```

## Telemetry

The `_track` context manager (inherited from `MemoryProtocol`) automatically logs operation latency. Use it in every operation:

```python
async def write(self, content, **kwargs):
    async with self._track("write") as record:
        # ... your logic ...
    return WriteResult(key=k, success=True, latency_ms=record["latency_ms"])
```

This data feeds the `latency_cost` and `scalability` metrics automatically.

## Tips

- Always call `super().__init__()` in your constructor
- Wrap every operation in `async with self._track("op_name")`
- Return `None` from `read()` if the key doesn't exist (not an exception)
- Implement `reset()` for clean test isolation between scenarios
- If your backend doesn't support `consolidate`, implement it manually (read + merge + write + delete originals)
