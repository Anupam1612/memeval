"""Tests for the InMemoryAdapter."""

import pytest

from memeval.adapters.in_memory import InMemoryAdapter
from memeval.protocol.types import MemoryMetadata, MemoryType, SearchFilters


@pytest.fixture
def adapter():
    return InMemoryAdapter()


@pytest.mark.asyncio
async def test_write_and_read(adapter):
    result = await adapter.write("User likes Python", key="pref_1")
    assert result.success
    assert result.key == "pref_1"

    entry = await adapter.read("pref_1")
    assert entry is not None
    assert entry.content == "User likes Python"
    assert entry.key == "pref_1"


@pytest.mark.asyncio
async def test_write_auto_key(adapter):
    result = await adapter.write("Some fact")
    assert result.success
    assert result.key.startswith("mem_")

    entry = await adapter.read(result.key)
    assert entry is not None
    assert entry.content == "Some fact"


@pytest.mark.asyncio
async def test_read_nonexistent(adapter):
    entry = await adapter.read("nonexistent_key")
    assert entry is None


@pytest.mark.asyncio
async def test_search_substring(adapter):
    await adapter.write("User's favorite color is blue", key="color")
    await adapter.write("User lives in San Francisco", key="city")
    await adapter.write("User works as an engineer", key="job")

    results = await adapter.search("favorite color")
    assert len(results) >= 1
    assert any("blue" in r.entry.content for r in results)
    assert results[0].rank == 1


@pytest.mark.asyncio
async def test_search_with_filters(adapter):
    await adapter.write(
        "Secret for user A",
        key="a_secret",
        metadata=MemoryMetadata(user_id="user_a"),
    )
    await adapter.write(
        "Secret for user B",
        key="b_secret",
        metadata=MemoryMetadata(user_id="user_b"),
    )

    results = await adapter.search(
        "Secret",
        filters=SearchFilters(user_id="user_a"),
    )
    assert len(results) == 1
    assert results[0].entry.metadata.user_id == "user_a"


@pytest.mark.asyncio
async def test_update(adapter):
    await adapter.write("User is vegetarian", key="diet")

    result = await adapter.update("diet", "User is vegan")
    assert result.success

    entry = await adapter.read("diet")
    assert entry is not None
    assert "vegan" in entry.content
    assert entry.updated_at is not None


@pytest.mark.asyncio
async def test_update_nonexistent(adapter):
    result = await adapter.update("nonexistent", "new content")
    assert not result.success


@pytest.mark.asyncio
async def test_delete(adapter):
    await adapter.write("Temporary fact", key="temp")
    assert await adapter.delete("temp")

    entry = await adapter.read("temp")
    assert entry is None


@pytest.mark.asyncio
async def test_delete_nonexistent(adapter):
    assert not await adapter.delete("nonexistent")


@pytest.mark.asyncio
async def test_list_all(adapter):
    await adapter.write("Fact 1", key="f1")
    await adapter.write("Fact 2", key="f2")
    await adapter.write("Fact 3", key="f3")

    entries = await adapter.list_all()
    assert len(entries) == 3


@pytest.mark.asyncio
async def test_list_all_with_filters(adapter):
    await adapter.write("A's fact", key="a", metadata=MemoryMetadata(user_id="alice"))
    await adapter.write("B's fact", key="b", metadata=MemoryMetadata(user_id="bob"))

    entries = await adapter.list_all(filters=SearchFilters(user_id="alice"))
    assert len(entries) == 1
    assert entries[0].metadata.user_id == "alice"


@pytest.mark.asyncio
async def test_consolidate_merge(adapter):
    await adapter.write("Fact A", key="a")
    await adapter.write("Fact B", key="b")

    result = await adapter.consolidate(["a", "b"], strategy="merge")
    assert result.success

    # Originals should be gone
    assert await adapter.read("a") is None
    assert await adapter.read("b") is None

    # Merged entry should exist
    entry = await adapter.read(result.key)
    assert entry is not None
    assert "Fact A" in entry.content
    assert "Fact B" in entry.content


@pytest.mark.asyncio
async def test_consolidate_deduplicate(adapter):
    await adapter.write("Same content", key="a")
    await adapter.write("Same content", key="b")
    await adapter.write("Different", key="c")

    result = await adapter.consolidate(["a", "b", "c"], strategy="deduplicate")
    assert result.success

    entry = await adapter.read(result.key)
    assert entry is not None
    assert entry.content.count("Same content") == 1


@pytest.mark.asyncio
async def test_memory_type(adapter):
    await adapter.write("Event happened", key="e1", memory_type=MemoryType.EPISODIC)
    entry = await adapter.read("e1")
    assert entry is not None
    assert entry.memory_type == MemoryType.EPISODIC


@pytest.mark.asyncio
async def test_reset(adapter):
    await adapter.write("Fact 1", key="f1")
    await adapter.write("Fact 2", key="f2")

    await adapter.reset()

    entries = await adapter.list_all()
    assert len(entries) == 0


@pytest.mark.asyncio
async def test_operation_log(adapter):
    await adapter.write("Test", key="t")
    await adapter.search("Test")
    await adapter.read("t")

    log = adapter.get_operation_log()
    assert len(log) == 3
    assert log[0]["operation"] == "write"
    assert log[1]["operation"] == "search"
    assert log[2]["operation"] == "read"
    assert all(r["latency_ms"] >= 0 for r in log)
