"""Tests for the top-level evaluate() function."""

import pytest

from memeval import evaluate, InMemoryAdapter
from memeval.metrics import RecallAccuracyMetric


@pytest.mark.asyncio
async def test_evaluate_builtin():
    adapter = InMemoryAdapter()
    results = await evaluate(adapter=adapter, scenarios="builtin")

    assert len(results) >= 5
    for r in results:
        assert r.scenario.name
        assert isinstance(r.step_results, list)


@pytest.mark.asyncio
async def test_evaluate_with_explicit_metrics():
    adapter = InMemoryAdapter()
    metrics = [RecallAccuracyMetric(threshold=0.3)]
    results = await evaluate(adapter=adapter, scenarios="builtin", metrics=metrics)

    for r in results:
        if r.get_all_search_results():
            assert "recall_accuracy" in r.metric_results
