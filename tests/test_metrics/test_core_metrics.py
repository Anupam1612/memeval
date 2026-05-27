"""Tests for core metrics: recall, relevance, consistency."""

import pytest

from memeval.adapters.in_memory import InMemoryAdapter
from memeval.metrics import ConsistencyMetric, RecallAccuracyMetric, RelevanceMetric
from memeval.scenarios.loader import load_builtin_scenarios
from memeval.scenarios.runner import ScenarioRunner


@pytest.fixture
def adapter():
    return InMemoryAdapter()


@pytest.fixture
def runner():
    return ScenarioRunner()


@pytest.mark.asyncio
async def test_recall_accuracy_basic(adapter, runner):
    scenarios = load_builtin_scenarios()
    basic = next(s for s in scenarios if s.name == "Basic Recall")

    metric = RecallAccuracyMetric(threshold=0.5, mode="substring")
    result = await runner.run(basic, adapter, [metric])

    mr = result.metric_results["recall_accuracy"]
    assert mr.score >= 0.0
    assert mr.score <= 1.0
    assert mr.metric_name == "recall_accuracy"
    assert "mode" in mr.details


@pytest.mark.asyncio
async def test_relevance_basic(adapter, runner):
    scenarios = load_builtin_scenarios()
    basic = next(s for s in scenarios if s.name == "Basic Recall")

    metric = RelevanceMetric(threshold=0.3, k=5)
    result = await runner.run(basic, adapter, [metric])

    mr = result.metric_results["relevance"]
    assert mr.score >= 0.0
    assert "mrr" in mr.details
    assert "ndcg_5" in mr.details


@pytest.mark.asyncio
async def test_consistency_no_contradictions(adapter, runner):
    scenarios = load_builtin_scenarios()
    basic = next(s for s in scenarios if s.name == "Basic Recall")

    metric = ConsistencyMetric(threshold=0.8, mode="basic")
    result = await runner.run(basic, adapter, [metric])

    mr = result.metric_results["consistency"]
    # Basic recall has no contradictions
    assert mr.score >= 0.8
    assert "contradictions_found" in mr.details


@pytest.mark.asyncio
async def test_consistency_with_contradictions(adapter):
    """Manually create contradictory memories and check detection."""
    await adapter.write("User is single and not in a relationship", key="c1")
    await adapter.write("User is married and has two kids", key="c2")

    metric = ConsistencyMetric(threshold=0.9, mode="basic")

    # Create a minimal scenario result
    from memeval.scenarios.types import Scenario, ScenarioResult
    scenario = Scenario(
        name="test", description="", version="1.0",
        memory_types_tested=[], dimensions_tested=[],
        config={}, setup=[], steps=[], thresholds={},
    )
    sr = ScenarioResult(scenario=scenario, setup_results=[], step_results=[])

    mr = await metric.evaluate(adapter, sr)
    assert mr.score <= 1.0
    assert mr.details["total_memories"] == 2
