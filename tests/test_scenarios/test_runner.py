"""Tests for the scenario runner."""

import pytest

from memeval.adapters.in_memory import InMemoryAdapter
from memeval.metrics import RecallAccuracyMetric
from memeval.scenarios.loader import load_builtin_scenarios
from memeval.scenarios.runner import ScenarioRunner


@pytest.fixture
def adapter():
    return InMemoryAdapter()


@pytest.fixture
def runner():
    return ScenarioRunner()


@pytest.mark.asyncio
async def test_run_basic_recall(adapter, runner):
    scenarios = load_builtin_scenarios()
    basic = next(s for s in scenarios if s.name == "Basic Recall")

    metrics = [RecallAccuracyMetric(threshold=0.5)]
    result = await runner.run(basic, adapter, metrics)

    assert len(result.setup_results) > 0
    assert len(result.step_results) > 0
    assert "recall_accuracy" in result.metric_results

    # InMemoryAdapter with substring search should do well on basic recall
    recall = result.metric_results["recall_accuracy"]
    assert recall.score > 0


@pytest.mark.asyncio
async def test_run_forgetting_scenario(adapter, runner):
    scenarios = load_builtin_scenarios()
    forgetting = next(s for s in scenarios if s.name == "Forgetting & Decay")

    result = await runner.run(forgetting, adapter, [])

    # Check that delete steps succeeded
    delete_steps = [sr for sr in result.step_results if sr.step_type.value == "delete"]
    assert len(delete_steps) == 3
    assert all(sr.success for sr in delete_steps)


@pytest.mark.asyncio
async def test_run_all_builtin_scenarios(adapter, runner):
    scenarios = load_builtin_scenarios()

    for scenario in scenarios:
        result = await runner.run(scenario, adapter, [])
        # Should not crash
        assert result.scenario.name == scenario.name
        assert isinstance(result.setup_results, list)
        assert isinstance(result.step_results, list)
