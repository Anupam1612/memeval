"""Tests for the cost estimation module and CostMetric."""

import pytest

from memeval.adapters.in_memory import InMemoryAdapter
from memeval.cost.estimator import CostProfile, estimate_tokens, get_profile
from memeval.cost.pricing import DEFAULT_PRICING, get_price
from memeval.metrics.cost import CostMetric
from memeval.scenarios.loader import load_builtin_scenarios
from memeval.scenarios.runner import ScenarioRunner


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_get_price_known_model():
    price = get_price("gpt-4o-mini")
    assert price == DEFAULT_PRICING["gpt-4o-mini"]


def test_get_price_partial_match():
    # "openai/gpt-4o-mini" should match "gpt-4o-mini"
    price = get_price("openai/gpt-4o-mini")
    assert price == DEFAULT_PRICING["gpt-4o-mini"]


def test_get_price_unknown_falls_back():
    price = get_price("some-unknown-model-xyz")
    assert price["input"] > 0


def test_get_price_override():
    price = get_price("gpt-4o-mini", pricing={"gpt-4o-mini": {"input": 9.0, "output": 9.0}})
    assert price["input"] == 9.0


def test_get_profile_known_adapters():
    assert get_profile("InMemoryAdapter").write_llm_input_mult == 0.0
    assert get_profile("Mem0Adapter").write_llm_input_mult > 0.0
    assert get_profile("LettaAdapter").write_llm_input_overhead > 0


def test_get_profile_unknown_adapter():
    profile = get_profile("SomeUnknownAdapter")
    assert "Unknown adapter" in profile.note


@pytest.fixture
def runner():
    return ScenarioRunner()


@pytest.fixture
def basic_scenario():
    scenarios = load_builtin_scenarios()
    return next(s for s in scenarios if s.name == "Basic Recall")


@pytest.mark.asyncio
async def test_cost_zero_for_in_memory(runner, basic_scenario):
    """InMemoryAdapter has no LLM/embedding calls -> zero cost."""
    adapter = InMemoryAdapter()
    metric = CostMetric()
    result = await runner.run(basic_scenario, adapter, [metric])

    mr = result.metric_results["cost"]
    assert mr.passed
    assert mr.score == 1.0
    assert mr.details["total_cost_usd"] == 0.0
    assert mr.details["source"] == "estimated"


@pytest.mark.asyncio
async def test_cost_with_llm_profile(runner, basic_scenario):
    """A Mem0-style profile should produce nonzero estimated cost."""
    adapter = InMemoryAdapter()
    profile = CostProfile(
        write_llm_input_mult=1.0,
        write_llm_input_overhead=150,
        write_llm_output_mult=0.3,
        write_embed_mult=0.4,
        search_embed_mult=1.0,
    )
    metric = CostMetric(profile=profile, llm_model="gpt-4o-mini")
    result = await runner.run(basic_scenario, adapter, [metric])

    mr = result.metric_results["cost"]
    assert mr.details["total_cost_usd"] > 0.0
    assert mr.details["tokens"]["llm_input_tokens"] > 0
    assert mr.details["tokens"]["embedding_tokens"] > 0
    assert mr.details["projected_monthly_usd"] > 0.0
    # Informational mode: always passes
    assert mr.passed


@pytest.mark.asyncio
async def test_cost_budget_enforcement(runner, basic_scenario):
    """An impossibly small budget should fail the metric."""
    adapter = InMemoryAdapter()
    profile = CostProfile(
        write_llm_input_mult=1.0,
        write_llm_output_mult=0.5,
        search_embed_mult=1.0,
    )
    metric = CostMetric(profile=profile, max_cost_usd=0.0000001)
    result = await runner.run(basic_scenario, adapter, [metric])

    mr = result.metric_results["cost"]
    assert not mr.passed
    assert mr.score < 1.0


@pytest.mark.asyncio
async def test_cost_budget_within_limit(runner, basic_scenario):
    """A generous budget should pass."""
    adapter = InMemoryAdapter()
    profile = CostProfile(write_llm_input_mult=1.0, search_embed_mult=1.0)
    metric = CostMetric(profile=profile, max_cost_usd=100.0)
    result = await runner.run(basic_scenario, adapter, [metric])

    mr = result.metric_results["cost"]
    assert mr.passed
    assert mr.score == 1.0


@pytest.mark.asyncio
async def test_cost_by_operation_breakdown(runner, basic_scenario):
    adapter = InMemoryAdapter()
    profile = CostProfile(write_llm_input_mult=1.0, search_embed_mult=1.0)
    metric = CostMetric(profile=profile)
    result = await runner.run(basic_scenario, adapter, [metric])

    by_op = result.metric_results["cost"].details["by_operation"]
    # Basic Recall has 5 writes (setup) + 5 assert_search steps
    assert by_op["write"]["count"] == 5
    assert by_op["assert_search"]["count"] == 5
    assert by_op["write"]["llm_input"] > 0
    assert by_op["assert_search"]["embedding"] > 0
