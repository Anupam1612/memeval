"""Tests for the self-contained HTML report."""

import pytest

from memeval.adapters.in_memory import InMemoryAdapter
from memeval.cost.estimator import CostProfile
from memeval.metrics import METRIC_REGISTRY
from memeval.metrics.cost import CostMetric
from memeval.reporting.html import render_benchmark_html, render_html_report
from memeval.scenarios.loader import _parse_scenario, load_builtin_scenarios
from memeval.scenarios.runner import ScenarioRunner


async def _run_scenarios(names: list[str], extra_metrics=None):
    adapter = InMemoryAdapter()
    runner = ScenarioRunner()
    scenarios = [s for s in load_builtin_scenarios() if s.name in names]
    results = []
    for scenario in scenarios:
        metrics = []
        for dim in scenario.dimensions_tested:
            metric_cls = METRIC_REGISTRY.get(dim)
            if metric_cls:
                metrics.append(metric_cls(threshold=scenario.thresholds.get(dim, 0.5)))
        if extra_metrics:
            metrics.extend(extra_metrics())
        results.append(await runner.run(scenario, adapter, metrics))
    return results


@pytest.mark.asyncio
async def test_render_html_report_basic():
    results = await _run_scenarios(["Basic Recall", "Privacy Isolation"])
    html = render_html_report(results, adapter_name="in_memory")

    assert "<!DOCTYPE html>" in html
    assert "in_memory" in html
    assert "Basic Recall" in html
    assert "Privacy Isolation" in html
    assert "Scorecard" in html
    assert "bar-track" in html  # dimension bars present
    assert "OVERALL SCORE".title() in html or "Overall score" in html
    # Self-contained: no external resources
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "cdn" not in html.lower()
    assert "@import" not in html
    assert "fonts.googleapis" not in html


@pytest.mark.asyncio
async def test_render_html_failed_scenario_has_timeline_details():
    # Basic Recall fails on InMemoryAdapter substring search
    results = await _run_scenarios(["Basic Recall"])
    html = render_html_report(results, adapter_name="in_memory")

    assert "FAILED" in html or "FAIL" in html
    assert "NOT FOUND" in html
    assert "Retrieved:" in html
    # Failed scenarios start expanded
    assert "open" in html


@pytest.mark.asyncio
async def test_render_html_with_cost():
    profile = CostProfile(write_llm_input_mult=1.0, search_embed_mult=1.0)
    results = await _run_scenarios(
        ["Basic Recall"],
        extra_metrics=lambda: [CostMetric(profile=profile)],
    )
    html = render_html_report(results, adapter_name="in_memory")

    assert "Run cost" in html
    assert "Projected monthly" in html
    assert "ops/day" in html
    assert "$" in html


@pytest.mark.asyncio
async def test_render_html_without_cost_omits_cost_cards():
    results = await _run_scenarios(["Basic Recall"])
    html = render_html_report(results, adapter_name="in_memory")

    assert "Run cost" not in html
    assert "Projected monthly" not in html


@pytest.mark.asyncio
async def test_render_benchmark_html_has_comparison():
    results_a = await _run_scenarios(["Basic Recall"])
    results_b = await _run_scenarios(["Basic Recall"])
    html = render_benchmark_html({"in_memory": results_a, "other": results_b})

    assert "Provider Comparison" in html
    assert "in_memory" in html
    assert "other" in html


@pytest.mark.asyncio
async def test_render_html_escapes_content():
    scenario = _parse_scenario({
        "name": "Escape <Test>",
        "description": "markup in content",
        "setup": [
            {"write": {"key": "k1", "content": "<script>alert(1)</script> hello"}},
        ],
        "steps": [
            {"search": {"query": "hello"}},
        ],
    })
    adapter = InMemoryAdapter()
    runner = ScenarioRunner()
    result = await runner.run(scenario, adapter, [])
    html = render_html_report([result], adapter_name="in_memory")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html or "&lt;Test&gt;" in html
