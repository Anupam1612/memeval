"""memeval — Evaluation framework for agent memory systems.

Quick start:
    from memeval import evaluate, InMemoryAdapter
    from memeval.metrics import RecallAccuracyMetric

    adapter = InMemoryAdapter()
    results = await evaluate(adapter=adapter, scenarios="builtin")

CLI:
    memeval run --adapter in_memory
    memeval benchmark --adapters mem0 --adapters zep
"""

from __future__ import annotations

__version__ = "0.1.1"

from memeval.adapters.in_memory import InMemoryAdapter
from memeval.metrics import METRIC_REGISTRY
from memeval.metrics.base import BaseMetric, MetricResult
from memeval.protocol.base import MemoryProtocol
from memeval.protocol.types import (
    MemoryEntry,
    MemoryMetadata,
    MemoryType,
    SearchFilters,
    SearchResult,
    WriteResult,
)
from memeval.scenarios.loader import load_builtin_scenarios, load_scenario, load_scenarios_from_dir
from memeval.scenarios.runner import ScenarioRunner
from memeval.scenarios.types import Scenario, ScenarioResult


async def evaluate(
    adapter: MemoryProtocol,
    scenarios: str | list[Scenario] = "builtin",
    metrics: list[BaseMetric] | None = None,
) -> list[ScenarioResult]:
    """Run evaluation scenarios against a memory adapter.

    Args:
        adapter: The memory backend to evaluate.
        scenarios: "builtin" for built-in suite, a directory path, or a list of Scenario objects.
        metrics: Optional explicit metrics. If None, uses each scenario's dimensions_tested.

    Returns:
        List of ScenarioResult for each scenario run.
    """
    if isinstance(scenarios, str):
        if scenarios == "builtin":
            scenario_list = load_builtin_scenarios()
        else:
            scenario_list = load_scenarios_from_dir(scenarios)
    else:
        scenario_list = scenarios

    runner = ScenarioRunner()
    results = []

    for scenario in scenario_list:
        # Resolve metrics
        run_metrics = metrics
        if run_metrics is None:
            run_metrics = []
            for dim in scenario.dimensions_tested:
                metric_cls = METRIC_REGISTRY.get(dim)
                if metric_cls:
                    threshold = scenario.thresholds.get(dim, 0.5)
                    run_metrics.append(metric_cls(threshold=threshold))

        result = await runner.run(scenario, adapter, run_metrics)
        results.append(result)

    return results


__all__ = [
    "__version__",
    "evaluate",
    "InMemoryAdapter",
    "MemoryProtocol",
    "MemoryEntry",
    "MemoryMetadata",
    "MemoryType",
    "SearchFilters",
    "SearchResult",
    "WriteResult",
    "Scenario",
    "ScenarioResult",
    "ScenarioRunner",
    "BaseMetric",
    "MetricResult",
    "METRIC_REGISTRY",
    "load_scenario",
    "load_scenarios_from_dir",
    "load_builtin_scenarios",
]
