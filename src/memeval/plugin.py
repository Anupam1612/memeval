"""Pytest plugin for memeval — auto-discovers YAML scenario files.

When memeval is installed, pytest automatically picks up this plugin
via the entry_points declaration in pyproject.toml.

Usage:
    pytest --memeval-adapter=in_memory
    pytest --memeval-adapter=mem0 tests/scenarios/
    pytest --memeval-adapter=in_memory --memeval-scenarios=./my_scenarios/
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from memeval.adapters.in_memory import InMemoryAdapter
from memeval.metrics import METRIC_REGISTRY
from memeval.protocol.base import MemoryProtocol
from memeval.scenarios.loader import load_scenario
from memeval.scenarios.runner import ScenarioRunner
from memeval.scenarios.types import Scenario


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("memeval", "Memory evaluation framework")
    group.addoption(
        "--memeval-adapter",
        default="in_memory",
        help="Memory adapter to test: in_memory, mem0, zep, letta (default: in_memory)",
    )
    group.addoption(
        "--memeval-scenarios",
        default=None,
        help="Path to custom scenarios directory (default: built-in scenarios)",
    )


def _get_adapter(adapter_name: str) -> MemoryProtocol:
    """Instantiate a memory adapter by name."""
    if adapter_name == "in_memory":
        return InMemoryAdapter()
    elif adapter_name == "mem0":
        from memeval.adapters.mem0 import Mem0Adapter

        return Mem0Adapter()
    elif adapter_name == "zep":
        from memeval.adapters.zep import ZepAdapter

        return ZepAdapter()
    elif adapter_name == "letta":
        from memeval.adapters.letta import LettaAdapter

        return LettaAdapter()
    else:
        raise ValueError(
            f"Unknown adapter: {adapter_name}. "
            "Available: in_memory, mem0, zep, letta"
        )


def pytest_collect_file(parent: Any, file_path: Path) -> ScenarioFile | None:
    """Auto-discover YAML scenario files."""
    if file_path.suffix == ".yaml" and file_path.parent.name != "__pycache__":
        # Only collect from scenarios directories or if explicitly specified
        return ScenarioFile.from_parent(parent, path=file_path)
    return None


class ScenarioFile(pytest.File):
    """A YAML file containing one memeval scenario."""

    def collect(self):  # type: ignore[override]
        try:
            scenario = load_scenario(self.path)
            yield ScenarioItem.from_parent(
                self, name=scenario.name, scenario=scenario
            )
        except Exception as e:
            raise pytest.UsageError(f"Error loading scenario {self.path}: {e}")


class ScenarioItem(pytest.Item):
    """A single memeval scenario test."""

    def __init__(self, scenario: Scenario, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scenario = scenario

    def runtest(self) -> None:
        adapter_name = self.config.getoption("memeval_adapter")
        adapter = _get_adapter(adapter_name)

        # Resolve metrics from scenario's dimensions_tested
        metrics = []
        for dim in self.scenario.dimensions_tested:
            metric_cls = METRIC_REGISTRY.get(dim)
            if metric_cls:
                threshold = self.scenario.thresholds.get(dim, 0.5)
                metrics.append(metric_cls(threshold=threshold))

        runner = ScenarioRunner()
        result = asyncio.run(runner.run(self.scenario, adapter, metrics))

        # Check assertions
        for sr in result.assertion_failures:
            details = sr.assertion_details
            raise AssertionError(
                f"Step {sr.step_index} assertion failed: {details}"
            )

        # Check metrics
        for mr in result.metric_failures:
            raise AssertionError(
                f"Metric [{mr.metric_name}] failed: "
                f"score {mr.score:.3f} < threshold {mr.threshold:.3f} — {mr.reason}"
            )

    def repr_failure(self, excinfo: Any) -> str:  # type: ignore[override]
        return f"MEMEVAL SCENARIO FAILED: {self.scenario.name}\n{excinfo.value}"

    def reportinfo(self) -> tuple:
        return self.path, 0, f"memeval: {self.scenario.name}"
