"""Data types for the scenario system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from memeval.metrics.base import MetricResult


class StepType(str, Enum):
    WRITE = "write"
    READ = "read"
    SEARCH = "search"
    UPDATE = "update"
    DELETE = "delete"
    CONSOLIDATE = "consolidate"
    CREATE_SESSION = "create_session"
    ADD_MESSAGE = "add_message"
    ASSERT_CONTEXT = "assert_context"
    ASSERT_READ = "assert_read"
    ASSERT_SEARCH = "assert_search"
    WAIT = "wait"


@dataclass
class StepResult:
    """Result of executing a single scenario step."""

    step_type: StepType
    step_index: int
    success: bool
    latency_ms: float
    assertion_passed: bool | None = None  # only for assert_* steps
    assertion_details: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)  # raw result from the operation


@dataclass
class Scenario:
    """A loaded test scenario."""

    name: str
    description: str
    version: str
    memory_types_tested: list[str]
    dimensions_tested: list[str]
    config: dict[str, Any]
    setup: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    thresholds: dict[str, float]
    source_path: str | None = None

    @property
    def all_steps(self) -> list[dict[str, Any]]:
        """Setup steps + test steps combined."""
        return self.setup + self.steps


@dataclass
class ScenarioResult:
    """Complete result of running a scenario."""

    scenario: Scenario
    setup_results: list[StepResult]
    step_results: list[StepResult]
    metric_results: dict[str, MetricResult] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if all assertions passed and all metrics met thresholds."""
        assertions_ok = all(
            sr.assertion_passed is not False
            for sr in self.step_results
        )
        metrics_ok = all(mr.passed for mr in self.metric_results.values())
        return assertions_ok and metrics_ok

    @property
    def assertion_failures(self) -> list[StepResult]:
        return [
            sr for sr in self.step_results
            if sr.assertion_passed is False
        ]

    @property
    def metric_failures(self) -> list[MetricResult]:
        return [mr for mr in self.metric_results.values() if not mr.passed]

    def get_all_search_results(self) -> list[StepResult]:
        """Get all search/assert_search step results."""
        return [
            sr for sr in self.setup_results + self.step_results
            if sr.step_type in (StepType.SEARCH, StepType.ASSERT_SEARCH)
        ]

    def get_all_write_results(self) -> list[StepResult]:
        """Get all write step results."""
        return [
            sr for sr in self.setup_results + self.step_results
            if sr.step_type == StepType.WRITE
        ]
