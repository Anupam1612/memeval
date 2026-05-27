"""Base metric class and result types for memeval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult


class MetricCategory(str, Enum):
    CORE = "core"
    LIFECYCLE = "lifecycle"
    OPERATIONAL = "operational"
    GOVERNANCE = "governance"


@dataclass(frozen=True)
class MetricResult:
    """Result of a single metric evaluation."""

    metric_name: str
    score: float  # 0.0 to 1.0
    passed: bool  # score >= threshold
    threshold: float
    reason: str  # human-readable explanation
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0  # time to compute this metric


class BaseMetric(ABC):
    """Abstract base class for all memeval metrics.

    Subclass and implement `evaluate` to create a custom metric.
    Register it in the METRIC_REGISTRY to make it available by name.
    """

    name: str = "base"
    category: MetricCategory = MetricCategory.CORE
    description: str = ""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    @abstractmethod
    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        """Run evaluation against a completed scenario.

        Args:
            adapter: The memory backend being evaluated.
            scenario_result: Results from running the scenario steps.

        Returns:
            MetricResult with score, pass/fail, and details.
        """
        ...

    def _result(
        self,
        score: float,
        reason: str,
        details: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
    ) -> MetricResult:
        """Helper to build a MetricResult with automatic pass/fail."""
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            reason=reason,
            details=details or {},
            latency_ms=latency_ms,
        )

    def assert_passed(self, result: MetricResult) -> None:
        """Raise AssertionError if the metric didn't pass."""
        if not result.passed:
            raise AssertionError(
                f"[{self.name}] Score {result.score:.3f} < threshold "
                f"{result.threshold:.3f}: {result.reason}"
            )
