"""Update Propagation metric — do corrections propagate correctly?

Pattern:
    1. Store fact A
    2. Store correction A' (supersedes A)
    3. Query for A — should return A', not A
    4. Query derived facts dependent on A — should reflect A'

Formula:
    propagation_rate = queries_returning_updated / total_dependent_queries
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult


class UpdatePropagationMetric(BaseMetric):
    """Tests whether corrections propagate through the memory system."""

    name = "update_propagation"
    category = MetricCategory.LIFECYCLE
    description = "Do memory corrections propagate to all retrieval paths?"

    def __init__(self, threshold: float = 0.9) -> None:
        super().__init__(threshold)

    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        start = time.perf_counter()

        # Analyze assertion steps for update propagation patterns
        total_checks = 0
        propagated = 0
        failures: list[dict] = []

        for step in scenario_result.step_results:
            if step.step_type.value not in ("assert_search", "assert_read"):
                continue

            expected_not = step.assertion_details.get("expected_not_contains", [])
            expected_yes = step.assertion_details.get("expected_contains", [])

            if not expected_not and not expected_yes:
                continue

            results = step.data.get("results", [])
            result_contents = " ".join(
                r.get("content", "") for r in results
            ).lower() if results else ""

            # For assert_read, check the direct content
            if step.step_type.value == "assert_read":
                result_contents = step.data.get("content", "").lower()

            # Check that old values are NOT present
            for old_val in expected_not:
                total_checks += 1
                if old_val.lower() not in result_contents:
                    propagated += 1
                else:
                    failures.append({
                        "step_index": step.step_index,
                        "type": "stale_value_found",
                        "stale_value": old_val,
                    })

            # Check that new values ARE present
            for new_val in expected_yes:
                total_checks += 1
                if new_val.lower() in result_contents:
                    propagated += 1
                else:
                    failures.append({
                        "step_index": step.step_index,
                        "type": "updated_value_missing",
                        "expected_value": new_val,
                    })

        if total_checks == 0:
            return self._result(1.0, "No update propagation checks defined", latency_ms=0)

        score = propagated / total_checks
        elapsed = (time.perf_counter() - start) * 1000

        return self._result(
            score=score,
            reason=f"{propagated}/{total_checks} update checks passed",
            details={
                "total_checks": total_checks,
                "propagated": propagated,
                "failures": failures,
            },
            latency_ms=elapsed,
        )
