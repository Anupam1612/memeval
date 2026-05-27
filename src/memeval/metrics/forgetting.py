"""Forgetting Quality metric — measures selective forgetting fidelity.

Good forgetting = outdated/irrelevant facts removed intentionally.
Bad forgetting = valid information lost unintentionally.

Formulas:
    forgetting_precision = |actually_forgotten ∩ should_forget| / |actually_forgotten|
    retention_rate       = |still_present ∩ should_retain| / |should_retain|
    FQ                   = harmonic_mean(forgetting_precision, retention_rate)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult


class ForgettingQualityMetric(BaseMetric):
    """Measures quality of selective forgetting."""

    name = "forgetting_quality"
    category = MetricCategory.LIFECYCLE
    description = "Does the system forget what it should and retain what it shouldn't?"

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__(threshold)

    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        start = time.perf_counter()

        # Extract should_forget and should_retain from scenario steps
        should_forget_keys: set[str] = set()
        should_retain_keys: set[str] = set()

        for step in scenario_result.step_results:
            details = step.assertion_details
            if "should_forget" in details:
                should_forget_keys.update(details["should_forget"])
            if "should_retain" in details:
                should_retain_keys.update(details["should_retain"])

        if not should_forget_keys and not should_retain_keys:
            # Try to infer from delete steps
            deleted_keys = {
                sr.data.get("key", "")
                for sr in scenario_result.step_results
                if sr.step_type.value == "delete" and sr.success
            }
            written_keys = {
                sr.data.get("key", "")
                for sr in scenario_result.setup_results + scenario_result.step_results
                if sr.step_type.value == "write" and sr.data.get("key")
            }
            should_forget_keys = deleted_keys
            should_retain_keys = written_keys - deleted_keys

        if not should_forget_keys and not should_retain_keys:
            return self._result(1.0, "No forgetting assertions defined", latency_ms=0)

        # Check what's actually in the store now
        all_memories = await adapter.list_all(limit=10000)
        present_keys = {m.key for m in all_memories}

        # Compute forgetting precision
        actually_forgotten = should_forget_keys - present_keys
        forgetting_precision = (
            len(actually_forgotten) / len(should_forget_keys)
            if should_forget_keys
            else 1.0
        )

        # Compute retention rate
        still_present = should_retain_keys & present_keys
        retention_rate = (
            len(still_present) / len(should_retain_keys)
            if should_retain_keys
            else 1.0
        )

        # Harmonic mean
        if forgetting_precision + retention_rate > 0:
            fq = (
                2 * forgetting_precision * retention_rate
                / (forgetting_precision + retention_rate)
            )
        else:
            fq = 0.0

        elapsed = (time.perf_counter() - start) * 1000

        return self._result(
            score=fq,
            reason=(
                f"Forgetting precision={forgetting_precision:.3f}, "
                f"Retention rate={retention_rate:.3f}"
            ),
            details={
                "forgetting_precision": forgetting_precision,
                "retention_rate": retention_rate,
                "should_forget": list(should_forget_keys),
                "should_retain": list(should_retain_keys),
                "actually_forgotten": list(actually_forgotten),
                "still_present": list(still_present),
                "leaked_keys": list(should_forget_keys & present_keys),
                "lost_keys": list(should_retain_keys - present_keys),
            },
            latency_ms=elapsed,
        )
