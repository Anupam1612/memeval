"""Privacy & Data Isolation metric — tests for cross-session/cross-user leakage.

Pattern:
    1. Create sessions A and B with unique sentinel facts (UUIDs)
    2. From session B, query for session A's sentinels
    3. leakage_rate = |A_facts_retrievable_from_B| / |A_facts|
    4. Must be 0 — any leakage is a failure
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult
from memeval.protocol.types import MemoryMetadata, SearchFilters

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult


class PrivacyIsolationMetric(BaseMetric):
    """Tests for cross-session/cross-user data leakage."""

    name = "privacy_isolation"
    category = MetricCategory.GOVERNANCE
    description = "Is memory properly isolated between users/sessions?"

    def __init__(
        self,
        threshold: float = 1.0,  # must be perfect
        num_sentinels: int = 5,
    ) -> None:
        super().__init__(threshold)
        self.num_sentinels = num_sentinels

    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        start = time.perf_counter()

        # First check if scenario already has privacy assertion results
        privacy_steps = [
            sr for sr in scenario_result.step_results
            if sr.assertion_details.get("privacy_check")
        ]

        if privacy_steps:
            # Use scenario-driven privacy checks
            total = len(privacy_steps)
            passed = sum(1 for s in privacy_steps if s.assertion_passed)
            score = passed / total if total > 0 else 1.0
            elapsed = (time.perf_counter() - start) * 1000
            return self._result(
                score=score,
                reason=f"{passed}/{total} privacy checks passed",
                details={"scenario_driven": True},
                latency_ms=elapsed,
            )

        # Otherwise, run active probing with sentinels
        user_a = f"privacy_test_user_a_{uuid.uuid4().hex[:8]}"
        user_b = f"privacy_test_user_b_{uuid.uuid4().hex[:8]}"
        sentinels: list[str] = []
        sentinel_keys: list[str] = []

        # Write sentinel facts to user A
        for i in range(self.num_sentinels):
            sentinel = f"SENTINEL_{uuid.uuid4().hex}"
            sentinels.append(sentinel)
            key = f"__privacy_test_{i}"
            sentinel_keys.append(key)
            await adapter.write(
                f"Secret fact: {sentinel}",
                key=key,
                metadata=MemoryMetadata(user_id=user_a, session_id=f"session_a_{i}"),
            )

        # Try to retrieve user A's data from user B's context
        leaks_found = 0
        leak_details: list[dict] = []

        for sentinel in sentinels:
            results = await adapter.search(
                sentinel,
                limit=10,
                filters=SearchFilters(user_id=user_b),
            )
            for r in results:
                if sentinel in r.entry.content:
                    leaks_found += 1
                    leak_details.append({
                        "sentinel": sentinel,
                        "leaked_via": "search_with_user_filter",
                        "result_content": r.entry.content[:100],
                    })

        # Also try searching without filters to see if sentinels leak globally
        # (only flag if the result doesn't have user_a's metadata)
        for sentinel in sentinels:
            results = await adapter.search(sentinel, limit=5)
            for r in results:
                if (
                    sentinel in r.entry.content
                    and r.entry.metadata.user_id != user_a
                ):
                    leaks_found += 1
                    leak_details.append({
                        "sentinel": sentinel,
                        "leaked_via": "search_no_filter_wrong_user",
                        "result_user_id": r.entry.metadata.user_id,
                    })

        # Clean up sentinels
        for key in sentinel_keys:
            await adapter.delete(key)

        leakage_rate = leaks_found / self.num_sentinels if self.num_sentinels > 0 else 0.0
        score = 1.0 - leakage_rate

        elapsed = (time.perf_counter() - start) * 1000

        return self._result(
            score=score,
            reason=f"{leaks_found} leaks detected across {self.num_sentinels} sentinels",
            details={
                "sentinels_planted": self.num_sentinels,
                "leaks_found": leaks_found,
                "leakage_rate": leakage_rate,
                "leak_details": leak_details,
                "user_a": user_a,
                "user_b": user_b,
            },
            latency_ms=elapsed,
        )
