"""Scalability metric — benchmarks performance degradation at scale.

Inserts memories at increasing scale points and measures retrieval
latency and accuracy at each level.

Score based on degradation ratio:
    ratio < 2x  → 1.0 (excellent)
    ratio 2-5x  → 0.7 (acceptable)
    ratio 5-10x → 0.4 (poor)
    ratio > 10x → 0.1 (failing)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult


class ScalabilityMetric(BaseMetric):
    """Benchmarks performance degradation as memory store grows."""

    name = "scalability"
    category = MetricCategory.OPERATIONAL
    description = "How does performance degrade as the memory store grows?"

    def __init__(
        self,
        threshold: float = 0.6,
        scale_points: tuple[int, ...] = (100, 500, 1000),
        search_queries: list[str] | None = None,
        searches_per_point: int = 5,
    ) -> None:
        super().__init__(threshold)
        self.scale_points = scale_points
        self.search_queries = search_queries or [
            "What does the user prefer?",
            "Tell me about recent events",
            "What was discussed earlier?",
        ]
        self.searches_per_point = searches_per_point

    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        start = time.perf_counter()

        # Save and clear current state
        original_log = adapter.get_operation_log()

        results_per_scale: dict[int, dict] = {}
        current_count = len(await adapter.list_all(limit=1))

        for scale in self.scale_points:
            # Insert memories up to this scale point
            to_insert = scale - current_count
            if to_insert > 0:
                for i in range(to_insert):
                    await adapter.write(
                        f"Scale test memory #{current_count + i}: "
                        f"This is a test fact about topic_{i % 50} with value_{i}.",
                        key=f"__scale_test_{current_count + i}",
                    )
                current_count = scale

            # Benchmark search latency at this scale
            search_latencies: list[float] = []
            for q in self.search_queries[: self.searches_per_point]:
                t0 = time.perf_counter()
                await adapter.search(q, limit=10)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                search_latencies.append(elapsed_ms)

            results_per_scale[scale] = {
                "mean_latency_ms": float(np.mean(search_latencies)),
                "p95_latency_ms": float(np.percentile(search_latencies, 95))
                if len(search_latencies) >= 2
                else float(np.mean(search_latencies)),
                "searches": len(search_latencies),
            }

        # Clean up scale test memories
        for i in range(max(self.scale_points)):
            await adapter.delete(f"__scale_test_{i}")

        # Compute degradation ratio
        scales = sorted(results_per_scale.keys())
        if len(scales) >= 2:
            first_latency = results_per_scale[scales[0]]["mean_latency_ms"]
            last_latency = results_per_scale[scales[-1]]["mean_latency_ms"]
            if first_latency > 0:
                degradation_ratio = last_latency / first_latency
            else:
                degradation_ratio = 1.0
        else:
            degradation_ratio = 1.0

        # Score from degradation ratio
        if degradation_ratio < 2:
            score = 1.0
        elif degradation_ratio < 5:
            score = 0.7
        elif degradation_ratio < 10:
            score = 0.4
        else:
            score = 0.1

        # Restore original log
        adapter._operation_log = original_log

        total_elapsed = (time.perf_counter() - start) * 1000

        return self._result(
            score=score,
            reason=f"Degradation ratio: {degradation_ratio:.2f}x across {scales}",
            details={
                "scale_points": results_per_scale,
                "degradation_ratio": degradation_ratio,
            },
            latency_ms=total_elapsed,
        )
