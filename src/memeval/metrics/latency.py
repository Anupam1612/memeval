"""Latency & Cost metric — measures operational performance.

Tracks per-operation:
    - Latency: p50, p95, p99 in milliseconds
    - Token cost: input/output tokens per operation (if tracked by adapter)

Score = 1.0 if p95 < target, degrades linearly to 0.0 at 3x target.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult


def _compute_latency_stats(latencies_ms: list[float]) -> dict:
    """Compute latency statistics from a list of latencies."""
    if not latencies_ms:
        return {
            "p50": 0.0, "p95": 0.0, "p99": 0.0,
            "mean": 0.0, "min": 0.0, "max": 0.0, "count": 0,
        }
    arr = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
    }


def _latency_score(p95: float, target: float) -> float:
    """Score a p95 latency against a target. 1.0 at target, 0.0 at 5x target."""
    if p95 <= target:
        return 1.0
    elif p95 >= target * 5:
        return 0.0
    else:
        return 1.0 - (p95 - target) / (target * 4)


class LatencyCostMetric(BaseMetric):
    """Measures latency and cost of memory operations."""

    name = "latency_cost"
    category = MetricCategory.OPERATIONAL
    description = "Are memory operations fast and cost-efficient?"

    def __init__(
        self,
        threshold: float = 0.8,
        target_p95_ms: float = 500.0,
    ) -> None:
        super().__init__(threshold)
        self.target_p95_ms = target_p95_ms

    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        start = time.perf_counter()

        op_log = adapter.get_operation_log()
        if not op_log:
            return self._result(1.0, "No operations recorded", latency_ms=0)

        # Group latencies by operation type
        by_op: dict[str, list[float]] = defaultdict(list)
        all_latencies: list[float] = []
        total_tokens = 0

        for record in op_log:
            latency = record.get("latency_ms", 0.0)
            op = record.get("operation", "unknown")
            by_op[op].append(latency)
            all_latencies.append(latency)
            total_tokens += record.get("tokens_used", 0)

        overall_stats = _compute_latency_stats(all_latencies)
        per_op_stats = {op: _compute_latency_stats(lats) for op, lats in by_op.items()}

        # Score read/search operations separately from write operations.
        # Writes often involve LLM calls (extraction, embedding) and are expected
        # to be slower. Reads and searches are the user-facing latency.
        read_ops = by_op.get("search", []) + by_op.get("read", [])
        write_ops = by_op.get("write", []) + by_op.get("update", [])

        if read_ops:
            read_stats = _compute_latency_stats(read_ops)
            read_p95 = read_stats["p95"]
        else:
            read_p95 = 0.0

        if write_ops:
            write_stats = _compute_latency_stats(write_ops)
            write_p95 = write_stats["p95"]
        else:
            write_p95 = 0.0

        # Score read latency against target (weighted 70%)
        # Score write latency against 5x target — writes are expected to be slower (weighted 30%)
        read_score = _latency_score(read_p95, self.target_p95_ms)
        write_score = _latency_score(write_p95, self.target_p95_ms * 5)

        if read_ops and write_ops:
            score = 0.7 * read_score + 0.3 * write_score
        elif read_ops:
            score = read_score
        elif write_ops:
            score = write_score
        else:
            score = 1.0

        p95 = overall_stats["p95"]

        elapsed = (time.perf_counter() - start) * 1000

        return self._result(
            score=score,
            reason=(
                f"read_p95={read_p95:.0f}ms, write_p95={write_p95:.0f}ms "
                f"(target: {self.target_p95_ms:.0f}ms)"
            ),
            details={
                "overall": overall_stats,
                "per_operation": per_op_stats,
                "read_p95_ms": read_p95,
                "write_p95_ms": write_p95,
                "read_score": read_score if read_ops else None,
                "write_score": write_score if write_ops else None,
                "total_operations": len(all_latencies),
                "total_tokens": total_tokens,
                "target_p95_ms": self.target_p95_ms,
            },
            latency_ms=elapsed,
        )
