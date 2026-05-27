"""Relevance metric — does the system return the RIGHT memories?

Measures ranked retrieval quality using:
    - MRR (Mean Reciprocal Rank)
    - NDCG@k (Normalized Discounted Cumulative Gain)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult


def _compute_mrr(queries: list[dict]) -> float:
    """Compute Mean Reciprocal Rank.

    Each query dict has:
        - results: list of retrieved contents (in rank order)
        - expected: list of expected contents
    """
    if not queries:
        return 0.0

    rr_sum = 0.0
    for q in queries:
        expected_lower = {e.lower().strip() for e in q["expected"]}
        for rank, content in enumerate(q["results"], 1):
            if any(exp in content.lower() for exp in expected_lower):
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(queries)


def _compute_ndcg(relevances: list[float], k: int) -> float:
    """Compute NDCG@k from a relevance vector."""
    if not relevances:
        return 0.0
    relevances_k = relevances[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevances_k))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


class RelevanceMetric(BaseMetric):
    """Measures ranked retrieval quality of memory search."""

    name = "relevance"
    category = MetricCategory.CORE
    description = "Does the system return the most relevant memories first?"

    def __init__(self, threshold: float = 0.7, k: int = 5) -> None:
        super().__init__(threshold)
        self.k = k

    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        start = time.perf_counter()

        search_steps = scenario_result.get_all_search_results()
        if not search_steps:
            return self._result(1.0, "No search steps to evaluate", latency_ms=0)

        queries: list[dict] = []
        ndcg_scores: list[float] = []

        for step in search_steps:
            expected = step.assertion_details.get("expected_contains", [])
            results = step.data.get("results", [])
            result_contents = [r["content"] for r in results]

            if not expected:
                continue

            queries.append({"expected": expected, "results": result_contents})

            # Compute binary relevance for NDCG
            expected_lower = {e.lower().strip() for e in expected}
            relevances = []
            for content in result_contents:
                is_relevant = any(exp in content.lower() for exp in expected_lower)
                relevances.append(1.0 if is_relevant else 0.0)

            ndcg_scores.append(_compute_ndcg(relevances, self.k))

        mrr = _compute_mrr(queries)
        avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

        # Combined score: weighted average of MRR and NDCG
        combined = 0.5 * mrr + 0.5 * avg_ndcg
        elapsed = (time.perf_counter() - start) * 1000

        return self._result(
            score=combined,
            reason=f"MRR={mrr:.3f}, NDCG@{self.k}={avg_ndcg:.3f}",
            details={
                "mrr": mrr,
                f"ndcg_{self.k}": avg_ndcg,
                "per_query_ndcg": ndcg_scores,
                "num_queries": len(queries),
            },
            latency_ms=elapsed,
        )
