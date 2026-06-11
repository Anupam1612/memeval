"""Cost metric -- estimates the token cost of memory operations.

Answers the question nobody else measures: how much does your agent's
memory actually cost per operation, per scenario, per month?

Hybrid counting:
    - measured:  if the adapter reports tokens (WriteResult.tokens_used)
    - estimated: otherwise, from content length via per-adapter profiles
      (see memeval.cost.estimator)

Scoring:
    - No budget set (default): score 1.0, informational only.
      Cost details appear in the report.
    - Budget set (max_cost_usd): score = min(1.0, budget / actual).
      Fails when the scenario costs more than the budget.
      Useful as a CI gate: "fail the build if memory costs exceed $0.05
      per scenario run."

This metric measures and reports. It does not recommend optimizations.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from memeval.cost.estimator import CostProfile, estimate_tokens, get_profile
from memeval.cost.pricing import get_price
from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult

# Step types that hit the write path
_WRITE_STEPS = {"write", "update", "add_message"}
# Step types that hit the search path
_SEARCH_STEPS = {"search", "assert_search", "assert_context"}


class CostMetric(BaseMetric):
    """Estimates token cost of all memory operations in a scenario."""

    name = "cost"
    category = MetricCategory.OPERATIONAL
    description = "How much do the memory operations cost in tokens and USD?"

    def __init__(
        self,
        threshold: float = 0.0,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        max_cost_usd: float | None = None,
        ops_per_day: int = 10_000,
        pricing: dict[str, dict[str, float]] | None = None,
        profile: CostProfile | None = None,
    ) -> None:
        # When a budget is set, the metric must fail if cost exceeds it:
        # score = budget/actual < 1.0 when over budget, so threshold is 1.0.
        if max_cost_usd is not None and threshold == 0.0:
            threshold = 1.0
        super().__init__(threshold)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.max_cost_usd = max_cost_usd
        self.ops_per_day = ops_per_day
        self.pricing = pricing
        self.profile = profile

    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        start = time.perf_counter()

        profile = self.profile or get_profile(adapter.__class__.__name__)
        llm_price = get_price(self.llm_model, self.pricing)
        embed_price = get_price(self.embedding_model, self.pricing)

        totals = {
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "embedding_tokens": 0,
        }
        by_op: dict[str, dict[str, Any]] = {}
        measured_ops = 0
        estimated_ops = 0

        all_steps = scenario_result.setup_results + scenario_result.step_results
        for sr in all_steps:
            step_type = sr.step_type.value
            op_bucket = by_op.setdefault(
                step_type,
                {"count": 0, "llm_input": 0, "llm_output": 0, "embedding": 0},
            )
            op_bucket["count"] += 1

            measured = sr.data.get("tokens_used")
            if measured is not None:
                # Adapter reported real usage; treat as LLM input tokens
                totals["llm_input_tokens"] += int(measured)
                op_bucket["llm_input"] += int(measured)
                measured_ops += 1
                continue

            if step_type in _WRITE_STEPS:
                chars = sr.data.get("content_chars", 0)
                content_tokens = max(1, chars // 4) if chars else 0
                if content_tokens == 0:
                    continue
                llm_in = int(
                    content_tokens * profile.write_llm_input_mult
                    + (profile.write_llm_input_overhead
                       if profile.write_llm_input_mult > 0 else 0)
                )
                llm_out = int(content_tokens * profile.write_llm_output_mult)
                embed = int(content_tokens * profile.write_embed_mult)
                totals["llm_input_tokens"] += llm_in
                totals["llm_output_tokens"] += llm_out
                totals["embedding_tokens"] += embed
                op_bucket["llm_input"] += llm_in
                op_bucket["llm_output"] += llm_out
                op_bucket["embedding"] += embed
                estimated_ops += 1

            elif step_type in _SEARCH_STEPS:
                query_tokens = estimate_tokens(sr.data.get("query") or "")
                if sr.data.get("query_chars"):
                    query_tokens = max(1, sr.data["query_chars"] // 4)
                if query_tokens == 0:
                    continue
                embed = int(query_tokens * profile.search_embed_mult)
                llm_in = int(query_tokens * profile.search_llm_input_mult)
                llm_out = int(query_tokens * profile.search_llm_output_mult)
                totals["embedding_tokens"] += embed
                totals["llm_input_tokens"] += llm_in
                totals["llm_output_tokens"] += llm_out
                op_bucket["embedding"] += embed
                op_bucket["llm_input"] += llm_in
                op_bucket["llm_output"] += llm_out
                estimated_ops += 1

        # USD cost (prices are per 1M tokens)
        llm_input_cost = totals["llm_input_tokens"] * llm_price["input"] / 1e6
        llm_output_cost = totals["llm_output_tokens"] * llm_price.get("output", 0.0) / 1e6
        embedding_cost = totals["embedding_tokens"] * embed_price["input"] / 1e6
        total_cost = llm_input_cost + llm_output_cost + embedding_cost

        total_ops = sum(b["count"] for b in by_op.values())
        cost_per_op = total_cost / total_ops if total_ops else 0.0
        projected_monthly = cost_per_op * self.ops_per_day * 30

        source = "measured" if estimated_ops == 0 and measured_ops > 0 else "estimated"
        if measured_ops > 0 and estimated_ops > 0:
            source = "mixed"

        # Scoring
        if self.max_cost_usd is not None and total_cost > 0:
            score = min(1.0, self.max_cost_usd / total_cost)
            reason = (
                f"${total_cost:.4f} {source} "
                f"(budget: ${self.max_cost_usd:.4f})"
            )
        else:
            score = 1.0
            reason = f"${total_cost:.4f} {source}, {total_ops} ops"

        elapsed = (time.perf_counter() - start) * 1000

        return self._result(
            score=score,
            reason=reason,
            details={
                "total_cost_usd": round(total_cost, 6),
                "cost_per_operation_usd": round(cost_per_op, 6),
                "projected_monthly_usd": round(projected_monthly, 2),
                "ops_per_day_assumption": self.ops_per_day,
                "tokens": totals,
                "by_operation": by_op,
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model,
                "source": source,
                "profile_note": profile.note,
                "budget_usd": self.max_cost_usd,
            },
            latency_ms=elapsed,
        )
