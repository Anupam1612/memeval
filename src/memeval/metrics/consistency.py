"""Consistency metric — detects contradictions in the memory store.

Uses either:
    - NLI (Natural Language Inference) model for pairwise contradiction detection
    - Simple heuristic (keyword-based negation detection) as fallback

Formula:
    consistency_score = 1 - (contradictions_found / pairs_checked)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult


# Common negation/update patterns for heuristic detection
_CONTRADICTION_PATTERNS = [
    ("is", "is not"),
    ("was", "was not"),
    ("likes", "dislikes"),
    ("prefers", "does not prefer"),
    ("vegetarian", "vegan"),
    ("yes", "no"),
    ("true", "false"),
    ("active", "inactive"),
    ("enabled", "disabled"),
]


def _heuristic_contradiction_check(mem_a: str, mem_b: str) -> float:
    """Simple heuristic: returns a contradiction score 0-1."""
    a_lower = mem_a.lower()
    b_lower = mem_b.lower()

    # Check if they're about the same topic but have conflicting values
    a_words = set(a_lower.split())
    b_words = set(b_lower.split())
    overlap = a_words & b_words

    if len(overlap) < 2:
        return 0.0  # probably about different topics

    for pos, neg in _CONTRADICTION_PATTERNS:
        if (pos in a_lower and neg in b_lower) or (neg in a_lower and pos in b_lower):
            return 0.8

    # Check for direct negation
    if "not" in a_lower and "not" not in b_lower and overlap:
        return 0.6
    if "not" in b_lower and "not" not in a_lower and overlap:
        return 0.6

    return 0.0


def _nli_contradiction_check(
    memories: list[str], model_name: str, threshold: float
) -> list[dict[str, Any]]:
    """Use an NLI model to detect contradictions pairwise."""
    try:
        from transformers import pipeline

        nli = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,  # CPU
        )
    except ImportError:
        raise ImportError(
            "transformers is required for NLI-based consistency checking. "
            "Install: pip install memeval[nli]"
        )

    contradictions = []
    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            result = nli(
                f"{memories[i]}. {memories[j]}",
                candidate_labels=["entailment", "neutral", "contradiction"],
            )
            labels = result["labels"]
            scores = result["scores"]
            contradiction_score = scores[labels.index("contradiction")]

            if contradiction_score > threshold:
                contradictions.append({
                    "memory_a_index": i,
                    "memory_b_index": j,
                    "memory_a": memories[i],
                    "memory_b": memories[j],
                    "contradiction_score": contradiction_score,
                })

    return contradictions


class ConsistencyMetric(BaseMetric):
    """Detects contradictions in the memory store."""

    name = "consistency"
    category = MetricCategory.CORE
    description = "Are stored memories free of contradictions?"

    def __init__(
        self,
        threshold: float = 0.95,
        mode: str = "heuristic",
        nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        contradiction_threshold: float = 0.7,
    ) -> None:
        super().__init__(threshold)
        self.mode = mode
        self.nli_model = nli_model
        self.contradiction_threshold = contradiction_threshold

    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        start = time.perf_counter()

        # Get all memories currently in the store
        all_memories = await adapter.list_all(limit=1000)
        contents = [m.content for m in all_memories]

        if len(contents) < 2:
            return self._result(
                1.0, "Fewer than 2 memories — no contradictions possible", latency_ms=0
            )

        if self.mode == "nli":
            contradictions = _nli_contradiction_check(
                contents, self.nli_model, self.contradiction_threshold
            )
        else:
            contradictions = []
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    score = _heuristic_contradiction_check(contents[i], contents[j])
                    if score >= self.contradiction_threshold:
                        contradictions.append({
                            "memory_a_index": i,
                            "memory_b_index": j,
                            "memory_a": contents[i],
                            "memory_b": contents[j],
                            "contradiction_score": score,
                        })

        total_pairs = len(contents) * (len(contents) - 1) // 2
        score = 1.0 - (len(contradictions) / total_pairs) if total_pairs > 0 else 1.0
        elapsed = (time.perf_counter() - start) * 1000

        return self._result(
            score=score,
            reason=f"{len(contradictions)} contradictions in {total_pairs} pairs",
            details={
                "mode": self.mode,
                "total_memories": len(contents),
                "total_pairs_checked": total_pairs,
                "contradictions_found": len(contradictions),
                "contradictions": contradictions[:10],  # cap for readability
            },
            latency_ms=elapsed,
        )
