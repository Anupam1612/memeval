"""Recall Accuracy metric — can the system retrieve what was stored?

Two modes:
    - exact: normalized string matching
    - semantic: embedding cosine similarity (requires sentence-transformers)

Formulas:
    recall    = |retrieved ∩ expected| / |expected|
    precision = |retrieved ∩ expected| / |retrieved|
    f1        = 2 * P * R / (P + R)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult


def _normalize(text: str) -> str:
    return text.strip().lower()


def _compute_exact_recall(expected: list[str], retrieved: list[str]) -> float:
    if not expected:
        return 1.0
    if not retrieved:
        return 0.0
    expected_norm = {_normalize(e) for e in expected}
    retrieved_norm = {_normalize(r) for r in retrieved}
    matched = expected_norm & retrieved_norm
    return len(matched) / len(expected_norm)


def _compute_semantic_scores(
    expected: list[str],
    retrieved: list[str],
    model: object,
    threshold: float,
) -> dict:
    """Compute semantic recall/precision/F1 using embeddings."""
    if not expected:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0, "max_similarities": []}
    if not retrieved:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "max_similarities": []}

    exp_emb = model.encode(expected, normalize_embeddings=True)  # type: ignore[attr-defined]
    ret_emb = model.encode(retrieved, normalize_embeddings=True)  # type: ignore[attr-defined]

    # Cosine similarity matrix (already L2-normalized, so dot = cosine)
    sim_matrix = np.dot(exp_emb, ret_emb.T)

    # Recall: for each expected fact, find max similarity in retrieved
    max_sims_recall = sim_matrix.max(axis=1)
    recall = float(np.mean(max_sims_recall >= threshold))

    # Precision: for each retrieved fact, find max similarity in expected
    max_sims_precision = sim_matrix.max(axis=0)
    precision = float(np.mean(max_sims_precision >= threshold))

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "max_similarities": max_sims_recall.tolist(),
    }


class RecallAccuracyMetric(BaseMetric):
    """Measures whether stored memories can be retrieved via search."""

    name = "recall_accuracy"
    category = MetricCategory.CORE
    description = "Can the memory system retrieve what was stored?"

    def __init__(
        self,
        threshold: float = 0.8,
        mode: str = "substring",
        similarity_cutoff: float = 0.85,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
    ) -> None:
        super().__init__(threshold)
        self.mode = mode
        self.similarity_cutoff = similarity_cutoff
        self.embedding_model_name = embedding_model
        self._embedder = None

    def _get_embedder(self) -> object:
        if self._embedder is None:
            if self.mode == "semantic":
                try:
                    from sentence_transformers import SentenceTransformer

                    self._embedder = SentenceTransformer(self.embedding_model_name)
                except ImportError:
                    raise ImportError(
                        "sentence-transformers required for semantic mode. "
                        "Install: pip install memeval[embeddings]"
                    )
        return self._embedder

    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        start = time.perf_counter()

        search_steps = scenario_result.get_all_search_results()
        if not search_steps:
            return self._result(1.0, "No search steps to evaluate", latency_ms=0)

        scores: list[float] = []
        details_per_step: list[dict] = []

        for step in search_steps:
            expected = step.assertion_details.get("expected_contains", [])
            retrieved_contents = [
                r["content"] for r in step.data.get("results", [])
            ]

            if not expected:
                continue

            if self.mode == "semantic":
                embedder = self._get_embedder()
                result = _compute_semantic_scores(
                    expected, retrieved_contents, embedder, self.similarity_cutoff
                )
                scores.append(result["recall"])
                details_per_step.append(result)
            else:
                # Substring matching: check if each expected string appears
                # in any retrieved content
                matched = 0
                for exp in expected:
                    exp_lower = _normalize(exp)
                    if any(exp_lower in _normalize(ret) for ret in retrieved_contents):
                        matched += 1
                recall = matched / len(expected) if expected else 1.0
                scores.append(recall)
                details_per_step.append({
                    "recall": recall,
                    "expected": expected,
                    "retrieved_count": len(retrieved_contents),
                    "matched": matched,
                })

        overall = float(np.mean(scores)) if scores else 1.0
        elapsed = (time.perf_counter() - start) * 1000

        return self._result(
            score=overall,
            reason=f"Recall {overall:.3f} across {len(scores)} search assertions",
            details={
                "mode": self.mode,
                "per_step": details_per_step,
                "num_evaluated": len(scores),
            },
            latency_ms=elapsed,
        )
