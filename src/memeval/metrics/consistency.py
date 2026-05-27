"""Consistency metric -- detects contradictions in the memory store.

Three modes (in order of credibility):
    1. "nli"       -- NLI model (DeBERTa). Most accurate. Requires pip install memoryeval[nli].
    2. "embedding" -- Embedding-based (default). Groups memories by topic similarity,
                      then flags pairs that are topically close but semantically divergent.
                      Requires pip install memoryeval[embeddings].
    3. "basic"     -- Keyword overlap + negation detection. Least reliable. No dependencies.

The embedding approach works as follows:
    - Embed all memories using sentence-transformers.
    - For each pair (i, j), compute cosine similarity.
    - If similarity is HIGH (> topic_threshold), they're about the same topic.
    - For same-topic pairs, extract the "assertion" by removing shared words
      and compare assertion embeddings.
    - If topic similarity is high but assertion similarity is low, flag as contradiction.

    Example:
        "User earns $80,000 per year" vs "User earns $120,000 per year"
        -> topic similarity: 0.95 (same subject)
        -> full similarity: 0.88 (slightly different)
        -> different numeric values in same-topic = potential contradiction

Formula:
    consistency_score = 1 - (contradictions_found / pairs_checked)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from memeval.metrics.base import BaseMetric, MetricCategory, MetricResult

if TYPE_CHECKING:
    from memeval.protocol.base import MemoryProtocol
    from memeval.scenarios.types import ScenarioResult


def _embedding_contradiction_check(
    memories: list[str],
    model_name: str,
    topic_threshold: float,
    contradiction_threshold: float,
) -> list[dict[str, Any]]:
    """Detect contradictions using embedding similarity analysis.

    Logic:
        1. Embed all memories.
        2. For each pair, compute cosine similarity.
        3. If pair similarity > topic_threshold, they're about the same topic.
        4. For same-topic pairs, check for divergent details:
           a. Extract key entities/values (numbers, proper nouns, short tokens)
              from each memory.
           b. If the topics match but the extracted values differ significantly,
              flag as a potential contradiction.
        5. Additionally, check for explicit negation patterns within same-topic
           pairs (e.g., one says "not" where the other doesn't).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required for embedding-based consistency. "
            "Install: pip install memoryeval[embeddings]"
        )

    model = SentenceTransformer(model_name)
    embeddings = model.encode(memories, normalize_embeddings=True)
    sim_matrix = np.dot(embeddings, embeddings.T)

    contradictions = []

    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            topic_sim = float(sim_matrix[i][j])

            # Only check pairs that are about the same topic
            if topic_sim < topic_threshold:
                continue

            # Same-topic pair found. Now check for divergent assertions.
            score = _compute_contradiction_score(
                memories[i], memories[j], topic_sim, model
            )

            if score >= contradiction_threshold:
                contradictions.append({
                    "memory_a_index": i,
                    "memory_b_index": j,
                    "memory_a": memories[i],
                    "memory_b": memories[j],
                    "topic_similarity": round(topic_sim, 4),
                    "contradiction_score": round(score, 4),
                })

    return contradictions


def _compute_contradiction_score(
    mem_a: str, mem_b: str, topic_sim: float, model: Any
) -> float:
    """Score how contradictory two same-topic memories are.

    Returns 0.0 (no contradiction) to 1.0 (strong contradiction).

    Signals checked:
        1. Negation asymmetry: one contains negation, the other doesn't.
        2. Numeric divergence: same structure but different numbers.
        3. Value divergence: shared context words but different value words.
    """
    a_lower = mem_a.lower()
    b_lower = mem_b.lower()
    a_words = set(a_lower.split())
    b_words = set(b_lower.split())

    signals: list[float] = []

    # Signal 1: Negation asymmetry
    negation_words = {"not", "no", "never", "neither", "cannot", "doesn't", "don't",
                      "isn't", "aren't", "won't", "wouldn't", "shouldn't", "can't"}
    a_has_neg = bool(a_words & negation_words)
    b_has_neg = bool(b_words & negation_words)
    if a_has_neg != b_has_neg:
        signals.append(0.8)

    # Signal 2: Numeric divergence
    a_numbers = _extract_numbers(a_lower)
    b_numbers = _extract_numbers(b_lower)
    if a_numbers and b_numbers and a_numbers != b_numbers:
        # Same topic, different numbers = likely contradiction
        # Stronger signal if the context (non-number words) is very similar
        shared_words = a_words & b_words
        context_overlap = len(shared_words) / max(len(a_words | b_words), 1)
        if context_overlap > 0.4:
            signals.append(0.7)

    # Signal 3: Value divergence via embedding
    # Extract the "different" parts of each memory
    shared = a_words & b_words
    a_unique = " ".join(w for w in a_lower.split() if w not in shared)
    b_unique = " ".join(w for w in b_lower.split() if w not in shared)

    if a_unique.strip() and b_unique.strip() and len(shared) >= 2:
        unique_embs = model.encode(
            [a_unique, b_unique], normalize_embeddings=True
        )
        value_sim = float(np.dot(unique_embs[0], unique_embs[1]))

        # Low similarity between the unique parts of same-topic memories
        # means they're saying different things about the same subject.
        # Scale the signal strength by topic similarity -- higher topic
        # overlap makes value divergence more meaningful.
        if value_sim < 0.3:
            signals.append(0.75)
        elif value_sim < 0.6:
            # Moderate divergence. Stronger signal if topic overlap is high.
            base = 0.5
            if topic_sim > 0.7:
                base = 0.65
            signals.append(base)

    # Signal 4: Structural substitution
    # If sentences share most words and differ in only 1-3 tokens,
    # those tokens are likely a value substitution ("CEO is X" vs "CEO is Y")
    a_ordered = a_lower.split()
    b_ordered = b_lower.split()
    if len(a_ordered) == len(b_ordered) and len(a_ordered) >= 3:
        diffs = [(a_w, b_w) for a_w, b_w in zip(a_ordered, b_ordered) if a_w != b_w]
        if 1 <= len(diffs) <= 3:
            # Most of the sentence is identical, only a small part differs
            # This is a strong substitution pattern (like "CEO is X" vs "CEO is Y")
            signals.append(0.75)

    if not signals:
        return 0.0

    # Combine signals: take the max and boost if multiple signals agree
    max_signal = max(signals)
    if len(signals) > 1:
        max_signal = min(1.0, max_signal + 0.1 * (len(signals) - 1))

    return max_signal


def _extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens from text."""
    numbers = set()
    for word in text.split():
        cleaned = word.strip("$,.()")
        try:
            float(cleaned.replace(",", ""))
            numbers.add(cleaned)
        except ValueError:
            continue
    return numbers


def _basic_contradiction_check(
    memories: list[str],
    contradiction_threshold: float,
) -> list[dict[str, Any]]:
    """Basic fallback: word overlap + negation detection. No ML models needed.

    Less reliable than embedding or NLI modes, but works without any
    extra dependencies. Checks:
        1. Whether two memories share significant word overlap (same topic).
        2. Whether one contains negation where the other doesn't.
        3. Whether they contain different numeric values in similar context.
    """
    negation_words = {"not", "no", "never", "neither", "cannot", "doesn't", "don't",
                      "isn't", "aren't", "won't", "wouldn't", "shouldn't", "can't"}

    contradictions = []

    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            a_lower = memories[i].lower()
            b_lower = memories[j].lower()
            a_words = set(a_lower.split())
            b_words = set(b_lower.split())
            shared = a_words & b_words

            # Need meaningful overlap to consider same-topic
            content_words = shared - {"the", "a", "an", "is", "are", "was", "were",
                                      "in", "on", "at", "to", "for", "of", "and",
                                      "or", "but", "with", "has", "have", "had",
                                      "user", "user's", "that", "this", "it"}
            if len(content_words) < 2:
                continue

            score = 0.0

            # Check negation asymmetry
            a_has_neg = bool(a_words & negation_words)
            b_has_neg = bool(b_words & negation_words)
            if a_has_neg != b_has_neg:
                score = max(score, 0.7)

            # Check numeric divergence
            a_nums = _extract_numbers(a_lower)
            b_nums = _extract_numbers(b_lower)
            if a_nums and b_nums and a_nums != b_nums:
                score = max(score, 0.6)

            if score >= contradiction_threshold:
                contradictions.append({
                    "memory_a_index": i,
                    "memory_b_index": j,
                    "memory_a": memories[i],
                    "memory_b": memories[j],
                    "contradiction_score": round(score, 4),
                })

    return contradictions


def _nli_contradiction_check(
    memories: list[str], model_name: str, threshold: float
) -> list[dict[str, Any]]:
    """Use an NLI model to detect contradictions pairwise."""
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "transformers required for NLI-based consistency. "
            "Install: pip install memoryeval[nli]"
        )

    nli = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,
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
                    "contradiction_score": round(contradiction_score, 4),
                })

    return contradictions


class ConsistencyMetric(BaseMetric):
    """Detects contradictions in the memory store.

    Modes (in order of reliability):
        - "nli": NLI model. Best accuracy. Requires memoryeval[nli].
        - "embedding": Embedding similarity analysis (default). Requires memoryeval[embeddings].
        - "basic": Word overlap + negation/numeric checks. No extra deps. Least reliable.
    """

    name = "consistency"
    category = MetricCategory.CORE
    description = "Are stored memories free of contradictions?"

    def __init__(
        self,
        threshold: float = 0.95,
        mode: str = "embedding",
        embedding_model: str = "all-MiniLM-L6-v2",
        nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        topic_threshold: float = 0.55,
        contradiction_threshold: float = 0.6,
    ) -> None:
        super().__init__(threshold)
        self.mode = mode
        self.embedding_model = embedding_model
        self.nli_model = nli_model
        self.topic_threshold = topic_threshold
        self.contradiction_threshold = contradiction_threshold

    async def evaluate(
        self,
        adapter: MemoryProtocol,
        scenario_result: ScenarioResult,
    ) -> MetricResult:
        start = time.perf_counter()

        all_memories = await adapter.list_all(limit=1000)
        contents = [m.content for m in all_memories]

        if len(contents) < 2:
            return self._result(
                1.0, "Fewer than 2 memories -- no contradictions possible", latency_ms=0
            )

        if self.mode == "nli":
            contradictions = _nli_contradiction_check(
                contents, self.nli_model, self.contradiction_threshold
            )
        elif self.mode == "embedding":
            try:
                contradictions = _embedding_contradiction_check(
                    contents,
                    self.embedding_model,
                    self.topic_threshold,
                    self.contradiction_threshold,
                )
            except ImportError:
                # Fall back to basic if embeddings not installed
                contradictions = _basic_contradiction_check(
                    contents, self.contradiction_threshold
                )
        else:
            contradictions = _basic_contradiction_check(
                contents, self.contradiction_threshold
            )

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
                "contradictions": contradictions[:10],
            },
            latency_ms=elapsed,
        )
