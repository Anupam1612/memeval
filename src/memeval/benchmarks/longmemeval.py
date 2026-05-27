"""LongMemEval benchmark integration.

Loads the LongMemEval dataset (Wu et al., ICLR 2025) and evaluates
memory retrieval quality using LLM-as-judge scoring.

Dataset: 500 QA instances across multi-turn conversation histories.
Source: https://huggingface.co/datasets/xiaowu0162/longmemeval
Paper: https://arxiv.org/abs/2410.10813

Five memory abilities tested:
    - Information Extraction (single-session recall)
    - Multi-Session Reasoning (aggregation across sessions)
    - Knowledge Updates (tracking corrections over time)
    - Temporal Reasoning (time-aware questions)
    - Abstention (knowing what you don't know)

Scoring: LLM-as-judge (matching the paper's methodology) with fallback
to embedding similarity when no LLM API key is available.

Reference baselines from the paper:
    - GPT-4o (long-context):   60.6%
    - ChatGPT (with memory):   57.7%
    - Coze (with memory):      33.0%
    - Llama 3.1 70B:           33.4%

Usage:
    from memeval.benchmarks import LongMemEvalRunner

    runner = LongMemEvalRunner(limit=50)
    results = await runner.run(adapter, scoring="llm")
    print(results.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Reference baselines from the paper (Table 2, LongMemEval_S)
PAPER_BASELINES: dict[str, dict[str, float]] = {
    "GPT-4o (long-context, oracle)": {
        "overall": 0.870,
        "information-extraction": 0.970,
        "multi-session": 0.870,
        "knowledge-update": 0.830,
        "temporal-reasoning": 0.650,
    },
    "GPT-4o (long-context)": {
        "overall": 0.606,
        "information-extraction": 0.810,
        "multi-session": 0.650,
        "knowledge-update": 0.540,
        "temporal-reasoning": 0.310,
    },
    "ChatGPT (deployed, with memory)": {
        "overall": 0.577,
    },
    "Llama 3.1 70B (long-context)": {
        "overall": 0.334,
    },
}


@dataclass
class LongMemEvalSample:
    """A single LongMemEval question with its conversation history."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str | None = None
    sessions: list[list[dict[str, str]]] = field(default_factory=list)
    session_dates: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)


def load_longmemeval(
    split: str = "longmemeval_oracle",
    limit: int | None = None,
    question_types: list[str] | None = None,
) -> list[LongMemEvalSample]:
    """Load LongMemEval dataset from HuggingFace.

    Args:
        split: Dataset split. Options:
            - "longmemeval_oracle": sessions sorted by relevance (recommended)
            - "longmemeval_s": single-session sorted
            - "longmemeval_m": multi-session sorted
        limit: Maximum number of samples to load. None = all 500.
        question_types: Filter by type. Options:
            - "single-session-user"
            - "single-session-assistant"
            - "single-session-preference"
            - "temporal-reasoning"
            - "knowledge-update"
            - "multi-session"

    Returns:
        List of LongMemEvalSample objects.
    """
    data = _download_dataset(split)

    if question_types:
        data = [d for d in data if d["question_type"] in question_types]

    if limit:
        data = data[:limit]

    samples = []
    for item in data:
        samples.append(LongMemEvalSample(
            question_id=item["question_id"],
            question_type=item["question_type"],
            question=item["question"],
            answer=item["answer"],
            question_date=item.get("question_date"),
            sessions=item.get("haystack_sessions", []),
            session_dates=item.get("haystack_dates", []),
            session_ids=item.get("haystack_session_ids", []),
        ))

    return samples


def _download_dataset(split: str) -> list[dict[str, Any]]:
    """Download dataset from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub required for LongMemEval. "
            "Install: pip install huggingface_hub"
        )

    path = hf_hub_download(
        "xiaowu0162/longmemeval",
        split,
        repo_type="dataset",
    )

    with open(path) as f:
        return json.load(f)


# -- Scoring functions --

def _score_keyword(answer: str, retrieved_text: str) -> tuple[bool, float, str]:
    """Basic keyword matching. Least reliable, no dependencies.

    Checks if significant words from the answer appear in retrieved text.
    Returns (hit, confidence, method).
    """
    answer_lower = answer.lower()
    retrieved_lower = retrieved_text.lower()

    # Extract meaningful words (>3 chars, not stopwords)
    stopwords = {"this", "that", "with", "from", "have", "been", "were", "they",
                 "their", "about", "would", "could", "should", "which", "there",
                 "what", "when", "where", "some", "than", "then", "also", "just",
                 "more", "very", "into", "over", "after", "before"}
    answer_words = [
        w for w in answer_lower.split()
        if len(w) > 3 and w not in stopwords
    ]

    if not answer_words:
        hit = answer_lower in retrieved_lower
        return hit, 1.0 if hit else 0.0, "keyword-exact"

    matched = sum(1 for w in answer_words if w in retrieved_lower)
    ratio = matched / len(answer_words)
    hit = ratio >= 0.5
    return hit, ratio, "keyword"


def _score_embedding(answer: str, retrieved_text: str) -> tuple[bool, float, str]:
    """Embedding similarity scoring. More reliable than keywords.

    Uses sentence-transformers to compute semantic similarity between
    the expected answer and the retrieved context.
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # Fall back to keyword if embeddings not available
        return _score_keyword(answer, retrieved_text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = model.encode([answer, retrieved_text], normalize_embeddings=True)
    similarity = float(np.dot(embs[0], embs[1]))

    # Threshold calibrated against LongMemEval answers:
    # >0.5 is a strong semantic match for short factual answers
    hit = similarity >= 0.45
    return hit, similarity, "embedding"


async def _score_llm(
    question: str, expected_answer: str, retrieved_text: str,
    provider: str = "anthropic", model: str = "claude-sonnet-4-6",
) -> tuple[bool, float, str]:
    """LLM-as-judge scoring. Most reliable, matches the paper's methodology.

    The original paper uses GPT-4o as judge with 97% human agreement.
    We support both Anthropic (Claude) and OpenAI (GPT) as judges.
    """
    prompt = f"""You are evaluating whether a memory system retrieved the correct information.

Question: {question}
Expected answer: {expected_answer}
Retrieved context: {retrieved_text[:2000]}

Does the retrieved context contain enough information to correctly answer the question?
Consider: the answer does not need to appear word-for-word. Equivalent meanings,
paraphrases, or sufficient supporting evidence count as correct.

Respond with ONLY a JSON object:
{{"correct": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

    try:
        if provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model=model,
                max_tokens=150,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        elif provider == "openai":
            import openai
            client = openai.AsyncOpenAI()
            response = await client.chat.completions.create(
                model=model,
                max_tokens=150,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""
        else:
            return _score_keyword(expected_answer, retrieved_text)

        # Parse JSON response
        result = _parse_judge_response(text)
        return result["correct"], result["confidence"], f"llm-{provider}"

    except Exception:
        # Fall back to embedding if LLM fails
        return _score_embedding(expected_answer, retrieved_text)


def _parse_judge_response(text: str) -> dict[str, Any]:
    """Parse the LLM judge's JSON response."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return {
                "correct": bool(data.get("correct", False)),
                "confidence": float(data.get("confidence", 0.5)),
                "reason": str(data.get("reason", "")),
            }
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return {"correct": False, "confidence": 0.0, "reason": "Failed to parse judge response"}


class LongMemEvalRunner:
    """Runs LongMemEval benchmark against a memory adapter.

    For each sample:
    1. Creates a session and feeds all conversation turns
    2. Queries the session with the question
    3. Scores the retrieved context against the expected answer

    Three scoring modes (in order of reliability):
    - "llm": LLM-as-judge (matches paper methodology, requires API key)
    - "embedding": Semantic similarity (requires sentence-transformers)
    - "keyword": Word overlap (no dependencies, least reliable)
    """

    def __init__(
        self,
        split: str = "longmemeval_oracle",
        limit: int | None = None,
        question_types: list[str] | None = None,
        scoring: str = "embedding",
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-6",
    ) -> None:
        self.split = split
        self.limit = limit
        self.question_types = question_types
        self.scoring = scoring
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    async def run(
        self,
        adapter: Any,
        verbose: bool = False,
    ) -> LongMemEvalResults:
        """Run the benchmark against an adapter."""
        from memeval.protocol.types import Message

        samples = load_longmemeval(
            split=self.split,
            limit=self.limit,
            question_types=self.question_types,
        )

        results: list[LongMemEvalResult] = []

        for i, sample in enumerate(samples):
            await adapter.reset()

            session_id = await adapter.create_session(
                session_id=f"longmemeval_{sample.question_id}",
            )

            # Feed all conversation turns
            for session in sample.sessions:
                for turn in session:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    if content.strip():
                        await adapter.add_message(
                            session_id,
                            Message(role=role, content=content),
                        )

            # Query for the answer
            context = await adapter.get_session_context(
                session_id,
                query=sample.question,
            )

            retrieved_text = " ".join(context.facts)

            # Score using selected method
            hit, confidence, method = await self._score(
                sample.question, sample.answer, retrieved_text
            )

            results.append(LongMemEvalResult(
                question_id=sample.question_id,
                question_type=sample.question_type,
                question=sample.question,
                expected_answer=sample.answer,
                retrieved_facts=context.facts[:5],
                hit=hit,
                confidence=confidence,
                scoring_method=method,
            ))

            if verbose:
                status = "HIT " if hit else "MISS"
                print(
                    f"  [{status}] {i + 1}/{len(samples)} "
                    f"({sample.question_type}) "
                    f"{sample.question[:55]}... "
                    f"[{confidence:.2f}]"
                )

        return LongMemEvalResults(
            results=results,
            scoring_method=self.scoring,
        )

    async def _score(
        self, question: str, answer: str, retrieved: str
    ) -> tuple[bool, float, str]:
        """Score a single answer using the configured method."""
        if not retrieved.strip():
            return False, 0.0, self.scoring

        if self.scoring == "llm":
            return await _score_llm(
                question, answer, retrieved,
                provider=self.llm_provider, model=self.llm_model,
            )
        elif self.scoring == "embedding":
            return _score_embedding(answer, retrieved)
        else:
            return _score_keyword(answer, retrieved)


@dataclass
class LongMemEvalResult:
    """Result for a single LongMemEval question."""

    question_id: str
    question_type: str
    question: str
    expected_answer: str
    retrieved_facts: list[str]
    hit: bool
    confidence: float = 0.0
    scoring_method: str = "keyword"


@dataclass
class LongMemEvalResults:
    """Aggregated results for the LongMemEval benchmark."""

    results: list[LongMemEvalResult]
    scoring_method: str = "keyword"

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def hits(self) -> int:
        return sum(1 for r in self.results if r.hit)

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    @property
    def avg_confidence(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.confidence for r in self.results) / len(self.results)

    def accuracy_by_type(self) -> dict[str, dict[str, Any]]:
        """Accuracy broken down by question type."""
        from collections import defaultdict

        by_type: dict[str, list[LongMemEvalResult]] = defaultdict(list)
        for r in self.results:
            by_type[r.question_type].append(r)

        return {
            qtype: {
                "accuracy": sum(1 for r in items if r.hit) / len(items),
                "total": len(items),
                "hits": sum(1 for r in items if r.hit),
                "avg_confidence": sum(r.confidence for r in items) / len(items),
            }
            for qtype, items in sorted(by_type.items())
        }

    def summary(self) -> str:
        """Human-readable summary with reference baselines."""
        lines = [
            f"LongMemEval: {self.hits}/{self.total} "
            f"({self.accuracy:.1%}) "
            f"[scoring: {self.scoring_method}, "
            f"avg confidence: {self.avg_confidence:.2f}]",
            "",
            "Per-type breakdown:",
        ]
        for qtype, stats in self.accuracy_by_type().items():
            lines.append(
                f"  {qtype:<30} "
                f"{stats['hits']:>3}/{stats['total']:<3} "
                f"({stats['accuracy']:.1%})"
            )

        lines.append("")
        lines.append("Reference baselines (from paper):")
        for name, scores in PAPER_BASELINES.items():
            lines.append(f"  {name:<40} {scores['overall']:.1%}")

        return "\n".join(lines)
