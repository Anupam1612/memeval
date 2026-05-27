"""LLM-as-Judge utility for semantic correctness evaluation.

Supports:
    - Anthropic (Claude) — recommended
    - OpenAI (GPT)

Usage:
    judge = LLMJudge(provider="anthropic", model="claude-sonnet-4-6")
    score, reason = await judge.judge("User is vegan", "User follows a vegan diet")
    # score=1.0, reason="Identical meaning expressed differently"
"""

from __future__ import annotations

import json
import statistics
from typing import Any


JUDGE_PROMPT = """You are evaluating whether a retrieved memory matches an expected fact.

Expected fact: {expected}
Retrieved memory: {retrieved}

Rate semantic equivalence on a scale of 1-5:
  5 = identical meaning (possibly different wording)
  4 = mostly equivalent with minor omissions or additions
  3 = partially correct, captures the gist but misses key details
  2 = related topic but wrong or misleading information
  1 = unrelated or contradictory

Return ONLY a JSON object with no other text:
{{"score": <1-5>, "reason": "<brief explanation>"}}"""


class LLMJudge:
    """Semantic correctness evaluation using an LLM judge."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        trials: int = 1,
        max_tokens: int = 200,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.trials = trials
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.provider == "anthropic":
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic()
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install: pip install memeval[llm-judge-anthropic]"
                )
        elif self.provider == "openai":
            try:
                import openai

                self._client = openai.AsyncOpenAI()
            except ImportError:
                raise ImportError(
                    "openai package required. Install: pip install memeval[llm-judge-openai]"
                )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return self._client

    async def _call_llm(self, expected: str, retrieved: str) -> dict:
        """Make a single LLM call and parse the JSON response."""
        client = self._get_client()
        prompt = JUDGE_PROMPT.format(expected=expected, retrieved=retrieved)

        if self.provider == "anthropic":
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        elif self.provider == "openai":
            response = await client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return {"score": 3, "reason": f"Could not parse judge response: {text[:100]}"}

    async def judge(self, expected: str, retrieved: str) -> tuple[float, str]:
        """Judge semantic equivalence between expected and retrieved.

        Returns:
            Tuple of (normalized_score 0.0-1.0, reason string).
            Score mapping: 1→0.0, 2→0.25, 3→0.5, 4→0.75, 5→1.0
        """
        scores: list[int] = []
        last_reason = ""

        for _ in range(self.trials):
            result = await self._call_llm(expected, retrieved)
            raw_score = int(result.get("score", 3))
            raw_score = max(1, min(5, raw_score))
            scores.append(raw_score)
            last_reason = result.get("reason", "")

        median_score = statistics.median(scores)
        normalized = (median_score - 1) / 4.0  # map 1-5 to 0.0-1.0

        return normalized, last_reason

    async def judge_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[float, str]]:
        """Judge multiple (expected, retrieved) pairs.

        For cost efficiency, pairs are judged sequentially.
        Override for parallel execution if needed.
        """
        results = []
        for expected, retrieved in pairs:
            score, reason = await self.judge(expected, retrieved)
            results.append((score, reason))
        return results
