"""Model pricing table for cost estimation.

Prices are USD per 1 million tokens, based on public provider pricing
as of mid-2026. Users can override any price via the `pricing` parameter
on CostMetric.

These prices change over time. They are estimates for relative
comparison, not billing-grade numbers.
"""

from __future__ import annotations

# USD per 1M tokens
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    # Anthropic
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # Local / self-hosted models have no per-token API cost
    "local": {"input": 0.0, "output": 0.0},
}

_FALLBACK = {"input": 1.00, "output": 3.00}


def get_price(model: str, pricing: dict[str, dict[str, float]] | None = None) -> dict[str, float]:
    """Look up pricing for a model.

    Args:
        model: Model name (e.g. "gpt-4o-mini").
        pricing: Optional override table merged over DEFAULT_PRICING.

    Returns:
        Dict with "input" and "output" prices per 1M tokens.
        Unknown models fall back to a mid-range default.
    """
    table = dict(DEFAULT_PRICING)
    if pricing:
        table.update(pricing)

    if model in table:
        return table[model]

    # Partial match (e.g. "openai/gpt-4o-mini" -> "gpt-4o-mini").
    # Longest key first so "gpt-4o-mini" wins over "gpt-4o".
    for known in sorted(table, key=len, reverse=True):
        if known in model:
            return table[known]

    return dict(_FALLBACK)
