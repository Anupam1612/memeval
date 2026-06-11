"""Token estimation with per-adapter cost profiles.

Most memory SDKs (Mem0, Zep, CrewAI) make LLM and embedding calls
internally but do not expose token usage in their responses. This module
estimates token consumption from content length using per-adapter
profiles that model what each provider does under the hood.

Estimation rule of thumb: 1 token is approximately 4 characters of
English text. Profiles apply multipliers on top of that for LLM
extraction, embedding, and fixed prompt overhead.

When an adapter CAN report real token usage (WriteResult.tokens_used),
the measured value takes precedence over the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass


def estimate_tokens(text: str | None) -> int:
    """Estimate token count from text length. ~4 chars per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class CostProfile:
    """Models the token consumption pattern of a memory adapter.

    All *_mult fields are multipliers applied to the content's estimated
    token count. Overhead fields are fixed token counts added per call
    (system prompts, instructions).
    """

    # Write path
    write_llm_input_mult: float = 0.0     # LLM extraction input (x content tokens)
    write_llm_input_overhead: int = 0     # fixed prompt overhead per write
    write_llm_output_mult: float = 0.0    # LLM extraction output (x content tokens)
    write_embed_mult: float = 0.0         # embedding tokens (x content tokens)

    # Search path
    search_embed_mult: float = 0.0        # query embedding (x query tokens)
    search_llm_input_mult: float = 0.0    # rerank input, if any
    search_llm_output_mult: float = 0.0   # rerank output, if any

    note: str = ""


# Profiles model what each adapter's provider does per operation.
# These are estimates -- documented, conservative, and labeled as such.
PROFILES: dict[str, CostProfile] = {
    # Pure dict store: no LLM, no embeddings (substring mode)
    "InMemoryAdapter": CostProfile(
        note="No LLM or embedding calls in default substring mode.",
    ),
    # Mem0 self-hosted: every write runs LLM fact extraction + embeds
    # the extracted fact; every search embeds the query.
    "Mem0Adapter": CostProfile(
        write_llm_input_mult=1.0,
        write_llm_input_overhead=150,
        write_llm_output_mult=0.3,
        write_embed_mult=0.4,
        search_embed_mult=1.0,
        note="Mem0 runs LLM extraction on every write.",
    ),
    # Zep Cloud: processing happens server-side and is billed by Zep's
    # subscription, not the user's LLM API key. Estimated equivalents
    # shown for relative comparison.
    "ZepAdapter": CostProfile(
        write_embed_mult=1.0,
        search_embed_mult=1.0,
        note="Zep processing is server-side (subscription-billed); "
             "token figures are relative estimates.",
    ),
    # Letta: every message goes through the agent's LLM with core
    # memory context attached.
    "LettaAdapter": CostProfile(
        write_llm_input_mult=1.0,
        write_llm_input_overhead=400,
        write_llm_output_mult=0.5,
        search_embed_mult=1.0,
        note="Letta sends each message through the agent LLM with "
             "core memory overhead.",
    ),
    # LangGraph InMemoryStore: no LLM, no embeddings by default.
    "LangGraphAdapter": CostProfile(
        note="Default InMemoryStore has no semantic index; zero token cost. "
             "Stores configured with embeddings will differ.",
    ),
    # CrewAI Memory: remember() runs LLM analysis + embedding,
    # recall() embeds the query.
    "CrewAIAdapter": CostProfile(
        write_llm_input_mult=1.0,
        write_llm_input_overhead=100,
        write_llm_output_mult=0.3,
        write_embed_mult=1.0,
        search_embed_mult=1.0,
        note="CrewAI runs memory-save analysis via LLM on remember().",
    ),
}

# Conservative default for unknown adapters: embedding-only pattern.
_DEFAULT_PROFILE = CostProfile(
    write_embed_mult=1.0,
    search_embed_mult=1.0,
    note="Unknown adapter; assuming embedding-only cost pattern. "
         "Pass an explicit profile to CostMetric for accuracy.",
)


def get_profile(adapter_class_name: str) -> CostProfile:
    """Return the cost profile for an adapter class name."""
    return PROFILES.get(adapter_class_name, _DEFAULT_PROFILE)
