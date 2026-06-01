![memeval](https://raw.githubusercontent.com/Anupam1612/memeval/main/assets/logo-light.svg)

[![PyPI](https://img.shields.io/pypi/v/memoryeval?color=blue&label=pypi)](https://pypi.org/project/memoryeval/) [![Python](https://img.shields.io/pypi/pyversions/memoryeval)](https://pypi.org/project/memoryeval/) [![License](https://img.shields.io/github/license/Anupam1612/memeval)](https://github.com/Anupam1612/memeval/blob/main/LICENSE) [![CI](https://img.shields.io/github/actions/workflow/status/Anupam1612/memeval/ci.yml?label=CI)](https://github.com/Anupam1612/memeval/actions) [![Downloads](https://img.shields.io/pypi/dm/memoryeval)](https://pypistats.org/packages/memoryeval)

**Find out why your AI agent forgot.**

Your agent told a customer they're vegan -- then recommended a steak restaurant. Your support bot asked for the account ID the customer already gave. Your personal assistant lost the project deadline mentioned three messages ago.

These are memory failures. They happen in production every day. And until now, there was no way to test for them before they reach your users.

memeval detects memory corruption, measures retrieval quality, and catches context loss in multi-turn conversations -- across any memory provider.

```bash
pip install memoryeval
memeval run --adapter in_memory
```

```
Session: Mid-Conversation Correction
  [turn 1] User: "Book a flight to Tokyo"
  [turn 3] User: "Actually, change that to Seoul"
  [assert] Query: "Where does the user want to fly?"
  [result] Retrieved: "Seoul"  -- PASS

Preference Update
  [setup]  Stored: "User is vegetarian"
  [step 1] Stored: "User switched to vegan"
  [assert] Query: "dietary preferences"
  [result] Retrieved: "vegetarian" AND "vegan"  -- FAIL: old fact not replaced
```

## What it catches

**Contradiction retention** -- your memory stores "User earns $80K" and "User earns $120K" side by side. memeval detects this using embedding similarity analysis, not keyword matching.

**Context loss in conversations** -- a fact shared in turn 1 disappears by turn 10. memeval tests recall depth across 10+ turn conversations.

**Stale data** -- user corrected a preference but the old value still appears in search results. memeval verifies that updates propagate.

**Cross-user data leakage** -- user A's private data shows up in user B's session. memeval plants sentinel values and probes for leaks.

**Silent forgetting** -- deleted facts are gone, but so are facts that should have survived. memeval measures both forgetting precision and retention rate.

## Quick start

```bash
pip install memoryeval

# Run 30 built-in scenarios against the test adapter
memeval run --adapter in_memory

# Test against real Mem0 (requires OPENAI_API_KEY)
pip install memoryeval[mem0]
memeval run --adapter mem0

# Compare providers side-by-side
memeval benchmark --adapters in_memory --adapters mem0
```

## Multi-turn conversation testing

memeval tests what users actually care about -- does the agent remember what was said earlier in this conversation?

```yaml
name: "Customer Support Multi-Turn"
steps:
  - create_session:
      session_id: "ticket_789"

  - add_message:
      session_id: "ticket_789"
      role: "user"
      content: "I was charged $99 but my plan is Basic at $29"

  - add_message:
      session_id: "ticket_789"
      role: "user"
      content: "My account email is frank@email.com"

  - add_message:
      session_id: "ticket_789"
      role: "user"
      content: "Please refund the difference"

  # 3 turns later, does the agent still know the issue?
  - assert_context:
      session_id: "ticket_789"
      query: "What is the billing issue?"
      expected_contains: ["99"]
```

Each adapter maps sessions to the provider's native concept:
- **Mem0**: `run_id`
- **Zep**: threads
- **Letta**: agent message sequence

## 30 built-in scenarios

| Category | Scenarios | What they test |
|----------|-----------|---------------|
| **Session** (6) | Basic recall, correction, 10-turn depth, isolation, support, preferences | Multi-turn conversation memory |
| **Core** (7) | Basic recall, adversarial, multi-hop, entity resolution, negation, high-volume, scale | Fact storage and retrieval |
| **Lifecycle** (6) | Preference update, contradiction, rapid updates, stale data, forgetting, GDPR deletion | Memory evolution over time |
| **Governance** (3) | Privacy isolation, multi-user isolation, cross-session recall | Data boundaries |
| **Edge cases** (2) | UTF-8 characters, empty/boundary conditions | Robustness |
| **Operations** (6) | Cascading deletion, consolidation, support handoff, context restoration, soft contradictions, scale stress | Memory management |

## What it measures

7 evaluation dimensions, each with concrete metrics:

| Dimension | What it catches | How |
|-----------|----------------|-----|
| **Recall** | "Agent forgot what I said" | Store facts, search for them, measure hit rate |
| **Relevance** | "Agent returned the wrong memory" | MRR and NDCG@k on ranked results |
| **Consistency** | "Agent has contradictory facts" | Embedding similarity to detect same-topic divergence |
| **Update propagation** | "Old value still appears after correction" | Update fact, verify old value is gone |
| **Forgetting quality** | "Deleted the wrong things" | Selective deletion precision + retention rate |
| **Latency** | "Memory operations are too slow" | p50/p95/p99 for reads vs writes separately |
| **Privacy** | "User A's data leaked to User B" | Sentinel-based cross-session probing |

## Use in CI/CD

```bash
# JSON reports for pipeline integration
memeval run --adapter mem0 --output report.json

# Reproducible multi-run benchmarks
python scripts/run_benchmark.py --adapter mem0 --runs 3 --output results/
```

```yaml
# .github/workflows/memeval.yml
- name: Memory evaluation
  run: memeval run --adapter mem0 --output report.json
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Use with pytest

```bash
pytest --memeval-adapter=mem0
```

memeval auto-discovers YAML scenario files. Write your own or use the 30 built-in ones.

## Use in Python

```python
import asyncio
from memeval import evaluate, InMemoryAdapter

async def main():
    adapter = InMemoryAdapter()
    results = await evaluate(adapter=adapter, scenarios="builtin")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.scenario.name}")

asyncio.run(main())
```

## Supported providers

| Provider | Session model | Install |
|----------|--------------|---------|
| In-Memory (testing) | dict-based | Built-in |
| [Mem0](https://mem0.ai) | run_id | `pip install memoryeval[mem0]` |
| [Zep](https://getzep.com) | threads | `pip install memoryeval[zep]` |
| [Letta](https://letta.com) | agent state | `pip install memoryeval[letta]` |
| Custom | You define it | See [writing adapters](docs/writing-adapters.md) |

## Benchmark findings

Single-run results from 2026-05-27 (memeval 0.1.2, Python 3.14, macOS ARM64, Mem0 self-hosted with gpt-4o-mini). Not statistically significant -- see [methodology](docs/benchmark-methodology.md) for reproducible multi-run benchmarks.

| Finding | Detail |
|---------|--------|
| Mem0 recall: 1.000 | LLM fact extraction makes semantic retrieval excellent |
| Mem0 consistency: 0.917 | Stores both old and new facts -- does not auto-resolve contradictions |
| Mem0 write latency: p95 ~3,500ms | Every write calls OpenAI for extraction |
| Mem0 search latency: p95 ~500ms | Retrieval is fast once facts are indexed |
| Zep graph: async indexing | Facts not searchable immediately after write |

```bash
# Reproduce
python scripts/run_benchmark.py --adapter in_memory --adapter mem0 --output results/
```

## Documentation

- [Getting started](docs/getting-started.md)
- [Writing custom adapters](docs/writing-adapters.md)
- [Benchmark methodology](docs/benchmark-methodology.md)

## License

Apache 2.0
