# memeval

**pytest for agent memory** — evaluate how well your AI agent remembers, retrieves, forgets, and isolates stored information.

No existing tool lets teams answer: *"Is my agent's memory actually working?"* memeval fills that gap.

```
                    MEMEVAL COMPARATIVE BENCHMARK                     
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓
┃ Dimension          ┃ in_memory ┃  mem0 ┃   zep ┃ letta ┃   Best    ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩
│ recall_accuracy    │     0.858 │ 1.000 │ 0.858 │ 0.858 │   mem0    │
│ relevance          │     0.610 │ 0.904 │ 0.610 │ 0.610 │   mem0    │
│ update_propagation │     0.583 │ 1.000 │ 0.583 │ 0.583 │   mem0    │
│ latency_cost       │     1.000 │ 0.840 │ 1.000 │ 1.000 │   tie     │
│ consistency        │     0.917 │ 0.917 │ 0.917 │ 0.917 │   tie     │
│ forgetting_quality │     1.000 │ 1.000 │ 1.000 │ 1.000 │   tie     │
│ privacy_isolation  │     1.000 │ 1.000 │ 1.000 │ 1.000 │   tie     │
├────────────────────┼───────────┼───────┼───────┼───────┼───────────┤
│ OVERALL            │     0.853 │ 0.952 │ 0.853 │ 0.853 │   mem0    │
└────────────────────┴───────────┴───────┴───────┴───────┴───────────┘
```

## Why memeval?

Every layer of the AI agent stack has evaluation tools — **except memory**.

| Layer | Eval Tool | Exists? |
|-------|-----------|---------|
| LLM prompts | LangSmith, Braintrust | Yes |
| RAG retrieval | Ragas, DeepEval | Yes |
| API endpoints | Postman, pytest | Yes |
| **Agent memory** | **memeval** | **Now it does** |

## What it evaluates

memeval tests **8 dimensions** of memory quality:

| Dimension | What it measures |
|-----------|-----------------|
| **Recall Accuracy** | Can the system retrieve what was stored? |
| **Relevance** | Does it return the *right* memories? (MRR, NDCG@k) |
| **Consistency** | Are there contradictions in stored facts? |
| **Update Propagation** | Do corrections propagate correctly? |
| **Forgetting Quality** | Does it forget what it should, keep what it shouldn't? |
| **Latency & Cost** | p50/p95/p99 latency, token cost per operation |
| **Scalability** | How does performance degrade at scale? |
| **Privacy Isolation** | Does data leak between users/sessions? |

## Quick Start

```bash
pip install memoryeval
```

### Run built-in scenarios

```bash
# Test against the built-in in-memory adapter
memeval run --adapter in_memory

# Test against Mem0 (requires OPENAI_API_KEY)
memeval run --adapter mem0

# Compare providers side-by-side
memeval benchmark --adapters in_memory --adapters mem0 --adapters zep
```

### Use in Python

```python
import asyncio
from memeval import evaluate, InMemoryAdapter

async def main():
    adapter = InMemoryAdapter()
    results = await evaluate(adapter=adapter, scenarios="builtin")
    
    for r in results:
        print(f"{r.scenario.name}: {'PASS' if r.passed else 'FAIL'}")
        for name, mr in r.metric_results.items():
            print(f"  {name}: {mr.score:.3f}")

asyncio.run(main())
```

### Use with pytest

```bash
# Auto-discovers YAML scenario files
pytest --memeval-adapter=mem0

# Or run specific scenarios
pytest my_scenarios/ --memeval-adapter=mem0
```

### Write custom scenarios (YAML)

```yaml
name: "User Preference Update"
description: "Tests whether corrections propagate"
dimensions_tested: [recall_accuracy, consistency, update_propagation]

setup:
  - write:
      key: "diet"
      content: "User is vegetarian"

steps:
  - write:
      key: "diet_v2"
      content: "User switched to vegan diet"

  - assert_search:
      query: "What are the user's dietary preferences?"
      expected_contains: ["vegan"]
      expected_not_contains: ["vegetarian"]

thresholds:
  recall_accuracy: 0.9
  consistency: 1.0
```

### JSON reports for CI/CD

```bash
memeval run --adapter mem0 --output report.json
```

```json
{
  "summary": {
    "scenarios_run": 10,
    "scenarios_passed": 8,
    "overall_score": 0.952
  },
  "dimensions": {
    "recall_accuracy": {"score": 1.0, "passed": true},
    "latency_cost": {"score": 0.84, "passed": true}
  }
}
```

## Supported Memory Providers

| Provider | Adapter | Install |
|----------|---------|---------|
| In-Memory (testing) | `in_memory` | Built-in |
| [Mem0](https://mem0.ai) | `mem0` | `pip install memoryeval[mem0]` |
| [Zep](https://getzep.com) | `zep` | `pip install memoryeval[zep]` |
| [Letta](https://letta.com) | `letta` | `pip install memoryeval[letta]` |
| Custom | Implement `MemoryProtocol` | See [docs](docs/writing-adapters.md) |

### Writing a custom adapter

```python
from memeval.protocol import MemoryProtocol, MemoryEntry, WriteResult

class MyMemoryAdapter(MemoryProtocol):
    async def write(self, content, *, key=None, metadata=None, memory_type="semantic"):
        # Your implementation
        ...
    
    async def search(self, query, *, limit=10, filters=None):
        # Your implementation
        ...
    
    # ... implement all 7 SMP operations
```

## Architecture

memeval is built on the **Standard Memory Protocol (SMP)** — a 7-operation interface that any memory backend implements via an adapter:

```
┌─────────────────────────────────────────────┐
│  Standard Memory Protocol (SMP)              │
│  write | read | search | update | delete     │
│  list_all | consolidate                      │
├─────────────────────────────────────────────┤
│  Adapters: Mem0 | Zep | Letta | Custom      │
├─────────────────────────────────────────────┤
│  Evaluation Harness                          │
│  YAML scenarios + 8 metric dimensions        │
├─────────────────────────────────────────────┤
│  Reporting: Console | JSON | CI/CD           │
└─────────────────────────────────────────────┘
```

## Real findings from our benchmarks

Testing against real Mem0 (self-hosted, gpt-4o-mini):

- **Recall: 1.000** — LLM fact extraction makes retrieval excellent
- **Consistency: 0.917** — Mem0 stores both old and new facts, doesn't auto-resolve contradictions
- **Latency: write p95 = 3,500ms** — every write calls OpenAI for extraction
- **Update propagation: 1.000** — corrections do propagate through search

These are production-relevant insights no other tool surfaces.

## Project Structure

```
src/memeval/
├── protocol/       # Standard Memory Protocol (SMP)
├── adapters/       # Mem0, Zep, Letta, InMemory
├── metrics/        # 8 evaluation dimensions
├── scenarios/      # YAML loader + execution engine
├── reporting/      # Console scorecard, JSON, comparisons
├── datasets/       # 24 built-in test scenarios
├── cli.py          # memeval run/benchmark/init
└── plugin.py       # pytest auto-discovery
```

## License

Apache 2.0
