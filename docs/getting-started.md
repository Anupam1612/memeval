# Getting Started with memeval

## Installation

```bash
pip install memoryeval
```

To use with a specific memory provider:

```bash
pip install memoryeval[mem0]     # Mem0 adapter
pip install memoryeval[zep]      # Zep adapter
pip install memoryeval[letta]    # Letta adapter
pip install memoryeval[all]      # Everything
```

## Quick Start

### 1. Run built-in scenarios

The fastest way to try memeval:

```bash
memeval run --adapter in_memory
```

This runs 24 built-in test scenarios against the InMemoryAdapter (no API keys needed) and prints a scorecard.

### 2. Test against a real provider

Set your API key and run against Mem0:

```bash
export OPENAI_API_KEY=sk-...
memeval run --adapter mem0
```

### 3. Compare providers

```bash
memeval benchmark --adapters in_memory --adapters mem0
```

### 4. Use in your project

Initialize memeval in your project directory:

```bash
memeval init
```

This creates a `memeval_scenarios/` directory with a sample scenario. Edit it or add your own.

```bash
memeval run --adapter mem0 --scenarios memeval_scenarios/
```

## Writing Scenarios

Scenarios are YAML files that define memory operations and assertions:

```yaml
name: "My Custom Test"
description: "Tests that user preferences are remembered"
dimensions_tested: [recall_accuracy, consistency]

setup:
  - write:
      key: "user_lang"
      content: "User's preferred language is Python"

steps:
  - assert_search:
      query: "What programming language does the user prefer?"
      expected_contains: ["Python"]
      min_results: 1

thresholds:
  recall_accuracy: 0.8
```

### Available step types

| Step | Description |
|------|-------------|
| `write` | Store a memory (key, content, metadata, memory_type) |
| `read` | Retrieve by key |
| `search` | Semantic search |
| `update` | Modify existing memory |
| `delete` | Remove a memory |
| `consolidate` | Merge multiple memories |
| `assert_read` | Read and verify content contains/not-contains expected values |
| `assert_search` | Search and verify results meet expectations |

### Assert options

For `assert_search`:
- `expected_contains` -- list of strings that must appear in results
- `expected_not_contains` -- list of strings that must NOT appear
- `min_results` / `max_results` -- result count bounds
- `expected_results_count` -- exact count
- `max_latency_ms` -- latency threshold
- `filters` -- user_id, session_id, memory_type filters

For `assert_read`:
- `expected_contains` -- string or list that must appear in the read content
- `expected_not_contains` -- string or list that must NOT appear

## Using with pytest

memeval registers as a pytest plugin automatically. Any `.yaml` file is auto-discovered:

```bash
pytest --memeval-adapter=mem0
pytest memeval_scenarios/ --memeval-adapter=in_memory
```

## Using in Python code

```python
import asyncio
from memeval import evaluate, InMemoryAdapter
from memeval.metrics import RecallAccuracyMetric

async def main():
    adapter = InMemoryAdapter()
    results = await evaluate(adapter=adapter, scenarios="builtin")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.scenario.name}")
        for name, mr in r.metric_results.items():
            print(f"  {name}: {mr.score:.3f}")

asyncio.run(main())
```

## CI/CD Integration

### JSON reports

```bash
memeval run --adapter mem0 --output report.json
```

### GitHub Actions

Copy `docs/memeval-template.yml` to your repo at `.github/workflows/memeval.yml` and configure your adapter and secrets.

## Evaluation Dimensions

memeval evaluates 8 dimensions:

| Dimension | What it measures |
|-----------|-----------------|
| `recall_accuracy` | Can stored memories be retrieved? |
| `relevance` | Are the right memories returned first? (MRR, NDCG) |
| `consistency` | Are there contradictions in stored facts? |
| `update_propagation` | Do corrections propagate correctly? |
| `forgetting_quality` | Is forgetting selective (good) or lossy (bad)? |
| `latency_cost` | Operation latency at p50/p95/p99 |
| `scalability` | Performance degradation at scale |
| `privacy_isolation` | Data isolation between users/sessions |

## Environment Variables

| Variable | Required for | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | Mem0 (self-hosted) | OpenAI API key for LLM + embeddings |
| `MEM0_API_KEY` | Mem0 (hosted) | Mem0 platform API key |
| `ZEP_API_KEY` | Zep | Zep Cloud API key |
| `LETTA_API_KEY` | Letta | Letta Cloud API key |
