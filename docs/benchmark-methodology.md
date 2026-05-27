# Benchmark Methodology

This document describes how memeval benchmarks are conducted, what they measure, and how to reproduce them.

## Principles

1. **Reproducible** -- every benchmark includes environment metadata (versions, platform, timestamp) and can be re-run with a single command.
2. **Statistical** -- for meaningful results, benchmarks should be run multiple times. Single-run results are labeled as such.
3. **Transparent** -- raw results are not committed to the repo. Instead, we provide the script and conditions so anyone can verify.

## Running Benchmarks

### Quick (single run, no API keys)

```bash
python scripts/run_benchmark.py --adapter in_memory
```

### Against real providers

```bash
# Requires OPENAI_API_KEY in .env or environment
python scripts/run_benchmark.py --adapter mem0

# Compare multiple adapters
python scripts/run_benchmark.py --adapter in_memory --adapter mem0
```

### Statistically significant (multiple runs)

```bash
python scripts/run_benchmark.py --adapter mem0 --runs 5 --output results/
```

This runs 5 independent evaluation passes and reports mean, standard deviation, min, and max for each dimension.

### Save results

```bash
python scripts/run_benchmark.py --adapter mem0 --runs 3 --output results/
# Creates results/benchmark_20260527_163000.json
```

## What Is Measured

Each benchmark run executes all built-in scenarios (currently 24) against the specified adapter and computes 7 evaluation dimensions:

| Dimension | What it measures | How |
|-----------|-----------------|-----|
| recall_accuracy | Can stored memories be retrieved? | Store facts, search for them, measure hit rate |
| relevance | Are the right memories returned first? | MRR and NDCG@k on ranked search results |
| consistency | Are stored facts free of contradictions? | Embedding similarity analysis on same-topic pairs |
| update_propagation | Do corrections propagate? | Store fact, update it, verify old value is gone |
| forgetting_quality | Is forgetting selective? | Delete specific items, verify others survive |
| latency_cost | Operation speed | p50/p95/p99 latency for read and write operations |
| privacy_isolation | Is data isolated between users? | Store sentinels for user A, search from user B |

## Conditions That Affect Results

When interpreting or comparing benchmark results, note the following variables:

### Adapter configuration
- **Mem0 (self-hosted):** Results depend on the LLM model used for fact extraction. Default config uses `gpt-4o-mini`. Using `gpt-4o` would produce different extraction quality and latency.
- **Mem0 (hosted):** Uses Mem0's platform models, which may differ from self-hosted.
- **Zep:** Graph processing is asynchronous. Results depend on indexing delay -- facts written and immediately searched may not be found.
- **Letta:** Results depend on the agent model and available credits.

### Environment
- **Network latency:** Hosted providers (Mem0 platform, Zep Cloud, Letta Cloud) are affected by network conditions.
- **Hardware:** Local embedding models (for consistency metric) run faster on machines with more CPU/RAM.
- **API rate limits:** Some providers throttle requests, which affects latency scores.

### Scenario set
- Results are tied to the specific scenario suite used. Adding or modifying scenarios changes the scores.
- The built-in suite is versioned with the package. Pin your memeval version for comparable results over time.

## Reading the Results

### Single run

```
Dimension              Score     Threshold    Status
recall_accuracy        0.923     0.800        PASS
consistency            0.917     0.900        PASS
```

A single run gives a point estimate. Use it for quick checks but not for claims about provider quality.

### Multiple runs

```
Dimension              Score     Std       Threshold    Status
recall_accuracy        0.923     0.012     0.800        PASS
consistency            0.917     0.045     0.900        PASS
```

The standard deviation shows run-to-run variance. High std (>0.05) means the score is unstable -- likely due to non-deterministic behavior in the memory provider (e.g., LLM-based extraction producing different facts each time).

### Comparing providers

When comparing providers, ensure:
1. Same memeval version
2. Same scenario set
3. Same number of runs
4. Same machine / network conditions (or note the difference)
5. Same consistency mode

## Methodology for Published Results

Any benchmark results published in blog posts, README, or documentation follow these standards:

1. **Minimum 3 runs** per adapter for published comparisons.
2. **Environment documented** -- memeval version, Python version, platform, date.
3. **Adapter config documented** -- which model, hosted vs self-hosted, any non-default settings.
4. **Raw data available** -- the `--output` JSON is referenced or linked.
5. **Single-run results labeled** -- if only 1 run, explicitly state "single run, not statistically significant."

## Reproducing README Results

The results shown in the README were produced with:

```bash
# Environment: memeval 0.1.1, Python 3.14, macOS ARM64
# Mem0: self-hosted with gpt-4o-mini, OPENAI_API_KEY required
# Date: 2026-05-27
# Note: single run, not statistically significant

python scripts/run_benchmark.py \
    --adapter in_memory \
    --adapter mem0 \
    --adapter zep \
    --adapter letta \
    --consistency-mode basic \
    --output results/
```

To reproduce: install memeval, set your API keys in `.env`, and run the command above. Your results will differ based on network conditions, API model versions, and non-deterministic LLM behavior.
