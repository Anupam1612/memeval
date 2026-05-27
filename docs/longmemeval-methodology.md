# LongMemEval Integration Methodology

## About LongMemEval

LongMemEval (Wu et al., ICLR 2025) is a benchmark for evaluating long-term memory in AI chat assistants. It contains 500 question-answer pairs derived from multi-turn conversation histories, testing 5 memory abilities.

- **Paper**: https://arxiv.org/abs/2410.10813
- **Dataset**: https://huggingface.co/datasets/xiaowu0162/longmemeval
- **Original code**: https://github.com/xiaowu0162/LongMemEval

## How we use it

### What the original paper tests

The paper evaluates end-to-end QA: given a conversation history, an LLM generates a natural language answer, and GPT-4o judges correctness. The system under test is the full pipeline (retrieval + generation).

### What memeval tests

We test **retrieval only**: given a conversation history stored via `add_message()`, can the memory adapter retrieve the relevant facts when queried with `get_session_context()`?

This is a deliberate choice. memeval evaluates memory systems, not LLMs. The question is: "Did the memory system surface the right information?" not "Did the LLM generate the right answer?"

### Key differences from the original evaluation

| Aspect | Original paper | memeval |
|--------|---------------|---------|
| What's tested | Full pipeline (retrieval + LLM generation) | Memory retrieval only |
| Input format | Conversation history loaded into LLM context | Conversation fed via `add_message()` API |
| Scoring | GPT-4o judges generated answer | LLM judges retrieved context (or embedding similarity) |
| Question being answered | "Can the system answer correctly?" | "Did the memory surface the right facts?" |
| Cost | High (LLM generates full answers) | Lower (only judges retrieval) |

### Why this matters

The paper shows that even with perfect retrieval (oracle mode), there's a 10-point generation gap. By testing retrieval separately, we isolate memory quality from LLM generation quality. A memory system that retrieves the right facts but pairs with a weak LLM will score poorly in the paper's methodology but correctly in ours.

## Scoring methods

### LLM-as-judge (--scoring llm)

Matches the paper's approach most closely. An LLM (Claude or GPT) evaluates whether the retrieved context contains enough information to answer the question correctly. This does not require exact string matching.

```bash
memeval longmemeval --adapter mem0 --scoring llm --limit 50
```

Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY.

### Embedding similarity (--scoring embedding, default)

Computes cosine similarity between the expected answer and the retrieved context using sentence-transformers. Threshold: 0.45 (calibrated against LongMemEval answer lengths).

```bash
memeval longmemeval --adapter mem0 --scoring embedding --limit 50
```

Requires: pip install sentence-transformers

### Keyword matching (--scoring keyword)

Checks if significant words from the expected answer appear in the retrieved context. Least reliable. No extra dependencies.

```bash
memeval longmemeval --adapter in_memory --scoring keyword --limit 50
```

## Reference baselines

From the paper (Table 2, LongMemEval_S):

| System | Overall | IE | MR | KU | TR |
|--------|---------|----|----|----|----|
| GPT-4o (oracle context) | 87.0% | 97.0% | 87.0% | 83.0% | 65.0% |
| GPT-4o (long-context) | 60.6% | 81.0% | 65.0% | 54.0% | 31.0% |
| ChatGPT (with memory) | 57.7% | -- | -- | -- | -- |
| Llama 3.1 70B | 33.4% | -- | -- | -- | -- |

IE = Information Extraction, MR = Multi-Session Reasoning,
KU = Knowledge Updates, TR = Temporal Reasoning.

Note: these numbers test end-to-end QA (retrieval + generation). Our retrieval-only scores are not directly comparable but provide a useful signal for how well the memory system surfaces relevant information.

## Running the benchmark

```bash
# Quick test (10 samples, keyword scoring, no API keys needed)
memeval longmemeval --adapter in_memory --limit 10 --scoring keyword

# Standard run (50 samples, embedding scoring)
memeval longmemeval --adapter mem0 --limit 50 --scoring embedding

# Full benchmark (all 500 samples, LLM scoring)
memeval longmemeval --adapter mem0 --limit 500 --scoring llm --verbose

# Filter by question type
memeval longmemeval --adapter mem0 --types temporal-reasoning,multi-session
```

## Interpreting results

- **Keyword scoring** will undercount hits because it misses paraphrases and equivalent expressions. Use it for quick sanity checks only.
- **Embedding scoring** is the recommended default. It captures semantic equivalence without requiring an API key for the judge.
- **LLM scoring** is the most accurate and matches the paper's methodology. Use it for publishable results.
- **InMemoryAdapter** is expected to score low (30-50%) because it uses substring matching, not semantic retrieval. This is the baseline floor.
- **Mem0** should score significantly higher due to LLM-powered fact extraction and vector search.
