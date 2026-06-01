# Contributing to memeval

Thanks for your interest in contributing. This document covers the process and guidelines.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/memeval.git
   cd memeval
   ```
3. Create a virtual environment and install dev dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
4. Verify everything works:
   ```bash
   ruff check src/ tests/
   mypy src/memeval/ --ignore-missing-imports
   pytest tests/ -v
   ```

## Making Changes

### Branch naming

Create a branch from `main` with a descriptive name:

```bash
git checkout -b fix/consistency-metric
git checkout -b feature/redis-adapter
git checkout -b docs/langgraph-guide
```

### Before submitting a PR

Every PR must pass these checks (they also run in CI):

```bash
# Lint
ruff check src/ tests/

# Type check
mypy src/memeval/ --ignore-missing-imports

# Tests
pytest tests/ -v

# Run the built-in scenarios (optional but recommended)
memeval run --adapter in_memory
```

### PR requirements

- **Branch protection is enabled on `main`**. All changes go through PRs.
- CI must pass (lint, type check, tests) before merging.
- Keep PRs focused. One feature or fix per PR.
- Write a clear PR description explaining what changed and why.

## What to Contribute

### New adapters

If you want to add support for a new memory provider:

1. Create `src/memeval/adapters/your_provider.py`
2. Implement the `MemoryProtocol` (7 core operations + 3 session operations)
3. Add it to the CLI factory in `src/memeval/cli.py`
4. Add optional dependency in `pyproject.toml`
5. See `docs/writing-adapters.md` for the full guide

### New scenarios

Add YAML files to `src/memeval/datasets/builtin/`:

```yaml
name: "Your Scenario Name"
description: "What this tests"
dimensions_tested: [recall_accuracy, consistency]

setup:
  - write:
      key: "fact_1"
      content: "Some fact to store"

steps:
  - assert_search:
      query: "search for the fact"
      expected_contains: ["fact"]

thresholds:
  recall_accuracy: 0.8
```

### New metrics

1. Create `src/memeval/metrics/your_metric.py`
2. Extend `BaseMetric` and implement `evaluate()`
3. Register in `src/memeval/metrics/__init__.py`

### Bug fixes

- Open an issue first describing the bug
- Reference the issue number in your PR

## Code Style

- **Linting**: ruff (configured in `pyproject.toml`)
- **Type checking**: mypy
- **Line length**: 100 characters
- **Python**: 3.10+
- No trailing whitespace, no unused imports
- Keep it simple. Don't add abstractions for one-time operations.

## Testing

- All new features need tests in `tests/`
- Run `pytest tests/ -v` before submitting
- If you add a new scenario, verify it works: `memeval run --adapter in_memory`

## Commit Messages

Write clear, concise commit messages:

```
Add Redis adapter with session support

- Maps to Redis hash + sorted set for memory storage
- Session support via Redis streams
- All SMP operations implemented
```

Not:

```
updated stuff
```

## Releases

Releases are handled by maintainers. The process:

1. Version bump in `pyproject.toml` and `src/memeval/__init__.py`
2. PR, CI pass, merge
3. Create GitHub release (triggers PyPI publish automatically)

## Questions?

Open an issue on GitHub. For larger changes, open an issue to discuss the approach before writing code.
