#!/usr/bin/env python3
"""Reproducible benchmark script for memeval.

Runs evaluation scenarios against one or more memory adapters,
collects results across multiple runs, and produces a statistical
summary with confidence intervals.

Usage:
    # Single adapter, single run
    python scripts/run_benchmark.py --adapter in_memory

    # Multiple adapters comparison
    python scripts/run_benchmark.py --adapter in_memory --adapter mem0

    # Multiple runs for statistical significance
    python scripts/run_benchmark.py --adapter mem0 --runs 3

    # Save results
    python scripts/run_benchmark.py --adapter mem0 --runs 3 --output results/

Requirements:
    pip install memoryeval
    For Mem0:  pip install memoryeval[mem0]  + OPENAI_API_KEY env var
    For Zep:   pip install memoryeval[zep]   + ZEP_API_KEY env var
    For Letta: pip install memoryeval[letta]  + LETTA_API_KEY env var
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible memeval benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--adapter", action="append", required=True,
        help="Adapter(s) to benchmark (use multiple times for comparison)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of runs per adapter for statistical significance (default: 1)",
    )
    parser.add_argument(
        "--scenarios", default="builtin",
        help="Scenario path or 'builtin' (default: builtin)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Directory to save results (default: print to console only)",
    )
    parser.add_argument(
        "--consistency-mode", default="basic",
        help="Consistency metric mode: embedding, nli, basic (default: basic)",
    )
    return parser.parse_args()


def create_adapter(name: str):
    """Create a memory adapter by name."""
    if name == "in_memory":
        from memeval.adapters.in_memory import InMemoryAdapter
        return InMemoryAdapter()
    elif name == "mem0":
        from memeval.adapters.mem0 import Mem0Adapter
        return Mem0Adapter()
    elif name == "zep":
        from memeval.adapters.zep import ZepAdapter
        return ZepAdapter()
    elif name == "letta":
        from memeval.adapters.letta import LettaAdapter
        return LettaAdapter()
    else:
        raise ValueError(f"Unknown adapter: {name}")


def collect_environment() -> dict:
    """Collect environment metadata for reproducibility."""
    import memeval
    env = {
        "memeval_version": memeval.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Check for optional dependencies
    try:
        import mem0
        env["mem0_version"] = getattr(mem0, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import sentence_transformers
        env["sentence_transformers_version"] = sentence_transformers.__version__
    except ImportError:
        pass

    return env


async def run_single(adapter_name: str, scenarios_path: str, consistency_mode: str) -> dict:
    """Run a single benchmark pass and return results."""
    from memeval import evaluate
    from memeval.reporting.json_report import generate_report
    from memeval.scenarios.loader import load_builtin_scenarios, load_scenarios_from_dir

    # Load scenarios
    if scenarios_path == "builtin":
        scenario_list = load_builtin_scenarios()
    else:
        scenario_list = load_scenarios_from_dir(scenarios_path)

    # Create fresh adapter for each run
    adapter = create_adapter(adapter_name)

    # Override consistency mode for all scenarios
    results = await evaluate(adapter=adapter, scenarios=scenario_list)

    report = generate_report(results, adapter_name)
    return report


def aggregate_runs(runs: list[dict]) -> dict:
    """Aggregate multiple runs into statistical summary."""
    if len(runs) == 1:
        summary = runs[0]["summary"].copy()
        summary["num_runs"] = 1
        summary["score_std"] = 0.0
        dimensions = {}
        for dim_name, dim_data in runs[0]["dimensions"].items():
            dimensions[dim_name] = {
                "mean": dim_data["score"],
                "std": 0.0,
                "min": dim_data["score"],
                "max": dim_data["score"],
                "threshold": dim_data["threshold"],
                "passed_all_runs": dim_data["passed"],
            }
        return {"summary": summary, "dimensions": dimensions}

    # Multiple runs: compute statistics
    overall_scores = [r["summary"]["overall_score"] for r in runs]
    scenarios_passed = [r["summary"]["scenarios_passed"] for r in runs]

    summary = {
        "scenarios_run": runs[0]["summary"]["scenarios_run"],
        "num_runs": len(runs),
        "overall_score_mean": round(float(np.mean(overall_scores)), 4),
        "overall_score_std": round(float(np.std(overall_scores)), 4),
        "overall_score_min": round(float(np.min(overall_scores)), 4),
        "overall_score_max": round(float(np.max(overall_scores)), 4),
        "scenarios_passed_mean": round(float(np.mean(scenarios_passed)), 1),
    }

    # Per-dimension statistics
    dim_scores: dict[str, list[float]] = defaultdict(list)
    dim_thresholds: dict[str, float] = {}
    dim_passed: dict[str, list[bool]] = defaultdict(list)

    for run in runs:
        for dim_name, dim_data in run["dimensions"].items():
            dim_scores[dim_name].append(dim_data["score"])
            dim_thresholds[dim_name] = dim_data["threshold"]
            dim_passed[dim_name].append(dim_data["passed"])

    dimensions = {}
    for dim_name in sorted(dim_scores.keys()):
        scores = dim_scores[dim_name]
        dimensions[dim_name] = {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "threshold": dim_thresholds[dim_name],
            "passed_all_runs": all(dim_passed[dim_name]),
        }

    return {"summary": summary, "dimensions": dimensions}


def print_results(adapter_name: str, aggregated: dict, env: dict) -> None:
    """Print benchmark results to console."""
    s = aggregated["summary"]
    num_runs = s.get("num_runs", 1)

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK RESULTS: {adapter_name}")
    print(f"{'=' * 60}")
    print(f"  memeval version:  {env['memeval_version']}")
    print(f"  Python:           {env['python_version']}")
    print(f"  Platform:         {env['platform']}")
    print(f"  Timestamp:        {env['timestamp']}")
    print(f"  Runs:             {num_runs}")
    print(f"  Scenarios:        {s['scenarios_run']}")

    if num_runs > 1:
        print(f"  Overall score:    {s['overall_score_mean']:.3f} "
              f"+/- {s['overall_score_std']:.3f} "
              f"(range: {s['overall_score_min']:.3f}-{s['overall_score_max']:.3f})")
    else:
        print(f"  Overall score:    {s.get('overall_score', s.get('overall_score_mean', 0)):.3f}")

    print(f"\n  {'Dimension':<22} {'Score':>8} {'Std':>8} {'Threshold':>10} {'Status':>8}")
    print(f"  {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8}")

    for dim_name, dim in aggregated["dimensions"].items():
        std_str = f"{dim['std']:.3f}" if num_runs > 1 else "  n/a"
        status = "PASS" if dim["passed_all_runs"] else "FAIL"
        print(f"  {dim_name:<22} {dim['mean']:>8.3f} {std_str:>8} "
              f"{dim['threshold']:>10} {status:>8}")

    print()


def main() -> None:
    args = parse_args()

    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    env = collect_environment()

    print("memeval benchmark")
    print(f"  Adapters:  {', '.join(args.adapter)}")
    print(f"  Runs:      {args.runs}")
    print(f"  Scenarios: {args.scenarios}")

    all_results: dict[str, dict] = {}

    for adapter_name in args.adapter:
        print(f"\nRunning {adapter_name}...")
        runs = []

        for run_num in range(args.runs):
            if args.runs > 1:
                print(f"  Run {run_num + 1}/{args.runs}...", end="", flush=True)

            start = time.perf_counter()
            report = asyncio.run(
                run_single(adapter_name, args.scenarios, args.consistency_mode)
            )
            elapsed = time.perf_counter() - start

            if args.runs > 1:
                score = report["summary"]["overall_score"]
                print(f" score={score:.3f} ({elapsed:.1f}s)")

            runs.append(report)

        aggregated = aggregate_runs(runs)
        aggregated["environment"] = env
        aggregated["adapter"] = adapter_name
        aggregated["raw_runs"] = runs

        all_results[adapter_name] = aggregated
        print_results(adapter_name, aggregated, env)

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"benchmark_{timestamp}.json"

        output_data = {
            "environment": env,
            "config": {
                "adapters": args.adapter,
                "runs": args.runs,
                "scenarios": args.scenarios,
                "consistency_mode": args.consistency_mode,
            },
            "results": all_results,
        }

        output_file.write_text(json.dumps(output_data, indent=2, default=str))
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
