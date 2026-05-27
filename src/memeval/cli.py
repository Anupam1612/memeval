"""CLI for memeval — run evaluations and benchmarks from the command line.

Usage:
    memeval run --adapter in_memory
    memeval run --adapter mem0 --scenarios ./my_scenarios/
    memeval benchmark --adapters mem0 --adapters zep --adapters letta
    memeval init
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

# Load .env file if present (for API keys)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from memeval.reporting.console import print_comparative, print_scorecard


@click.group()
@click.version_option(version="0.1.0", prog_name="memeval")
def app() -> None:
    """memeval — Evaluation framework for agent memory systems."""


@app.command()
@click.option(
    "--adapter",
    default="in_memory",
    help="Adapter: in_memory, mem0, zep, letta",
    show_default=True,
)
@click.option(
    "--scenarios",
    default="builtin",
    help="Path to scenarios dir, or 'builtin' for built-in suite",
    show_default=True,
)
@click.option("--output", default=None, help="Output JSON report path")
@click.option("--verbose", is_flag=True, help="Show detailed per-step results")
def run(adapter: str, scenarios: str, output: str | None, verbose: bool) -> None:
    """Run memory evaluation scenarios."""
    asyncio.run(_run_eval(adapter, scenarios, output, verbose))


async def _run_eval(
    adapter_name: str, scenarios_path: str, output: str | None, verbose: bool
) -> None:
    from rich.console import Console

    from memeval.metrics import METRIC_REGISTRY
    from memeval.reporting.json_report import generate_report
    from memeval.scenarios.loader import load_builtin_scenarios, load_scenarios_from_dir
    from memeval.scenarios.runner import ScenarioRunner

    console = Console()

    # Load adapter
    adapter = _create_adapter(adapter_name)
    console.print(f"[bold]Adapter:[/bold] {adapter_name}")

    # Load scenarios
    if scenarios_path == "builtin":
        scenario_list = load_builtin_scenarios()
        console.print(f"[bold]Scenarios:[/bold] {len(scenario_list)} built-in")
    else:
        scenario_list = load_scenarios_from_dir(scenarios_path)
        console.print(f"[bold]Scenarios:[/bold] {len(scenario_list)} from {scenarios_path}")

    if not scenario_list:
        console.print("[yellow]No scenarios found.[/yellow]")
        sys.exit(1)

    # Run each scenario
    runner = ScenarioRunner()
    all_results = []

    for scenario in scenario_list:
        metrics = []
        for dim in scenario.dimensions_tested:
            metric_cls = METRIC_REGISTRY.get(dim)
            if metric_cls:
                threshold = scenario.thresholds.get(dim, 0.5)
                metrics.append(metric_cls(threshold=threshold))

        result = await runner.run(scenario, adapter, metrics)
        all_results.append(result)

        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        console.print(f"  {status} {scenario.name}")

        if verbose and not result.passed:
            for f in result.assertion_failures:
                console.print(f"    [red]Assertion failed at step {f.step_index}[/red]")
            for mf in result.metric_failures:
                console.print(
                    f"    [red]{mf.metric_name}: {mf.score:.3f} < {mf.threshold:.3f}[/red]"
                )

    # Print scorecard
    console.print()
    print_scorecard(all_results, adapter_name, console)

    # Output JSON report
    if output:
        report = generate_report(all_results, adapter_name)
        Path(output).write_text(json.dumps(report, indent=2, default=str))
        console.print(f"\n[bold]Report saved:[/bold] {output}")

    # Exit with error if any failed
    if not all(r.passed for r in all_results):
        sys.exit(1)


@app.command()
@click.option(
    "--adapters",
    multiple=True,
    required=True,
    help="Adapters to compare (use multiple times)",
)
@click.option("--scenarios", default="builtin", show_default=True)
@click.option("--output", default="benchmark_results.json")
def benchmark(adapters: tuple[str, ...], scenarios: str, output: str) -> None:
    """Run comparative benchmark across memory providers."""
    asyncio.run(_run_benchmark(adapters, scenarios, output))


async def _run_benchmark(
    adapter_names: tuple[str, ...], scenarios_path: str, output: str
) -> None:
    from rich.console import Console

    from memeval.metrics import METRIC_REGISTRY
    from memeval.reporting.json_report import generate_report
    from memeval.scenarios.loader import load_builtin_scenarios, load_scenarios_from_dir
    from memeval.scenarios.runner import ScenarioRunner

    console = Console()

    # Load scenarios
    if scenarios_path == "builtin":
        scenario_list = load_builtin_scenarios()
    else:
        scenario_list = load_scenarios_from_dir(scenarios_path)

    n_scenarios = len(scenario_list)
    n_adapters = len(adapter_names)
    console.print(f"[bold]Benchmark:[/bold] {n_scenarios} scenarios x {n_adapters} adapters\n")

    all_adapter_results: dict[str, list] = {}

    for adapter_name in adapter_names:
        console.print(f"[bold]Running: {adapter_name}[/bold]")
        adapter = _create_adapter(adapter_name)
        runner = ScenarioRunner()
        results = []

        for scenario in scenario_list:
            metrics = []
            for dim in scenario.dimensions_tested:
                metric_cls = METRIC_REGISTRY.get(dim)
                if metric_cls:
                    threshold = scenario.thresholds.get(dim, 0.5)
                    metrics.append(metric_cls(threshold=threshold))

            result = await runner.run(scenario, adapter, metrics)
            results.append(result)

            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            console.print(f"  {status} {scenario.name}")

        all_adapter_results[adapter_name] = results
        console.print()

    # Print comparative table
    print_comparative(all_adapter_results, console)

    # Save benchmark report
    benchmark_report = {}
    for name, results in all_adapter_results.items():
        benchmark_report[name] = generate_report(results, name)

    Path(output).write_text(json.dumps(benchmark_report, indent=2, default=str))
    console.print(f"\n[bold]Benchmark report saved:[/bold] {output}")


@app.command()
def init() -> None:
    """Initialize memeval in the current project."""
    from rich.console import Console

    console = Console()

    scenarios_dir = Path("memeval_scenarios")
    scenarios_dir.mkdir(exist_ok=True)

    sample = scenarios_dir / "sample_recall.yaml"
    if not sample.exists():
        sample.write_text(
            """name: "Sample Recall Test"
description: "A sample scenario — customize for your use case"
version: "1.0"
memory_types_tested: [semantic]
dimensions_tested: [recall_accuracy]

setup:
  - write:
      key: "test_fact"
      content: "The user's favorite programming language is Python"
      memory_type: semantic

steps:
  - assert_search:
      query: "What is the user's favorite programming language?"
      expected_contains: ["Python"]
      min_results: 1

thresholds:
  recall_accuracy: 0.8
"""
        )

    console.print("[green]memeval initialized![/green]")
    console.print(f"  Created: {scenarios_dir}/")
    console.print(f"  Sample scenario: {sample}")
    console.print()
    console.print("Next steps:")
    console.print("  1. Edit or add YAML scenarios in memeval_scenarios/")
    console.print("  2. Run: memeval run --adapter in_memory --scenarios memeval_scenarios/")
    console.print("  3. Or via pytest: pytest memeval_scenarios/ --memeval-adapter=in_memory")


@app.command()
@click.option(
    "--adapter",
    default="in_memory",
    help="Adapter: in_memory, mem0, zep, letta",
    show_default=True,
)
@click.option(
    "--scenarios",
    default="builtin",
    help="Path to scenarios dir, or 'builtin'",
    show_default=True,
)
@click.option(
    "--failures-only",
    is_flag=True,
    help="Only show failed scenarios",
)
def diagnose(adapter: str, scenarios: str, failures_only: bool) -> None:
    """Visualize memory failures with detailed timelines.

    Shows what was stored, what was retrieved, and where things went wrong.
    """
    asyncio.run(_run_diagnose(adapter, scenarios, failures_only))


async def _run_diagnose(
    adapter_name: str, scenarios_path: str, failures_only: bool
) -> None:

    from rich.console import Console

    from memeval.metrics import METRIC_REGISTRY
    from memeval.scenarios.loader import load_builtin_scenarios, load_scenarios_from_dir
    from memeval.scenarios.runner import ScenarioRunner
    from memeval.visualizer import TerminalVisualizer

    console = Console(stderr=True)
    viz = TerminalVisualizer(console)

    adapter = _create_adapter(adapter_name)
    console.print(f"[bold]Diagnosing:[/bold] {adapter_name}")

    if scenarios_path == "builtin":
        scenario_list = load_builtin_scenarios()
    else:
        scenario_list = load_scenarios_from_dir(scenarios_path)

    runner = ScenarioRunner()
    all_results = []

    for scenario in scenario_list:
        metrics = []
        for dim in scenario.dimensions_tested:
            metric_cls = METRIC_REGISTRY.get(dim)
            if metric_cls:
                threshold = scenario.thresholds.get(dim, 0.5)
                metrics.append(metric_cls(threshold=threshold))

        result = await runner.run(scenario, adapter, metrics)
        all_results.append(result)

    if failures_only:
        viz.show_failures(all_results)
    else:
        viz.show_summary(all_results)
        viz.show_failures(all_results)

    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    console.print(f"[bold]{passed}/{total} scenarios passed[/bold]")


def _create_adapter(name: str):  # type: ignore[return]
    """Factory for memory adapters."""
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
        raise click.BadParameter(
            f"Unknown adapter: {name}. Available: in_memory, mem0, zep, letta"
        )
