"""Rich console output for memeval — scorecards and comparative tables."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from memeval.scenarios.types import ScenarioResult


def print_scorecard(
    results: list[ScenarioResult],
    adapter_name: str,
    console: Console | None = None,
) -> None:
    """Print a scorecard summarizing all scenario results."""
    console = console or Console()

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    # Aggregate metric scores across scenarios
    metric_scores: dict[str, list[float]] = defaultdict(list)
    metric_thresholds: dict[str, float] = {}

    for result in results:
        for name, mr in result.metric_results.items():
            metric_scores[name].append(mr.score)
            metric_thresholds[name] = mr.threshold

    table = Table(
        title=f"MEMEVAL SCORECARD — {adapter_name}",
        caption=f"{passed}/{total} scenarios passed",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Dimension", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status", justify="center")

    overall_scores: list[float] = []

    for name in sorted(metric_scores.keys()):
        scores = metric_scores[name]
        avg = sum(scores) / len(scores)
        threshold = metric_thresholds[name]
        overall_scores.append(avg)

        if name == "latency_cost":
            # Show p95 latency in the score column for readability
            status = "[green]PASS[/green]" if avg >= threshold else "[red]FAIL[/red]"
            table.add_row(name, f"{avg:.3f}", f"{threshold:.3f}", status)
        else:
            status = "[green]PASS[/green]" if avg >= threshold else "[red]FAIL[/red]"
            table.add_row(name, f"{avg:.3f}", f"{threshold:.3f}", status)

    # Overall row
    if overall_scores:
        overall = sum(overall_scores) / len(overall_scores)
        overall_status = "[green]PASS[/green]" if passed == total else "[red]FAIL[/red]"
        table.add_section()
        table.add_row(
            "[bold]OVERALL[/bold]",
            f"[bold]{overall:.3f}[/bold]",
            "",
            overall_status,
        )

    console.print()
    console.print(table)
    console.print()


def print_comparative(
    all_results: dict[str, list[ScenarioResult]],
    console: Console | None = None,
) -> None:
    """Print a comparative benchmark table across adapters."""
    console = console or Console()
    adapter_names = list(all_results.keys())

    # Collect all dimension names
    all_dimensions: set[str] = set()
    adapter_dim_scores: dict[str, dict[str, float]] = {}

    for adapter_name, results in all_results.items():
        dim_scores: dict[str, list[float]] = defaultdict(list)
        for result in results:
            for name, mr in result.metric_results.items():
                dim_scores[name].append(mr.score)
                all_dimensions.add(name)
        adapter_dim_scores[adapter_name] = {
            name: sum(scores) / len(scores) for name, scores in dim_scores.items()
        }

    table = Table(
        title="MEMEVAL COMPARATIVE BENCHMARK",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Dimension", style="bold")
    for name in adapter_names:
        table.add_column(name, justify="right")
    table.add_column("Best", justify="center", style="bold green")

    for dim in sorted(all_dimensions):
        row = [dim]
        scores = {}
        for adapter_name in adapter_names:
            score = adapter_dim_scores.get(adapter_name, {}).get(dim, 0.0)
            scores[adapter_name] = score
            row.append(f"{score:.3f}")

        if scores:
            best = max(scores, key=lambda k: scores[k])
            row.append(best)
        else:
            row.append("-")

        table.add_row(*row)

    # Overall row
    table.add_section()
    overall_row = ["[bold]OVERALL[/bold]"]
    overall_scores: dict[str, float] = {}
    for adapter_name in adapter_names:
        dims = adapter_dim_scores.get(adapter_name, {})
        avg = sum(dims.values()) / len(dims) if dims else 0.0
        overall_scores[adapter_name] = avg
        overall_row.append(f"[bold]{avg:.3f}[/bold]")

    if overall_scores:
        best_overall = max(overall_scores, key=lambda k: overall_scores[k])
        overall_row.append(f"[bold]{best_overall}[/bold]")
    else:
        overall_row.append("-")

    table.add_row(*overall_row)

    console.print()
    console.print(table)
    console.print()
