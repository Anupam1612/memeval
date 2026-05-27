"""Terminal-based memory failure visualizer using Rich.

Shows a conversation timeline with:
- What was stored (writes, messages)
- What was retrieved (searches, context queries)
- Where things went wrong (assertion failures, contradictions)
- Operation latency and cost

Usage:
    from memeval.visualizer import TerminalVisualizer

    viz = TerminalVisualizer()
    viz.show_scenario(scenario_result)
    viz.show_failures(all_results)
    viz.show_timeline(scenario_result)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

if TYPE_CHECKING:
    from memeval.scenarios.types import ScenarioResult, StepResult


class TerminalVisualizer:
    """Rich-based terminal visualizer for memory evaluation results."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def show_scenario(self, result: ScenarioResult) -> None:
        """Show a detailed visualization of a single scenario run."""
        passed = result.passed
        status = "PASSED" if passed else "FAILED"
        style = "green" if passed else "red"

        # Header
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{result.scenario.name}[/bold]\n"
                f"{result.scenario.description}",
                title=f"[{style}]{status}[/{style}]",
                border_style=style,
            )
        )

        # Timeline
        self._show_timeline(result)

        # Metrics
        if result.metric_results:
            self._show_metrics(result)

        self.console.print()

    def show_failures(self, results: list[ScenarioResult]) -> None:
        """Show only the failed scenarios with failure details."""
        failures = [r for r in results if not r.passed]

        if not failures:
            self.console.print("\n[green]All scenarios passed.[/green]\n")
            return

        self.console.print(
            f"\n[bold red]{len(failures)} scenario(s) failed:[/bold red]\n"
        )

        for result in failures:
            self.show_scenario(result)

    def show_summary(self, results: list[ScenarioResult]) -> None:
        """Show a compact summary of all results."""
        self.console.print()

        table = Table(
            title="Scenario Results",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Status", justify="center", width=6)
        table.add_column("Scenario", min_width=30)
        table.add_column("Failures", justify="right")
        table.add_column("Details", max_width=50)

        for result in results:
            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"

            failure_count = len(result.assertion_failures) + len(result.metric_failures)
            failure_str = str(failure_count) if failure_count > 0 else ""

            # Build details string
            details_parts: list[str] = []
            for mf in result.metric_failures:
                details_parts.append(f"{mf.metric_name}: {mf.score:.3f} < {mf.threshold}")
            for af in result.assertion_failures:
                step_type = af.step_type.value
                details_parts.append(f"step {af.step_index} ({step_type})")

            details = "; ".join(details_parts[:3])
            if len(details_parts) > 3:
                details += f" +{len(details_parts) - 3} more"

            table.add_row(status, result.scenario.name, failure_str, details)

        self.console.print(table)
        self.console.print()

    def _show_timeline(self, result: ScenarioResult) -> None:
        """Show the step-by-step timeline of a scenario execution."""
        tree = Tree("[bold]Timeline[/bold]")

        # Setup steps
        if result.setup_results:
            setup_branch = tree.add("[dim]Setup[/dim]")
            for sr in result.setup_results:
                self._add_step_to_tree(setup_branch, sr, is_setup=True)

        # Test steps
        steps_branch = tree.add("[bold]Steps[/bold]")
        for sr in result.step_results:
            self._add_step_to_tree(steps_branch, sr, is_setup=False)

        self.console.print(tree)

    def _add_step_to_tree(
        self, branch: Tree, sr: StepResult, is_setup: bool
    ) -> None:
        """Add a single step to the timeline tree."""
        step_type = sr.step_type.value
        latency = f"{sr.latency_ms:.0f}ms" if sr.latency_ms > 1 else "<1ms"

        if step_type == "write":
            content = sr.data.get("key", "")
            label = f"[cyan]WRITE[/cyan] {content} ({latency})"

        elif step_type == "add_message":
            role = sr.data.get("role", "?")
            content = sr.data.get("content", "")
            truncated = content[:60] + "..." if len(content) > 60 else content
            label = f"[cyan]MSG[/cyan] [{role}] {truncated} ({latency})"

        elif step_type == "create_session":
            sid = sr.data.get("session_id", "?")
            label = f"[cyan]SESSION[/cyan] {sid}"

        elif step_type == "delete":
            key = sr.data.get("key", "?")
            deleted = sr.data.get("deleted", False)
            status = "removed" if deleted else "not found"
            label = f"[yellow]DELETE[/yellow] {key} -- {status}"

        elif step_type == "consolidate":
            keys = sr.data.get("source_keys", [])
            label = f"[yellow]CONSOLIDATE[/yellow] {len(keys)} memories merged"

        elif step_type in ("search", "assert_search"):
            query = sr.data.get("query", "?")
            count = sr.data.get("count", 0)
            truncated_q = query[:50] + "..." if len(query) > 50 else query

            if sr.assertion_passed is False:
                label = f"[red]SEARCH FAILED[/red] \"{truncated_q}\" -> {count} results"
                node = branch.add(label)
                self._add_failure_details(node, sr)
                return
            else:
                label = f"[green]SEARCH[/green] \"{truncated_q}\" -> {count} results ({latency})"

        elif step_type in ("read", "assert_read"):
            key = sr.data.get("key", "?")
            found = sr.data.get("found", False)

            if sr.assertion_passed is False:
                label = f"[red]READ FAILED[/red] {key}"
                node = branch.add(label)
                self._add_failure_details(node, sr)
                return
            else:
                status_str = "found" if found else "not found"
                label = f"[green]READ[/green] {key} -- {status_str}"

        elif step_type == "assert_context":
            query = sr.data.get("query") or sr.assertion_details.get("query", "")
            facts_count = sr.data.get("facts_count", 0)

            if sr.assertion_passed is False:
                truncated_q = query[:50] + "..." if len(query) > 50 else query
                label = f"[red]CONTEXT FAILED[/red] \"{truncated_q}\" -> {facts_count} facts"
                node = branch.add(label)
                self._add_failure_details(node, sr)
                return
            else:
                truncated_q = query[:50] + "..." if len(query) > 50 else query
                label = (
                    f"[green]CONTEXT[/green] \"{truncated_q}\""
                    f" -> {facts_count} facts ({latency})"
                )

        elif step_type == "update":
            key = sr.data.get("key", "?")
            label = f"[cyan]UPDATE[/cyan] {key} ({latency})"

        else:
            label = f"[dim]{step_type}[/dim] ({latency})"

        branch.add(label)

    def _add_failure_details(self, node: Tree, sr: StepResult) -> None:
        """Add failure details to a tree node."""
        details = sr.assertion_details

        if "expected_contains" in details:
            for exp in details["expected_contains"]:
                # Check if this specific expectation was met
                results = sr.data.get("results", [])
                facts = sr.data.get("facts", [])
                all_content = " ".join(
                    [r.get("content", "") for r in results] + facts
                ).lower()

                if exp.lower() in all_content:
                    node.add(f"[green]expected \"{exp}\" -- found[/green]")
                else:
                    node.add(f"[red]expected \"{exp}\" -- NOT FOUND[/red]")

        if "expected_not_contains" in details:
            for exp in details["expected_not_contains"]:
                results = sr.data.get("results", [])
                facts = sr.data.get("facts", [])
                all_content = " ".join(
                    [r.get("content", "") for r in results] + facts
                ).lower()

                if exp.lower() in all_content:
                    node.add(f"[red]should NOT contain \"{exp}\" -- LEAKED[/red]")
                else:
                    node.add(f"[green]correctly excluded \"{exp}\"[/green]")

        # Show what was actually retrieved
        results = sr.data.get("results", [])
        if results:
            retrieved_node = node.add("[dim]Retrieved:[/dim]")
            for r in results[:5]:
                content = r.get("content", "?")
                score = r.get("score", 0)
                truncated = content[:70] + "..." if len(content) > 70 else content
                retrieved_node.add(f"[dim]{truncated} (score: {score:.2f})[/dim]")

        facts = sr.data.get("facts", [])
        if facts and not results:
            facts_node = node.add("[dim]Facts in context:[/dim]")
            for f in facts[:5]:
                truncated = f[:70] + "..." if len(f) > 70 else f
                facts_node.add(f"[dim]{truncated}[/dim]")

    def _show_metrics(self, result: ScenarioResult) -> None:
        """Show metric results for a scenario."""
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Score", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Detail", max_width=40)

        for name, mr in result.metric_results.items():
            status = "[green]PASS[/green]" if mr.passed else "[red]FAIL[/red]"
            detail = mr.reason[:40] if mr.reason else ""
            table.add_row(name, f"{mr.score:.3f}", f"{mr.threshold}", status, detail)

        self.console.print(table)
