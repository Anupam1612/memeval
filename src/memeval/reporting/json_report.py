"""JSON report generation for CI/CD integration."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memeval.scenarios.types import ScenarioResult


def generate_report(
    results: list[ScenarioResult],
    adapter_name: str,
) -> dict[str, Any]:
    """Generate a machine-readable JSON report."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    # Aggregate dimension scores
    dim_scores: dict[str, list[float]] = defaultdict(list)
    dim_thresholds: dict[str, float] = {}

    for result in results:
        for name, mr in result.metric_results.items():
            dim_scores[name].append(mr.score)
            dim_thresholds[name] = mr.threshold

    dimensions: dict[str, Any] = {}
    for name, scores in dim_scores.items():
        avg = sum(scores) / len(scores)
        dimensions[name] = {
            "score": round(avg, 4),
            "threshold": dim_thresholds[name],
            "passed": avg >= dim_thresholds[name],
            "num_scenarios": len(scores),
        }

    overall = (
        sum(sum(s) / len(s) for s in dim_scores.values()) / len(dim_scores)
        if dim_scores
        else 0.0
    )

    scenarios: list[dict[str, Any]] = []
    for result in results:
        scenario_entry: dict[str, Any] = {
            "name": result.scenario.name,
            "passed": result.passed,
            "metrics": {},
            "assertion_failures": len(result.assertion_failures),
        }
        for name, mr in result.metric_results.items():
            scenario_entry["metrics"][name] = {
                "score": round(mr.score, 4),
                "threshold": mr.threshold,
                "passed": mr.passed,
                "reason": mr.reason,
            }
        scenarios.append(scenario_entry)

    return {
        "memeval_version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "adapter": {"name": adapter_name},
        "summary": {
            "scenarios_run": total,
            "scenarios_passed": passed,
            "overall_score": round(overall, 4),
            "overall_passed": passed == total,
        },
        "dimensions": dimensions,
        "scenarios": scenarios,
    }
