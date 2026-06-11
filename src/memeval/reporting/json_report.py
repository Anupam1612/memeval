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

    report: dict[str, Any] = {
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

    cost_section = _build_cost_section(results)
    if cost_section:
        report["cost"] = cost_section

    return report


def _build_cost_section(results: list[ScenarioResult]) -> dict[str, Any] | None:
    """Aggregate cost metric details across scenarios, if present."""
    total_cost = 0.0
    projected_monthly = 0.0
    tokens = {"llm_input_tokens": 0, "llm_output_tokens": 0, "embedding_tokens": 0}
    per_scenario: list[dict[str, Any]] = []
    source = None
    llm_model = None

    for result in results:
        mr = result.metric_results.get("cost")
        if mr is None:
            continue
        d = mr.details
        total_cost += d.get("total_cost_usd", 0.0)
        projected_monthly += d.get("projected_monthly_usd", 0.0)
        for k in tokens:
            tokens[k] += d.get("tokens", {}).get(k, 0)
        source = d.get("source", source)
        llm_model = d.get("llm_model", llm_model)
        per_scenario.append({
            "scenario": result.scenario.name,
            "cost_usd": d.get("total_cost_usd", 0.0),
            "cost_per_operation_usd": d.get("cost_per_operation_usd", 0.0),
        })

    if not per_scenario:
        return None

    return {
        "total_cost_usd": round(total_cost, 6),
        "projected_monthly_usd": round(projected_monthly, 2),
        "tokens": tokens,
        "source": source,
        "llm_model": llm_model,
        "per_scenario": per_scenario,
    }
