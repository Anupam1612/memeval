"""Scenario execution engine for memeval.

Runs setup steps, test steps, collects assertions, and evaluates metrics.
"""

from __future__ import annotations

import time
from typing import Any

from memeval.metrics.base import BaseMetric
from memeval.protocol.base import MemoryProtocol
from memeval.protocol.types import MemoryMetadata, MemoryType, SearchFilters
from memeval.scenarios.types import Scenario, ScenarioResult, StepResult, StepType


def _parse_memory_type(raw: str) -> MemoryType:
    try:
        return MemoryType(raw)
    except ValueError:
        return MemoryType.SEMANTIC


def _parse_metadata(raw: dict | None) -> MemoryMetadata | None:
    if not raw:
        return None
    return MemoryMetadata(
        source=raw.get("source"),
        user_id=raw.get("user_id"),
        session_id=raw.get("session_id"),
        agent_id=raw.get("agent_id"),
        tags=tuple(raw.get("tags", [])),
        extra={k: v for k, v in raw.items()
               if k not in ("source", "user_id", "session_id", "agent_id", "tags", "timestamp")},
    )


def _parse_filters(raw: dict | None) -> SearchFilters | None:
    if not raw:
        return None
    return SearchFilters(
        user_id=raw.get("user_id"),
        session_id=raw.get("session_id"),
        agent_id=raw.get("agent_id"),
        memory_type=MemoryType(raw["memory_type"]) if "memory_type" in raw else None,
    )


class ScenarioRunner:
    """Executes a scenario against a memory adapter and collects results."""

    async def run(
        self,
        scenario: Scenario,
        adapter: MemoryProtocol,
        metrics: list[BaseMetric] | None = None,
    ) -> ScenarioResult:
        """Run a complete scenario.

        Args:
            scenario: The scenario to execute.
            adapter: The memory backend to test.
            metrics: Optional list of metrics to compute. If None, uses
                     the scenario's dimensions_tested to select metrics.

        Returns:
            ScenarioResult with setup results, step results, and metric results.
        """
        # Reset adapter state for clean test
        await adapter.reset()

        # Execute setup steps
        setup_results = []
        for i, step in enumerate(scenario.setup):
            result = await self._execute_step(step, adapter, step_index=i)
            setup_results.append(result)

        # Execute test steps
        step_results = []
        for i, step in enumerate(scenario.steps):
            result = await self._execute_step(
                step, adapter, step_index=len(scenario.setup) + i
            )
            step_results.append(result)

        scenario_result = ScenarioResult(
            scenario=scenario,
            setup_results=setup_results,
            step_results=step_results,
        )

        # Run metrics
        if metrics:
            for metric in metrics:
                # Apply scenario-specific threshold override
                if metric.name in scenario.thresholds:
                    metric.threshold = scenario.thresholds[metric.name]
                mr = await metric.evaluate(adapter, scenario_result)
                scenario_result.metric_results[metric.name] = mr

        return scenario_result

    async def _execute_step(
        self,
        step: dict[str, Any],
        adapter: MemoryProtocol,
        step_index: int,
    ) -> StepResult:
        """Execute a single scenario step."""
        # Determine step type — the key in the dict
        step_type, step_params = self._parse_step(step)

        start = time.perf_counter()

        if step_type == StepType.WRITE:
            return await self._exec_write(step_params, adapter, step_index, start)
        elif step_type == StepType.READ:
            return await self._exec_read(step_params, adapter, step_index, start)
        elif step_type == StepType.SEARCH:
            return await self._exec_search(step_params, adapter, step_index, start)
        elif step_type == StepType.UPDATE:
            return await self._exec_update(step_params, adapter, step_index, start)
        elif step_type == StepType.DELETE:
            return await self._exec_delete(step_params, adapter, step_index, start)
        elif step_type == StepType.ASSERT_READ:
            return await self._exec_assert_read(step_params, adapter, step_index, start)
        elif step_type == StepType.ASSERT_SEARCH:
            return await self._exec_assert_search(step_params, adapter, step_index, start)
        else:
            elapsed = (time.perf_counter() - start) * 1000
            return StepResult(
                step_type=step_type,
                step_index=step_index,
                success=False,
                latency_ms=elapsed,
                data={"error": f"Unknown step type: {step_type}"},
            )

    def _parse_step(self, step: dict) -> tuple[StepType, dict]:
        """Extract step type and parameters from a YAML step dict."""
        for key in StepType:
            if key.value in step:
                return key, step[key.value]
        # Fallback: first key
        first_key = next(iter(step))
        try:
            return StepType(first_key), step[first_key]
        except ValueError:
            return StepType.WRITE, step.get(first_key, {})

    async def _exec_write(
        self, params: dict, adapter: MemoryProtocol, idx: int, start: float
    ) -> StepResult:
        result = await adapter.write(
            content=params["content"],
            key=params.get("key"),
            metadata=_parse_metadata(params.get("metadata")),
            memory_type=_parse_memory_type(params.get("memory_type", "semantic")),
        )
        elapsed = (time.perf_counter() - start) * 1000
        return StepResult(
            step_type=StepType.WRITE,
            step_index=idx,
            success=result.success,
            latency_ms=elapsed,
            data={"key": result.key, "latency_ms": result.latency_ms},
        )

    async def _exec_read(
        self, params: dict, adapter: MemoryProtocol, idx: int, start: float
    ) -> StepResult:
        entry = await adapter.read(params["key"])
        elapsed = (time.perf_counter() - start) * 1000
        return StepResult(
            step_type=StepType.READ,
            step_index=idx,
            success=entry is not None,
            latency_ms=elapsed,
            data={
                "key": params["key"],
                "content": entry.content if entry else None,
                "found": entry is not None,
            },
        )

    async def _exec_search(
        self, params: dict, adapter: MemoryProtocol, idx: int, start: float
    ) -> StepResult:
        results = await adapter.search(
            query=params["query"],
            limit=params.get("limit", 10),
            filters=_parse_filters(params.get("filters")),
        )
        elapsed = (time.perf_counter() - start) * 1000
        return StepResult(
            step_type=StepType.SEARCH,
            step_index=idx,
            success=True,
            latency_ms=elapsed,
            data={
                "query": params["query"],
                "results": [
                    {"content": r.entry.content, "key": r.entry.key, "score": r.score, "rank": r.rank}
                    for r in results
                ],
                "count": len(results),
            },
        )

    async def _exec_update(
        self, params: dict, adapter: MemoryProtocol, idx: int, start: float
    ) -> StepResult:
        result = await adapter.update(
            key=params["key"],
            content=params["content"],
            metadata=_parse_metadata(params.get("metadata")),
        )
        elapsed = (time.perf_counter() - start) * 1000
        return StepResult(
            step_type=StepType.UPDATE,
            step_index=idx,
            success=result.success,
            latency_ms=elapsed,
            data={"key": params["key"]},
        )

    async def _exec_delete(
        self, params: dict, adapter: MemoryProtocol, idx: int, start: float
    ) -> StepResult:
        deleted = await adapter.delete(params["key"])
        elapsed = (time.perf_counter() - start) * 1000
        return StepResult(
            step_type=StepType.DELETE,
            step_index=idx,
            success=deleted,
            latency_ms=elapsed,
            data={"key": params["key"], "deleted": deleted},
        )

    async def _exec_assert_read(
        self, params: dict, adapter: MemoryProtocol, idx: int, start: float
    ) -> StepResult:
        entry = await adapter.read(params["key"])
        elapsed = (time.perf_counter() - start) * 1000

        assertion_details: dict[str, Any] = {}
        passed = True

        content = entry.content if entry else ""

        if "expected_contains" in params:
            expected = params["expected_contains"]
            if isinstance(expected, str):
                expected = [expected]
            assertion_details["expected_contains"] = expected
            for exp in expected:
                if exp.lower() not in content.lower():
                    passed = False

        if "expected_not_contains" in params:
            not_expected = params["expected_not_contains"]
            if isinstance(not_expected, str):
                not_expected = [not_expected]
            assertion_details["expected_not_contains"] = not_expected
            for exp in not_expected:
                if exp.lower() in content.lower():
                    passed = False

        return StepResult(
            step_type=StepType.ASSERT_READ,
            step_index=idx,
            success=entry is not None,
            latency_ms=elapsed,
            assertion_passed=passed,
            assertion_details=assertion_details,
            data={"key": params["key"], "content": content, "found": entry is not None},
        )

    async def _exec_assert_search(
        self, params: dict, adapter: MemoryProtocol, idx: int, start: float
    ) -> StepResult:
        results = await adapter.search(
            query=params["query"],
            limit=params.get("limit", 10),
            filters=_parse_filters(params.get("filters")),
        )
        elapsed = (time.perf_counter() - start) * 1000

        result_contents = [r.entry.content for r in results]
        all_content = " ".join(result_contents).lower()
        assertion_details: dict[str, Any] = {}
        passed = True

        if "expected_contains" in params:
            expected = params["expected_contains"]
            if isinstance(expected, str):
                expected = [expected]
            assertion_details["expected_contains"] = expected
            for exp in expected:
                if exp.lower() not in all_content:
                    passed = False

        if "expected_not_contains" in params:
            not_expected = params["expected_not_contains"]
            if isinstance(not_expected, str):
                not_expected = [not_expected]
            assertion_details["expected_not_contains"] = not_expected
            for exp in not_expected:
                if exp.lower() in all_content:
                    passed = False

        if "min_results" in params:
            assertion_details["min_results"] = params["min_results"]
            if len(results) < params["min_results"]:
                passed = False

        if "max_results" in params:
            assertion_details["max_results"] = params["max_results"]
            if len(results) > params["max_results"]:
                passed = False

        if "expected_results_count" in params:
            assertion_details["expected_results_count"] = params["expected_results_count"]
            if len(results) != params["expected_results_count"]:
                passed = False

        if "max_latency_ms" in params:
            assertion_details["max_latency_ms"] = params["max_latency_ms"]
            if elapsed > params["max_latency_ms"]:
                passed = False

        return StepResult(
            step_type=StepType.ASSERT_SEARCH,
            step_index=idx,
            success=True,
            latency_ms=elapsed,
            assertion_passed=passed,
            assertion_details=assertion_details,
            data={
                "query": params["query"],
                "results": [
                    {"content": r.entry.content, "key": r.entry.key, "score": r.score, "rank": r.rank}
                    for r in results
                ],
                "count": len(results),
            },
        )
