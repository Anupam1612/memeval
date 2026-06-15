"""Renders evaluation results into a self-contained HTML report.

No JS chart library, no CDN, no external fonts fetched at view time. The
report is a single HTML file that works offline and in air-gapped CI.

Usage:
    from memeval.reporting.html import render_html_report, render_benchmark_html

    html = render_html_report(results, adapter_name="mem0")
    Path("report.html").write_text(html)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memeval.scenarios.types import ScenarioResult, StepResult

_TEMPLATE_PATH = Path(__file__).parent / "template.html.j2"


def _get_version() -> str:
    from memeval import __version__

    return __version__


def _load_template() -> Any:
    from jinja2 import Environment, select_autoescape

    env = Environment(autoescape=select_autoescape(["html"]))
    return env.from_string(_TEMPLATE_PATH.read_text())


def render_html_report(
    results: list[ScenarioResult],
    adapter_name: str,
) -> str:
    """Render a single-adapter run into a self-contained HTML report."""
    context = _build_context(results, adapter_key="adapter", adapter_label=adapter_name)
    return _load_template().render(**context)


def render_benchmark_html(
    all_results: dict[str, list[ScenarioResult]],
) -> str:
    """Render a multi-adapter benchmark into a self-contained HTML report."""
    # Use the first adapter's results for the scenario detail section;
    # the comparison table covers the cross-adapter view.
    first_name = next(iter(all_results))
    context = _build_context(
        all_results[first_name],
        adapter_key="benchmark",
        adapter_label=" vs ".join(all_results),
    )
    context["comparison"] = _build_comparison(all_results)
    return _load_template().render(**context)


# -- context builders --


def _build_context(
    results: list[ScenarioResult], adapter_key: str, adapter_label: str
) -> dict[str, Any]:
    passed = sum(1 for r in results if r.passed)

    # Aggregate dimension scores
    dim_scores: dict[str, list[float]] = defaultdict(list)
    dim_thresholds: dict[str, float] = {}
    for r in results:
        for name, mr in r.metric_results.items():
            dim_scores[name].append(mr.score)
            dim_thresholds[name] = mr.threshold

    bars = []
    overall_scores = []
    for name in sorted(dim_scores):
        if name == "cost":
            continue  # cost has its own stat cards
        avg = sum(dim_scores[name]) / len(dim_scores[name])
        overall_scores.append(avg)
        threshold = dim_thresholds[name]
        value = max(0.0, min(1.0, avg))
        bars.append({
            "label": name,
            "value": avg,
            "pct": round(value * 100, 2),
            "tick_pct": round(threshold * 100, 2) if 0 < threshold <= 1 else None,
            "passed": avg >= threshold,
        })

    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    cost = _build_cost(results)

    return {
        "adapter_key": adapter_key,
        "adapter_label": adapter_label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version": _get_version(),
        "subtitle": _build_subtitle(adapter_key, adapter_label, len(results), len(bars), cost),
        "scenarios_total": len(results),
        "scenarios_passed": passed,
        "scenarios_failed": len(results) - passed,
        "overall_score": overall,
        "dimensions_count": len(bars),
        "bars": bars,
        "cost": cost,
        "scenarios": [_build_scenario(r) for r in results],
        "comparison": None,
    }


def _build_subtitle(
    adapter_key: str, adapter_label: str, n_scenarios: int, n_dims: int,
    cost: dict[str, Any] | None,
) -> str:
    if adapter_key == "benchmark":
        target = f"comparing {adapter_label}"
    else:
        target = f"executed against the {adapter_label} adapter"
    parts = [
        f"{n_scenarios} scenario{'s' if n_scenarios != 1 else ''}",
        f"{n_dims} quality dimension{'s' if n_dims != 1 else ''}",
    ]
    if cost:
        parts.append("token cost tracking")
    return f"Standardized memory test suite {target}. {', '.join(parts)}."


def _build_cost(results: list[ScenarioResult]) -> dict[str, Any] | None:
    total = 0.0
    projected = 0.0
    total_tokens = 0
    source = "estimated"
    ops_per_day = 10_000
    found = False

    for r in results:
        mr = r.metric_results.get("cost")
        if mr is None:
            continue
        found = True
        d = mr.details
        total += d.get("total_cost_usd", 0.0)
        projected += d.get("projected_monthly_usd", 0.0)
        total_tokens += sum(d.get("tokens", {}).values())
        source = d.get("source", source)
        ops_per_day = d.get("ops_per_day_assumption", ops_per_day)

    if not found:
        return None

    return {
        "total_usd": total,
        "projected_monthly_usd": projected,
        "total_tokens": total_tokens,
        "source": source,
        "ops_per_day": ops_per_day,
    }


def _build_comparison(
    all_results: dict[str, list[ScenarioResult]],
) -> dict[str, Any]:
    adapter_names = list(all_results)
    all_dims: set[str] = set()
    scores: dict[str, dict[str, float]] = {}

    for name, results in all_results.items():
        dim_scores: dict[str, list[float]] = defaultdict(list)
        for r in results:
            for dim, mr in r.metric_results.items():
                if dim == "cost":
                    continue
                dim_scores[dim].append(mr.score)
                all_dims.add(dim)
        scores[name] = {
            d: sum(v) / len(v) for d, v in dim_scores.items()
        }

    rows = []
    for dim in sorted(all_dims):
        values = [scores.get(n, {}).get(dim, 0.0) for n in adapter_names]
        best = adapter_names[values.index(max(values))] if values else ""
        rows.append({"dimension": dim, "scores": values, "best": best})

    return {"adapters": adapter_names, "rows": rows}


# -- scenario timeline --


def _build_scenario(result: ScenarioResult) -> dict[str, Any]:
    events = result.setup_results + result.step_results

    # Pad keys so previews line up in the monospace timeline.
    key_lens = [
        len(str(sr.data.get("key", "")))
        for sr in events
        if sr.step_type.value in ("write", "update", "delete")
    ]
    pad = max(key_lens, default=0)

    timeline: list[dict[str, Any]] = []
    for sr in events:
        timeline.extend(_event_lines(sr, pad))

    metrics = [
        {
            "name": name,
            "score": mr.score,
            "threshold": mr.threshold,
            "passed": mr.passed,
            "reason": mr.reason[:120],
        }
        for name, mr in result.metric_results.items()
    ]

    return {
        "name": result.scenario.name,
        "description": result.scenario.description,
        "passed": result.passed,
        "timeline": timeline,
        "metrics": metrics,
    }


def _truncate(text: str, limit: int = 70) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _line(segs: list[tuple[str, str]], indent: int = 0) -> dict[str, Any]:
    return {"segs": [{"t": t, "css": css} for t, css in segs], "indent": indent}


def _preview(sr: StepResult) -> str:
    preview = sr.data.get("content_preview", "")
    if sr.data.get("content_chars", 0) > len(preview):
        preview += "..."
    return preview


def _event_lines(sr: StepResult, pad: int) -> list[dict[str, Any]]:
    step = sr.step_type.value
    key = str(sr.data.get("key", ""))

    if step == "write":
        return [_line([
            ("WRITE", "amber"),
            (f"  {key.ljust(pad)} ", "bright"),
            (f'"{_preview(sr)}"', "dim"),
        ])]

    if step == "update":
        return [_line([
            ("UPDATE", "amber"),
            (f" {key.ljust(pad)} ", "bright"),
            (f'"{_preview(sr)}"', "dim"),
        ])]

    if step == "add_message":
        role = sr.data.get("role", "?")
        content = _truncate(sr.data.get("content", ""), 60)
        return [_line([
            ("MSG", "amber"),
            (f" [{role}] ", "bright"),
            (content, "dim"),
        ])]

    if step == "create_session":
        return [_line([
            ("SESSION", "amber"),
            (f" {sr.data.get('session_id', '')}", "bright"),
        ])]

    if step == "delete":
        status = "removed" if sr.data.get("deleted") else "not found"
        return [_line([
            ("DELETE", "amber"),
            (f" {key.ljust(pad)} ", "bright"),
            (f"-- {status}", "dim"),
        ])]

    if step == "consolidate":
        n = len(sr.data.get("source_keys", []))
        return [_line([
            ("CONSOLIDATE", "amber"),
            (f" {n} memories merged", "dim"),
        ])]

    if step in ("search", "assert_search", "assert_context"):
        query = _truncate(sr.data.get("query") or "", 55)
        count = sr.data.get("count", sr.data.get("facts_count", 0))
        label = "CONTEXT" if step == "assert_context" else "SEARCH"

        if sr.assertion_passed is False:
            lines = [_line([
                (f"{label} FAILED", "fail"),
                (f' "{query}"  ', "bright"),
                (f"-> {count} results", "dim"),
            ])]
            lines.extend(_failure_lines(sr))
            return lines
        return [_line([
            (label, "pass"),
            (f' "{query}"  ', "bright"),
            (f"-> {count} results", "dim"),
        ])]

    if step in ("read", "assert_read"):
        if sr.assertion_passed is False:
            lines = [_line([
                ("READ FAILED", "fail"),
                (f" {key}", "bright"),
            ])]
            lines.extend(_failure_lines(sr))
            return lines
        status = "found" if sr.data.get("found") else "not found"
        return [_line([
            ("READ", "pass"),
            (f" {key} ", "bright"),
            (f"-- {status}", "dim"),
        ])]

    return [_line([(step, "dim")])]


def _failure_lines(sr: StepResult) -> list[dict[str, Any]]:
    """Expected vs actual details for a failed assertion."""
    lines: list[dict[str, Any]] = []
    details = sr.assertion_details

    results = sr.data.get("results", [])
    facts = sr.data.get("facts", [])
    all_content = " ".join(
        [r.get("content", "") for r in results] + facts
    ).lower()

    for exp in details.get("expected_contains", []):
        if exp.lower() in all_content:
            lines.append(_line([(f'expected "{exp}" -- found', "pass")], indent=1))
        else:
            lines.append(_line([(f'expected "{exp}" -- NOT FOUND', "fail")], indent=1))

    for exp in details.get("expected_not_contains", []):
        if exp.lower() in all_content:
            lines.append(
                _line([(f'should NOT contain "{exp}" -- LEAKED', "fail")], indent=1)
            )
        else:
            lines.append(_line([(f'correctly excluded "{exp}"', "pass")], indent=1))

    shown = results[:5] if results else [{"content": f} for f in facts[:5]]
    if shown:
        lines.append(_line([("Retrieved:", "dim")], indent=1))
        for r in shown:
            content = _truncate(r.get("content", ""), 70)
            score = r.get("score")
            segs: list[tuple[str, str]] = [(f"{content}  ", "muted")]
            if isinstance(score, float):
                segs.append((f"score {score:.2f}", "amber"))
            lines.append(_line(segs, indent=2))

    return lines
