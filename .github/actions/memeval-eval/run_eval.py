"""Runner script for the memeval GitHub Action.

Executes evaluation scenarios, writes JSON report, sets GitHub Action outputs,
and generates a step summary for the Actions UI.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="memeval GitHub Action runner")
    parser.add_argument("--adapter", default="in_memory")
    parser.add_argument("--scenarios", default="builtin")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--report", default="memeval-report.json")
    parser.add_argument("--github-output", default=None)
    parser.add_argument("--github-step-summary", default=None)
    return parser.parse_args()


async def run_evaluation(args):
    from memeval import evaluate
    from memeval.reporting.json_report import generate_report

    # Create adapter
    adapter = _create_adapter(args.adapter)

    # Run evaluation
    results = await evaluate(adapter=adapter, scenarios=args.scenarios)

    # Generate report
    report = generate_report(results, args.adapter)

    # Write JSON report
    Path(args.report).write_text(json.dumps(report, indent=2, default=str))

    return report, results


def _create_adapter(name):
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


def write_github_output(path, report, threshold):
    """Write outputs for subsequent GitHub Action steps."""
    if not path:
        return

    summary = report["summary"]
    overall_passed = summary["overall_passed"] and summary["overall_score"] >= threshold

    with open(path, "a") as f:
        f.write(f"overall-score={summary['overall_score']:.4f}\n")
        f.write(f"passed={'true' if overall_passed else 'false'}\n")
        f.write(f"scenarios-passed={summary['scenarios_passed']}\n")
        f.write(f"scenarios-total={summary['scenarios_run']}\n")


def write_step_summary(path, report, adapter_name, threshold):
    """Write a markdown summary for the GitHub Actions UI."""
    if not path:
        return

    summary = report["summary"]
    overall_passed = summary["overall_passed"] and summary["overall_score"] >= threshold
    status = "PASSED" if overall_passed else "FAILED"

    lines = [
        f"## memeval: {status}",
        "",
        f"**Adapter:** `{adapter_name}` | "
        f"**Overall score:** {summary['overall_score']:.3f} | "
        f"**Threshold:** {threshold} | "
        f"**Scenarios:** {summary['scenarios_passed']}/{summary['scenarios_run']} passed",
        "",
        "### Dimensions",
        "",
        "| Dimension | Score | Threshold | Status |",
        "|-----------|-------|-----------|--------|",
    ]

    for dim_name, dim in report["dimensions"].items():
        dim_status = "PASS" if dim["passed"] else "FAIL"
        lines.append(f"| {dim_name} | {dim['score']:.3f} | {dim['threshold']} | {dim_status} |")

    lines.append("")
    lines.append("### Scenarios")
    lines.append("")

    for sc in report["scenarios"]:
        sc_status = "PASS" if sc["passed"] else "FAIL"
        lines.append(f"- [{sc_status}] **{sc['name']}**")
        for m_name, m_val in sc["metrics"].items():
            m_detail = ""
            if not m_val["passed"]:
                m_detail = f" -- below {m_val['threshold']}"
            lines.append(f"  - {m_name}: {m_val['score']:.3f}{m_detail}")

    with open(path, "a") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    print(f"memeval: adapter={args.adapter}, scenarios={args.scenarios}, threshold={args.threshold}")

    report, results = asyncio.run(run_evaluation(args))

    summary = report["summary"]
    overall_passed = summary["overall_passed"] and summary["overall_score"] >= args.threshold

    # Print results to console
    print(f"\nResults: {summary['scenarios_passed']}/{summary['scenarios_run']} scenarios passed")
    print(f"Overall score: {summary['overall_score']:.3f} (threshold: {args.threshold})")
    print(f"Status: {'PASSED' if overall_passed else 'FAILED'}")
    print(f"Report: {args.report}")

    for sc in report["scenarios"]:
        status = "PASS" if sc["passed"] else "FAIL"
        print(f"  [{status}] {sc['name']}")

    # Write GitHub outputs
    write_github_output(args.github_output, report, args.threshold)
    write_step_summary(args.github_step_summary, report, args.adapter, args.threshold)

    # Exit with failure if threshold not met
    if not overall_passed:
        print(f"\nFAILED: overall score {summary['overall_score']:.3f} below threshold {args.threshold}")
        sys.exit(1)


if __name__ == "__main__":
    main()
