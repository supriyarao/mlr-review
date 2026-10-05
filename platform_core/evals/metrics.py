"""Generic eval metrics.

Why these live in the platform layer: recall, precision, routing accuracy and latency percentiles
mean the same thing for any use case. What is *domain-specific* is which class matters most --
for MLR that is blocker recall, because a missed blocker is the expensive error -- so the headline
metric is chosen by the use case (``mlr/eval_metrics.py``) and computed from these primitives.

Deterministic code.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from platform_core.evals.schema import CaseResult


def recall(matched: int, expected: int) -> float:
    """Of the things that should have been found, how many were.

    An empty expectation set returns 1.0: a case with nothing to find cannot fail on recall, and
    returning 0.0 would drag the aggregate down for cases that passed perfectly.
    """
    return 1.0 if expected == 0 else matched / expected


def precision(matched: int, produced: int) -> float:
    """Of the things that were found, how many should have been."""
    return 1.0 if produced == 0 else matched / produced


def f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile by nearest rank. Explicit rather than numpy, so the arithmetic is readable."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = min(int(round(q / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return float(ordered[index])


def recall_by_class(cases: Sequence[CaseResult], severity: str = "") -> dict[str, float | int]:
    """Recall over expected findings, optionally restricted to one severity.

    This is the primitive the MLR layer calls with ``severity="blocker"`` to get its headline
    number, and with ``severity="advisory"`` for the secondary one.
    """
    expected = matched = 0
    for case in cases:
        for m in case.matches:
            if severity and m.expected.severity != severity:
                continue
            expected += 1
            matched += int(m.matched)
    return {"expected": expected, "matched": matched, "recall": round(recall(matched, expected), 4)}


def finding_precision(cases: Sequence[CaseResult]) -> dict[str, float | int]:
    """How many produced findings were accounted for by an expectation.

    Trade-off: an "unexpected" finding is not automatically wrong -- a real reviewer might agree
    with it, and a golden set never enumerates every defensible observation. So this is reported
    as a signal to read alongside the false-alarm count (which *is* unambiguous, because
    ``must_not_flag`` is an explicit assertion), not as a verdict on its own.
    """
    produced = sum(len(c.findings) for c in cases)
    unexpected = sum(len(c.unexpected_findings) for c in cases)
    accounted = produced - unexpected
    return {
        "produced": produced,
        "accounted_for": accounted,
        "unexpected": unexpected,
        "precision": round(precision(accounted, produced), 4),
    }


def routing_accuracy(cases: Sequence[CaseResult]) -> dict[str, float | int]:
    """Did the run land on the reviewer tier the case expected."""
    scored = [c for c in cases if c.expected_route]
    correct = sum(1 for c in scored if c.route_correct)
    return {
        "scored": len(scored),
        "correct": correct,
        "accuracy": round(precision(correct, len(scored)), 4),
    }


def severity_mismatches(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Findings detected correctly but graded at the wrong severity.

    Reported separately from misses because the fix is different: a mismatch means the tenant's
    severity rules need calibrating, a miss means the check did not fire at all.
    """
    rows = [
        {
            "case_id": c.case_id,
            "check_type": m.expected.check_type,
            "expected": m.expected.severity,
            "actual": m.actual_severity,
        }
        for c in cases
        for m in c.matches
        if m.matched and m.severity_mismatch
    ]
    return {"count": len(rows), "rows": rows}


def false_alarms(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """must_not_flag entries that were flagged anyway. Unambiguous precision failures."""
    rows = [{"case_id": c.case_id, "flagged": item} for c in cases for item in c.false_alarms]
    return {"count": len(rows), "rows": rows}


def cost_and_latency(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Per-run cost and latency distribution.

    p95 as well as p50 because a review pipeline that is usually fast and occasionally very slow
    is a different product from one that is uniformly medium, and the mean hides that.
    """
    latencies = [float(c.latency_ms) for c in cases if c.latency_ms]
    costs = [c.cost_usd for c in cases]
    return {
        "runs": len(cases),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "latency_mean_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "tokens_in_total": sum(c.tokens_in for c in cases),
        "tokens_out_total": sum(c.tokens_out for c in cases),
        "cost_usd_total": round(sum(costs), 6),
        "cost_usd_per_run": round(statistics.fmean(costs), 6) if costs else 0.0,
    }


def pass_rate(cases: Sequence[CaseResult]) -> dict[str, float | int]:
    passed = sum(1 for c in cases if c.passed)
    return {
        "cases": len(cases),
        "passed": passed,
        "pass_rate": round(precision(passed, len(cases)), 4),
    }


def generic_metrics(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Every use-case-agnostic metric, in one dict."""
    return {
        "pass_rate": pass_rate(cases),
        "recall_all": recall_by_class(cases),
        "finding_precision": finding_precision(cases),
        "routing": routing_accuracy(cases),
        "severity_mismatches": severity_mismatches(cases),
        "false_alarms": false_alarms(cases),
        "cost_and_latency": cost_and_latency(cases),
    }


def mean_and_std(values: Sequence[float]) -> dict[str, float]:
    """Mean and standard deviation, for ``--repeats N``.

    Models are nondeterministic, so a single run of an eval set reports a point estimate with
    unknown variance. Reporting the spread is what makes "recall went from 0.85 to 0.80" an
    answerable question rather than a coin flip read as a regression.
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def per_group(cases: Sequence[CaseResult], key: str = "tenant_id") -> dict[str, dict[str, Any]]:
    """Metrics broken down by a case attribute -- in practice, per tenant.

    An aggregate can hide that the system works for one tenant's policy and not another's, which
    is precisely the failure a multi-tenant platform needs to catch.
    """
    groups: dict[str, list[CaseResult]] = {}
    for case in cases:
        groups.setdefault(str(getattr(case, key, "")), []).append(case)
    return {name: generic_metrics(members) for name, members in sorted(groups.items())}
