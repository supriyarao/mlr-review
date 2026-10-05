"""MLR-specific eval metrics and the matching rules behind them.

The generic machinery -- recall, precision, percentiles, per-group breakdown -- lives in
``platform_core/evals/metrics.py``. What is domain-specific is *which* number is the headline, and
why:

**Blocker recall is the headline metric.** The two errors this system can make are not symmetric. A
missed blocker means non-compliant material goes out; a false escalation means a reviewer spends ten
minutes confirming something was fine. One is a regulatory event, the other is a cost. So blocker
recall is the number the regression gate guards, and false escalation rate is reported beside it as
the price being paid -- never traded against it silently.

**False escalation rate is the honest counterweight.** A system that escalates everything scores
perfectly on recall and is worthless, because reviewers stop reading escalations that are usually
noise. Reporting both is what makes the confidence-threshold sweep meaningful rather than a way to
look good.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from platform_core.evals.metrics import precision, recall
from platform_core.evals.schema import CaseResult, ExpectedFinding, FindingMatch, GoldenCase

# Routes that count as "a human above the standard reviewer looked at this".
ESCALATION_TIERS_ARE_ANYTHING_BUT = "standard_reviewer"


def finding_matches(expected: ExpectedFinding, finding: dict[str, Any], case: GoldenCase) -> bool:
    """Does an actual finding satisfy an expected one?

    The rule, stated in the spec and implemented here: same ``check_type``, and the claim text
    contains ``claim_contains`` case-insensitively. Severity is compared **separately** so a severity
    disagreement is reported as a policy-calibration problem rather than counted as a miss -- those
    need different fixes, and conflating them hides which one you have.

    ``claim_contains`` is matched against the finding's quote *and* its issue text. A check can
    legitimately quote a shorter span than the case author had in mind while describing the same
    problem, and failing that would measure transcription rather than detection.
    """
    if finding.get("check_type") != expected.check_type:
        return False

    needle = expected.claim_contains.lower().strip()
    if not needle:
        return True

    haystack = " ".join(
        str(finding.get(field) or "") for field in ("quote", "issue", "suggested_fix")
    ).lower()
    return needle in haystack


def match_findings(case: GoldenCase, findings: Sequence[dict[str, Any]]) -> list[FindingMatch]:
    """Pair each expected finding with an actual one, at most one each way.

    Greedy, highest-confidence first. One actual finding cannot satisfy two expectations: that would
    let a single vague finding claim credit for several distinct problems.
    """
    remaining = sorted(findings, key=lambda f: -int(f.get("confidence") or 0))
    claimed: set[int] = set()
    matches: list[FindingMatch] = []

    for expected in case.expected_findings:
        hit: dict[str, Any] | None = None
        hit_index = -1
        for index, finding in enumerate(remaining):
            if index in claimed:
                continue
            if finding_matches(expected, finding, case):
                hit, hit_index = finding, index
                break

        if hit is None:
            matches.append(FindingMatch(expected=expected, matched=False))
            continue

        claimed.add(hit_index)
        actual_severity = str(hit.get("severity") or hit.get("severity_final") or "")
        matches.append(
            FindingMatch(
                expected=expected,
                matched=True,
                severity_mismatch=bool(expected.severity and actual_severity != expected.severity),
                actual_severity=actual_severity,
                matched_finding_id=str(hit.get("id") or ""),
            )
        )

    return matches


# A finding whose quote is a fragment of a forbidden sentence only counts as flagging it when the
# quote is a substantial part of it. Without this, the trademark rule -- whose quote is the single word
# "Zelvoran" -- was reported as a false alarm against every must_not_flag sentence containing the brand
# name, which is most of them. A one-word quote is not a finding *about* a sentence.
MIN_FRAGMENT_CHARS = 30
MIN_FRAGMENT_RATIO = 0.5


def false_alarms(case: GoldenCase, findings: Sequence[dict[str, Any]]) -> list[str]:
    """``must_not_flag`` entries that were flagged anyway.

    Unambiguous precision failures, unlike a merely unexpected finding: the case author asserted that
    flagging this text would be wrong.

    A finding flags a forbidden entry if it quotes that text, or quotes a substantial fragment of it.
    Matching on any shared substring in either direction was too loose and produced false positives
    from short quotes -- which would have made the metric that keeps precision honest the least honest
    number in the set.
    """
    flagged = []
    for forbidden in case.must_not_flag:
        needle = forbidden.lower().strip()
        if not needle:
            continue
        for finding in findings:
            quote = str(finding.get("quote") or "").lower().strip()
            if not quote:
                continue
            if needle in quote:
                flagged.append(forbidden)
                break
            substantial = len(quote) >= max(
                MIN_FRAGMENT_CHARS, int(MIN_FRAGMENT_RATIO * len(needle))
            )
            if substantial and quote in needle:
                flagged.append(forbidden)
                break
    return flagged


def unexpected(case: GoldenCase, findings: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Findings no expectation accounts for.

    Not automatically wrong -- a real reviewer might agree with one, and a golden set never enumerates
    every defensible observation. Reported as a signal to read next to the false-alarm count, which
    *is* unambiguous.
    """
    matched_ids = {
        m.matched_finding_id for m in match_findings(case, findings) if m.matched_finding_id
    }
    return [
        {
            "id": f.get("id"),
            "check_type": f.get("check_type"),
            "severity": f.get("severity"),
            "quote": str(f.get("quote") or "")[:120],
            "issue": str(f.get("issue") or "")[:160],
        }
        for f in findings
        if f.get("id") not in matched_ids
    ]


def expected_rules_fired(case: GoldenCase, findings: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Did the tenant rules the case expects actually fire?

    Checked separately from ``expected_findings`` because "the right rule fired" and "the finding
    reads correctly" are different failures with different fixes.
    """
    fired = {str(f.get("rule_id")) for f in findings if f.get("rule_id")}
    missing = [r for r in case.expected_rule_ids if r not in fired]
    return {
        "expected": list(case.expected_rule_ids),
        "fired": sorted(fired),
        "missing": missing,
        "all_fired": not missing,
    }


def blocker_recall(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """The headline metric. A missed blocker is the expensive error."""
    expected = matched = 0
    missed: list[dict[str, str]] = []
    for case in cases:
        for m in case.matches:
            if m.expected.severity != "blocker":
                continue
            expected += 1
            if m.matched:
                matched += 1
            else:
                missed.append(
                    {
                        "case_id": case.case_id,
                        "check_type": m.expected.check_type,
                        "claim_contains": m.expected.claim_contains,
                    }
                )
    return {
        "expected": expected,
        "matched": matched,
        "recall": round(recall(matched, expected), 4),
        # Named, not just counted: a missed blocker is the thing to go and look at.
        "missed": missed,
    }


def advisory_recall(cases: Sequence[CaseResult]) -> dict[str, Any]:
    expected = matched = 0
    for case in cases:
        for m in case.matches:
            if m.expected.severity == "advisory":
                expected += 1
                matched += int(m.matched)
    return {
        "expected": expected,
        "matched": matched,
        "recall": round(recall(matched, expected), 4),
    }


def false_escalation_rate(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """How often a piece that should have gone to a standard reviewer was escalated.

    The speed-versus-safety counterweight to blocker recall. Reported alongside it always: a system
    that escalates everything has perfect recall and is useless, because reviewers learn to skim
    escalations that are usually noise.
    """
    standard_expected = [c for c in cases if c.expected_route == ESCALATION_TIERS_ARE_ANYTHING_BUT]
    escalated = [
        c
        for c in standard_expected
        if c.actual_route and c.actual_route != ESCALATION_TIERS_ARE_ANYTHING_BUT
    ]
    return {
        "standard_expected": len(standard_expected),
        "escalated": len(escalated),
        "rate": round(precision(len(escalated), len(standard_expected)), 4)
        if standard_expected
        else 0.0,
        "cases": [{"case_id": c.case_id, "routed_to": c.actual_route} for c in escalated],
    }


def grounding_rate(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """What proportion of findings passed the grounding gate, and by which mode.

    Broken down by mode because the three are different strengths of claim: evidence in a tenant
    source is a stronger basis than a rule's own wording. One aggregate number would mix them.
    """
    findings = [f for c in cases for f in c.findings]
    by_mode: dict[str, int] = {}
    for finding in findings:
        by_mode[str(finding.get("grounded_by") or "none")] = (
            by_mode.get(str(finding.get("grounded_by") or "none"), 0) + 1
        )
    grounded = sum(1 for f in findings if f.get("grounded"))
    return {
        "findings": len(findings),
        "grounded": grounded,
        "rate": round(precision(grounded, len(findings)), 4),
        "by_mode": dict(sorted(by_mode.items())),
    }


def rule_finding_coverage(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """How many findings came from tenant business rules, and from which.

    Worth tracking separately from the core checks: rule findings are the part a compliance owner
    controls without engineering, so a rule that stops firing after an edit is their problem to see
    rather than something buried in an aggregate.
    """
    rule_findings = [f for c in cases for f in c.findings if f.get("rule_id")]
    by_rule: dict[str, int] = {}
    for finding in rule_findings:
        rule_id = str(finding["rule_id"])
        by_rule[rule_id] = by_rule.get(rule_id, 0) + 1
    return {
        "rule_findings": len(rule_findings),
        "distinct_rules": len(by_rule),
        "by_rule": dict(sorted(by_rule.items())),
    }


def severity_calibration(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Findings detected correctly but graded at the wrong severity.

    A policy-calibration problem, not a detection problem, and the fix is a tenant's severity_rules
    rather than a prompt. Conflating the two hides which one you have.
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


def mlr_metrics(cases: Sequence[CaseResult]) -> dict[str, Any]:
    """Every MLR-specific metric. Passed to the harness as its ``metrics_hook``."""
    return {
        # Headline first: this is the number the regression gate guards.
        "blocker_recall": blocker_recall(cases),
        "false_escalation_rate": false_escalation_rate(cases),
        "advisory_recall": advisory_recall(cases),
        "grounding_rate": grounding_rate(cases),
        "severity_calibration": severity_calibration(cases),
        "rule_findings": rule_finding_coverage(cases),
    }


# The metric the regression gate guards, as a (group, key) path for
# ``platform_core.evals.harness.compare_to_baseline``.
GUARD_METRICS: list[tuple[str, str]] = [("blocker_recall", "recall")]
