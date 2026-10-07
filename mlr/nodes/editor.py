"""Dedupe overlapping findings, merge their evidence, and calibrate confidence.

**Deterministic code. No model involved**, which is a deliberate choice worth defending.

Two or more checks legitimately reach the same conclusion about the same sentence: the substantiation
check notices a subgroup figure presented generally, and the tenant's promotional-code reviewer
notices the same thing against its own clause. Showing a reviewer two findings for one problem wastes
their attention and makes the report look padded.

The merge rule is **same claim and the same offending text**. That is narrow on purpose. A model could
be asked "do these two findings describe the same root issue?", and it would sometimes merge two
genuinely different problems that happen to quote the same sentence -- losing a finding, silently,
which is the one failure mode this system must not have. An exact-overlap rule under-merges instead:
the reviewer occasionally sees two related findings, which costs attention but never costs a problem.

Confidence calibration is also arithmetic rather than judgement:

* independent agreement raises confidence, capped, because two checks reaching the same conclusion by
  different routes is real corroboration;
* failing the grounding gate caps it low, because an unverifiable finding should not look certain.
"""

from __future__ import annotations

from typing import Any

from mlr.schemas import Citation, Finding
from platform_core.rules.deterministic import normalize_whitespace

# Confidence granted by independent corroboration, and the ceiling it may reach. Not tuned: these are
# starting values, and the eval harness can measure whether they help.
AGREEMENT_BONUS = 6
AGREEMENT_CEILING = 97
# An ungrounded finding cannot be presented as near-certain whatever the model claimed.
UNGROUNDED_CEILING = 45


def merge_key(finding: Finding) -> tuple[str, str]:
    """What makes two findings the same problem: the same offending text.

    Keyed on the quote alone, not on the quote plus the claim id. The quote identifies the text more
    precisely than a claim id does, and including the claim id made the merge depend on every producer
    remembering to fill that field -- which the rule checkers do not reliably do, because a rule is
    about the copy rather than about a claim. The result was a reference-document finding and a
    substantiation finding quoting the identical sentence and arriving as two separate items, which is
    exactly the duplication this node exists to remove.

    An absence has no quote, so it is keyed by what produced it: two checks both reporting a missing
    safety section are reporting one problem, but a missing disclaimer and a missing safety section are
    not.
    """
    if finding.is_absence:
        return ("absence", f"{finding.check_type}:{finding.rule_id}")
    return ("quote", normalize_whitespace(finding.quote).lower())


def _merge(group: list[Finding]) -> Finding:
    """Combine findings about the same problem, keeping every citation.

    The surviving finding is the most severe, then the most confident -- so a merge can never soften a
    blocker into an advisory. Citations from all of them are kept: a reviewer assessing a merged
    finding should see every independent line of evidence, which is the whole value of corroboration.
    """
    ordered = sorted(group, key=lambda f: (f.is_blocker, f.confidence), reverse=True)
    primary, others = ordered[0], ordered[1:]

    citations: list[Citation] = list(primary.citations)
    seen = {(c.source_id, normalize_whitespace(c.passage)) for c in citations}
    for other in others:
        for citation in other.citations:
            key = (citation.source_id, normalize_whitespace(citation.passage))
            if key not in seen:
                seen.add(key)
                citations.append(citation)

    # Corroboration only counts when it comes from a different check. Two findings from one check are
    # the same opinion twice, not two opinions.
    distinct_checks = {f.check_type for f in group}
    confidence = primary.confidence
    if len(distinct_checks) > 1:
        confidence = min(confidence + AGREEMENT_BONUS, AGREEMENT_CEILING)

    issue = primary.issue
    extra = [f.issue for f in others if f.issue and f.issue != primary.issue]
    if extra:
        issue = primary.issue + " Also noted: " + " ".join(extra)

    return primary.model_copy(
        update={
            "citations": citations,
            "confidence": confidence,
            "issue": issue,
            "merged_from": sorted({f.id for f in others}),
            # A merged finding keeps the strictest rule attribution it had, so the reviewer can still
            # see which tenant rule was involved.
            "rule_id": primary.rule_id or next((f.rule_id for f in others if f.rule_id), ""),
            "rule_version": primary.rule_version
            or next((f.rule_version for f in others if f.rule_id), 0),
        }
    )


def tidy(findings: list[Finding], state: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    """Produce the report's finding list: deduped, merged, calibrated.

    A part of the ``consolidate`` node rather than a node of its own."""
    if not findings:
        return [], {"in": 0, "out": 0}

    groups: dict[tuple[str, str], list[Finding]] = {}
    for finding in findings:
        groups.setdefault(merge_key(finding), []).append(finding)

    merged: list[Finding] = []
    for group in groups.values():
        finding = _merge(group) if len(group) > 1 else group[0]
        if not finding.grounded:
            finding = finding.model_copy(
                update={"confidence": min(finding.confidence, UNGROUNDED_CEILING)}
            )
        merged.append(finding)

    # Blockers first, then by confidence: the reviewer's attention should land on the most
    # serious finding, not on whichever check happened to finish first.
    merged.sort(key=lambda f: (f.is_blocker, f.confidence), reverse=True)

    summary = {
        "in": len(findings),
        "out": len(merged),
        "merged_groups": sum(1 for g in groups.values() if len(g) > 1),
        "blockers": sum(1 for f in merged if f.is_blocker),
        "confidence_capped_for_ungrounded": sum(1 for f in merged if not f.grounded),
    }
    return merged, summary
