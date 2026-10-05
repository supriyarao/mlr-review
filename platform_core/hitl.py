"""Human-in-the-loop: the escalation contract.

Why this exists: the system is not allowed to approve anything. Every run ends at a person, and
which person depends on what the system found and on the tenant's policy. This module owns the
shape of that handoff -- what a reviewer is shown, what they are asked, and what their answer must
contain to be recorded as accountable.

Deterministic code. No model involved: deciding *who* reviews is policy, and policy is data.

The design point worth defending: a reviewer is never shown a bare verdict. The interrupt payload
carries the reasons the run escalated and the evidence behind each finding, because a reviewer who
cannot see why the system is unsure cannot meaningfully take responsibility for overriding it. And
``ReviewDecision`` records a per-finding verdict, not just an overall one -- partly so the reviewer
can disagree with one finding while accepting the rest, and partly because those disagreements are
the training signal for improving the system (see the eval candidates flywheel).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["approve", "reject", "request_changes"]
FindingVerdict = Literal["agree", "disagree", "edit"]


class EscalationReason(BaseModel):
    """One reason a run needed a human above the standard tier.

    Kept as a structured object rather than a string so the UI can group reasons, the router can
    match them against tenant escalation rules, and the eval harness can measure which reason
    types drive false escalations.
    """

    model_config = ConfigDict(extra="forbid")

    # Matched against tenant escalation rules, e.g. "off_label", "ungrounded", "schema_failure".
    code: str
    # Written for the reviewer, not the developer.
    explanation: str
    # Ids of the findings or items this reason refers to.
    items: list[str] = Field(default_factory=list)


class ReviewTask(BaseModel):
    """A pending human review: what is being asked, of whom, and why.

    This is the payload handed to ``interrupt()``. It is deliberately self-contained -- a reviewer
    opening it should need nothing else to decide.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    tenant_id: str
    # Reviewer tier this task is addressed to, from tenant policy.
    tier: str
    # Why it came to this tier rather than the standard one. Empty for a routine review.
    reasons: list[EscalationReason] = Field(default_factory=list)
    # The report under review. Opaque at the platform layer: the use case decides its shape.
    report: dict[str, Any] = Field(default_factory=dict)
    # Supporting passages, so the reviewer can check a citation without leaving the screen.
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_escalation(self) -> bool:
        return bool(self.reasons)

    def reason_codes(self) -> list[str]:
        return [r.code for r in self.reasons]

    def headline(self) -> str:
        """One line for a queue listing."""
        if not self.reasons:
            return f"Routine review for {self.tenant_id}"
        codes = ", ".join(dict.fromkeys(self.reason_codes()))
        return f"Escalated to {self.tier}: {codes}"


class FindingReview(BaseModel):
    """A reviewer's verdict on one finding.

    A disagreement here is the most valuable data the system produces: it is a labelled example of
    the system being wrong, supplied by a qualified person, for free. Those become draft golden
    cases rather than being discarded.
    """

    model_config = ConfigDict(extra="forbid")

    finding_id: str
    verdict: FindingVerdict
    note: str = ""
    # Present when verdict == "edit": the reviewer's corrected text.
    edited_issue: str = ""
    edited_severity: str = ""


class ReviewDecision(BaseModel):
    """A reviewer's answer, resumed into the graph.

    ``reviewer`` and ``tier`` are required, not optional: an unattributed decision is not an
    accountable one, and accountability is the whole point of the human step.
    """

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    reviewer: str
    tier: str
    per_finding: list[FindingReview] = Field(default_factory=list)
    note: str = ""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def disagreements(self) -> list[FindingReview]:
        """Findings the reviewer rejected or corrected -- the eval-candidate signal."""
        return [f for f in self.per_finding if f.verdict in ("disagree", "edit")]

    def as_audit_output(self) -> dict[str, Any]:
        """Flattened form for the audit log, so a human step reads like any other event."""
        return {
            "decision": self.decision,
            "reviewer": self.reviewer,
            "tier": self.tier,
            "note": self.note,
            "per_finding": [f.model_dump(mode="json") for f in self.per_finding],
            "disagreement_count": len(self.disagreements()),
            "decided_at": self.decided_at.isoformat(),
        }


def build_review_task(
    *,
    run_id: str,
    tenant_id: str,
    tier: str,
    report: dict[str, Any],
    reasons: list[EscalationReason] | None = None,
    evidence: list[dict[str, Any]] | None = None,
) -> ReviewTask:
    """Assemble the interrupt payload for a human review."""
    return ReviewTask(
        run_id=run_id,
        tenant_id=tenant_id,
        tier=tier,
        reasons=reasons or [],
        report=report,
        evidence=evidence or [],
    )


def parse_decision(payload: Any) -> ReviewDecision:
    """Validate whatever came back from the resume call.

    A resume payload arrives from outside the graph -- an HTTP request from the UI -- so it is
    validated rather than trusted. An invalid decision must fail loudly: the alternative is a run
    that records an approval nobody actually gave.
    """
    if isinstance(payload, ReviewDecision):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"review decision must be a mapping, got {type(payload).__name__}")
    return ReviewDecision.model_validate(payload)
