"""The human-in-the-loop contract: what a reviewer is shown, and what counts as an answer."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from platform_core.hitl import (
    EscalationReason,
    FindingReview,
    ReviewDecision,
    ReviewTask,
    build_review_task,
    parse_decision,
)


def test_a_routine_review_is_still_a_human_review():
    """No findings does not mean no reviewer. The system never approves anything itself."""
    task = build_review_task(
        run_id="r1",
        tenant_id="acme_pharma",
        tier="standard_reviewer",
        report={"findings": [], "summary": "no issues found"},
    )
    assert task.is_escalation is False
    assert task.tier == "standard_reviewer"
    assert task.headline() == "Routine review for acme_pharma"


def test_an_escalation_tells_the_reviewer_why_it_reached_them():
    """A reviewer who cannot see why the system is unsure cannot take responsibility for it."""
    task = build_review_task(
        run_id="r1",
        tenant_id="borealis_bio",
        tier="medical_director",
        report={"findings": [{"id": "f1"}]},
        reasons=[
            EscalationReason(
                code="off_label",
                explanation="Copy implies use in children; the label covers adults only.",
                items=["f1"],
            )
        ],
    )
    assert task.is_escalation is True
    assert task.reason_codes() == ["off_label"]
    assert "Escalated to medical_director" in task.headline()
    assert "label covers adults only" in task.reasons[0].explanation


def test_the_payload_carries_evidence_so_a_citation_can_be_checked_in_place():
    task = build_review_task(
        run_id="r1",
        tenant_id="acme_pharma",
        tier="senior_regulatory",
        report={},
        evidence=[{"source_id": "label-1", "passage": "in adults"}],
    )
    assert task.evidence[0]["source_id"] == "label-1"


def test_multiple_reasons_are_all_preserved():
    task = build_review_task(
        run_id="r1",
        tenant_id="t",
        tier="legal",
        report={},
        reasons=[
            EscalationReason(code="comparative_claim", explanation="superiority claim"),
            EscalationReason(code="ungrounded", explanation="no verifiable citation"),
        ],
    )
    assert task.reason_codes() == ["comparative_claim", "ungrounded"]
    assert "comparative_claim, ungrounded" in task.headline()


# --- Decisions ----------------------------------------------------------------------------------


def test_a_decision_must_be_attributed():
    """An unattributed decision is not an accountable one."""
    with pytest.raises(ValidationError):
        ReviewDecision(decision="approve")  # type: ignore[call-arg]


def test_only_the_three_defined_decisions_are_accepted():
    for decision in ("approve", "reject", "request_changes"):
        assert ReviewDecision(decision=decision, reviewer="r", tier="t").decision == decision
    with pytest.raises(ValidationError):
        ReviewDecision(decision="auto_approve", reviewer="r", tier="t")  # type: ignore[arg-type]


def test_disagreements_are_isolated_from_agreements():
    """These become draft golden cases: a qualified human labelling the system's mistakes."""
    decision = ReviewDecision(
        decision="request_changes",
        reviewer="a.reviewer",
        tier="senior_regulatory",
        per_finding=[
            FindingReview(finding_id="f1", verdict="agree"),
            FindingReview(finding_id="f2", verdict="disagree", note="This is on-label."),
            FindingReview(finding_id="f3", verdict="edit", edited_severity="advisory"),
        ],
    )
    assert [f.finding_id for f in decision.disagreements()] == ["f2", "f3"]


def test_a_reviewer_can_accept_most_findings_and_reject_one():
    decision = ReviewDecision(
        decision="approve",
        reviewer="r",
        tier="t",
        per_finding=[
            FindingReview(finding_id="f1", verdict="agree"),
            FindingReview(finding_id="f2", verdict="agree"),
            FindingReview(finding_id="f3", verdict="disagree", note="wrong"),
        ],
    )
    assert len(decision.disagreements()) == 1


def test_the_audit_form_flattens_a_human_step_like_any_other_event():
    decision = ReviewDecision(
        decision="reject",
        reviewer="a.reviewer",
        tier="legal",
        note="Needs head-to-head data.",
        per_finding=[FindingReview(finding_id="f1", verdict="agree")],
    )
    row = decision.as_audit_output()
    assert row["decision"] == "reject"
    assert row["reviewer"] == "a.reviewer"
    assert row["disagreement_count"] == 0
    assert row["decided_at"]


# --- Resume payload validation ------------------------------------------------------------------


def test_a_resume_payload_from_outside_is_validated_not_trusted():
    """It arrives over HTTP. An invalid decision must fail loudly, never record a phantom approval."""
    good = parse_decision({"decision": "approve", "reviewer": "r", "tier": "standard_reviewer"})
    assert good.decision == "approve"

    with pytest.raises(ValidationError):
        parse_decision({"decision": "approve"})  # no reviewer
    with pytest.raises(ValidationError):
        parse_decision({"decision": "yes", "reviewer": "r", "tier": "t"})
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_decision("approve")


def test_an_already_parsed_decision_passes_through():
    decision = ReviewDecision(decision="approve", reviewer="r", tier="t")
    assert parse_decision(decision) is decision


def test_unknown_fields_in_a_resume_payload_are_rejected():
    """Guards against a UI change silently sending a field the graph ignores."""
    with pytest.raises(ValidationError):
        parse_decision({"decision": "approve", "reviewer": "r", "tier": "t", "auto_approved": True})


def test_review_task_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ReviewTask(run_id="r", tenant_id="t", tier="x", bypass_human=True)  # type: ignore[call-arg]
