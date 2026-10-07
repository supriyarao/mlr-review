"""Canonical example payloads, built from the real models. **Single source of truth.**

Why this module exists rather than hand-written JSON: the API fixtures under ``tests/fixtures/api/``
and the stub service in :mod:`mlr.service` both need example data, and a parallel session coding
against a hand-written fixture would be coding against something that can silently diverge from the
schema it claims to illustrate. Building the examples through the Pydantic models means a fixture
cannot be schema-invalid, and ``scripts/gen_api_fixtures.py`` regenerates the JSON from here.

The example run is deliberately the hardest case in the golden set rather than a happy path: a
subgroup result presented as a general one, under Acme. A parallel session building the UI needs to
see a blocker, a citation, an escalation and a rule finding -- an empty happy-path payload would let
them build something that falls over on the first real review.

Deterministic: every value is fixed, so regenerating the fixtures produces no diff.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mlr.schemas import (
    AppliedRule,
    Citation,
    Claim,
    Escalation,
    EscalationReason,
    Finding,
    GateResult,
    ReviewReport,
    ReviewTask,
    RunCost,
)

# Fixed, not generated: a fixture whose timestamp moved on every regeneration would show a diff
# on every run and the drift test would be useless.
CREATED_AT = datetime(2026, 10, 5, 12, 42, tzinfo=UTC)

RUN_ID = "a1b2c3d4e5f6"
TENANT_ID = "acme_pharma"
PRODUCT_ID = "zelvoran"
LABEL_VERSION = "1.0"
POLICY_VERSION = "a6062d2216a1137f"

# The quote is lifted verbatim from corpus/promo/zelvoran-hcp-subgroup-claim.md, and the citation
# passage verbatim from corpus/references/zel-302.md. That is not decoration: the grounding gate
# requires exactly this, so an example that cheated here would teach the wrong shape.
OFFENDING_QUOTE = "Nearly 6 in 10 adults treated with Zelvoran achieved PASI-90"
EVIDENCE_PASSAGE = (
    "The 58.1% figure applies only to the subgroup of patients with prior biologic exposure. "
    "The result in the overall randomised population was 44.2%."
)


def example_claims() -> list[Claim]:
    return [
        Claim(
            id="c1",
            original_text="Nearly 6 in 10 adults treated with Zelvoran achieved PASI-90",
            normalized_claim=(
                "Nearly 6 in 10 adults treated with Zelvoran achieved PASI-90 at Week 16 in the "
                "overall population."
            ),
            location="Efficacy",
            char_span=(214, 274),
            claim_type="efficacy",
            reference_markers=["2"],
        ),
        Claim(
            id="c2",
            original_text="Sustained response: PASI-90 was maintained at 12 weeks in the extension study",
            normalized_claim=(
                "Zelvoran maintained PASI-90 response at 12 weeks in the extension study."
            ),
            location="Efficacy",
            char_span=(352, 429),
            claim_type="efficacy",
            reference_markers=["3"],
        ),
    ]


def example_findings() -> list[Finding]:
    """One blocker from a core check, one from a tenant rule, one advisory.

    Covers every display case the UI has to handle: a grounded blocker with evidence, a rule finding
    whose authority is the rule's own wording, and an advisory.
    """
    return [
        Finding(
            id="f1",
            check_type="substantiation",
            claim_id="c1",
            quote=OFFENDING_QUOTE,
            char_span=(214, 274),
            issue=(
                "Presents the prior-biologic subgroup result (58.1%) as a general result. PASI-90 "
                "in the overall randomised population was 44.2%."
            ),
            suggested_fix=(
                "State the overall result, or name the subgroup: 'In adults with prior biologic "
                "exposure, 58.1% achieved PASI-90 at Week 16'."
            ),
            citations=[
                Citation(
                    source_id="acme_pharma:reference:zel-302:2",
                    passage=EVIDENCE_PASSAGE,
                    char_span=(812, 968),
                    kind="evidence",
                    title="ZEL-302 - Prespecified subgroup analysis by prior biologic exposure",
                )
            ],
            severity_proposed="blocker",
            severity_final="blocker",
            severity_rule_id="ACME-SEV-SUBSTANTIATION",
            confidence=92,
            grounded=True,
            grounded_by="evidence",
            model="claude-sonnet-5",
            prompt_version="substantiation-v2",
        ),
        Finding(
            id="f2",
            check_type="business_rule",
            claim_id="c1",
            quote=OFFENDING_QUOTE,
            char_span=(214, 274),
            issue=(
                "Breaches the Acme Brand and Promotional Code: a subgroup figure is presented as "
                "the result in the overall population."
            ),
            suggested_fix="Describe the result as a subgroup result, naming the subgroup.",
            citations=[
                Citation(
                    source_id="ACME-DOC-001 v1",
                    passage=(
                        "A result observed in a subgroup must be described as a subgroup result, "
                        "naming the subgroup. A subgroup figure must never be presented as the "
                        "result in the overall population."
                    ),
                    kind="doc_clause",
                    title="Acme Brand and Promotional Code, clause 2",
                    rule_id="ACME-DOC-001",
                    rule_version=1,
                )
            ],
            severity_proposed="blocker",
            severity_final="blocker",
            severity_rule_id="ACME-SEV-DEFAULT",
            confidence=88,
            grounded=True,
            grounded_by="doc_clause",
            rule_id="ACME-DOC-001",
            rule_version=1,
            model="claude-sonnet-5",
            prompt_version="rule-check-v1",
        ),
        Finding(
            id="f3",
            check_type="substantiation",
            claim_id="c2",
            quote="PASI-90 was maintained at 12 weeks in the extension study",
            char_span=(372, 429),
            issue=(
                "Cites Reference 3 (ZEL-303) for PASI-90 at 12 weeks. ZEL-303 reports PASI-75 at "
                "Week 16 and Week 52, did not assess PASI-90, and made no Week 12 assessment."
            ),
            suggested_fix="Cite a study that reports the endpoint and timepoint claimed.",
            citations=[
                Citation(
                    source_id="acme_pharma:reference:zel-303:1",
                    passage="ZEL-303 did not assess PASI-90.",
                    char_span=(640, 671),
                    kind="evidence",
                    title="ZEL-303 - Endpoints not assessed",
                )
            ],
            severity_proposed="blocker",
            severity_final="blocker",
            severity_rule_id="ACME-SEV-SUBSTANTIATION",
            confidence=95,
            grounded=True,
            grounded_by="evidence",
            model="claude-sonnet-5",
            prompt_version="substantiation-v2",
        ),
    ]


def example_gate_results() -> list[GateResult]:
    """Gates record a pass as well as a failure, so the trail shows what ran."""
    return [
        GateResult(
            gate="schema",
            passed=True,
            details="all 2 node output(s) validated on the first attempt",
            reason_code="schema_failure",
        ),
        GateResult(
            gate="grounding",
            passed=True,
            details="all 3 finding(s) carry a citation verified against a source of this tenant",
            reason_code="ungrounded",
        ),
        GateResult(
            gate="confidence",
            passed=True,
            details="all 3 finding(s) met the tenant threshold of 70",
            reason_code="low_confidence",
        ),
    ]


def example_rules_applied() -> list[AppliedRule]:
    return [
        AppliedRule(
            id="ACME-BAN-001",
            version=1,
            kind="deterministic",
            severity="blocker",
            title="Prohibited efficacy and safety language",
        ),
        AppliedRule(
            id="ACME-BAN-002",
            version=1,
            kind="deterministic",
            severity="advisory",
            title="Unqualified superlatives",
        ),
        AppliedRule(
            id="ACME-DOC-001",
            version=1,
            kind="reference_doc",
            severity="blocker",
            title="Compliance with the Acme Brand and Promotional Code",
        ),
        AppliedRule(
            id="ACME-ISI-001",
            version=1,
            kind="deterministic",
            severity="blocker",
            title="Important Safety Information must be present",
        ),
        AppliedRule(
            id="ACME-LLM-002",
            version=1,
            kind="llm",
            severity="blocker",
            title="Depicted population must match the indication",
        ),
        AppliedRule(
            id="ACME-REQ-001",
            version=1,
            kind="deterministic",
            severity="blocker",
            title="Prescribing Information signpost",
        ),
        AppliedRule(
            id="ACME-TM-001",
            version=1,
            kind="deterministic",
            severity="advisory",
            title="Trademark symbol on first mention",
        ),
    ]


def example_escalation() -> Escalation:
    return Escalation(
        tier="senior_regulatory",
        reasons=[
            EscalationReason(
                code="blocker",
                explanation=(
                    "Three blocking findings: a subgroup result presented as a general result, the "
                    "same issue under the promotional code, and a citation that does not measure "
                    "the claimed endpoint."
                ),
                items=["f1", "f2", "f3"],
            )
        ],
        evidence=[
            {
                "source_id": "acme_pharma:reference:zel-302:2",
                "title": "ZEL-302 - Prespecified subgroup analysis",
                "passage": EVIDENCE_PASSAGE,
            }
        ],
    )


def example_report(*, status: str = "awaiting_human") -> ReviewReport:
    """The canonical report. Every API example derives from this."""
    return ReviewReport(
        run_id=RUN_ID,
        tenant_id=TENANT_ID,
        product_id=PRODUCT_ID,
        label_version=LABEL_VERSION,
        audience="hcp",
        channel="sales_aid",
        market="us",
        status=status,  # type: ignore[arg-type]
        route="senior_regulatory",
        escalation=example_escalation(),
        findings=example_findings(),
        gate_results=example_gate_results(),
        rules_applied=example_rules_applied(),
        claims_extracted=2,
        preapproved_matches={},
        policy_version=POLICY_VERSION,
        config_version="2026.1",
        backend="anthropic",
        cost=RunCost(
            tokens_in=18420, tokens_out=3180, cost_usd=0.06864, latency_ms=11840, model_calls=9
        ),
    )


def example_review_task() -> ReviewTask:
    """What the reviewer is shown when the graph interrupts."""
    report = example_report()
    return ReviewTask(
        run_id=RUN_ID,
        tenant_id=TENANT_ID,
        tier="senior_regulatory",
        reasons=example_escalation().reasons,
        report={
            "summary": report.summary_line(),
            "blockers": len(report.blockers()),
            "advisories": len(report.advisories()),
            "findings": [f.model_dump(mode="json") for f in report.findings],
        },
        evidence=example_escalation().evidence,
        created_at=CREATED_AT,
    )


def example_audit_events() -> list[dict[str, Any]]:
    """An ordered audit trail for the example run.

    Hand-built rather than read from a database, because the fixture has to be stable: a real trail
    carries timestamps that would change the fixture on every regeneration.
    """
    base = {
        "run_id": RUN_ID,
        "tenant_id": TENANT_ID,
        "policy_version": POLICY_VERSION,
        "label_version": LABEL_VERSION,
    }
    steps = [
        ("ingest", "node", "", "", 0, 0, 0.0, 12),
        (
            "extract_claims",
            "node",
            "claude-sonnet-5",
            "extract_claims-v1",
            4120,
            680,
            0.01504,
            3210,
        ),
        (
            "preapproved_filter",
            "node",
            "claude-haiku-4-5",
            "preapproved-v1",
            1840,
            120,
            0.00244,
            740,
        ),
        (
            "substantiation",
            "node",
            "claude-sonnet-5",
            "substantiation-v2",
            6840,
            1180,
            0.02640,
            5100,
        ),
        ("house_rules", "node", "claude-haiku-4-5", "rule-check-v1", 3120, 410, 0.00517, 1180),
        ("consolidate", "gate", "", "", 0, 0, 0.0, 21),
        ("route", "node", "", "", 0, 0, 0.0, 3),
        ("human_review", "human", "", "", 0, 0, 0.0, 0),
    ]
    events = []
    for seq, (node, kind, model, prompt_version, tin, tout, cost, latency) in enumerate(steps, 1):
        events.append(
            {
                **base,
                "id": seq,
                "seq": seq,
                "ts": f"2026-10-05T12:{30 + seq:02d}:00+00:00",
                "node": node,
                "event_type": kind,
                "input_hash": f"{seq:016x}",
                "model": model,
                "prompt_version": prompt_version,
                "tokens_in": tin,
                "tokens_out": tout,
                "cost_usd": cost,
                "latency_ms": latency,
                "error": "",
            }
        )
    return events
