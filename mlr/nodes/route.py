"""Decide which human sees this review, and why.

**Deterministic code. No model involved.** Who reviews what is policy, read from the tenant's
escalation configuration.

The invariant this module exists to guarantee: **every path ends at a person.** There is no branch in
which the system approves a piece. A review with no findings at all still routes to a standard
reviewer with "no issues found" -- because "the system found nothing" and "this material is approved"
are different statements, and only a human may make the second one.

Escalation reasons are collected rather than reduced to a single cause, because a reviewer handed an
escalation needs the whole picture: a piece can be escalated for a blocker *and* an unverifiable
finding *and* a check that failed to run, and each of those changes what they should look at.
"""

from __future__ import annotations

from typing import Any

from mlr.nodes.base import node_run, rollup_cost
from mlr.schemas import Escalation, EscalationReason, Finding
from platform_core.gates import GateResult
from platform_core.rules.registry import RuleRegistry
from platform_core.tenancy import load_tenant


def flag_low_confidence(
    findings: list[Finding], state: dict[str, Any]
) -> tuple[list[Finding], dict[str, Any], list[GateResult]]:
    """Flag findings below the tenant's confidence threshold.

    A part of the ``consolidate`` node rather than a node of its own. A low-confidence finding is
    kept and escalated, never discarded. The threshold is the safety-versus-throughput dial: Acme
    accepts machine judgement at 70, Borealis wants a human to look at anything below 80.

    Still the single place the threshold is applied, which is what lets the eval harness sweep it
    and recompute routing without calling a model again. Folding it into ``consolidate`` moved
    where it runs, not how many places know about it.

    Worth knowing when reading the sweep: the confidences feeding this are the reviewers' own
    self-reports, and nothing calibrates them against observed correctness. The dial is in the
    right place; what it is dialling has not been measured.
    """
    threshold = load_tenant(str(state["tenant_id"])).confidence_threshold
    if not findings:
        return (
            [],
            {"findings": 0, "threshold": threshold},
            [
                GateResult(
                    gate="confidence",
                    passed=True,
                    details="no findings to assess",
                    reason_code="low_confidence",
                )
            ],
        )

    flagged: list[Finding] = []
    below: list[str] = []
    for finding in findings:
        is_low = finding.confidence < threshold
        if is_low:
            below.append(finding.id)
        flagged.append(finding.model_copy(update={"low_confidence": is_low}))

    gate = GateResult(
        gate="confidence",
        passed=not below,
        details=(
            f"all {len(flagged)} finding(s) met the tenant threshold of {threshold}"
            if not below
            else f"{len(below)} of {len(flagged)} finding(s) fell below the tenant threshold "
            f"of {threshold}"
        ),
        failed_items=below,
        on_fail_route="escalate" if below else None,
        reason_code="low_confidence",
    )
    return flagged, {"findings": len(flagged), "threshold": threshold, "below": len(below)}, [gate]


def _escalation_reasons(state: dict[str, Any], findings: list[Finding]) -> list[EscalationReason]:
    """Every reason this review needs more than a standard reviewer.

    Reason codes are what the tenant's escalation rules match against, which is how Borealis sends
    off-label findings to a medical director and comparative claims to legal without the routing code
    knowing what either of those things is.
    """
    reasons: list[EscalationReason] = []

    blockers = [f for f in findings if f.is_blocker]
    if blockers:
        reasons.append(
            EscalationReason(
                code="blocker",
                explanation=(
                    f"{len(blockers)} blocking finding(s) that cannot be resolved by rewording "
                    f"alone."
                ),
                items=[f.id for f in blockers],
            )
        )

    # Specific finding types a tenant may route to a specialist.
    for code, check_type, description in (
        ("off_label", "off_label", "Promotion outside the approved label, explicit or implied."),
        (
            "comparative_claim",
            "comparative",
            "A comparative or superiority claim, which carries competitor-litigation exposure.",
        ),
    ):
        matched = [f for f in findings if f.check_type == check_type]
        if matched:
            reasons.append(
                EscalationReason(code=code, explanation=description, items=[f.id for f in matched])
            )

    # A substantiation finding on a claim the piece frames comparatively is a comparative claim even
    # though no check is named "comparative": the claim type comes from extraction.
    comparative_claims = {
        c.id for c in (state.get("claims") or []) if getattr(c, "claim_type", "") == "comparative"
    }
    if comparative_claims and not any(r.code == "comparative_claim" for r in reasons):
        matched = [f for f in findings if f.claim_id in comparative_claims]
        if matched:
            reasons.append(
                EscalationReason(
                    code="comparative_claim",
                    explanation=(
                        "A finding on a comparative claim, which carries competitor-litigation "
                        "exposure."
                    ),
                    items=[f.id for f in matched],
                )
            )

    # The graded risk judgement, but only if this tenant asked to route on it. The flag names the
    # lowest level that escalates, so a tenant sets `escalate_on_risk_judgement: serious` and gets
    # exactly that. A tenant that sets nothing is unaffected, which is why adding this signal could
    # not change an existing routing decision.
    #
    # Trade-off: this is the one escalation reason that comes from a judgement rather than a fact.
    # It is opt-in for that reason, and the explanation says so in words a reviewer reads, because
    # "a model thought this looked serious" and "this claim has no evidence" deserve different
    # amounts of trust even when they route to the same person.
    from mlr.prompts.risk_judgement import LEVELS

    escalate_from = str(
        load_tenant(str(state["tenant_id"])).policy_flags.get("escalate_on_risk_judgement", "")
    ).strip()
    level = str(state.get("risk_level") or "")
    if (
        escalate_from in LEVELS
        and level in LEVELS
        and LEVELS.index(level) >= LEVELS.index(escalate_from)
    ):
        reasons.append(
            EscalationReason(
                code="high_risk_judgement",
                explanation=(
                    f"Graded {level!r} for overall regulatory exposure, at or above this "
                    f"tenant's {escalate_from!r} threshold. This is a judgement about the "
                    f"piece as a whole, not a finding about any one claim."
                ),
                items=[],
            )
        )

    # Gate failures. Each carries its own reason code so a tenant can route on it.
    for gate in state.get("gate_results") or []:
        if gate.passed or not gate.reason_code:
            continue
        if any(r.code == gate.reason_code for r in reasons):
            continue
        reasons.append(
            EscalationReason(
                code=gate.reason_code,
                explanation=gate.details or f"The {gate.gate} gate did not pass.",
                items=list(gate.failed_items),
            )
        )

    # A node that failed outright. Recorded as a reason so the reviewer is told the review is
    # incomplete rather than being handed a partial result that looks whole.
    errors = state.get("errors") or []
    if errors and not any(r.code == "schema_failure" for r in reasons):
        reasons.append(
            EscalationReason(
                code="incomplete_review",
                explanation=(
                    f"{len(errors)} step(s) did not complete, so this review is not exhaustive: "
                    + "; ".join(str(e)[:120] for e in errors[:2])
                ),
                items=[],
            )
        )

    return reasons


def _tier_for(state: dict[str, Any], reasons: list[EscalationReason]) -> str:
    """The reviewer tier, honouring a per-rule override before tenant defaults.

    A tenant can send one specific rule's findings to a named tier via ``params.escalate_to`` -- used
    by Borealis to route comparative-language violations to legal. Checked first because it is the
    most specific statement of intent available.
    """
    config = load_tenant(str(state["tenant_id"]))
    codes = [r.code for r in reasons]

    from mlr.state import report_findings

    rule_ids = {f.rule_id for f in report_findings(state) if f.rule_id and f.is_blocker}
    if rule_ids:
        try:
            registry = RuleRegistry(str(state["tenant_id"]))
            for rule_id in sorted(rule_ids):
                override = registry.get(rule_id).escalate_to
                if override:
                    return override
        except Exception:  # pragma: no cover - a missing rule must not break routing
            pass

    return config.escalation.tier_for(codes)


def route(state: dict[str, Any]) -> dict[str, Any]:
    """Choose the destination. Always a human."""
    with node_run(state, "route") as run:
        from mlr.state import report_findings

        findings = report_findings(state)
        config = load_tenant(str(state["tenant_id"]))
        reasons = _escalation_reasons(state, findings)

        if not reasons:
            # Clean, or advisory-only with everything grounded and confident. Still a person.
            destination = config.escalation.standard_tier
            run.output = {
                "route": destination,
                "escalated": False,
                "findings": len(findings),
                "summary": "no issues found" if not findings else "advisory findings only",
            }
            return {
                "route": destination,
                "escalation": None,
                "status": "awaiting_human",
                **rollup_cost(str(state.get("run_id", ""))),
            }

        tier = _tier_for(state, reasons)
        evidence = [
            {
                "source_id": c.source_id,
                "title": c.title,
                "passage": c.passage,
                "kind": c.kind,
                "finding_id": f.id,
            }
            for f in findings
            for c in f.citations
        ][:12]

        escalation = Escalation(tier=tier, reasons=reasons, evidence=evidence)
        run.output = {
            "route": tier,
            "escalated": True,
            "reasons": [r.code for r in reasons],
            "blockers": sum(1 for f in findings if f.is_blocker),
        }
        return {
            "route": tier,
            "escalation": escalation,
            "status": "awaiting_human",
            **rollup_cost(str(state.get("run_id", ""))),
        }
