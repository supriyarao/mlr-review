"""Stop and ask a person. This is the node the whole system exists to reach.

**Deterministic code**, and the one place the graph deliberately refuses to continue on its own.

``interrupt()`` suspends the run; LangGraph persists the checkpoint and ``invoke`` returns with the
payload the reviewer is being shown. Days later, ``Command(resume=...)`` continues from exactly here
-- nothing before this point is re-run or re-charged, which is what makes a human step affordable.

The payload is deliberately self-contained: the report, the reasons it escalated, and the evidence
behind each finding. A reviewer who cannot see *why* the system is unsure cannot meaningfully take
responsibility for overriding it, and taking responsibility is the entire point of their being here.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from mlr.nodes.base import node_run
from mlr.schemas import ReviewTask
from platform_core.hitl import parse_decision


def build_task(state: dict[str, Any]) -> ReviewTask:
    """Assemble what the reviewer is shown."""
    from mlr.state import report_findings

    findings = report_findings(state)
    escalation = state.get("escalation")
    blockers = [f for f in findings if f.is_blocker]

    return ReviewTask(
        run_id=str(state.get("run_id", "")),
        tenant_id=str(state.get("tenant_id", "")),
        tier=str(state.get("route") or "standard_reviewer"),
        reasons=list(escalation.reasons) if escalation else [],
        report={
            "summary": (
                "No issues found. A reviewer must still confirm."
                if not findings
                else f"{len(blockers)} blocker(s), {len(findings) - len(blockers)} advisory."
            ),
            "product_id": state.get("product_id", ""),
            "label_version": state.get("label_version", ""),
            "audience": state.get("audience", ""),
            "blockers": len(blockers),
            "advisories": len(findings) - len(blockers),
            "findings": [f.model_dump(mode="json") for f in findings],
            "rules_applied": [r.model_dump(mode="json") for r in state.get("rules_applied") or []],
            "gate_results": [g.model_dump(mode="json") for g in state.get("gate_results") or []],
            "policy_version": state.get("policy_version", ""),
            "incomplete_steps": list(state.get("errors") or []),
        },
        evidence=list(escalation.evidence) if escalation else [],
    )


def human_review(state: dict[str, Any]) -> dict[str, Any]:
    """Suspend the run until a person decides.

    The decision is validated before it is accepted. It arrives from outside the graph -- an HTTP
    request from the UI -- so trusting it would mean a malformed payload could record an approval
    nobody gave.
    """
    task = build_task(state)

    # Everything above this line runs again on resume, because LangGraph re-enters the node. Keep it
    # cheap and side-effect free: building the payload twice is harmless, writing an audit event twice
    # would corrupt the trail.
    answer = interrupt(task.model_dump(mode="json"))

    with node_run(state, "human_review", event_type="human") as run:
        decision = parse_decision(answer)
        run.output = decision.as_audit_output()
        return {
            "human_decision": decision.model_dump(mode="json"),
            "status": "complete",
        }
