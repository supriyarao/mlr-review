"""Record the outcome, and turn reviewer disagreements into draft eval cases.

**Deterministic code. No model involved.**

The second job is the interesting one. When a qualified reviewer marks a finding "disagree", that is a
labelled example of the system being wrong, produced for free by exactly the person whose judgement
the system is trying to approximate. Discarding it would be throwing away the most valuable data the
system generates.

So each disagreement is written to ``evals/candidates/`` as a **draft** golden case. Draft, not
active: a single reviewer's disagreement is one opinion, and promoting it straight into the regression
suite would let one person's call silently become the system's definition of correct. A human curates
them into ``evals/golden/``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlr.nodes.base import node_run, rollup_cost
from platform_core.hitl import ReviewDecision

CANDIDATES_DIR = Path(__file__).resolve().parent.parent.parent / "evals" / "candidates"


def candidate_path(run_id: str, finding_id: str) -> Path:
    return CANDIDATES_DIR / f"{run_id}-{finding_id}.json"


def write_candidates(state: dict[str, Any], decision: ReviewDecision) -> list[Path]:
    """Write one draft eval case per reviewer disagreement.

    JSON rather than the golden-set YAML on purpose: a candidate is not yet a case. It records what
    happened and what the reviewer said, and a human decides whether it becomes a case -- at which
    point they write the ``why_hard`` that the golden schema requires and that only a person can.
    """
    from mlr.state import report_findings

    disagreements = decision.disagreements()
    if not disagreements:
        return []

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    findings = {f.id: f for f in report_findings(state)}
    written: list[Path] = []

    for review in disagreements:
        finding = findings.get(review.finding_id)
        if finding is None:
            continue
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_run_id": state.get("run_id", ""),
            "tenant_id": state.get("tenant_id", ""),
            "product_id": state.get("product_id", ""),
            "label_version": state.get("label_version", ""),
            "audience": state.get("audience", ""),
            "channel": state.get("channel", ""),
            "policy_version": state.get("policy_version", ""),
            "reviewer": decision.reviewer,
            "reviewer_tier": decision.tier,
            "verdict": review.verdict,
            "reviewer_note": review.note,
            "reviewer_edit": {
                "issue": review.edited_issue,
                "severity": review.edited_severity,
            },
            "system_finding": finding.model_dump(mode="json"),
            # Left blank deliberately. A case needs a human to say what makes it hard, and
            # auto-generating that sentence would produce a golden set full of filler.
            "why_hard": "",
            "status": "draft",
            "note": (
                "Draft eval candidate generated from a reviewer disagreement. Promote to "
                "evals/golden/ only after a human writes why_hard and confirms the expected outcome."
            ),
        }
        path = candidate_path(str(state.get("run_id", "")), review.finding_id)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def finalize(state: dict[str, Any]) -> dict[str, Any]:
    """Close out the run."""
    with node_run(state, "finalize") as run:
        from mlr.state import report_findings

        raw = state.get("human_decision")
        findings = report_findings(state)

        if not raw:
            # Reached without a decision. Status stays non-terminal rather than being marked complete:
            # a run nobody decided is not a finished run.
            run.output = {"note": "no human decision recorded", "findings": len(findings)}
            return {"status": "awaiting_human"}

        decision = ReviewDecision.model_validate(raw)
        written = write_candidates(state, decision)

        run.output = {
            "decision": decision.decision,
            "reviewer": decision.reviewer,
            "tier": decision.tier,
            "findings": len(findings),
            "blockers": sum(1 for f in findings if f.is_blocker),
            "disagreements": len(decision.disagreements()),
            "eval_candidates_written": [p.name for p in written],
        }
        return {"status": "complete", **rollup_cost(str(state.get("run_id", "")))}
