"""Turn what the reviewers said into the list a person will read.

**Deterministic code. No model involved** -- in any of the four stages, deliberately. If the thing
deciding whether to trust model output were itself model output, none of it would have independent
value.

This was four nodes: grounding_gate, severity, editor, confidence_gate. They ran one after another,
in a fixed order, each consuming the previous one's output and all four writing the same state key.
Four boxes in the timeline for "tidy up what the reviewers found" told a reader nothing, and the
reader who wants the detail gets it from the finding itself -- every finding says how it was
grounded, which policy line graded it, and whether it was merged.

So they are four stages of one node, in the order they have to run in:

1. **verify**  -- nothing a reviewer said is trusted until the quote is found in the piece and the
   citation is checked against a source this tenant owns. A finding that fails is marked, never
   dropped: discarding it would turn "I could not verify this" into "there is no problem here".
2. **grade**   -- the tenant's severity rules decide blocker or advisory. This is where Acme and
   Borealis disagree about the same finding.
3. **tidy**    -- two reviewers who flagged the same words produce one finding, not two.
4. **flag**    -- anything below the tenant's confidence threshold is marked, kept and escalated.

Order matters and is not arrangeable. Grading before verifying would grade findings that turn out
to be unverifiable; merging before grading would merge findings the tenant grades differently; and
flagging confidence before merging would flag a confidence the merge is about to revise upward.

**What is lost by merging, and what is not.** The audit trail used to carry four events and now
carries one, so the per-stage detail is in that event's output under ``grounding``, ``severity``,
``editor`` and ``confidence`` rather than spread across four rows -- the same information, one
fewer join. What a stepped demo loses is three pause points; that was the real cost and it was
judged worth paying for a graph a reader can follow.
"""

from __future__ import annotations

from typing import Any

from mlr.nodes.base import node_run
from mlr.nodes.editor import tidy
from mlr.nodes.grounding_gate import verify
from mlr.nodes.route import flag_low_confidence
from mlr.nodes.severity import grade
from mlr.schemas import Finding
from platform_core.gates import GateResult


def consolidate(state: dict[str, Any]) -> dict[str, Any]:
    """Verify, grade, merge and flag. One node, four stages, no model."""
    with node_run(state, "consolidate", event_type="gate") as run:
        # Read the raw accumulator, not report_findings: this node is the first thing to run after
        # the fan-out, so `final_findings` is empty and `findings` is what every reviewer appended.
        findings: list[Finding] = list(state.get("findings") or [])
        gates: list[GateResult] = []
        summary: dict[str, Any] = {}

        findings, summary["grounding"], part_gates = verify(findings, state)
        gates += part_gates

        findings, summary["severity"] = grade(findings, state)
        findings, summary["editor"] = tidy(findings, state)

        findings, summary["confidence"], part_gates = flag_low_confidence(findings, state)
        gates += part_gates

        summary["findings"] = len(findings)
        run.output = summary
        run.citations = [{"source_id": c.source_id} for f in findings for c in f.citations]

        # Written to `final_findings`, never to `findings`. That key has an append reducer so the
        # parallel reviewers can each contribute, and a reducer applies to every write -- returning
        # this list there would append it to the originals and silently double every finding.
        return {"final_findings": findings, "gate_results": gates}
