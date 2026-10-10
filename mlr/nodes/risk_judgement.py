"""Grade the piece's overall regulatory exposure, for a tenant that wants to route on it.

**A decision-model call.** One score question, three ordered levels, no generated text.

Why it is a node and not part of ``route``: ``route`` must stay deterministic. Who reviews what is
policy read from tenant config, an auditor has to be able to replay a routing decision by hand, and
the eval's threshold sweep recomputes routing from stored findings with no model calls at all. Put a
model inside ``route`` and all three of those stop being true. So the judgement is made here, lands
in state as a fact, and ``route`` reads it the same way it reads every other fact.

Why it runs after ``consolidate``: the question is about the piece *and* what the checks found, so
it needs the merged findings rather than the raw ones.

**It decides nothing on its own.** It writes a level into state. Whether that level escalates is a
tenant policy flag, and a tenant that sets nothing is unaffected -- which is why adding this could
not change any existing routing decision.
"""

from __future__ import annotations

from typing import Any

from mlr.nodes.base import node_run
from mlr.prompts import risk_judgement as prompt
from mlr.state import report_findings
from platform_core import decisions


def risk_judgement(state: dict[str, Any], *, decider: Any = None) -> dict[str, Any]:
    """Record how much regulatory exposure the piece carries, as a graded level."""
    with node_run(state, "risk_judgement") as run:
        document = str(state.get("document") or "")
        if not document:
            run.output = {"level": "", "note": "empty document"}
            return {"risk_level": "", "risk_score": 0.0}

        findings = [f.model_dump(mode="json") for f in report_findings(state)]

        try:
            result = decisions.decide(
                state=prompt.build_state(document=document, findings=findings),
                questions=prompt.build_questions(),
                node="risk_judgement",
                version=prompt.QUESTION_VERSION,
                fake=decider,
            )
            # Reading the answer is inside the guard, not after it. A missing answer, or one of the
            # wrong kind, is the same situation as a failed call: there is no judgement. Leaving it
            # outside meant a wrong-kind answer raised through node_run, which re-raises -- so an
            # additive signal could kill the review it was only ever meant to annotate.
            answer = result[prompt.QUESTION_NAME]
            level = answer.level()
        except decisions.DecisionError as exc:
            # No judgement means no extra reason code, so the run routes exactly as it would have
            # without this node. Failing closed here means failing *silent*: the signal is additive,
            # so its absence removes nothing a reviewer was relying on. Recorded, not hidden.
            run.error = f"risk judgement unavailable: {exc}"
            run.output = {"level": "", "failed": True}
            return {"risk_level": "", "risk_score": 0.0}

        run.record_decision(result, version=prompt.QUESTION_VERSION)

        run.output = {
            "level": level,
            # The probability-weighted value as well as the level, because they can disagree in a
            # way worth seeing: a 1.4 reported as "worth a second look" is a different piece from a
            # 1.0 reported the same way.
            "score": round(answer.value, 4),
            "confidence": round(answer.confidence, 4),
            "levels": list(prompt.LEVELS),
        }
        return {"risk_level": level, "risk_score": round(answer.value, 4)}
