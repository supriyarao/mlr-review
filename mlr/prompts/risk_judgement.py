"""How much regulatory exposure a piece carries, as a graded judgement.

The question the reason codes cannot ask. Routing works from codes -- ``blocker``, ``off_label``,
``ungrounded`` -- and each is a fact about a finding. None of them can say "nothing here is
individually blocking, but the whole piece reads wrong", which is a judgement a senior reviewer
makes in a second and a rule cannot express.

So this is deliberately **additive**. It does not decide anything. It produces one more reason code
that a tenant may route on, exactly like the others, and a tenant that configures nothing is
unaffected. The routing table stays deterministic, the audit trail still names the rule that fired,
and the eval's threshold sweep still recomputes routing without calling a model.

A score rather than a yes/no, because "how much" is the question. The answer is one of three
ordered levels and the model's confidence in it.
"""

from __future__ import annotations

from typing import Any

from platform_core import decisions

QUESTION_VERSION = "risk_judgement-v1"

QUESTION_NAME = "risk_level"

# Ordered, lowest first. Three rather than five: the levels have to mean something distinct to a
# reviewer, and "slightly serious" is not a thing anyone acts on differently.
LEVELS: tuple[str, ...] = (
    "routine",
    "worth a second look",
    "serious",
)

INSTRUCTIONS = (
    "How much regulatory exposure does this promotional piece carry, taken as a whole? Judge the "
    "piece a reviewer would actually receive, not any single sentence in it. Weigh how a reader "
    "would come away: what they would believe about how well the product works, what they would "
    "understand about its risks, and whether the balance between those two is honest. A piece can "
    "carry serious exposure without any one line being indefensible."
)

# How much of the piece to send. The endpoint allows 32k tokens for the state and the longest
# question together; this is far below that and keeps the call cheap on a long leave piece.
#
# Trade-off: a truncated piece could hide a problem in its tail, which is why this judgement is only
# ever an additional signal and never the only thing standing between a piece and a reviewer. The
# checks that must not miss anything read the whole document.
MAX_DOCUMENT_CHARS = 12_000


def build_state(*, document: str, findings: list[dict[str, Any]]) -> dict[str, Any]:
    """The piece, plus what the checks already found.

    The findings travel as counts and quotes rather than full objects. The question is about the
    piece as a whole, and handing over five paragraphs of per-finding explanation would invite the
    model to re-judge the findings instead of judging the piece.
    """
    by_check: dict[str, int] = {}
    blockers = 0
    for finding in findings:
        check = str(finding.get("check_type") or "unknown")
        by_check[check] = by_check.get(check, 0) + 1
        if (finding.get("severity_final") or finding.get("severity_proposed")) == "blocker":
            blockers += 1

    return {
        "piece": document[:MAX_DOCUMENT_CHARS],
        "truncated": len(document) > MAX_DOCUMENT_CHARS,
        "findings_so_far": {
            "total": len(findings),
            "blocking": blockers,
            "by_check": by_check,
            # The offending text only. Enough to see what was flagged, not enough to re-argue it.
            "quotes": [str(f.get("quote") or "")[:160] for f in findings[:12] if f.get("quote")],
        },
    }


def build_questions() -> dict[str, dict[str, Any]]:
    """One score question, graded over :data:`LEVELS`."""
    return {QUESTION_NAME: decisions.score(INSTRUCTIONS, list(LEVELS))}
