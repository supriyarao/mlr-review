"""Pre-approved claim equivalence. One question: do these two claims say the same thing?

Two versions live here during the migration from a chat model to a decision model:

``preapproved-v1``  the chat prompt. A fast chat model is asked for ``{equivalent, reason}`` as JSON,
                    one call per candidate. Still the live path until the node is switched over.
``preapproved-v2``  the same question as a typed yes/no for the decisions endpoint. The answer is a
                    probability rather than a boolean, and one request carries a question per
                    candidate instead of one call each.

v1 is deleted when the node stops calling it. Keeping both for one step is deliberate: it makes the
switch a one-file change that can be reverted without touching the question wording, and it keeps
the recorded v1 cassettes meaningful until there are v2 ones to replace them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from platform_core import decisions

PROMPT_VERSION = "preapproved-v1"


class EquivalenceVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    equivalent: bool = Field(
        description="True only if the submitted claim says the same thing as the approved claim."
    )
    reason: str = Field(default="", description="One sentence.")


SYSTEM = """You compare a submitted promotional claim against a claim that has already been approved \
by a medical, legal and regulatory committee. You decide only whether they say the same thing."""

TEMPLATE = """Does the submitted claim assert the same thing as the approved claim?

Answer true only if a reader would take away the same meaning: the same population, the same
endpoint, the same magnitude, the same timepoint and the same comparator. Wording may differ.

Answer false if the submitted claim is broader in any respect -- a wider population, a stronger
result, a dropped comparator, a dropped timepoint, a dropped qualifier. A claim that is merely
*similar* is not approved; it is new.

<approved_claim>
{approved}
</approved_claim>

<submitted_claim>
{submitted}
</submitted_claim>

Return JSON matching the schema."""


def build(*, approved: str, submitted: str) -> str:
    return TEMPLATE.format(approved=approved.strip(), submitted=submitted.strip())


# --- preapproved-v2: the same question, typed ------------------------------------------------

QUESTION_VERSION = "preapproved-v2"

# The equivalence test, carried verbatim from v1's wording into the provider's ``criteria`` block.
# Same words on purpose: the point of this step is to change the transport, not the question, so any
# difference the eval shows afterwards is attributable to the model rather than to a rewrite.
TRUE_MEANS = (
    "A reader would take away the same meaning: the same population, the same endpoint, the same "
    "magnitude, the same timepoint and the same comparator. Wording may differ."
)
FALSE_MEANS = (
    "The submitted claim is broader in any respect -- a wider population, a stronger result, a "
    "dropped comparator, a dropped timepoint, a dropped qualifier. A claim that is merely similar "
    "is not approved; it is new."
)

# How a candidate's question is named, and the key it carries in the state. Numbered rather than
# named after the source id: the id is a path-shaped string, and a question name is what comes back
# in the answers map, so a short stable key keeps the response readable in an audit record.
CANDIDATE_KEY = "candidate_{index}"


def build_state(*, submitted: str, candidates: list[str]) -> dict[str, Any]:
    """The shared evidence for one submitted claim: the claim, plus every candidate to test it against.

    All candidates travel in one state so the request pays for the submitted claim once rather than
    once per candidate. That is the whole saving: the questions are small, the claim is not.
    """
    return {
        "submitted_claim": submitted.strip(),
        "approved_claims": {
            CANDIDATE_KEY.format(index=index): text.strip() for index, text in enumerate(candidates)
        },
    }


def build_questions(*, count: int) -> dict[str, dict[str, Any]]:
    """One yes/no question per candidate, each naming the candidate it is about.

    Each question names its candidate key explicitly. Relying on question order to say which
    approved claim is meant would make the answers map ambiguous the moment a candidate is dropped.
    """
    return {
        CANDIDATE_KEY.format(index=index): decisions.noul(
            "Does the submitted claim assert the same thing as the approved claim at "
            f"approved_claims.{CANDIDATE_KEY.format(index=index)}?",
            true_means=TRUE_MEANS,
            false_means=FALSE_MEANS,
        )
        for index in range(count)
    }
