"""Pre-approved claim equivalence prompt. Fast model: this is a matching task, not a judgement."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

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
