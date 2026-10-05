"""Claim extraction prompt.

Bump PROMPT_VERSION on any change: it is recorded in the audit log and in eval results, and it keys
the response cassettes -- so a prompt edit invalidates its recordings rather than silently replaying
answers to a different question.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

PROMPT_VERSION = "extract_claims-v1"


class ExtractedClaim(BaseModel):
    """One claim, as the model reports it.

    No character offsets are requested. Models are unreliable at counting characters, and a wrong
    offset would make a finding point at the wrong words -- worse than having no offset. The node
    locates ``original_text`` in the document itself, which is exact or it is a detectable failure.
    """

    model_config = ConfigDict(extra="forbid")

    original_text: str = Field(description="The claim exactly as it appears, copied verbatim.")
    normalized_claim: str = Field(
        description="The claim rewritten as a self-contained sentence, with the product, "
        "population, endpoint and timepoint made explicit from elsewhere in the piece."
    )
    claim_type: str = Field(
        description="One of: efficacy, safety, indication, dosing, comparative, economic, other."
    )
    location: str = Field(default="", description="The section heading the claim appears under.")
    reference_markers: list[str] = Field(
        default_factory=list,
        description="Reference numbers cited for this claim, e.g. ['2']. Empty if none.",
    )


class ExtractedClaims(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ExtractedClaim] = Field(default_factory=list)


SYSTEM = """You identify promotional claims in pharmaceutical marketing material for regulatory \
review. You extract claims; you do not judge them. Another reviewer decides whether each claim is \
supported."""

TEMPLATE = """Identify every promotional claim in the piece below.

A claim is any statement that asserts something about the product: how well it works, how safe it is,
who it is for, how it is taken, or how it compares with something else. Include claims made in
headlines, body copy, captions and callouts.

Do NOT extract:
  - the Important Safety Information block, which is required risk text rather than a claim
  - reference list entries
  - the Prescribing Information signpost
  - storage, handling or logistical instructions

For each claim:
  - copy `original_text` VERBATIM from the piece, character for character. Do not tidy, trim or
    rephrase it. It is used to locate the claim in the document, so an inexact copy loses it.
  - write `normalized_claim` as a standalone sentence. A headline like "Nearly 6 in 10" means
    nothing alone, so resolve it using the rest of the piece: which product, which population,
    which endpoint, which timepoint. If the piece does not say, leave that part out rather than
    guessing.
  - record `reference_markers` exactly as cited, so "[2]" becomes ["2"].

<piece>
{document}
</piece>

Return JSON matching the schema. Return an empty list if the piece contains no claims."""


def build(*, document: str) -> str:
    return TEMPLATE.format(document=document)
