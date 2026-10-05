"""Response schema and shared instructions for the rule-checking nodes.

The prompt body itself is built by :mod:`platform_core.rules.prompting`, which owns the delimiting and
neutralising of tenant-authored text. What lives here is the MLR-specific part: the response shape,
and the instruction that ties a reported violation back to a claim id.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from platform_core.gates import GateResult

PROMPT_VERSION = "rule-response-v1"


class ReportedRuleFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(description="The rule being applied. Must be one of the ids listed.")
    quote: str = Field(
        description="The offending text, copied VERBATIM from the content. Empty only when the "
        "violation is that something required is absent."
    )
    issue: str = Field(description="Why that text violates that rule, in one sentence.")
    suggested_fix: str = Field(default="")
    confidence: int = Field(ge=0, le=100, default=0)
    claim_id: str = Field(
        default="",
        description="The claim id this concerns, if it maps to one of the listed claims.",
    )
    clause: str = Field(
        default="",
        description="For a reference-document review only: the exact clause text, copied verbatim "
        "from the document.",
    )


class RuleCheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[ReportedRuleFinding] = Field(default_factory=list)


def rule_response_instructions(claims: list[Any]) -> str:
    """Appended to a rule prompt so a violation can be tied to a claim.

    Claim ids are offered rather than required: a rule about a missing disclaimer is not about any one
    claim, and forcing an id would invite the model to pick an arbitrary one.
    """
    if not claims:
        return (
            "\n\nReport each violation with its rule_id and a verbatim quote. "
            "Leave claim_id empty. Return JSON matching the schema."
        )

    listed = "\n".join(
        f"  {getattr(c, 'id', '')}: {getattr(c, 'original_text', '')[:140]}" for c in claims
    )
    return (
        "\n\nThe claims identified in this piece are:\n"
        f"{listed}\n\n"
        "If a violation concerns one of them, set claim_id to its id. If it concerns the piece as a "
        "whole -- something missing, or a layout problem -- leave claim_id empty.\n\n"
        "Return JSON matching the schema."
    )


def schema_failure_gate(node: str, exc: Exception) -> GateResult:
    """The schema gate's failure record.

    A node whose output could not be parsed has not produced a reviewable result, so this fails with
    a route: the run escalates rather than continuing as though the check found nothing.
    """
    return GateResult(
        gate="schema",
        passed=False,
        details=f"{node} output failed validation after a retry: {exc}",
        failed_items=[node],
        on_fail_route="escalate",
        reason_code="schema_failure",
    )
