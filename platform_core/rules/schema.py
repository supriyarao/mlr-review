"""Rule data model.

Deterministic code. These are the shapes a compliance owner authors in YAML and edits in the UI.

Two design points worth defending:

* **``description`` is the citable text.** A rule finding has to tell a reviewer which rule it
  broke, in the rule owner's own words. So the plain-English statement is a first-class field that
  gets quoted into the citation, not a comment.
* **Examples are mandatory, and executable.** Every rule ships at least one passing and one failing
  example, and :mod:`platform_core.rules.validation` runs the rule against them before it can be
  activated. A rule nobody can demonstrate is a rule nobody can trust -- and it is the cheapest
  possible regression test for a rule that gets edited later.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RuleKind = Literal["deterministic", "llm", "reference_doc"]
Severity = Literal["blocker", "advisory"]
Status = Literal["draft", "active", "retired"]
Audience = Literal["hcp", "patient", "any"]

# The deterministic evaluators, by params.type. Kept here so schema validation can reject an
# unknown type at author time rather than at review time.
DETERMINISTIC_TYPES = frozenset(
    {
        "banned_terms",
        "required_text",
        "trademark_first_mention",
        "required_section",
        "max_claims_without_reference",
    }
)


class RuleExample(BaseModel):
    """A worked example of the rule, used as its own unit test.

    ``expected`` is what the rule should conclude about ``text``: "pass" means no violation,
    "fail" means the rule should fire.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    expected: Literal["pass", "fail"]
    note: str = ""


class RuleScope(BaseModel):
    """When a rule applies.

    Scope is what makes two tenants' rule sets genuinely different rather than just differently
    worded: the same disclaimer requirement can be patient-facing-only for one tenant and
    unconditional for another.

    Empty list means "no restriction on this dimension" -- the common case, and more readable in
    YAML than repeating a wildcard.
    """

    model_config = ConfigDict(extra="forbid")

    audience: Audience = "any"
    channels: list[str] = Field(default_factory=list)
    product_ids: list[str] = Field(default_factory=list)
    markets: list[str] = Field(default_factory=list)

    def applies_to(
        self,
        *,
        audience: str = "any",
        channel: str = "",
        product_id: str = "",
        market: str = "",
    ) -> bool:
        """Does this scope cover the piece being reviewed?

        A rule scoped to ``patient`` does not apply to an HCP piece. A rule scoped to ``any``
        applies to both -- and a submission whose audience is unknown ("any") only picks up
        unscoped rules, because applying a patient-facing rule to a piece that might be for
        clinicians would generate false blockers.
        """
        if self.audience != "any" and self.audience != audience:
            return False
        if self.channels and channel not in self.channels:
            return False
        if self.product_ids and product_id not in self.product_ids:
            return False
        return not (self.markets and market not in self.markets)

    def summary(self) -> str:
        """One line for a rules table."""
        parts = [f"audience={self.audience}"]
        for name, values in (
            ("channels", self.channels),
            ("products", self.product_ids),
            ("markets", self.markets),
        ):
            if values:
                parts.append(f"{name}={','.join(values)}")
        return " ".join(parts)


class Rule(BaseModel):
    """One tenant-authored business rule, at one version."""

    model_config = ConfigDict(extra="forbid")

    # Stable across versions: this is the identity a finding cites and a reviewer recognises.
    id: str
    version: int = Field(ge=1, default=1)
    tenant_id: str
    title: str
    # The plain-English statement. Quoted into the citation of any finding this rule produces, so
    # it is written for a reviewer rather than as an internal note.
    description: str
    kind: RuleKind
    severity: Severity
    # Maps into the use case's finding categories, or "business_rule" for anything that does not
    # fit one. Deliberately a free string: the platform must not enumerate a use case's checks.
    check_type: str = "business_rule"
    scope: RuleScope = Field(default_factory=RuleScope)
    params: dict[str, Any] = Field(default_factory=dict)
    examples: list[RuleExample] = Field(default_factory=list)
    owner: str = ""
    effective_from: date
    effective_to: date | None = None
    status: Status = "draft"
    # Clause or section reference when the rule is derived from a reference document.
    source_ref: str | None = None

    @field_validator("id")
    @classmethod
    def _id_is_a_stable_identifier(cls, v: str) -> str:
        if not v or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", v):
            raise ValueError(f"rule id must be alphanumeric with - or _, got {v!r}")
        return v

    @field_validator("description")
    @classmethod
    def _description_is_substantive(cls, v: str) -> str:
        """It becomes the citation a reviewer reads, so a placeholder is not acceptable."""
        if len(v.strip()) < 10:
            raise ValueError("description is the citable text of the rule; write a full sentence")
        return v

    @model_validator(mode="after")
    def _kind_matches_params(self) -> Rule:
        """Reject a rule whose params do not match its kind, at author time."""
        if self.kind == "deterministic":
            kind_type = self.params.get("type")
            if kind_type not in DETERMINISTIC_TYPES:
                raise ValueError(
                    f"deterministic rule needs params.type from {sorted(DETERMINISTIC_TYPES)}, "
                    f"got {kind_type!r}"
                )
        if self.kind == "reference_doc" and not str(self.params.get("doc_path", "")).strip():
            raise ValueError("reference_doc rule needs params.doc_path")
        return self

    @model_validator(mode="after")
    def _effective_window_is_ordered(self) -> Rule:
        if self.effective_to and self.effective_to < self.effective_from:
            raise ValueError("effective_to cannot precede effective_from")
        return self

    @property
    def escalate_to(self) -> str:
        """Optional tenant override routing this rule's findings to a specific reviewer tier.

        Lets a tenant send comparative-claim violations to legal while everything else goes to
        regulatory, without the routing code knowing anything about comparative claims.
        """
        return str(self.params.get("escalate_to", "") or "")

    def is_effective_on(self, when: date) -> bool:
        """Is this rule within its effective window on a given date?

        Checked per run rather than at load time, so a rule can be authored ahead of the date a
        new code of practice takes effect, and stops applying on its own afterwards.
        """
        if when < self.effective_from:
            return False
        return not (self.effective_to and when > self.effective_to)

    def key(self) -> tuple[str, str, int]:
        """Identity of this exact version, used in the policy hash and the audit log."""
        return (self.tenant_id, self.id, self.version)

    def label(self) -> str:
        return f"{self.id} v{self.version}"


class Segment(BaseModel):
    """A span of the submitted text the caller wants judged individually.

    The seam that keeps ``max_claims_without_reference`` platform code: it needs to iterate over
    "the claims", but a claim is a use-case concept. The use case maps its claims onto segments and
    the platform stays generic.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    char_span: tuple[int, int] = (0, 0)


class RuleInput(BaseModel):
    """Everything a rule evaluator is allowed to see."""

    model_config = ConfigDict(extra="forbid")

    text: str
    # Section offsets and layout hints, produced by the use case's ingest step.
    doc_meta: dict[str, Any] = Field(default_factory=dict)
    segments: list[Segment] = Field(default_factory=list)
    audience: str = "any"
    channel: str = ""
    product_id: str = ""
    market: str = ""


class RuleViolation(BaseModel):
    """A rule firing. Neutral: the use case maps this onto its own finding schema.

    ``quote`` is the offending text, which must appear verbatim in the submitted piece -- that is
    what the grounding gate checks, and it is why a rule finding cannot be fabricated out of thin
    air. ``citation_text`` is the rule's own wording (or a document clause), which is what the
    reviewer is shown as the authority for the finding.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    rule_version: int
    tenant_id: str
    severity: Severity
    check_type: str
    # The offending text, verbatim from the submitted piece. Empty when the violation is an
    # absence (a missing disclaimer has nothing to quote) -- see citation_text instead.
    quote: str = ""
    char_span: tuple[int, int] = (0, 0)
    # Written for a reviewer: what is wrong and why.
    message: str = ""
    suggested_fix: str = ""
    # The authority: the rule's description, or the exact clause of a reference document.
    citation_text: str = ""
    # "rule" or "doc_clause" -- which grounding mode applies to this violation.
    citation_kind: Literal["rule", "doc_clause"] = "rule"
    citation_source_id: str = ""
    # Which segment triggered it, when the rule iterates segments.
    segment_id: str = ""
    confidence: int = Field(ge=0, le=100, default=100)

    @property
    def is_absence(self) -> bool:
        """True when the violation is something missing rather than something present.

        Matters to the grounding gate: an absence has no quote to find in the document, so
        demanding one would make every missing-disclaimer finding ungrounded.
        """
        return not self.quote.strip()
