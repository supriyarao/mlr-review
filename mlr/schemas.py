"""MLR domain schemas. **This is a contract.**

Other sessions build against these shapes in parallel with the graph that produces them, so the
field names and types here are the interface. See ``docs/contracts.md`` for the narrative version
and for what is allowed to change.

Two decisions in here are load-bearing and easy to get wrong:

* **A finding carries two severities.** ``severity_proposed`` is what the check suggested;
  ``severity_final`` is what the tenant's policy decided. Keeping both means the audit log can show
  that Acme downgraded something Borealis blocks, which is the whole point of multi-tenancy -- and
  it stops a reviewer being told "advisory" with no trace of the check having said otherwise.
* **Grounding is recorded as a mode, not a boolean.** A finding can be grounded in retrieved
  evidence, in a tenant rule, or in a clause of a tenant reference document. These are different
  strengths of claim, so collapsing them into ``grounded: True`` would make "grounding rate" a
  number that silently mixes three things.

``GateResult`` is deliberately **not redefined here**. It lives in ``platform_core.gates`` because a
gate is a platform concept; it is re-exported so a consumer of this module has one import to make.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Re-exported, not redefined: the definition belongs to the platform layer.
from platform_core.gates import GateResult
from platform_core.hitl import EscalationReason, ReviewDecision, ReviewTask

__all__ = [
    "Citation",
    "Claim",
    "CheckType",
    "ClaimType",
    "CitationKind",
    "Escalation",
    "EscalationReason",
    "Finding",
    "GateResult",
    "GroundedBy",
    "ReviewDecision",
    "ReviewReport",
    "ReviewTask",
    "RunCost",
    "RunStatus",
    "Severity",
    "AppliedRule",
]

# The five specialist checks, plus one bucket for tenant business rules that do not map onto a
# specific check. A rule may declare any of these as its check_type, which is why the rules engine
# treats check_type as an opaque string: the platform must not enumerate a use case's checks.
CheckType = Literal[
    "substantiation",
    "reference_support",
    "off_label",
    "fair_balance",
    "safety_prominence",
    "business_rule",
]

Severity = Literal["blocker", "advisory"]

ClaimType = Literal[
    "efficacy",
    "safety",
    "indication",
    "dosing",
    "comparative",
    "economic",
    "other",
]

# Where a citation points. See the grounding discussion in the module docstring.
CitationKind = Literal["evidence", "rule", "doc_clause"]

# Which grounding mode actually verified a finding. "none" means it failed the grounding gate and
# is therefore escalating -- it is never dropped.
GroundedBy = Literal["evidence", "rule", "doc_clause", "none"]

RunStatus = Literal["queued", "running", "awaiting_human", "complete", "failed"]

# What a human decided. Mirrors platform_core.hitl.Decision; restated so the API contract is
# readable without chasing an import.
HumanDecision = Literal["approve", "reject", "request_changes"]


class Citation(BaseModel):
    """Evidence for a finding, as something a reviewer can go and check.

    ``passage`` must appear **verbatim** in the named source, modulo whitespace. That is what the
    grounding gate verifies, and it is the reason a model cannot assert a citation into existence.
    """

    model_config = ConfigDict(extra="forbid")

    # Identifies the source: a retrieval source_id, a rule label, or a reference document path.
    source_id: str
    # The exact text being relied on.
    passage: str
    # Offset of the passage inside its source, so the UI can show it highlighted in place.
    char_span: tuple[int, int] = (0, 0)
    kind: CitationKind = "evidence"
    # Human-readable source name for display, e.g. "Zelvoran label v1.0 - 6.2 ZEL-302".
    title: str = ""
    # Set when kind is "rule" or "doc_clause", so a finding can link to the rule that produced it.
    rule_id: str = ""
    rule_version: int = 0


class Claim(BaseModel):
    """One promotional assertion extracted from the submitted document.

    ``normalized_claim`` is the self-contained form: a headline reading "Nearly 6 in 10" becomes
    "Nearly 6 in 10 adults treated with Zelvoran achieved PASI-90". Checks run against the
    normalised form because a fragment cannot be judged, while ``original_text`` is what the
    reviewer is shown, so the finding points at words that are actually on the page.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    original_text: str
    normalized_claim: str
    # Section heading the claim was found under, for display and for placement rules.
    location: str = ""
    char_span: tuple[int, int] = (0, 0)
    claim_type: ClaimType = "other"
    # Reference markers the copy attached to this claim, e.g. ["1", "3"]. The reference_support
    # check needs these to know which study it is being asked to verify against.
    reference_markers: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    """One problem a check or a rule identified.

    ``claim_id`` is empty for a document-level finding: a missing disclaimer or absent safety
    section is not about any single claim.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    check_type: CheckType
    claim_id: str = ""

    # The offending text, verbatim from the submitted document. Empty when the finding is an
    # absence -- a missing disclaimer has nothing to quote.
    quote: str = ""
    char_span: tuple[int, int] = (0, 0)
    # Written for a reviewer: what is wrong, in one or two sentences.
    issue: str
    suggested_fix: str = ""
    citations: list[Citation] = Field(default_factory=list)

    # Proposed by the check that produced the finding.
    severity_proposed: Severity = "advisory"
    # Decided by the severity node from the tenant's severity_rules. None until that node runs.
    severity_final: Severity | None = None
    # Which tenant rule set the final severity, so the decision is traceable to a policy line.
    severity_rule_id: str = ""

    confidence: int = Field(ge=0, le=100, default=0)
    # Set by the confidence gate when confidence is below the tenant's threshold. Such a finding is
    # kept and escalated, never discarded.
    low_confidence: bool = False

    grounded: bool = False
    grounded_by: GroundedBy = "none"

    # Set when a tenant business rule produced this finding.
    rule_id: str = ""
    rule_version: int = 0

    # Provenance. Empty for deterministic findings, which is itself informative: a reviewer can see
    # at a glance which findings involved a model and which are plain code.
    model: str = ""
    prompt_version: str = ""

    # Ids of findings the editor merged into this one. Both originals' citations are kept, so a
    # merge never loses evidence.
    merged_from: list[str] = Field(default_factory=list)

    @property
    def severity(self) -> Severity:
        """The severity that counts: the tenant's decision, falling back to the proposal."""
        return self.severity_final or self.severity_proposed

    @property
    def is_blocker(self) -> bool:
        return self.severity == "blocker"

    @property
    def is_absence(self) -> bool:
        """True when the finding is about something missing rather than something written."""
        return not self.quote.strip()

    @property
    def from_rule(self) -> bool:
        return bool(self.rule_id)


class AppliedRule(BaseModel):
    """A tenant business rule that was in force for a run.

    Recorded per run, with its version, because "which rules applied to this piece" is a question an
    auditor asks and the answer changes as rules are edited.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    version: int
    kind: Literal["deterministic", "llm", "reference_doc"]
    severity: Severity
    title: str = ""


class Escalation(BaseModel):
    """Why a run went above the standard reviewer, and to whom."""

    model_config = ConfigDict(extra="forbid")

    tier: str
    reasons: list[EscalationReason] = Field(default_factory=list)
    # Supporting passages, so the reviewer can check a citation without leaving the screen.
    evidence: list[dict] = Field(default_factory=list)

    def reason_codes(self) -> list[str]:
        return [r.code for r in self.reasons]


class RunCost(BaseModel):
    """What one review cost. Surfaced next to the findings, not buried in a log.

    Cost per review is the number that decides whether a system like this is deployable at scale, so
    it is part of the report rather than an operational afterthought.
    """

    model_config = ConfigDict(extra="forbid")

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    model_calls: int = 0


class ReviewReport(BaseModel):
    """The complete result of one review. This is the shape the API returns and the UI renders."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    tenant_id: str
    product_id: str
    label_version: str = ""
    audience: str = "any"
    channel: str = ""
    market: str = ""

    status: RunStatus
    # The reviewer tier this run is routed to. Always set once routing has run -- there is no
    # "approved" terminal state, because the system never approves anything.
    route: str = ""
    escalation: Escalation | None = None

    findings: list[Finding] = Field(default_factory=list)
    gate_results: list[GateResult] = Field(default_factory=list)
    rules_applied: list[AppliedRule] = Field(default_factory=list)

    claims_extracted: int = 0
    preapproved_matches: dict[str, str] = Field(default_factory=dict)

    # The exact conditions that produced this result, mirrored from the audit log so a report is
    # self-contained when exported.
    policy_version: str = ""
    config_version: str = ""
    backend: str = ""
    cost: RunCost = Field(default_factory=RunCost)

    # Set when status is "failed".
    error: str = ""

    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.is_blocker]

    def advisories(self) -> list[Finding]:
        return [f for f in self.findings if not f.is_blocker]

    def ungrounded(self) -> list[Finding]:
        return [f for f in self.findings if not f.grounded]

    def summary_line(self) -> str:
        blockers, advisories = len(self.blockers()), len(self.advisories())
        if not self.findings:
            return f"No issues found. Routed to {self.route} for confirmation."
        return f"{blockers} blocker(s), {advisories} advisory. Routed to {self.route}."
