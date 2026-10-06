"""Prompts for the three specialist checks.

One module rather than three, because they share a response schema and a set of standing rules. The
differences between the checks are their question and their evidence, which is exactly what varies
between the entries in ``CHECKS`` -- keeping them side by side makes those differences legible, and
makes it obvious when one check's instructions drift away from the others'.

Each check carries its own version string, so editing one does not invalidate the others' cassettes.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


class ReportedCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="The source id exactly as given in the evidence block.")
    passage: str = Field(
        description="The exact sentence or sentences from that source that establish the problem, "
        "copied verbatim."
    )


class ReportedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(
        default="",
        description="The claim id this concerns, or '' if the finding "
        "is about the piece as a whole.",
    )
    quote: str = Field(
        description="The offending text, copied VERBATIM from the piece. Leave empty only when the "
        "finding is that something required is missing."
    )
    issue: str = Field(
        description="What is wrong, in one or two sentences, written for a reviewer."
    )
    suggested_fix: str = Field(default="", description="How to correct it.")
    severity: str = Field(default="advisory", description="blocker or advisory.")
    concern: str = Field(
        default="",
        description="Only when the question names more than one concern: which of them this "
        "finding is. Use the exact word given. Leave empty otherwise.",
    )
    confidence: int = Field(ge=0, le=100, description="0-100. Be honest: a low number is useful.")
    citations: list[ReportedCitation] = Field(
        default_factory=list,
        description="The evidence. At least one unless nothing in the evidence block bears on it.",
    )


class CheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[ReportedFinding] = Field(default_factory=list)


# Standing rules for every check. Stated once so they cannot drift apart between checks, and so the
# grounding contract is visible in the prompt the model actually reads.
_SHARED_OPEN = """Rules that apply to every finding you report:

  - Quote the offending text VERBATIM from <piece>. An approximate quote cannot be located in the
    document and the finding will be rejected as unverifiable."""

# Only for a check that is given an evidence block. Telling a check with no <evidence> to cite from
# <evidence> is an instruction it cannot follow, and the model's choices are then to invent a source
# or to report nothing -- both worse than saying plainly what it may cite.
_SHARED_EVIDENCE = """
  - Cite evidence VERBATIM from <evidence>, using the source_id exactly as given. Do not paraphrase a
    passage and do not cite a source that is not in the evidence block.
  - If the evidence does not let you establish the problem, report the finding with no citations and
    a low confidence rather than inventing support. "I cannot verify this" is a useful answer and is
    handled correctly downstream; a fabricated citation is not."""

# For a check whose authority is the piece and the measurements taken from it.
_SHARED_MEASUREMENT = """
  - You have no source documents to cite for this question, and you are not expected to find any.
    Where a finding rests on a measurement from <tool_results>, cite it: set the citation's
    source_id to the tool's name exactly as given, kind to "measurement", and passage to the figure
    you relied on, copied as reported (for example "risk_benefit_word_ratio: 0.198"). A figure no
    tool reported will be rejected.
  - A finding that rests only on reading the piece, with no measurement behind it, has nothing to
    cite. Report it anyway, with no citations and an honest confidence. It will be marked
    unverified and sent to a senior reviewer, which is the right outcome -- but prefer a
    measurement where one applies, because a finding a reviewer can check lands better than one
    they have to take on trust."""

_SHARED_CLOSE = """
  - Do not report a problem you are not seeing. An empty findings list is the right answer for
    compliant copy, and a false alarm costs a reviewer's trust in every other finding.
  - Judge only your assigned question. Another reviewer is checking the others."""


@dataclass(frozen=True)
class CheckPrompt:
    """One specialist check's question and framing."""

    name: str
    version: str
    tier: str  # "strong" or "fast"
    system: str
    question: str

    # What this check is given, stated as four separate facts rather than one "document level"
    # flag. That flag used to mean all four at once, and the checks do not actually agree: a
    # fair-balance judgement is about the piece as a whole *and* needs the claims in front of it,
    # which the single flag made impossible to express.
    #
    # Every finding is either an annotation on a claim or a comment on the piece, so a check that
    # cannot see the claims can only ever produce the second kind.
    shows_claims: bool = True
    # Whether to retrieve from the tenant's corpus at all. A check that is not comparing the copy
    # against a source does not need evidence, and eight irrelevant passages are not free.
    needs_evidence: bool = True
    # Whether ingest's layout measurements are included.
    needs_layout: bool = False
    # Whether the check still runs on a piece that yielded no claims. A piece with nothing
    # extractable can still be missing its safety information.
    runs_without_claims: bool = False

    # The check_type values this check's findings may carry, most serious first. Empty means the
    # findings are simply the check's own name, which is the usual case.
    #
    # One node answering two questions is not the same as one policy covering both: a tenant grades
    # by check_type, and ``presentation`` exists precisely because two questions share an input,
    # not because two concerns share a severity. Acme treats imbalance as editorial and a missing
    # ISI as a blocker, and that difference is the whole of golden cases 007 and 008.
    #
    # Ordered most serious first because that is the fallback for a finding the model did not tag:
    # a mislabelled finding should land on the concern a tenant is likelier to take seriously, not
    # the lenient one.
    emits: tuple[str, ...] = ()

    def check_type_for(self, concern: str) -> str:
        """Which check_type a reported finding carries."""
        if not self.emits:
            return self.name
        return concern if concern in self.emits else self.emits[0]

    def build(
        self,
        *,
        document: str,
        claims_block: str,
        evidence_block: str,
        extra: str = "",
        tools_block: str = "",
    ) -> str:
        parts = [
            self.question,
            "",
            _SHARED_OPEN
            + (_SHARED_EVIDENCE if self.needs_evidence else _SHARED_MEASUREMENT)
            + _SHARED_CLOSE,
        ]
        if extra:
            parts += ["", extra]
        if tools_block:
            parts += ["", tools_block]
        parts += [
            "",
            f"<piece>\n{document}\n</piece>",
        ]
        if self.shows_claims:
            parts += ["", f"<claims>\n{claims_block}\n</claims>"]
        if self.needs_evidence:
            parts += ["", f"<evidence>\n{evidence_block}\n</evidence>"]
        parts += ["", "Return JSON matching the schema."]
        return "\n".join(parts)


SUBSTANTIATION = CheckPrompt(
    name="substantiation",
    version="substantiation-v2",
    tier="strong",
    system=(
        "You are a regulatory reviewer checking whether promotional claims are supported by the "
        "approved label and the cited evidence, at the strength the copy states, and whether each "
        "reference the copy cites reports what the claim citing it says."
    ),
    # v2 merges the former ``reference_support`` check into this one. Both asked their question of
    # the same retrieved passages -- the label, the study summaries and the cleared-claim library all
    # sit in one tenant collection -- and both routinely landed on the same claim, leaving the editor
    # to merge near-duplicate findings. One reviewer, one evidence block, one finding per problem.
    #
    # Trade-off: a reviewer can no longer tell the two failure families apart from ``check_type``
    # alone -- "the claim overstates the evidence" and "the footnote points at the wrong study" are
    # now both ``substantiation``. They need different fixes, so the distinction has to survive in
    # the finding's ``issue`` and ``suggested_fix``, which is why the question below asks for it
    # explicitly. It was worth giving up: both tenants already graded the two identically (blocker),
    # so no routing or severity decision depended on the split.
    question="""For each claim, decide whether the evidence supports it AS STATED, and whether any
reference the claim cites reports what the claim says.

Two kinds of failure. Report either one; a claim can have both.

**A. The claim overstates the evidence.** In order of how often they matter:

  1. A result from a subgroup presented as a general result. If the evidence reports a figure for a
     subgroup -- prior treatment, disease severity, age band -- and the claim applies it to patients
     generally, that is an overstatement even though the number itself is real. This is the single
     most common way a technically accurate claim misleads.
  2. A stronger endpoint than was measured, or a longer timeframe than was studied.
  3. A dropped comparator, so a relative result reads as an absolute one.
  4. A dropped population limit, so a result in a defined population reads as applying to everyone.
  5. A claim with no support anywhere in the evidence. Report it with no citation and say you could
     not verify it. Do not treat absence of evidence as support.

**B. The cited reference does not report the claim.** For a claim carrying a reference marker, check
that the cited study reports the endpoint, the population and the timepoint the claim states. A
citation fails if the study measured a different endpoint (PASI-90 where the study reports PASI-75),
reported a different timepoint (Week 12 where the study assessed Week 16), studied a different
population, or used a different comparator. The study being broadly about the product is not enough:
the specific thing claimed has to be the thing measured.

Reference markers in the claims map to the numbered reference list in the piece, and from there to
the sources in the evidence block by study id.

These two failures need different corrections, so say which one you found. A claim the evidence
cannot support at any strength has to be weakened or withdrawn; a claim whose footnote points at the
wrong study may only need the right citation. Make ``suggested_fix`` say which.

Some claims are marked ``cleared="yes"``. The company's review committee has already approved that
wording against the evidence, so do not re-judge it under A -- report it only if its citation fails
under B.""",
)

OFF_LABEL = CheckPrompt(
    name="off_label",
    version="off_label-v1",
    tier="strong",
    system=(
        "You are a regulatory reviewer checking for promotion outside the approved label, including "
        "by implication."
    ),
    question="""Decide whether the piece promotes use outside the approved indication.

Explicit off-label promotion is rare and easy. What you are mainly looking for is IMPLICATION:

  - imagery, captions or testimonials depicting a population the label does not cover -- children
    where the indication is adults, for instance
  - language widening the population: "the whole family", "at every age", "anyone with"
  - a dropped population limit, so a result in a defined group reads as general
  - an indication, dose or duration the label does not carry

Judge the piece as a reader encounters it, including captions and copy adjacent to a claim, not the
claim sentence in isolation. A claim that is itself pre-approved can still be off-label by
implication once it is placed next to an image or a line that widens it -- the context is exactly
what was not reviewed when the claim was approved.""",
)

PRESENTATION = CheckPrompt(
    name="presentation",
    version="presentation-v1",
    tier="fast",
    system=(
        "You are a regulatory reviewer checking how a promotional piece presents risk: whether the "
        "required safety information is there and readable, and whether it is given weight "
        "comparable to the benefit claims."
    ),
    # One node, two concerns, because they share every input and differ only in the question asked
    # of it. Both read the piece, the claims, ingest's layout measurements and the company's own
    # validators; neither needs the label or the studies. Run apart they made two model calls over
    # near-identical context and routinely reported the same buried ISI twice, leaving the editor
    # to merge it.
    #
    # Trade-off: the two concerns keep their own check_type, so a tenant still grades them
    # separately -- see ``emits``. Merging the policy as well as the node would have cost the
    # clearest thing the two tenants demonstrate, which is Acme calling imbalance editorial while
    # Borealis calls it blocking.
    question="""Two questions about how this piece presents risk. Answer both.

**safety_prominence -- is the required safety information there and readable?**
An absolute question about the safety content itself. The realistic failure is not absence but
burial: safety information reduced to a single line, placed after the footnotes or the references,
or separated from the efficacy content by several pages. Present-but-buried is still misleading, so
report it. Also report content that is present but too thin to let a reader weigh the risk.

**fair_balance -- is the risk given weight comparable to the benefit claims?**
A relative question, answered claim by claim. For each claim below: is the risk that qualifies it
close enough, prominent enough and plainly enough worded that a reader meets both? Consider how a
reader actually encounters the piece, how much space risk gets beside the benefit content, and
whether the language minimises it.

A piece can fail one and pass the other, and they need different corrections, so **set ``concern``
on every finding** to exactly ``safety_prominence`` or ``fair_balance``.

Attach each finding to the claim it concerns with claim_id. Leave claim_id empty only for a problem
genuinely about the piece as a whole -- the safety block being too short, for instance, belongs to
no single claim.

Neither question is about whether the safety text is accurate, or whether the claims are supported.
Another reviewer is checking those. You are given no source evidence here because these questions
do not need any: everything they turn on is in the piece, the claims, and the measurements below.

A <tool_results> block carries measurements from the company's own validators. Treat those numbers
as established fact -- they were computed by qualified code, not estimated, and you must not
contradict one or recompute it yourself. What they cannot do is weigh up. A piece can clear every
threshold and still bury its risk information, and a piece can miss one and still read fairly; that
judgement is yours, and it is the reason you are being asked. Where a measurement is
"inconclusive", the company's code could not take it: that is not a pass. Say so in any finding
that depends on it, and lower your confidence accordingly.""",
    shows_claims=True,
    needs_evidence=False,
    needs_layout=True,
    runs_without_claims=True,
    emits=("safety_prominence", "fair_balance"),
)

CHECKS: dict[str, CheckPrompt] = {c.name: c for c in (SUBSTANTIATION, OFF_LABEL, PRESENTATION)}

# Exposed so the audit log and eval provenance can record every prompt version in one place.
PROMPT_VERSIONS: dict[str, str] = {c.name: c.version for c in CHECKS.values()}

# Which node produces a given check_type. Not the identity map any more: ``presentation`` answers
# two concerns, so the step view and the eval oracle both need to be told rather than assuming a
# finding's check_type names the node that made it.
NODE_FOR_CHECK_TYPE: dict[str, str] = {
    concern: check.name for check in CHECKS.values() for concern in (check.emits or (check.name,))
}
