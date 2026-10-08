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
        description="What is wrong, in one or two sentences, written for a reviewer who already "
        "knows the vocabulary."
    )
    explanation: str = Field(
        description="The same finding in plain English, for someone who is not a regulatory "
        "specialist -- a brand manager or an agency writer. Two or three sentences covering: what "
        "triggered this, what the piece actually says, and why that is a problem. Name the rule or "
        "the check in ordinary words. No jargon: not 'unsubstantiated at the stated strength' but "
        "'the study does not show this'. Do not repeat the issue sentence verbatim."
    )
    suggested_fix: str = Field(
        default="",
        description="What to change, concretely enough to act on. Where you can, give the wording "
        "that would be acceptable rather than describing it.",
    )
    severity: str = Field(default="advisory", description="blocker or advisory.")
    concern: str = Field(
        default="",
        description="Only when the question names more than one concern: which of them this "
        "finding is. Use the exact word given. Leave empty otherwise.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Only the exact tag words the question tells you to attach, and only when the "
        "condition it describes holds. Leave empty otherwise. Do not invent tags.",
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
  - Every finding needs all three of issue, explanation and suggested_fix. They are read by
    different people: the issue by a regulatory reviewer, the explanation by whoever has to change
    the copy, and the fix by both. An explanation that just restates the issue in the same
    vocabulary has not done its job.
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
    # Whether the approved label's safety sections are included. Distinct from needs_evidence: this
    # fetches one specific document by kind, with no ranking, because a completeness check asks
    # what the label *contains* and relevance scoring would drop the section nobody mentioned --
    # which is the one being looked for.
    needs_label_safety: bool = False
    # Whether the check still runs on a piece that yielded no claims. A piece with nothing
    # extractable can still be missing its safety information.
    runs_without_claims: bool = False

    # Tags this check is allowed to attach. A tag lets a tenant grade a subset of one check type
    # differently -- see SeverityRule.when_tag -- and is filtered against this list on the way in,
    # so a model cannot mint a tag and with it a severity nobody configured.
    tags: tuple[str, ...] = ()

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
    version="substantiation-v4",
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

Three kinds of failure. Report any that apply; a claim can have more than one.

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

**B. The claim compares figures that were not measured the same way.** A comparison between two
products is only meaningful if the numbers come from the same trial, or from trials that match on
everything which moves the number. When a claim sets one product's figure against another's, check
each of these against the evidence and report any that differ:

  - **population** -- severity, setting, age band, comorbidity. A sicker or hospitalised population
    records a worse result under any treatment, so a comparison across them measures the patients,
    not the products.
  - **treatment duration and the timepoint** the endpoint was assessed at.
  - **the analysis population** the figure is drawn from. A per-protocol or clinically evaluable
    figure is systematically higher than an intent-to-treat one from the same trial, so pairing one
    product's per-protocol number with another's intent-to-treat number widens the gap twice over.
  - **the endpoint definition** itself.

Two separate trials are not a head-to-head trial, and a superiority claim needs one. A footnote
disclosing the differences does not cure the comparison: the piece still draws the conclusion, and
the headline is what a reader takes away. Report it even when the piece discloses the mismatch
itself -- especially then, because the evidence that refutes the claim is already in the document.

**C. The cited reference does not report the claim.** For a claim carrying a reference marker, check
that the cited study reports the endpoint, the population and the timepoint the claim states. A
citation fails if the study measured a different endpoint (PASI-90 where the study reports PASI-75),
reported a different timepoint (Week 12 where the study assessed Week 16), studied a different
population, or used a different comparator. The study being broadly about the product is not enough:
the specific thing claimed has to be the thing measured.

Reference markers in the claims map to the numbered reference list in the piece, and from there to
the sources in the evidence block by study id.

These failures need different corrections, so say which one you found. A claim the evidence cannot
support at any strength has to be weakened or withdrawn; a cross-study comparison has to drop the
comparative framing or be replaced by a head-to-head trial; a claim whose footnote points at the
wrong study may only need the right citation. Make ``suggested_fix`` say which.

Some claims are marked ``cleared="yes"``. The company's review committee has already approved that
wording against the evidence, so do not re-judge it under A -- report it only if its citation fails
under B.""",
)

OFF_LABEL = CheckPrompt(
    name="off_label",
    version="off_label-v3",
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
what was not reviewed when the claim was approved.

**The market this piece is for.** A <tool_results> block carries a lookup against the company's
regulatory system: whether this product is approved in this market, and for which indications.
Approval is per market, so the same piece can be acceptable in one and unacceptable in another, and
the label in <evidence> is not the whole answer.

  - ``approved``: judge the claims against the indications the lookup lists for THIS market, which
    may be narrower than the label you were given. A claim for an indication not on that list is
    off-label.
  - ``not_approved``: every claim of effectiveness or safety for this product is promotion of an
    unapproved product. Report each one and attach the tag ``unapproved_market``. This is not a
    wording problem and ``suggested_fix`` should not suggest rewording: the material cannot run in
    this market at all. Say which market, and say the product is not approved there.
  - ``unknown``: the record is missing, which is NOT the same as unapproved. Do not report the
    piece as promoting an unapproved product. Say the approval status could not be established for
    this market and report low confidence, so a person checks it.

Attach ``unapproved_market`` only in the ``not_approved`` case, and only to findings about that
product.""",
    tags=("unapproved_market",),
)

PRESENTATION = CheckPrompt(
    name="presentation",
    version="presentation-v3",
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

**safety_completeness -- does the piece carry the risks the label obliges it to?**
A question about coverage, answered against the <label_safety> block below, which lists the safety
sections of the approved label. Report a finding for each required topic the piece does not convey
-- a Boxed Warning the piece never mentions, contraindications it omits, warnings and precautions
it leaves out, adverse reactions it does not state.

This is not a test of reproduction. A piece need not quote the label, and need not be exhaustive at
any given length; it has to let a reader meet the risk the label describes. A piece that lists only
the mild, common reactions and is silent on everything serious has failed this question even though
it has a safety section, and that failure is invisible to the other two: the section exists, it is
long enough, and it sits in the right place.

Where a topic is conveyed in the piece's own words, that is a pass -- say so by not reporting it.
Cite the label section id when you report a topic as missing.

**fair_balance -- is the risk given weight comparable to the benefit claims?**
A relative question, answered claim by claim. For each claim below: is the risk that qualifies it
close enough, prominent enough and plainly enough worded that a reader meets both? Consider how a
reader actually encounters the piece, how much space risk gets beside the benefit content, and
whether the language minimises it.

A piece can fail one and pass the others, and they need different corrections, so **set
``concern`` on every finding** to exactly ``safety_prominence``, ``safety_completeness`` or
``fair_balance``.

Attach each finding to the claim it concerns with claim_id. Leave claim_id empty only for a problem
genuinely about the piece as a whole -- the safety block being too short, for instance, belongs to
no single claim.

None of these questions is about whether the claims are supported, or whether the figures in the
safety text are right. Another reviewer is checking those. The only source you are given is the
label's safety sections, and they are here for one purpose: to say what this piece was obliged to
carry. Everything else turns on the piece, the claims and the measurements below.

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
    needs_label_safety=True,
    runs_without_claims=True,
    # safety_prominence stays first. It is the fallback for a finding the model did not tag, and
    # moving it would silently relabel every untagged finding in golden cases 004, 007 and 008 --
    # a change to three passing cases made as a side effect of adding a fourth concern. The two
    # are comparable in seriousness anyway; fair_balance is the mild one and stays last.
    emits=("safety_prominence", "safety_completeness", "fair_balance"),
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
