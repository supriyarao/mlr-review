"""Prompts for the four specialist checks.

One module rather than four, because they share a response schema and a set of standing rules. The
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
_SHARED = """Rules that apply to every finding you report:

  - Quote the offending text VERBATIM from <piece>. An approximate quote cannot be located in the
    document and the finding will be rejected as unverifiable.
  - Cite evidence VERBATIM from <evidence>, using the source_id exactly as given. Do not paraphrase a
    passage and do not cite a source that is not in the evidence block.
  - If the evidence does not let you establish the problem, report the finding with no citations and
    a low confidence rather than inventing support. "I cannot verify this" is a useful answer and is
    handled correctly downstream; a fabricated citation is not.
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
    # Whether the check looks at individual claims or at the piece as a whole.
    document_level: bool = False

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
            _SHARED,
        ]
        if extra:
            parts += ["", extra]
        if tools_block:
            parts += ["", tools_block]
        parts += [
            "",
            f"<piece>\n{document}\n</piece>",
        ]
        if not self.document_level:
            parts += ["", f"<claims>\n{claims_block}\n</claims>"]
        parts += [
            "",
            f"<evidence>\n{evidence_block}\n</evidence>",
            "",
            "Return JSON matching the schema.",
        ]
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

FAIR_BALANCE = CheckPrompt(
    name="fair_balance",
    version="fair_balance-v1",
    tier="fast",
    system=(
        "You are a regulatory reviewer checking whether risk information is presented with "
        "comparable prominence to benefit claims."
    ),
    question="""Decide whether the risk information in this piece is presented comparably to the
benefit claims.

Consider how a reader actually meets the piece: whether risk appears near the claims it qualifies or
only at the end, how much space it gets relative to the benefit content, and whether the language
minimises it. Layout metrics for this piece are given below if available.

This is a judgement about prominence, not about whether the safety text is accurate.

A <tool_results> block may carry measurements from the company's own fair-balance validator. Treat
those numbers as established fact -- they were computed by qualified code, not estimated, and you
must not contradict one or recompute it yourself. What they cannot do is weigh up. A piece can clear
every threshold and still bury its risk information, and a piece can miss one and still read fairly;
that judgement is yours, and it is the reason you are being asked.

Where a measurement is "inconclusive", the company's code could not take it. That is not a pass. Say
so in any finding that depends on it, and lower your confidence accordingly.""",
    document_level=True,
)

SAFETY_PROMINENCE = CheckPrompt(
    name="safety_prominence",
    version="safety_prominence-v1",
    tier="fast",
    system=(
        "You are a regulatory reviewer checking that required safety information is present and "
        "readable where a reader will meet it."
    ),
    question="""Decide whether the required safety information is present, complete enough to be
useful, and positioned where a reader will encounter it alongside the claims it qualifies.

The realistic failure is not absence but burial: safety information reduced to a single line, placed
after the footnotes, or separated from the efficacy content by several pages. Present-but-buried is
still misleading, so report it. Layout metrics are given below.""",
    document_level=True,
)

CHECKS: dict[str, CheckPrompt] = {
    c.name: c for c in (SUBSTANTIATION, OFF_LABEL, FAIR_BALANCE, SAFETY_PROMINENCE)
}

# Exposed so the audit log and eval provenance can record every prompt version in one place.
PROMPT_VERSIONS: dict[str, str] = {c.name: c.version for c in CHECKS.values()}
