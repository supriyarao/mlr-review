"""The four specialist checks, run through one shared implementation.

**LLM decisions.** Each check asks a different question of the same piece, so what differs between
them is their prompt, their evidence, and which claims they see -- not their mechanics. One runner
parameterised by :class:`~mlr.prompts.checks.CheckPrompt` keeps those differences in one visible place
and means a fix to grounding, retrieval or error handling lands in all five at once.

Cost tiering: substantiation and off_label use the strong model because they are inferential --
"is this subgroup result being generalised" is the hardest judgement in the system. fair_balance and
safety_prominence use the fast model because they mostly reason over layout numbers that ingest
already computed.
"""

from __future__ import annotations

import json
import re
from typing import Any

import mlr.tools  # noqa: F401  -- importing registers the use case's tools
from mlr.nodes.base import node_run
from mlr.prompts.checks import CHECKS, CheckPrompt, CheckResponse
from mlr.prompts.rules import schema_failure_gate
from mlr.schemas import Citation, Claim, Finding
from platform_core import llm, tools
from platform_core.retrieval import Passage, TenantRetriever

# Evidence passages per check. Enough for the model to find the relevant study without filling the
# context with near-duplicates -- and a check that cannot establish a problem from eight passages
# should say so rather than be given eighty.
EVIDENCE_K = 8

# substantiation answers two questions off one retrieval -- whether the evidence supports the claim,
# and whether the study it cites reports it -- so it gets a larger share. Before the two checks were
# merged they drew eight passages each; twelve for the one check is still a net saving, and starving
# it would show up as "could not verify" on claims whose study simply fell outside the window.
EVIDENCE_K_BY_CHECK: dict[str, int] = {"substantiation": 12}

# The doc_type the studies are indexed under, and the share of the window they get first refusal on.
#
# substantiation's second question names a specific study, so the study has to be in the window or
# the question cannot be answered. Giving studies first refusal makes that likely instead of
# accidental. The rest is left for the label, which is what the population and indication limits in
# failure family A are judged against -- take the whole window for studies and the check loses its
# ability to say "this result is real but the population is wider than the label allows".
STUDY_DOC_TYPE = "reference"
STUDY_SLOTS = 8

# Document types an evidence check must never be shown.
#
# A cleared claim is not evidence. Asking "does the evidence support this claim?" against a pool
# containing the cleared-claim library invites the circular answer -- supported, because wording
# like it was approved once before -- and the approval said nothing about whether the underlying
# study backs it. Matching against cleared claims is the preapproved_filter node's job, done
# separately and earlier, with doc_type narrowed the other way.
CIRCULAR_DOC_TYPES: tuple[str, ...] = ("preapproved_claim",)

# Tools each check calls before it reasons. See platform_core/tools.py for why the node invokes
# these rather than offering them for the model to choose: a measurement this check always needs is
# not a decision, and making the model ask for it buys a round trip and nondeterminism for nothing.
# The registry is the seam -- a model-driven call would be a change of caller, not of design.
TOOLS_BY_CHECK: dict[str, tuple[str, ...]] = {
    # Two, in order, both feeding the one model call that follows. They answer the check's two
    # concerns separately because a customer would own them separately.
    "presentation": ("safety_prominence_validator", "fair_balance_validator"),
}


# Study identifiers as they appear in a piece's reference list: ZEL-302, NEX-114.
_STUDY_ID = re.compile(r"\b[A-Z]{2,5}-\d{2,4}\b")

# A numbered reference-list entry: "3. ZEL-303. Open-label extension, 52 weeks, 610 adults."
_REFERENCE_ENTRY = re.compile(r"^\s*\d+\.\s+(.*)$", re.MULTILINE)


def _claims_for(check: str, state: dict[str, Any]) -> list[Claim]:
    """Which claims a given check examines.

    The only non-obvious entry is substantiation, whose rule has two halves because the check does.
    A pre-approved claim skips the evidence question -- the committee has already done that work --
    but it does *not* skip the citation question: clearing the wording said nothing about the
    footnote later attached to it. So a cleared claim is still examined if it cites a reference, and
    ``_format_claims`` marks it so the prompt knows to judge only that half.

    Every other check sees every claim, because approved wording in a new context can still mislead.
    """
    claims: list[Claim] = list(state.get("claims") or [])
    if check == "substantiation":
        matched = set(state.get("preapproved_matches") or {})
        return [c for c in claims if c.id not in matched or c.reference_markers]
    return claims


def _study_ids(document: str) -> list[str]:
    """Study identifiers named anywhere in the piece, in order of first appearance.

    Read off the piece rather than the claims because that is where they are written: a claim
    carries a marker ("[3]"), and only the piece's reference list says that 3 means ZEL-303. The
    markers alone are useless as query terms -- "3" matches nothing -- so expanding the query with
    the study ids is what gives retrieval a lexical handle on the documents the copy is actually
    relying on.

    Taken from the whole document, always, not from the claims in scope: a reference list sits at
    the end of the piece, far from the claim citing it, and a study named only in a footnote is
    still a study the checker should have in front of it.
    """
    # Prefer the numbered reference list when the piece has one. Scanning the whole document also
    # picks up endpoint scales -- PASI-90 has the same shape as ZEL-302 and no general rule tells
    # them apart -- which is harmless as a query term but makes the list a poor thing to reason
    # about. The reference list is where a study is named as a study.
    entries = "\n".join(_REFERENCE_ENTRY.findall(document))
    for text in (entries, document):
        seen: list[str] = []
        for match in _STUDY_ID.finditer(text):
            if match.group(0) not in seen:
                seen.append(match.group(0))
        if seen:
            return seen
    return []


def _evidence_query(check: str, claims: list[Claim], state: dict[str, Any]) -> str:
    """What to retrieve for this check.

    Built from the claims under examination rather than from the whole document, so retrieval is
    aimed at the assertions being judged instead of at the piece's general topic.
    """
    if check == "safety_prominence":
        return "contraindications warnings adverse reactions important safety information"
    if check == "off_label":
        return (
            "indications and usage limitations of use population age paediatric adults "
            + " ".join(c.normalized_claim for c in claims[:4])
        )
    if check == "substantiation":
        # The claims under examination, plus every study the piece names. See the note on
        # docs/known-issues.md #5: this biases retrieval towards the cited studies, it does not
        # resolve a marker to a source, so a study can still miss the window.
        studies = " ".join(_study_ids(str(state.get("document") or "")))
        text = " ".join(c.normalized_claim or c.original_text for c in claims[:6])
        return f"{text} {studies}".strip() or "indications"
    return " ".join(c.normalized_claim or c.original_text for c in claims[:6]) or "indications"


def _format_evidence(passages: list[Passage]) -> str:
    if not passages:
        return "(no evidence retrieved for this tenant and product)"
    return "\n\n".join(
        f'<source id="{p.source_id}" title="{p.title}">\n{p.text}\n</source>' for p in passages
    )


def _format_claims(claims: list[Claim], cleared: set[str] | None = None) -> str:
    """The claims block. ``cleared`` marks claims the tenant's committee has already approved.

    Only substantiation passes it, and only because it is the one check whose question is partly
    waived for an approved claim. Marking in the block rather than filtering in ``_claims_for``
    keeps the waiver visible to the model instead of hiding it in Python.
    """
    if not claims:
        return "(no claims to examine)"
    cleared = cleared or set()
    lines = []
    for c in claims:
        refs = f" cites=[{', '.join(c.reference_markers)}]" if c.reference_markers else ""
        mark = ' cleared="yes"' if c.id in cleared else ""
        lines.append(
            f'<claim id="{c.id}" type="{c.claim_type}" section="{c.location}"{refs}{mark}>\n'
            f"  as written: {c.original_text}\n"
            f"  in full:    {c.normalized_claim}\n"
            f"</claim>"
        )
    return "\n".join(lines)


def _layout_block(state: dict[str, Any]) -> str:
    """Layout numbers for the two document-level checks, computed by ingest.

    When ingest found no section structure the numbers below would all read as zero, and a zero is
    a measurement: "the safety block is 0 characters" says the safety information is absent, which
    is a different claim from "nobody could measure it". So that case says so instead, and tells the
    check to judge on the text alone and report low confidence rather than inventing a verdict.
    """
    meta = state.get("doc_meta") or {}
    if not (meta.get("sections") or {}):
        return (
            "<layout>\n"
            "  No section structure could be read from this document, so none of the usual layout\n"
            "  measurements are available. They are UNKNOWN, not zero. Judge only from the text of\n"
            "  the piece itself, say in your finding that the layout could not be measured, and\n"
            "  report low confidence -- a reviewer will be asked to look at the original.\n"
            "</layout>"
        )
    rows = [
        ("safety section", meta.get("safety_section") or "not found"),
        ("safety block size", f"{meta.get('safety_block_chars', 0)} characters"),
        ("efficacy block size", f"{meta.get('efficacy_block_chars', 0)} characters"),
        ("safety / efficacy size ratio", meta.get("safety_to_efficacy_ratio", "n/a")),
        ("characters between them", meta.get("safety_distance_chars", "n/a")),
        ("safety appears after efficacy", meta.get("safety_after_efficacy", "n/a")),
        ("safety appears after the footnotes", meta.get("safety_after_footnotes", "n/a")),
        ("safety appears after the references", meta.get("safety_after_references", "n/a")),
        ("section order", ", ".join(meta.get("section_order") or []) or "n/a"),
    ]
    body = "\n".join(f"  {name}: {value}" for name, value in rows)
    return f"<layout>\n{body}\n</layout>"


# Label sections that carry safety content. Used to decide what a piece is obliged to mention, so
# it is matched against headings rather than body text.
_SAFETY_HEADING_MARKERS = (
    "boxed warning",
    "contraindication",
    "warnings and precautions",
    "adverse reactions",
)


def _label_safety_sections(retriever: TenantRetriever, state: dict[str, Any]) -> list[Passage]:
    """The safety sections of the approved label, fetched whole rather than searched.

    A completeness check asks what the label *contains*, so ranking is the wrong tool: the section
    a piece forgot to mention is precisely the one a relevance score would drop. See
    ``TenantRetriever.passages_of_type``.
    """
    passages = retriever.passages_of_type(
        "label",
        product_id=str(state.get("product_id") or ""),
        label_version=str(state.get("label_version") or ""),
    )
    return [
        p for p in passages if any(marker in p.title.lower() for marker in _SAFETY_HEADING_MARKERS)
    ]


def _label_safety_block(sections: list[Passage]) -> str:
    """What the label obliges this piece to carry, shown to the presentation check.

    Empty when no label section could be found, and it says so rather than rendering nothing --
    the same rule the layout block follows. "The label carries no safety sections" and "nobody
    could look" lead to opposite findings, and silence reads as the first.
    """
    if not sections:
        return (
            "<label_safety>\n"
            "  The approved label's safety sections could not be retrieved, so there is nothing to\n"
            "  compare this piece against. Do not report a completeness failure on this run; say\n"
            "  the comparison could not be made and report low confidence.\n"
            "</label_safety>"
        )

    parts = []
    for section in sections:
        heading = section.title.split("\u2014")[-1].strip() or section.title
        parts.append(f"  [{section.source_id}] {heading}\n    {section.text.strip()}")
    body = "\n\n".join(parts)
    return (
        "<label_safety>\n"
        "  The safety sections the approved label carries. A piece is expected to convey each of\n"
        "  these, in its own words and at whatever length suits the format -- not to reproduce\n"
        "  them. Cite the section id when you report one as missing.\n\n"
        f"{body}\n"
        "</label_safety>"
    )


def _retrieve(
    check: str, retriever: TenantRetriever, query: str, state: dict[str, Any], k: int
) -> list[Passage]:
    """The evidence a check is shown.

    No check sees the cleared-claim library: that pool answers "has this wording been approved
    before", which is the preapproved_filter node's question, and it is the wrong answer to every
    question asked here. Ruling it out is enough on its own for three of the four checks -- their
    queries are already worded like the source they need, so the right passages win the ranking
    once the cleared claims stop taking slots.

    substantiation is the exception and fills its window in two passes, studies first. Its query is
    claim text, and a claim reads more like the label's indication statement than like a study's
    endpoint table, so in a single ranked pass the studies lose -- which is fatal for a check whose
    second question is "does the study this claim cites report it".
    """
    scope: dict[str, Any] = {
        "product_id": str(state.get("product_id") or ""),
        "label_version": str(state.get("label_version") or ""),
    }
    if check != "substantiation":
        return retriever.search(query, **scope, exclude_doc_types=CIRCULAR_DOC_TYPES, k=k)

    studies = retriever.search(query, **scope, doc_type=STUDY_DOC_TYPE, k=min(STUDY_SLOTS, k))
    others = retriever.search(
        query,
        **scope,
        exclude_doc_types=(*CIRCULAR_DOC_TYPES, STUDY_DOC_TYPE),
        k=k - len(studies),
    )

    # Whichever side came up short, the other takes the slack: a tenant with two studies and a long
    # label should still get a full window rather than a window with four empty slots in it.
    short = k - len(studies) - len(others)
    if short > 0:
        seen = {p.source_id for p in studies}
        extra = retriever.search(query, **scope, doc_type=STUDY_DOC_TYPE, k=k)
        studies += [p for p in extra if p.source_id not in seen][:short]

    return studies + others


def _run_tools(
    check: str,
    state: dict[str, Any],
    claims: list[Claim],
    label_safety: list[Passage] | None = None,
) -> list[tools.ToolResult]:
    """Call the tools this check depends on. Failures come back as results, not exceptions."""
    payload = {
        "document": str(state.get("document") or ""),
        "doc_meta": dict(state.get("doc_meta") or {}),
        "claims": [c.model_dump(mode="json") for c in claims],
        # Headings only. The validator decides which topics are *required* from what the label
        # carries; it does not need the label's text, and sending it would be a lot of context for
        # a keyword scan.
        "label_headings": [p.title for p in (label_safety or [])],
    }
    return [tools.call(name, payload) for name in TOOLS_BY_CHECK.get(check, ())]


def _format_tools(results: list[tools.ToolResult]) -> str:
    """The tool block the model is shown.

    A failed call is rendered, not dropped. The model needs to know the measurement was attempted
    and is unavailable -- silence would read as "nothing to report", which is the same mistake the
    layout block makes when it hands over zeros instead of saying UNKNOWN.
    """
    if not results:
        return ""
    parts = []
    for result in results:
        body = (
            json.dumps(result.data, indent=2, sort_keys=True)
            if result.ok
            else f"UNAVAILABLE: {result.error}"
        )
        parts.append(f'<tool name="{result.tool}" version="{result.version}">\n{body}\n</tool>')
    return "<tool_results>\n" + "\n".join(parts) + "\n</tool_results>"


def run_check(check_name: str, state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """Run one specialist check and return its findings."""
    check: CheckPrompt = CHECKS[check_name]

    with node_run(state, check_name) as run:
        claims = _claims_for(check_name, state)
        if not check.runs_without_claims and not claims:
            run.output = {"claims": 0, "findings": 0, "note": "nothing in scope for this check"}
            return {}

        # A check that declares no need for evidence does not open the collection at all. Skipping
        # the call rather than discarding the result is the point: retrieval is the slowest thing in
        # the node, and eight passages a check was never going to use still cost context.
        passages: list[Passage] = []
        if check.needs_evidence:
            passages = _retrieve(
                check_name,
                TenantRetriever(str(state["tenant_id"])),
                _evidence_query(check_name, claims, state),
                state,
                EVIDENCE_K_BY_CHECK.get(check_name, EVIDENCE_K),
            )

        cleared = (
            set(state.get("preapproved_matches") or {}) if check_name == "substantiation" else set()
        )

        # The label's safety sections, for the checks that ask what a piece left out. Fetched by
        # kind rather than by relevance, and separate from `passages` above: this check still does
        # no semantic retrieval, it reads one specific document.
        label_safety: list[Passage] = []
        if check.needs_label_safety:
            label_safety = _label_safety_sections(TenantRetriever(str(state["tenant_id"])), state)

        tool_results = _run_tools(check_name, state, claims, label_safety)
        extra = ""
        if check.needs_layout:
            extra = _layout_block(state)
        if check.needs_label_safety:
            extra = f"{extra}\n\n{_label_safety_block(label_safety)}".strip()

        prompt_text = check.build(
            document=str(state.get("document") or ""),
            claims_block=_format_claims(claims, cleared),
            evidence_block=_format_evidence(passages),
            extra=extra,
            tools_block=_format_tools(tool_results),
        )

        try:
            result = llm.call_structured(
                check.tier,  # type: ignore[arg-type]
                node=check_name,
                prompt=prompt_text,
                schema=CheckResponse,
                prompt_version=check.version,
                system=check.system,
                fake=fake,
            )
        except llm.SchemaFailure as exc:
            run.error = str(exc)
            # The check did not run. Recorded as a gate failure so the run escalates: a check that
            # could not complete must never be indistinguishable from a check that found nothing.
            return {
                "errors": [f"{check_name}: {exc}"],
                "gate_results": [schema_failure_gate(check_name, exc)],
            }

        run.record(result)
        known_claims = {c.id for c in claims}
        citation_kind = "evidence" if check.needs_evidence else "measurement"
        findings: list[Finding] = []

        for i, reported in enumerate(result.parsed.findings, start=1):
            severity = (
                reported.severity if reported.severity in ("blocker", "advisory") else "advisory"
            )
            findings.append(
                Finding(
                    id=f"{check_name[:3]}-{i}",
                    # Usually the check's own name. A check answering more than one question
                    # declares them in ``emits`` and the model says which this finding is, because
                    # the tenant grades by check_type and the concerns are graded differently.
                    check_type=check.check_type_for(reported.concern),  # type: ignore[arg-type]
                    # A claim id the check was not given is dropped rather than trusted: it is either
                    # a hallucination or a reference to a claim another check owns.
                    claim_id=reported.claim_id if reported.claim_id in known_claims else "",
                    quote=reported.quote,
                    issue=reported.issue,
                    explanation=reported.explanation,
                    suggested_fix=reported.suggested_fix,
                    citations=[
                        Citation(
                            source_id=c.source_id,
                            passage=c.passage,
                            # Derived from what the check was given, not asked of the model: a
                            # check either had an evidence block to cite or it had measurements,
                            # and we already know which. Asking would add a field the model can
                            # get wrong, and a mislabelled citation fails grounding silently.
                            kind=citation_kind,
                            title=next(
                                (p.title for p in passages if p.source_id == c.source_id), ""
                            ),
                        )
                        for c in reported.citations
                    ],
                    severity_proposed=severity,  # type: ignore[arg-type]
                    confidence=reported.confidence,
                    model=result.usage.model,
                    prompt_version=result.prompt_version,
                )
            )

        run.output = {
            "claims_examined": len(claims),
            "evidence_passages": len(passages),
            "findings": len(findings),
            "by_concern": {
                concern: sum(1 for f in findings if f.check_type == concern)
                for concern in (check.emits or (check_name,))
            },
            "retry_needed": bool(result.recovered_from),
            # Which of the company's own code ran, what version, and what it said. An auditor
            # asking "did our validator run on this piece" must not have to infer it.
            "tools": [
                {
                    "tool": t.tool,
                    "version": t.version,
                    "ok": t.ok,
                    "latency_ms": t.latency_ms,
                    "summary": t.data.get("summary", "") if t.ok else t.error,
                    "failed": t.data.get("failed", []) if t.ok else [],
                    "inconclusive": t.data.get("inconclusive", []) if t.ok else [],
                }
                for t in tool_results
            ],
        }
        run.citations = [
            {"source_id": c.source_id, "passage": c.passage[:160]}
            for f in findings
            for c in f.citations
        ]
        # Tool output goes into state as well as the audit log, because the grounding gate verifies
        # a "measurement" citation against it: a finding may cite a number its validator really
        # reported, and nothing else.
        published = [
            {"tool": t.tool, "version": t.version, "ok": t.ok, "data": t.data, "error": t.error}
            for t in tool_results
        ]
        return {"findings": findings, **({"tool_results": published} if published else {})}
