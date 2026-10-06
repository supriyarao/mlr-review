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

import re
from typing import Any

from mlr.nodes.base import node_run
from mlr.prompts.checks import CHECKS, CheckPrompt, CheckResponse
from mlr.prompts.rules import schema_failure_gate
from mlr.schemas import Citation, Claim, Finding
from platform_core import llm
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
    """Layout numbers for the two document-level checks, computed by ingest."""
    meta = state.get("doc_meta") or {}
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


def run_check(check_name: str, state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """Run one specialist check and return its findings."""
    check: CheckPrompt = CHECKS[check_name]

    with node_run(state, check_name) as run:
        claims = _claims_for(check_name, state)
        if not check.document_level and not claims:
            run.output = {"claims": 0, "findings": 0, "note": "nothing in scope for this check"}
            return {}

        retriever = TenantRetriever(str(state["tenant_id"]))
        passages = _retrieve(
            check_name,
            retriever,
            _evidence_query(check_name, claims, state),
            state,
            EVIDENCE_K_BY_CHECK.get(check_name, EVIDENCE_K),
        )

        cleared = (
            set(state.get("preapproved_matches") or {}) if check_name == "substantiation" else set()
        )
        prompt_text = check.build(
            document=str(state.get("document") or ""),
            claims_block=_format_claims(claims, cleared),
            evidence_block=_format_evidence(passages),
            extra=_layout_block(state) if check.document_level else "",
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
        findings: list[Finding] = []

        for i, reported in enumerate(result.parsed.findings, start=1):
            severity = (
                reported.severity if reported.severity in ("blocker", "advisory") else "advisory"
            )
            findings.append(
                Finding(
                    id=f"{check_name[:3]}-{i}",
                    check_type=check_name,  # type: ignore[arg-type]
                    # A claim id the check was not given is dropped rather than trusted: it is either
                    # a hallucination or a reference to a claim another check owns.
                    claim_id=reported.claim_id if reported.claim_id in known_claims else "",
                    quote=reported.quote,
                    issue=reported.issue,
                    suggested_fix=reported.suggested_fix,
                    citations=[
                        Citation(
                            source_id=c.source_id,
                            passage=c.passage,
                            kind="evidence",
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
            "retry_needed": bool(result.recovered_from),
        }
        run.citations = [
            {"source_id": c.source_id, "passage": c.passage[:160]}
            for f in findings
            for c in f.citations
        ]
        return {"findings": findings}
