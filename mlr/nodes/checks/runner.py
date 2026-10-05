"""The five specialist checks, run through one shared implementation.

**LLM decisions.** Each check asks a different question of the same piece, so what differs between
them is their prompt, their evidence, and which claims they see -- not their mechanics. One runner
parameterised by :class:`~mlr.prompts.checks.CheckPrompt` keeps those differences in one visible place
and means a fix to grounding, retrieval or error handling lands in all five at once.

Cost tiering: substantiation, reference_support and off_label use the strong model because they are
inferential -- "is this subgroup result being generalised" is the hardest judgement in the system.
fair_balance and safety_prominence use the fast model because they mostly reason over layout numbers
that ingest already computed.
"""

from __future__ import annotations

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


def _claims_for(check: str, state: dict[str, Any]) -> list[Claim]:
    """Which claims a given check examines.

    The only non-obvious entry is substantiation, which skips pre-approved claims: the committee has
    already done that work. Every other check still sees them, because approved wording in a new
    context can still mislead.
    """
    claims: list[Claim] = list(state.get("claims") or [])
    if check == "substantiation":
        matched = set(state.get("preapproved_matches") or {})
        return [c for c in claims if c.id not in matched]
    if check == "reference_support":
        # Nothing to verify for a claim that cites nothing. The missing-reference case is a tenant
        # business rule, not this check.
        return [c for c in claims if c.reference_markers]
    return claims


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
    return " ".join(c.normalized_claim or c.original_text for c in claims[:6]) or "indications"


def _format_evidence(passages: list[Passage]) -> str:
    if not passages:
        return "(no evidence retrieved for this tenant and product)"
    return "\n\n".join(
        f'<source id="{p.source_id}" title="{p.title}">\n{p.text}\n</source>' for p in passages
    )


def _format_claims(claims: list[Claim]) -> str:
    if not claims:
        return "(no claims to examine)"
    lines = []
    for c in claims:
        refs = f" cites=[{', '.join(c.reference_markers)}]" if c.reference_markers else ""
        lines.append(
            f'<claim id="{c.id}" type="{c.claim_type}" section="{c.location}"{refs}>\n'
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


def run_check(check_name: str, state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """Run one specialist check and return its findings."""
    check: CheckPrompt = CHECKS[check_name]

    with node_run(state, check_name) as run:
        claims = _claims_for(check_name, state)
        if not check.document_level and not claims:
            run.output = {"claims": 0, "findings": 0, "note": "nothing in scope for this check"}
            return {}

        retriever = TenantRetriever(str(state["tenant_id"]))
        passages = retriever.search(
            _evidence_query(check_name, claims, state),
            product_id=str(state.get("product_id") or ""),
            label_version=str(state.get("label_version") or ""),
            k=EVIDENCE_K,
        )

        prompt_text = check.build(
            document=str(state.get("document") or ""),
            claims_block=_format_claims(claims),
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
