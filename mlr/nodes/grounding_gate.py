"""Verify that every finding is grounded in something checkable.

**Deterministic code. No model involved** -- deliberately. If the thing deciding whether to trust
model output were itself model output, the check would have no independent value.

Two conditions, both required:

1. **The quote must appear verbatim in the submitted piece.** A finding whose quote cannot be located
   is describing text that is not there. This applies to rule findings as much as to core findings,
   which is what stops an injected rule fabricating a violation.
2. **The citation must be verifiable**, by one of three modes:

   ``evidence``     the passage appears verbatim in a retrieved source **belonging to this tenant**
   ``rule``         the rule id and version were in this run's applied set
   ``doc_clause``   the clause appears verbatim in a reference document owned by this tenant
   ``measurement``  the named tool ran on this piece, and the citation either quotes a figure it
                    reported or quotes its output verbatim (categorical measurements carry no
                    figures -- "missing: contraindications" is a measurement with no number in it)

The fourth mode exists because some questions have no source to cite. "Is the risk given comparable
weight to the benefit" is answered from the piece and from measurements, not from the label -- so
the ``presentation`` check is given no evidence block at all, and without this its findings would
every one be ungrounded and escalate every piece. What grounds them instead is stronger than a
retrieved passage: a number the company's own code computed, recorded with the tool's version, and
recomputable by hand.

An ungrounded finding is **never dropped.** It is marked ``grounded=False`` and forces escalation.
Discarding it would turn "I could not verify this" into "there is no problem here", which is the worst
thing a review system can do -- and it is a mistake no reviewer would ever see.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mlr.schemas import Finding, GateResult
from platform_core.retrieval import TenantRetriever
from platform_core.rules.deterministic import normalize_whitespace
from platform_core.tenancy import tenant_reference_docs_dir


def _quote_is_in_the_piece(finding: Finding, document: str) -> bool:
    """Is the offending text actually in the submitted document?

    An absence -- a missing disclaimer, an absent safety section -- has nothing to quote, so it passes
    this condition by definition. Demanding a quote would make every missing-X finding ungrounded.
    """
    if finding.is_absence:
        return True
    quote = normalize_whitespace(finding.quote)
    return bool(quote) and quote in normalize_whitespace(document)


def _verify_citations(
    finding: Finding,
    *,
    retriever: TenantRetriever,
    state: dict[str, Any],
    reference_docs: dict[str, str],
) -> str:
    """Return the grounding mode that verified this finding, or "" if none did."""
    applied = {(a.id, a.version) for a in (state.get("rules_applied") or [])}
    product_id = str(state.get("product_id") or "")
    measurements = _measurement_text(state)

    for citation in finding.citations:
        if citation.kind == "evidence":
            if citation.passage.strip() and retriever.contains_verbatim(
                citation.passage, product_id=product_id
            ):
                return "evidence"

        elif citation.kind == "rule":
            # A rule grounds a finding only if it was genuinely in force for this run. That is what
            # prevents a model naming a rule it was never given.
            if (citation.rule_id, citation.rule_version) in applied:
                return "rule"

        elif citation.kind == "measurement":
            # Same shape as the rule check above: the source must have genuinely run on this piece,
            # and the figure must be one it genuinely reported. A model cannot cite a measurement
            # into existence any more than it can cite a rule that was never applied.
            reported = measurements.get(citation.source_id)
            if reported is None:
                continue
            cited = _figures_in(citation.passage)
            # A figure cited must be one the tool reported. The `cited` test matters because an
            # empty set is a subset of anything, so "the ratio was too low" with no number in it
            # would otherwise verify against a tool that merely ran.
            if cited and cited <= _figures_in(reported):
                return "measurement"
            # Not every measurement is a number. A completeness check reports which topics are
            # missing -- `missing: ["contraindications", "warnings_and_precautions"]` -- and that
            # is as much a measurement as a ratio is, but it has no figures in it at all. Requiring
            # one sent every correctly-identified omission to the reviewer marked "could not be
            # verified", which reads as the system being unsure of a thing it measured exactly.
            #
            # The anti-fabrication property is kept by a different route: the quote has to appear
            # verbatim in what the tool actually emitted, so a model still cannot cite a
            # measurement into existence. The length floor stops a quote so short it would appear
            # in any tool's output from grounding a finding by coincidence.
            if _quotes_measurement(citation.passage, reported):
                return "measurement"

        elif citation.kind == "doc_clause":
            clause = normalize_whitespace(citation.passage)
            if clause and any(clause in body for body in reference_docs.values()):
                return "doc_clause"

    return ""


# Words short enough to appear in any tool's output carry no evidence that this tool reported
# anything in particular, so they are not counted. Two distinct long tokens is the floor.
_TERM = re.compile(r"[A-Za-z][A-Za-z_]{4,}")
_MIN_MEASUREMENT_TERMS = 2


def _terms_in(text: str) -> set[str]:
    """The substantive words in a string, lowercased. Underscores are kept, so
    ``warnings_and_precautions`` stays one token rather than three."""
    return {m.group(0).lower() for m in _TERM.finditer(text or "")}


def _quotes_measurement(passage: str, reported: str) -> bool:
    """Whether a citation names things the tool genuinely reported.

    Deliberately not a verbatim substring test. The tool's output is serialised with sorted keys,
    so quoting two fields that read as adjacent -- ``"missing": [...], "verdict": "fail"`` -- is
    not a contiguous substring of it, and asking a model to reproduce JSON byte for byte is a
    fragile thing to hang grounding on.

    What is checked instead is the same shape as the figures rule above: every substantive term in
    the citation has to be one the tool actually emitted. A model can reorder and re-punctuate; it
    cannot introduce a topic the tool never named, which is the property that matters.
    """
    cited = _terms_in(passage)
    if len(cited) < _MIN_MEASUREMENT_TERMS:
        return False
    return cited <= _terms_in(reported)


_FIGURE = re.compile(r"-?\d+(?:\.\d+)?")


def _figures_in(text: str) -> set[str]:
    """The numbers in a string, normalised so 0.20 and 0.2 compare equal."""
    out: set[str] = set()
    for raw in _FIGURE.findall(text or ""):
        try:
            out.add(repr(float(raw)))
        except ValueError:  # pragma: no cover - the pattern only matches numbers
            continue
    return out


def _measurement_text(state: dict[str, Any]) -> dict[str, str]:
    """What each tool reported on this run, keyed by the id a citation would name.

    Both ``name`` and ``name vX`` are accepted as the source_id, because a model asked to cite a
    tool will reasonably write either and the version is already recorded beside the result.
    """
    out: dict[str, str] = {}
    for result in state.get("tool_results") or []:
        if not result.get("ok"):
            continue
        body = json.dumps(result.get("data") or {}, sort_keys=True)
        out[str(result.get("tool", ""))] = body
        out[f"{result.get('tool', '')} v{result.get('version', '')}"] = body
    return out


def _load_reference_docs(tenant_id: str) -> dict[str, str]:
    """Whitespace-normalised reference documents for this tenant.

    Loaded once per run rather than per finding: a code of practice is small, and re-reading it for
    each citation would make the gate's cost scale with the number of findings.
    """
    directory = tenant_reference_docs_dir(tenant_id)
    if not directory.is_dir():
        return {}
    return {
        path.name: normalize_whitespace(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.md"))
    }


def verify(
    findings: list[Finding], state: dict[str, Any]
) -> tuple[list[Finding], dict[str, Any], list[GateResult]]:
    """Mark each finding with how it was grounded, and produce the gate's verdict.

    A part of the ``consolidate`` node rather than a node of its own."""
    if not findings:
        return (
            [],
            {"findings": 0, "grounded": 0},
            [
                GateResult(
                    gate="grounding",
                    passed=True,
                    details="no findings to verify",
                    reason_code="ungrounded",
                )
            ],
        )

    document = str(state.get("document") or "")
    tenant_id = str(state["tenant_id"])
    retriever = TenantRetriever(tenant_id)
    reference_docs = _load_reference_docs(tenant_id)

    verified: list[Finding] = []
    ungrounded_ids: list[str] = []
    reasons: dict[str, str] = {}

    for finding in findings:
        quote_ok = _quote_is_in_the_piece(finding, document)
        mode = (
            _verify_citations(
                finding, retriever=retriever, state=state, reference_docs=reference_docs
            )
            if quote_ok
            else ""
        )

        grounded = bool(mode)
        verified.append(
            finding.model_copy(
                update={"grounded": grounded, "grounded_by": mode if grounded else "none"}
            )
        )
        if not grounded:
            ungrounded_ids.append(finding.id)
            reasons[finding.id] = (
                "the quoted text does not appear in the submitted piece"
                if not quote_ok
                else "no citation could be verified against a source belonging to this tenant"
            )

    by_mode = {
        mode: sum(1 for f in verified if f.grounded_by == mode)
        for mode in ("evidence", "rule", "doc_clause", "measurement", "none")
    }
    passed = not ungrounded_ids

    gate = GateResult(
        gate="grounding",
        passed=passed,
        details=(
            f"all {len(verified)} finding(s) verified "
            f"({by_mode['evidence']} by evidence, {by_mode['rule']} by rule, "
            f"{by_mode['doc_clause']} by document clause, "
            f"{by_mode['measurement']} by measurement)"
            if passed
            else f"{len(ungrounded_ids)} of {len(verified)} finding(s) could not be verified: "
            + "; ".join(f"{fid} ({reasons[fid]})" for fid in ungrounded_ids[:3])
        ),
        failed_items=ungrounded_ids,
        on_fail_route="escalate" if ungrounded_ids else None,
        reason_code="ungrounded",
    )

    summary = {"findings": len(verified), "by_mode": by_mode, "passed": passed}

    # The verified findings replace the accumulator's contents via the editor's key, not this
    # one -- `findings` has an append reducer, so writing here would double everything. The
    # grounding flags therefore travel in `final_findings`, which the editor then refines.
    return verified, summary, [gate]
