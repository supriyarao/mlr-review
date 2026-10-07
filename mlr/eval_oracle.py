"""A synthetic model for offline evals, derived from each case's own expectations.

**Read this before trusting an offline number.**

Offline mode needs model responses, and there is no API key. So this builds responses from the golden
case itself: for each expected finding it emits a finding of that check type, quoting real text from
the submitted piece and citing a real passage from the tenant's evidence. In effect it simulates a
*perfect* model.

What that measures, genuinely:

* **Routing accuracy.** The route is a pure function of tenant policy over a given finding set, so
  offline mode tests it completely. This is where the per-tenant divergence lives.
* **Grounding.** The oracle quotes real document text and cites real evidence passages, so the gate is
  doing actual verbatim verification against the real corpus -- not being handed a pass.
* **False escalation.** The three clean cases get no findings from the oracle, so anything that
  escalates them is the pipeline inventing a problem.
* **Deterministic rules.** These make no model call at all, so their findings, severities and
  confidences are entirely real offline.
* **Dedupe and severity.** Both are deterministic code operating on the finding set.

What it does **not** measure, and must never be reported as measuring:

* **Whether a model would notice any of this.** Blocker recall for LLM-sourced findings is close to
  tautological here: the oracle reports what the case expects, so of course the case passes. The real
  number requires a real model, and that is what ``--mode live`` and recorded cassettes are for.

Every eval result records its mode, and the offline baseline is marked provisional for exactly this
reason. Deterministic rule findings are the one part of offline blocker recall that is a real
measurement.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mlr.nodes.ingest import build_doc_meta
from mlr.prompts.checks import NODE_FOR_CHECK_TYPE
from platform_core.evals.schema import GoldenCase
from platform_core.retrieval import TenantRetriever

# How much of a retrieved passage to cite. Long enough to be meaningful to a reviewer, short enough
# that it is certainly a verbatim substring of the source.
CITATION_CHARS = 220


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]


def find_quote(document: str, needle: str) -> str:
    """The sentence in the document containing ``needle``, verbatim.

    Verbatim matters: the grounding gate looks this up in the document, so a tidied or reflowed version
    would be marked ungrounded and the case would fail for the wrong reason.
    """
    lowered = needle.lower().strip()
    if not lowered:
        return ""
    for sentence in _sentences(document):
        if lowered in sentence.lower():
            # Strip markdown furniture the document itself carries, keeping the text identical.
            cleaned = sentence.lstrip("#*>- ").strip()
            return cleaned if cleaned in document else sentence
    # The needle may be a phrase the case author expects in the issue text rather than the copy.
    return ""


# How to name each check to someone who is not a regulatory specialist. Mirrors the friendly names
# the UI uses, because a reader meeting both should not have to work out that they are the same.
_PLAIN_CHECK_NAME: dict[str, str] = {
    "substantiation": "The evidence reviewer",
    "off_label": "The label reviewer",
    "fair_balance": "The balance reviewer",
    "safety_prominence": "The safety reviewer",
    "business_rule": "A company rule",
}


def find_measurement(
    check_type: str, document: str, doc_meta: dict[str, Any]
) -> dict[str, str] | None:
    """A real figure from the company's validators, as a measurement citation.

    The ``presentation`` check is given no evidence block -- its questions are not answered against
    the label -- so its findings are grounded on measurements instead. The oracle has to cite the
    same way a well-behaved model would, or every golden case with a balance or prominence
    expectation would come back ungrounded and escalate for the wrong reason.

    The figure is read back out of the tool's own output rather than written here, so a change to
    what the validator reports cannot leave the oracle citing a number nobody computed.
    """
    from platform_core import tools

    tool = (
        "fair_balance_validator" if check_type == "fair_balance" else "safety_prominence_validator"
    )
    result = tools.call(tool, {"document": document, "doc_meta": doc_meta, "claims": []})
    if not result.ok:
        return None

    # Prefer a figure from a check the validator actually failed: that is the measurement a finding
    # about this piece would rest on.
    checks = result.data.get("checks") or {}
    ordered = list(result.data.get("failed") or []) + list(checks)
    for name in ordered:
        for key, value in (checks.get(name) or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # No "kind": a reported citation carries only a source and a passage, and the
                # runner derives the kind from what the check was given. The oracle has to emit
                # exactly the shape a model would, or it stops testing the real path.
                return {"source_id": tool, "passage": f"{name}.{key}: {value}"}
    return None


def find_citation(case: GoldenCase, query: str) -> dict[str, str] | None:
    """A real evidence passage from this tenant's corpus, as a citation.

    Retrieved rather than invented, and truncated at a sentence boundary so the cited text is certainly
    present in the source. If retrieval returns nothing -- which is correct for the cross-tenant and
    unverifiable cases -- the oracle emits a finding with no citation, which is what a well-behaved
    model should do and what the grounding gate should then escalate.
    """
    try:
        passages = TenantRetriever(case.tenant_id).search(
            query, product_id=case.product_id, doc_type="", k=3
        )
    except Exception:  # pragma: no cover - depends on local seed state
        return None

    for passage in passages:
        if passage.doc_type == "preapproved_claim":
            continue
        text = passage.text.strip()
        # Cut at a sentence end inside the window so the citation reads as a quotation.
        window = text[:CITATION_CHARS]
        cut = window.rfind(". ")
        snippet = window[: cut + 1] if cut > 60 else window
        if snippet.strip():
            return {"source_id": passage.source_id, "passage": snippet.strip()}
    return None


# Expected findings whose supporting evidence is deliberately absent. The oracle must emit these with
# no citation, because that is the correct behaviour and the point of the case.
_UNVERIFIABLE_MARKERS = (
    "92% of adults maintained",
    "cardiovascular events by 22%",
    "works better than",
    "leading biologics",
    "leading therapies",
)


def _is_unverifiable(claim_contains: str) -> bool:
    lowered = claim_contains.lower()
    return any(marker.lower() in lowered for marker in _UNVERIFIABLE_MARKERS)


def build_script(case: GoldenCase, document: str) -> dict[str, list[str]]:
    """Scripted responses for one case, keyed by node name.

    Checks the case expects nothing from return an empty finding list, so a check that should stay
    quiet is tested for staying quiet.
    """
    claims_payload = _claims_for(case, document)
    script: dict[str, list[str]] = {
        "extract_claims": [json.dumps({"claims": claims_payload})],
        # Only case 009 relies on a pre-approved match; everywhere else a match would wrongly skip
        # substantiation, so the default is "not equivalent".
        "preapproved_filter": [
            json.dumps(
                {
                    "equivalent": "preapproved" in case.tags,
                    "reason": "Oracle verdict derived from the case tags.",
                }
            )
        ],
    }

    by_check: dict[str, list[dict[str, Any]]] = {}
    for index, expected in enumerate(case.expected_findings):
        # Business-rule expectations are produced by the rules engine, not by a check node.
        if expected.check_type == "business_rule":
            continue

        quote = find_quote(document, expected.claim_contains)
        claim_id = _claim_id_for(claims_payload, quote or expected.claim_contains)
        # Query with the case author's note as well as the claim text. The note says what is wrong,
        # which is what the apt passage talks about -- querying on the claim alone retrieves the
        # passage the claim resembles, which is usually the indication statement.
        query = " ".join(filter(None, (expected.claim_contains, expected.note))) or case.title
        if NODE_FOR_CHECK_TYPE.get(expected.check_type) == "presentation":
            # The case's own doc_meta where it pins one (001 does, so a parsing change cannot
            # look like a model regression), otherwise the map ingest would compute.
            citation = find_measurement(
                expected.check_type,
                document,
                dict(case.doc_meta) or build_doc_meta(document),
            )
        elif _is_unverifiable(expected.claim_contains):
            citation = None
        else:
            citation = find_citation(case, query)

        by_check.setdefault(expected.check_type, []).append(
            {
                "claim_id": claim_id,
                "quote": quote,
                "issue": (
                    expected.note
                    or f"{expected.check_type.replace('_', ' ').capitalize()} problem with this claim."
                ),
                # The oracle has to emit the shape a model would, including the plain-English
                # half, or the goldens stop exercising the field at all. It cannot write a real
                # one -- the case note is a reviewer's sentence and the oracle has nothing else to
                # work from -- but it can at least do what a plain explanation is supposed to do:
                # name the reviewer in ordinary words and say what the piece actually says, before
                # repeating why that is a problem. The fake backend is what the offline demo runs
                # on, so this text is what a reader sees there.
                "explanation": (
                    f"{_PLAIN_CHECK_NAME.get(expected.check_type, 'A reviewer')} flagged this. "
                    f"The piece says: {(quote or expected.claim_contains).strip()!r}. "
                    f"{expected.note or 'That does not hold up as written.'}"
                ),
                "suggested_fix": "",
                "severity": expected.severity or "advisory",
                # High but not perfect, and above both tenants' thresholds, so the confidence gate is
                # exercised without every case tripping it.
                "confidence": 92 - index,
                "citations": [citation] if citation else [],
            }
        )

    # Keyed by node, not by check_type: one node can answer several concerns, and when it does its
    # findings carry a ``concern`` so the runner can tag each one the way the model would have.
    from mlr.prompts.checks import CHECKS

    for node, check in CHECKS.items():
        concerns = check.emits or (node,)
        findings = [
            {**finding, **({"concern": concern} if check.emits else {})}
            for concern in concerns
            for finding in by_check.get(concern, [])
        ]
        script[node] = [json.dumps({"findings": findings})]

    # Inside house_rules the judged rules and the reference-document reviews are separate model
    # calls, so the script still needs one scripted response per call, keyed by the call's node
    # name rather than the graph node's.
    script["llm_rules"] = [json.dumps({"findings": _rule_findings(case, document, kind="llm")})]
    script["reference_doc_review"] = [
        json.dumps({"findings": _rule_findings(case, document, kind="reference_doc")})
    ]
    return script


def _claims_for(case: GoldenCase, document: str) -> list[dict[str, Any]]:
    """Claims the oracle reports: the ones the case's expectations refer to, plus the real first claim.

    Includes a claim even for the clean cases, because a pipeline handed zero claims would skip the
    per-claim checks entirely and the clean cases would pass for the wrong reason.
    """
    seen: list[str] = []
    for expected in case.expected_findings:
        quote = find_quote(document, expected.claim_contains)
        if quote and quote not in seen:
            seen.append(quote)

    if not seen:
        # A clean piece still contains claims. Take the first numeric efficacy-looking sentence.
        for sentence in _sentences(document):
            if re.search(r"\d", sentence) and re.search(r"%|PASI|MACE|HbA1c", sentence):
                cleaned = sentence.lstrip("#*>- ").strip()
                if cleaned in document:
                    seen.append(cleaned)
                    break

    claims = []
    for text in seen:
        refs = re.findall(r"\[(\d+)\]", text)
        claims.append(
            {
                "original_text": text,
                "normalized_claim": text,
                "claim_type": "comparative" if "better than" in text.lower() else "efficacy",
                "location": "Efficacy",
                "reference_markers": refs,
            }
        )
    return claims


def _claim_id_for(claims: list[dict[str, Any]], quote: str) -> str:
    for i, claim in enumerate(claims, start=1):
        if quote and quote in claim["original_text"]:
            return f"c{i}"
    return ""


def _rule_findings(case: GoldenCase, document: str, *, kind: str) -> list[dict[str, Any]]:
    """Rule findings the oracle reports for non-deterministic rules.

    Deterministic rules are excluded: they run as real code with no model, so scripting them would
    replace a genuine measurement with a fake one.
    """
    from platform_core.rules.registry import RuleRegistry

    try:
        registry = RuleRegistry(case.tenant_id)
        rules = {r.id: r for r in registry.active_rules()}
    except Exception:  # pragma: no cover
        return []

    out: list[dict[str, Any]] = []
    for rule_id in case.expected_rule_ids:
        rule = rules.get(rule_id)
        if rule is None or rule.kind != kind:
            continue

        # Prefer a business_rule expectation. Failing that, fall back to the case's first expectation,
        # so the rule finding lands on the SAME claim and the SAME quote as the core finding about it --
        # which is what makes the editor's merge path get exercised. Without this the rule finding has
        # no quote, reads as an absence, and silently never merges.
        expected = next(
            (e for e in case.expected_findings if e.check_type == "business_rule"),
            next(iter(case.expected_findings), None),
        )
        needle = expected.claim_contains if expected else ""
        quote = find_quote(document, needle)

        entry: dict[str, Any] = {
            "rule_id": rule_id,
            "quote": quote,
            "issue": (expected.note if expected else rule.description)[:300],
            "explanation": (
                f"This is {rule.label()}, one of the company's own rules. "
                f"It says: {rule.description.strip().rstrip('.')}."
            ),
            "suggested_fix": "",
            "confidence": 88,
            "claim_id": "",
        }
        if kind == "reference_doc":
            clauses = [str(c) for c in rule.params.get("clauses", []) or []]
            # Pick the clause the case is about, so the finding cites something that genuinely exists.
            entry["clause"] = next(
                (c for c in clauses if "longest timepoint" in c or "subgroup" in c),
                clauses[0] if clauses else "",
            )
        out.append(entry)
    return out
