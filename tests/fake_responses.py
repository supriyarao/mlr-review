"""Scripted model responses for end-to-end graph tests.

Why hand-written rather than recorded: these tests exercise the *graph*, not the model. A scripted
response lets a test say "suppose substantiation finds this" and then assert on grounding, severity,
dedupe and routing -- which is what the graph is responsible for. Recorded cassettes against a real
model are a different job, done by the eval harness.

The quotes and citation passages are copied verbatim from the real corpus, because the grounding gate
checks them. A response with an invented quote would make every graph test fail for the right reason
but the wrong cause.
"""

from __future__ import annotations

import json
from typing import Any

# Verbatim from corpus/promo/zelvoran-hcp-subgroup-claim.md.
SUBGROUP_QUOTE = "Nearly 6 in 10 adults treated with Zelvoran achieved PASI-90"
REF_QUOTE = "Sustained response: PASI-90 was maintained at 12 weeks in the extension study"

# Verbatim from corpus/references/zel-302.md and zel-303.md.
ZEL302_PASSAGE = (
    "The 58.1% figure applies only to the subgroup of patients with prior biologic exposure."
)
ZEL303_PASSAGE = "ZEL-303 did not assess PASI-90."


def claims(*entries: dict[str, Any]) -> str:
    return json.dumps({"claims": list(entries)})


def claim(
    text: str,
    *,
    normalized: str = "",
    claim_type: str = "efficacy",
    location: str = "Efficacy",
    refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "original_text": text,
        "normalized_claim": normalized or text,
        "claim_type": claim_type,
        "location": location,
        "reference_markers": refs or [],
    }


def findings(*entries: dict[str, Any]) -> str:
    return json.dumps({"findings": list(entries)})


def finding(
    *,
    quote: str,
    issue: str,
    claim_id: str = "",
    severity: str = "blocker",
    confidence: int = 90,
    citations: list[dict[str, str]] | None = None,
    fix: str = "",
    explanation: str = "",
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "quote": quote,
        "issue": issue,
        # Required of a real model, so required here: a scripted response that omits it would
        # stop these tests exercising the shape the graph actually receives.
        "explanation": explanation or f"In plain terms: {issue}",
        "suggested_fix": fix,
        "severity": severity,
        "confidence": confidence,
        "citations": citations or [],
    }


def rule_findings(*entries: dict[str, Any]) -> str:
    return json.dumps({"findings": list(entries)})


def rule_finding(
    *,
    rule_id: str,
    quote: str,
    issue: str,
    claim_id: str = "",
    confidence: int = 85,
    clause: str = "",
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "quote": quote,
        "issue": issue,
        "explanation": f"In plain terms: {issue}",
        "suggested_fix": "",
        "confidence": confidence,
        "claim_id": claim_id,
        "clause": clause,
    }


NO_FINDINGS = findings()
NO_RULE_FINDINGS = rule_findings()


# preapproved_filter asks a decision model, not a chat model, so its answers are deciders rather
# than scripted JSON. See tests/test_preapproved_node.py for the node's own coverage.
def decider(*, equivalent: bool) -> Any:
    """A FakeDecider that calls every candidate equivalent, or none of them."""
    from platform_core import decisions

    probability = 0.99 if equivalent else 0.01
    return decisions.FakeDecider(default_answer={"type": "noul", "noul": probability})


def subgroup_case_script() -> dict[str, list[str]]:
    """Responses for the reference case: a subgroup figure presented as a general result.

    Produces two substantiation blockers -- one for the overstated subgroup figure, one for the
    citation that reports something else -- plus a promotional-code finding on the same claim and the
    same quote as the first, which is what exercises the editor's merge path.
    """
    return {
        "extract_claims": [
            claims(
                claim(
                    SUBGROUP_QUOTE,
                    normalized=(
                        "Nearly 6 in 10 adults treated with Zelvoran achieved PASI-90 at Week 16 "
                        "in the overall population."
                    ),
                    refs=["2"],
                ),
                claim(
                    REF_QUOTE,
                    normalized="Zelvoran maintained PASI-90 at 12 weeks in the extension study.",
                    refs=["3"],
                ),
            )
        ],
        # One call, both failure families: the evidence question on c1 and the citation question
        # on c2. Before the checks merged these were two scripted responses from two nodes.
        "substantiation": [
            findings(
                finding(
                    claim_id="c1",
                    quote=SUBGROUP_QUOTE,
                    issue=(
                        "Presents the prior-biologic subgroup result (58.1%) as a general result. "
                        "PASI-90 in the overall population was 44.2%."
                    ),
                    fix="State the overall result, or name the subgroup.",
                    severity="blocker",
                    confidence=92,
                    citations=[
                        {"source_id": "acme_pharma:reference:zel-302:2", "passage": ZEL302_PASSAGE}
                    ],
                ),
                finding(
                    claim_id="c2",
                    quote=REF_QUOTE,
                    issue=(
                        "Cites Reference 3 for PASI-90 at 12 weeks. ZEL-303 reports PASI-75 at "
                        "Week 16 and did not assess PASI-90."
                    ),
                    severity="blocker",
                    confidence=95,
                    citations=[
                        {"source_id": "acme_pharma:reference:zel-303:1", "passage": ZEL303_PASSAGE}
                    ],
                ),
            )
        ],
        "off_label": [NO_FINDINGS],
        "presentation": [NO_FINDINGS],
        "llm_rules": [NO_RULE_FINDINGS],
        "reference_doc_review": [
            rule_findings(
                rule_finding(
                    rule_id="ACME-DOC-001",
                    claim_id="c1",
                    quote=SUBGROUP_QUOTE,
                    issue="A subgroup figure is presented as the result in the overall population.",
                    clause=(
                        "A result observed in a subgroup must be described as a subgroup result, "
                        "naming the subgroup. A subgroup figure must never be presented as the "
                        "result in the overall population."
                    ),
                )
            )
        ],
    }


def clean_script(quote: str, normalized: str) -> dict[str, list[str]]:
    """Responses for a compliant piece: one claim, no findings from any check.

    No longer carries a ``preapproved`` switch. Whether a claim counts as already approved is now
    decided by the ``decider`` passed to ``compile_with``, not by a scripted chat response, and the
    graph's default decider matches nothing -- so a claim scripted here always reaches
    substantiation. A test that wants the pre-approved path takes :func:`decider`.
    """
    return {
        "extract_claims": [claims(claim(quote, normalized=normalized, refs=["1"]))],
        "substantiation": [NO_FINDINGS],
        "off_label": [NO_FINDINGS],
        "presentation": [NO_FINDINGS],
        "llm_rules": [NO_RULE_FINDINGS],
        "reference_doc_review": [NO_RULE_FINDINGS],
    }
