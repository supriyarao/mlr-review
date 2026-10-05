"""Extract the promotional claims from the piece.

**An LLM decision**, on the strong model. Deciding what counts as a claim is genuinely interpretive:
a headline fragment, an image caption and a footnote can each be a claim or not depending on what
surrounds them, and no pattern captures that.

The node does one thing the model is not asked to do: it locates each claim in the document itself.
Models are unreliable at counting characters, and a wrong offset would make a finding point at the
wrong words -- worse than having no offset at all. A claim whose text cannot be found verbatim is
flagged rather than silently given a span of (0, 0), because an unlocatable claim means the model
paraphrased, and every finding about it will then fail grounding for a reason nobody can see.
"""

from __future__ import annotations

from typing import Any

from mlr.nodes.base import node_run
from mlr.prompts import extract_claims as prompt
from mlr.prompts.rules import schema_failure_gate
from mlr.schemas import Claim
from platform_core import llm
from platform_core.rules.deterministic import normalize_whitespace

VALID_CLAIM_TYPES = {
    "efficacy",
    "safety",
    "indication",
    "dosing",
    "comparative",
    "economic",
    "other",
}


def locate(document: str, needle: str) -> tuple[int, int] | None:
    """Find a claim's exact position in the document.

    Tries the literal text first, then a whitespace-tolerant search -- a model copying a claim that
    wraps across lines will reproduce the words but not the line breaks, and rejecting that would
    throw away correct extractions.
    """
    if not needle.strip():
        return None

    index = document.find(needle)
    if index >= 0:
        return (index, index + len(needle))

    # Whitespace-tolerant: walk the document comparing normalised forms.
    target = normalize_whitespace(needle)
    if not target:
        return None
    flat_doc = normalize_whitespace(document)
    if target not in flat_doc:
        return None

    # Map the normalised hit back to a real span by matching the first and last words.
    words = target.split()
    first, last = words[0], words[-1]
    start = document.find(first)
    while start >= 0:
        end = document.find(last, start)
        if end < 0:
            return None
        candidate = document[start : end + len(last)]
        if normalize_whitespace(candidate) == target:
            return (start, end + len(last))
        start = document.find(first, start + 1)
    return None


def extract_claims(state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """Identify claims and locate each one in the document."""
    with node_run(state, "extract_claims") as run:
        document = str(state.get("document") or "")
        if not document.strip():
            run.output = {"claims": 0, "note": "empty document"}
            return {"claims": [], "errors": ["extract_claims: the submitted document is empty"]}

        try:
            result = llm.call_structured(
                "strong",
                node="extract_claims",
                prompt=prompt.build(document=document),
                schema=prompt.ExtractedClaims,
                prompt_version=prompt.PROMPT_VERSION,
                system=prompt.SYSTEM,
                fake=fake,
            )
        except llm.SchemaFailure as exc:
            run.error = str(exc)
            # Without claims there is nothing for the per-claim checks to examine, so the run
            # continues to the document-level checks and escalates. It must not look clean.
            return {
                "claims": [],
                "errors": [f"extract_claims: {exc}"],
                "gate_results": [schema_failure_gate("extract_claims", exc)],
            }

        run.record(result)
        claims: list[Claim] = []
        unlocatable: list[str] = []

        for i, extracted in enumerate(result.parsed.claims, start=1):
            span = locate(document, extracted.original_text)
            if span is None:
                unlocatable.append(extracted.original_text[:80])
            claim_type = (
                extracted.claim_type if extracted.claim_type in VALID_CLAIM_TYPES else "other"
            )
            claims.append(
                Claim(
                    id=f"c{i}",
                    original_text=extracted.original_text,
                    normalized_claim=extracted.normalized_claim or extracted.original_text,
                    location=extracted.location,
                    char_span=span or (0, 0),
                    claim_type=claim_type,  # type: ignore[arg-type]
                    reference_markers=[str(m) for m in extracted.reference_markers],
                )
            )

        run.output = {
            "claims": len(claims),
            "by_type": {
                t: sum(1 for c in claims if c.claim_type == t) for t in sorted(VALID_CLAIM_TYPES)
            },
            "unlocatable": unlocatable,
        }

        updates: dict[str, Any] = {"claims": claims}
        if unlocatable:
            # Visible rather than silent: a paraphrased claim will make every finding about it fail
            # grounding, and without this the reviewer would see that with no explanation.
            updates["errors"] = [
                f"extract_claims: {len(unlocatable)} claim(s) could not be located verbatim in the "
                f"document, so findings about them may fail grounding: {unlocatable}"
            ]
        return updates
