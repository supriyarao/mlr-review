"""Match claims against the tenant's already-approved claims.

**A retrieval step plus an LLM decision** on the fast model: retrieval narrows to plausible matches,
and a cheap model confirms the two claims actually say the same thing. Pure string matching would
miss a reworded-but-identical claim, and pure retrieval similarity would match a claim that is merely
*about* the same result.

The policy that matters, and the talking point: a matched claim **skips substantiation but is still
checked for fair balance and off-label implication.** The committee cleared the wording, not the
context -- and the context is precisely what was not reviewed when the claim was approved. A
pre-approved efficacy sentence placed beside a photograph of a child is still off-label promotion.
"""

from __future__ import annotations

from typing import Any

from mlr.nodes.base import node_run
from mlr.prompts import preapproved as prompt
from mlr.schemas import Claim
from platform_core import llm
from platform_core.retrieval import TenantRetriever

# How many candidate approved claims to confirm per submitted claim. Retrieval is good at putting the
# right one first; checking three guards against a near-miss ranking above an exact match without
# paying for a model call on every approved claim in the library.
CANDIDATES_PER_CLAIM = 3


def preapproved_filter(state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """Record which submitted claims match an already-approved claim."""
    with node_run(state, "preapproved_filter") as run:
        claims: list[Claim] = list(state.get("claims") or [])
        if not claims:
            run.output = {"claims": 0, "matched": 0}
            return {"preapproved_matches": {}}

        retriever = TenantRetriever(str(state["tenant_id"]))
        matches: dict[str, str] = {}
        considered = 0

        for claim in claims:
            candidates = retriever.search(
                claim.normalized_claim or claim.original_text,
                product_id=str(state.get("product_id") or ""),
                # Only pre-approved claims. Searching all evidence would match the underlying study
                # and wrongly report a brand new claim as already cleared.
                doc_type="preapproved_claim",
                k=CANDIDATES_PER_CLAIM,
            )
            for candidate in candidates:
                considered += 1
                try:
                    result = llm.call_structured(
                        "fast",
                        node="preapproved_filter",
                        prompt=prompt.build(
                            approved=candidate.text,
                            submitted=claim.normalized_claim or claim.original_text,
                        ),
                        schema=prompt.EquivalenceVerdict,
                        prompt_version=prompt.PROMPT_VERSION,
                        system=prompt.SYSTEM,
                        fake=fake,
                    )
                except llm.SchemaFailure:
                    # Failing to confirm equivalence means the claim is treated as new, which routes
                    # it through substantiation. Failing closed in the safe direction: the cost is an
                    # unnecessary check, not a skipped one.
                    continue

                run.record(result)
                if result.parsed.equivalent:
                    matches[claim.id] = candidate.source_id.rsplit(":", 1)[-1]
                    break

        run.output = {
            "claims": len(claims),
            "candidates_considered": considered,
            "matched": len(matches),
            "matches": matches,
        }
        return {"preapproved_matches": matches}
