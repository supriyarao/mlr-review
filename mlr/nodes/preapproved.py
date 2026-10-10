"""Match claims against the tenant's already-approved claims.

**A retrieval step plus a decision-model call.** Retrieval narrows to plausible matches, and a
decision model confirms the two claims actually say the same thing. Pure string matching would miss
a reworded-but-identical claim, and pure retrieval similarity would match a claim that is merely
*about* the same result.

Why a decision model and not a chat model: the question is "do these two claims say the same thing",
which is a yes/no. A chat model answers it with a generative call and returns prose to be parsed; a
decision model returns a probability. This was the most-called model path in the system -- 142 calls
across the 22 golden cases, more than every other prompt combined -- so it is where the shape of the
question mattered most.

Two things changed with the transport, both improvements rather than side effects:

* **One request per claim, not one per candidate.** Every candidate travels as its own question in a
  single request, so the submitted claim is paid for once instead of three times.
* **The best match wins, not the first one over the line.** With all the answers in hand there is no
  reason to stop at the first acceptable candidate.

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
from platform_core import decisions
from platform_core.retrieval import TenantRetriever

# How many candidate approved claims to test per submitted claim. Retrieval is good at putting the
# right one first; checking three guards against a near-miss ranking above an exact match. Cheaper
# than it was: three candidates are now three questions in one request rather than three requests.
CANDIDATES_PER_CLAIM = 3

# How certain the model must be that two claims say the same thing before the claim is treated as
# already approved.
#
# Set high on purpose, and the asymmetry is the point. A false match skips substantiation on a claim
# nobody cleared, which is the one error this node must not make. A missed match only costs an
# unnecessary check. Measured separation on the golden corpus is wide -- 0.97 to 0.99 for genuinely
# equivalent claims, 0.02 to 0.15 for claims that differ -- so 0.85 sits in empty space rather than
# on a boundary.
#
# Trade-off: a module constant, not tenant configuration. It is a property of the model's
# calibration rather than a statement of a company's risk appetite, so it does not belong in a
# tenant YAML yet -- but if two tenants ever want different answers here, this is the line to move.
EQUIVALENCE_THRESHOLD = 0.85


def preapproved_filter(state: dict[str, Any], *, decider: Any = None) -> dict[str, Any]:
    """Record which submitted claims match an already-approved claim."""
    with node_run(state, "preapproved_filter") as run:
        claims: list[Claim] = list(state.get("claims") or [])
        if not claims:
            run.output = {"claims": 0, "matched": 0}
            return {"preapproved_matches": {}}

        retriever = TenantRetriever(str(state["tenant_id"]))
        matches: dict[str, str] = {}
        scores: dict[str, float] = {}
        considered = 0
        failures = 0

        for claim in claims:
            submitted = claim.normalized_claim or claim.original_text
            candidates = retriever.search(
                submitted,
                product_id=str(state.get("product_id") or ""),
                # Only pre-approved claims. Searching all evidence would match the underlying study
                # and wrongly report a brand new claim as already cleared.
                doc_type="preapproved_claim",
                k=CANDIDATES_PER_CLAIM,
            )
            if not candidates:
                continue
            considered += len(candidates)

            try:
                result = decisions.decide(
                    state=prompt.build_state(
                        submitted=submitted, candidates=[c.text for c in candidates]
                    ),
                    questions=prompt.build_questions(count=len(candidates)),
                    node="preapproved_filter",
                    version=prompt.QUESTION_VERSION,
                    fake=decider,
                )
            except decisions.DecisionError:
                # Failing to confirm equivalence means the claim is treated as new, which routes it
                # through substantiation. Failing closed in the safe direction: the cost is an
                # unnecessary check, not a skipped one.
                failures += 1
                continue

            run.record_decision(result, version=prompt.QUESTION_VERSION)

            # The most confident candidate above the threshold, not the first one. Ties go to the
            # earlier candidate, which is the better-ranked one.
            best_index, best_probability = -1, 0.0
            for index in range(len(candidates)):
                answer = result[prompt.CANDIDATE_KEY.format(index=index)]
                if answer.probability > best_probability:
                    best_index, best_probability = index, answer.probability

            if best_index >= 0 and best_probability >= EQUIVALENCE_THRESHOLD:
                matched = candidates[best_index]
                matches[claim.id] = matched.source_id.rsplit(":", 1)[-1]
                # Rounded, because the display precision is the only precision that means
                # anything here: the model reports two decimal places.
                scores[claim.id] = round(best_probability, 4)

        run.output = {
            "claims": len(claims),
            "candidates_considered": considered,
            "matched": len(matches),
            "matches": matches,
            # How sure the model was about each match it made. Only for claims that matched -- a
            # score beside a claim with no match would invite reading a near-miss as a result.
            "match_scores": scores,
            "threshold": EQUIVALENCE_THRESHOLD,
            # Worth surfacing rather than hiding: a run where every decision call failed matches
            # nothing, and "matched 0" alone would read as "nothing was pre-approved".
            "decision_failures": failures,
        }
        if failures:
            run.error = f"{failures} of {len(claims)} equivalence decision(s) could not be made"
        return {"preapproved_matches": matches}
