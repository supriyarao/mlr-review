"""The review graph's state. **This is a contract.**

Why the shape matters to other sessions: the API reads the final state to build a report, the UI
renders it, and the eval harness scores it. See ``docs/contracts.md``.

The one thing in here that is not obvious, and that is easy to get badly wrong:

**Append-only accumulation and replacement need separate keys.**

``findings`` carries an ``operator.add`` reducer because five check nodes run in parallel and each
returns its own list; the reducer concatenates them, which is the only way parallel branches can all
contribute without overwriting each other.

But a reducer applies to *every* write to that key. So the editor node, whose entire job is to
dedupe and calibrate, **cannot** replace ``findings`` -- returning the deduped list would append it
to the originals and double every finding. This is a standard LangGraph trap and it fails silently:
the counts just come out wrong.

So the pipeline is explicitly two-stage:

    findings        append-only, written by the parallel checks and the rule nodes
    final_findings  plain list, written once by the editor, replaces

Everything downstream of the editor -- severity, the confidence gate, routing, the report, the eval
scorer -- reads ``final_findings``. ``findings`` is kept rather than discarded because the audit
trail should be able to show what the checks originally said before the editor merged anything.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from mlr.schemas import (
    AppliedRule,
    Citation,
    Claim,
    Escalation,
    EscalationReason,
    Finding,
    GateResult,
    RunCost,
)


class ReviewState(TypedDict, total=False):
    """State threaded through the MLR review graph.

    ``total=False`` because nodes return partial updates: LangGraph merges each node's returned dict
    into the state, and requiring every key on every write would mean every node restating fields it
    does not touch.
    """

    # --- Set at submission ----------------------------------------------------------------------
    run_id: str
    tenant_id: str
    product_id: str
    label_version: str
    # Submission metadata that drives rule scope. A rule scoped to patient-facing material does not
    # apply to an HCP piece, so these are part of the input, not something inferred from the text.
    audience: str
    channel: str
    market: str

    # --- Produced by ingest (deterministic) -----------------------------------------------------
    # Extracted text, with front matter removed and markdown emphasis normalised. Findings quote
    # from this string, and the grounding gate checks quotes against it, so it must be the exact
    # text the checks saw.
    document: str
    # Section offsets and layout hints: {"sections": {name: {"start": int, "end": int}}, ...}.
    # Placement rules and the safety-prominence check read this.
    doc_meta: dict[str, Any]

    # --- Produced by extract_claims (LLM) -------------------------------------------------------
    claims: list[Claim]

    # --- Produced by preapproved_filter ---------------------------------------------------------
    # claim_id -> pre-approved claim id. A matched claim skips substantiation but is still checked
    # for fair balance and off-label implication: approved wording in a new context can still
    # mislead, and the context is exactly what was not reviewed when the claim was cleared.
    preapproved_matches: dict[str, str]

    # --- Produced by select_rules (deterministic) -----------------------------------------------
    rules_applied: list[AppliedRule]

    # --- Accumulated by the parallel branches ---------------------------------------------------
    # Append-only. Five check nodes plus the three rule nodes all write here concurrently.
    # DO NOT write a replacement list to this key -- see the module docstring.
    findings: Annotated[list[Finding], operator.add]
    # Append-only. Gates record their outcome whether they pass or fail, so the audit trail shows
    # that a gate ran and what it concluded rather than only recording failures.
    gate_results: Annotated[list[GateResult], operator.add]
    # Append-only. A node that fails records why and the run continues to a human rather than
    # dying: a partial review that escalates is more useful than a crash.
    errors: Annotated[list[str], operator.add]

    # --- Produced by the editor (replaces) ------------------------------------------------------
    # Deduped, merged and confidence-calibrated. Everything downstream reads this, not `findings`.
    final_findings: list[Finding]

    # --- Produced by route ----------------------------------------------------------------------
    # The reviewer tier this run goes to. There is no state in which a run has no destination:
    # a clean piece still routes to a standard reviewer.
    route: str
    escalation: Escalation | None

    # --- Produced by human_review / finalize ----------------------------------------------------
    # The resumed decision, validated by platform_core.hitl.parse_decision before it lands here.
    human_decision: dict[str, Any] | None

    # --- Bookkeeping ----------------------------------------------------------------------------
    status: str
    # Content hash of the tenant config plus the active rule set, from platform_core.policy.
    policy_version: str
    cost: RunCost


# Keys written by exactly one node, which may therefore be replaced rather than appended. Used by
# a test that asserts no node returns a replacement for a reduced key -- the silent-doubling trap.
REPLACEABLE_KEYS: frozenset[str] = frozenset(
    {
        "run_id",
        "tenant_id",
        "product_id",
        "label_version",
        "audience",
        "channel",
        "market",
        "document",
        "doc_meta",
        "claims",
        "preapproved_matches",
        "rules_applied",
        "final_findings",
        "route",
        "escalation",
        "human_decision",
        "status",
        "policy_version",
        "cost",
    }
)

# Keys with an append reducer. Writing a replacement to one of these concatenates instead.
ACCUMULATED_KEYS: frozenset[str] = frozenset({"findings", "gate_results", "errors"})


def initial_state(
    *,
    run_id: str,
    tenant_id: str,
    product_id: str,
    label_version: str = "",
    audience: str = "any",
    channel: str = "",
    market: str = "",
    document: str = "",
    doc_meta: dict[str, Any] | None = None,
) -> ReviewState:
    """Build the state a run starts from.

    The accumulator keys are initialised to empty lists explicitly. LangGraph's reducers handle a
    missing key, but starting them empty means a node can read them unconditionally and the eval
    harness can score a run that failed before any check executed.
    """
    return ReviewState(
        run_id=run_id,
        tenant_id=tenant_id,
        product_id=product_id,
        label_version=label_version,
        audience=audience,
        channel=channel,
        market=market,
        document=document,
        doc_meta=doc_meta or {},
        claims=[],
        preapproved_matches={},
        rules_applied=[],
        findings=[],
        gate_results=[],
        errors=[],
        final_findings=[],
        route="",
        escalation=None,
        human_decision=None,
        status="queued",
        policy_version="",
        cost=RunCost(),
    )


def report_findings(state: ReviewState) -> list[Finding]:
    """The findings a report should show.

    Prefers the editor's output. Falls back to the raw accumulator so a run that failed before the
    editor ran still reports what the checks found, rather than appearing to have found nothing --
    which would read as a clean piece.
    """
    final = state.get("final_findings")
    if final:
        return list(final)
    return list(state.get("findings") or [])


# Types that may appear inside a checkpoint, passed to
# ``platform_core.runtime.checkpointer(allowed_types=...)``. Declared here because this is the layer
# that knows what the state holds; the platform cannot import these and should not have to.
#
# Getting this wrong is quiet: an omitted type comes back from a checkpoint as a plain dict, and the
# resumed run fails with an AttributeError inside whichever node touches it next.
CHECKPOINT_TYPES: tuple[type, ...] = (
    AppliedRule,
    Citation,
    Claim,
    Escalation,
    EscalationReason,
    Finding,
    GateResult,
    RunCost,
)
