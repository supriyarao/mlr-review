"""The service façade the API and UI build against. **This is a contract.**

Why this module exists, and why it is a stub today: the API, the UI and the eval harness all need a
stable way to say "review this piece", "how is it going", "here is the reviewer's decision" and "show
me the trail". That interface can be settled now, which lets those three be built in parallel
worktrees while the graph itself is still being written.

**Deviation from the brief, stated plainly.** The request was to put ``start_run``/``get_status``/
``resume``/``get_audit`` in ``platform_core/runtime.py`` as signatures with a fixture-returning stub.
Two problems with doing exactly that:

1. Three of those four already exist there, implemented and covered by 13 passing tests. Replacing
   working code with stubs would be a regression, and the stubs would have to be un-stubbed later.
2. ``platform_core`` must not know what a claim or a tenant's promotional piece is -- the
   import-boundary test enforces it. A function that takes ``(tenant_id, product_id, audience,
   channel, text)`` and returns a ``ReviewReport`` is MLR-specific by definition.

So the low-level, graph-agnostic runtime stays where it is (and gained ``get_audit``), and the
use-case-shaped façade lives here, in the layer that is allowed to know about reviews. The function
names and payloads are exactly as the brief specified; only the module differs.

Everything here returns data built by :mod:`mlr.examples`, so a consumer gets schema-valid payloads
today and real ones after Phase 3 with no change on their side.
"""

from __future__ import annotations

from typing import Any

from mlr import examples
from mlr.schemas import ReviewDecision, ReviewReport, ReviewTask

# Flipped to False in Phase 3 when the graph is wired in. Exposed so the API can surface it: a demo
# that silently returns fixture data would be worse than one that says it is doing so.
IS_STUB: bool = True


class RunNotFoundError(Exception):
    """Raised when a run id is unknown. The API maps this to a 404."""


def start_run(
    *,
    tenant_id: str,
    product_id: str,
    document: str,
    label_version: str = "",
    audience: str = "any",
    channel: str = "",
    market: str = "",
    doc_meta: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> str:
    """Submit a piece for review and return its run id.

    Returns as soon as the run is registered; the graph executes in the background, so the caller
    polls :func:`get_status`. ``audience`` and ``channel`` are required inputs rather than inferred
    from the text, because they determine which tenant rules are in scope -- a patient-facing
    disclaimer rule must not be applied to a piece for clinicians.

    Args:
        tenant_id: Which tenant's policy and evidence apply. Never optional.
        product_id: Must be configured for this tenant.
        document: Extracted text of the piece, front matter removed.
        label_version: Defaults to the tenant's configured version for this product.
        audience: ``hcp``, ``patient`` or ``any``.
        channel: Free-form channel tag, e.g. ``journal_ad``, ``email``.
        market: Market code, used by rule scope.
        doc_meta: Section offsets from ingest; computed if omitted.
        run_id: Supplied only when replaying a known id, e.g. in evals.

    Raises:
        TenantConfigError: if the tenant or product is unknown.
    """
    return run_id or examples.RUN_ID


def get_status(run_id: str) -> ReviewReport:
    """Current state of a run, as the report the UI renders.

    The same shape at every stage: a running review returns a partial report with
    ``status="running"`` and whatever findings exist so far, rather than a different payload that the
    caller has to branch on. ``status`` is the only field whose value gates what is populated.

    Raises:
        RunNotFoundError: if the id is unknown.
    """
    if not run_id:
        raise RunNotFoundError("run_id is required")
    return examples.example_report()


def get_review_task(run_id: str) -> ReviewTask | None:
    """What the reviewer is being asked, when a run is suspended at the human step.

    None when the run is not waiting on anyone. Separate from :func:`get_status` because a reviewer
    queue needs the question and the evidence bundle, which a status view does not.
    """
    return examples.example_review_task()


def get_queue(tier: str = "") -> list[ReviewTask]:
    """Pending review tasks, optionally for one reviewer tier.

    Tier filtering is how the demo simulates logging in as a senior reviewer rather than a standard
    one. There is no authentication in this demo; a real deployment would derive the tier from the
    authenticated user rather than a query parameter.
    """
    task = examples.example_review_task()
    return [task] if not tier or task.tier == tier else []


def resume(run_id: str, decision: dict[str, Any] | ReviewDecision) -> ReviewReport:
    """Record a human decision and resume the suspended run.

    The decision is validated by ``platform_core.hitl.parse_decision`` before it reaches the graph:
    it arrives over HTTP, and an invalid decision must fail loudly rather than record an approval
    nobody gave.

    Resuming continues from the checkpoint, so nothing before the human step is re-run or re-charged.

    Raises:
        RunNotFoundError: if the id is unknown.
        ValidationError: if the decision payload is malformed.
    """
    from platform_core.hitl import parse_decision

    parse_decision(decision)
    return examples.example_report(status="complete")


def get_audit(run_id: str) -> list[dict[str, Any]]:
    """Ordered audit events for a run.

    One event per node, gate and human decision, each carrying the model, prompt version, policy
    version, tokens, cost and latency that produced it. This is what makes a decision defensible
    months later, so it is a first-class part of the interface rather than an admin endpoint.
    """
    return examples.example_audit_events()


def list_runs(tenant_id: str = "", status: str = "") -> list[ReviewReport]:
    """Recent runs, newest first, optionally filtered."""
    report = examples.example_report()
    if tenant_id and report.tenant_id != tenant_id:
        return []
    if status and report.status != status:
        return []
    return [report]


def get_graph_mermaid() -> str:
    """Mermaid source for the compiled graph.

    Rendered from the compiled graph rather than drawn by hand, so the diagram in the README and on
    the UI home page cannot drift from the code it documents. The stub returns the intended shape.
    """
    return (
        "graph TD\n"
        "    __start__ --> ingest\n"
        "    ingest --> extract_claims\n"
        "    extract_claims --> preapproved_filter\n"
        "    preapproved_filter --> select_rules\n"
        "    select_rules --> substantiation\n"
        "    select_rules --> reference_support\n"
        "    select_rules --> off_label\n"
        "    select_rules --> fair_balance\n"
        "    select_rules --> safety_prominence\n"
        "    select_rules --> deterministic_rules\n"
        "    select_rules --> llm_rules\n"
        "    select_rules --> reference_doc_review\n"
        "    substantiation --> grounding_gate\n"
        "    reference_support --> grounding_gate\n"
        "    off_label --> grounding_gate\n"
        "    fair_balance --> grounding_gate\n"
        "    safety_prominence --> grounding_gate\n"
        "    deterministic_rules --> grounding_gate\n"
        "    llm_rules --> grounding_gate\n"
        "    reference_doc_review --> grounding_gate\n"
        "    grounding_gate --> severity\n"
        "    severity --> editor\n"
        "    editor --> confidence_gate\n"
        "    confidence_gate --> route\n"
        "    route --> human_review\n"
        "    human_review --> finalize\n"
        "    finalize --> __end__\n"
    )
