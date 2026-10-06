"""Wire the MLR review graph.

Shape:

    ingest -> extract_claims -> preapproved_filter
        -> [ 4 specialist checks | deterministic_rules | llm_rules | reference_doc_review ... ]
        -> grounding_gate -> severity -> editor -> confidence_gate -> route
        -> human_review (interrupt) -> finalize

Three things about the structure are worth explaining.

**Which tenant rules apply is decided in ``ingest``, not in a node of its own.** Selection is a pure
function of the submission metadata ingest resolves, it calls no model, and it has to happen once
before the fan-out: three rule branches writing ``rules_applied`` concurrently would collide on a key
with no reducer. It also belongs beside ``policy_version``, which hashes the same active rule set.
Trade-off: one node now does two things, in exchange for a pipeline a reader can follow without being
told what a rulebook picker is.

**The fan-out is one conditional edge returning a list of ``Send``.** Seven branches run concurrently
and each appends to ``findings`` through its reducer. ``reference_doc_review`` gets one ``Send`` per
applicable document rather than being handed several codes of practice at once, because a reviewer
prompt carrying two unrelated documents does measurably worse at citing the right clause from either.

**Everything after the fan-out is sequential and deterministic.** Grounding, severity, dedupe, the
confidence gate and routing are all plain code, in a fixed order, each consuming the previous one's
output. That is what makes a decision reconstructable: given the findings the checks produced, the
route is a pure function of tenant policy, so an auditor can replay it by hand -- and the eval
harness can recompute routing at five different confidence thresholds without calling a model again.

Trade-off: the checks receive all their claims in one call rather than one call per claim. Per-claim
fan-out would parallelise further and isolate a schema failure to a single claim, at the cost of
re-sending the document and evidence for every claim. For pieces of this size the batching wins
clearly; a per-claim map-reduce is the documented next step for long documents.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from mlr import trace
from mlr.nodes import checks, finalize, human_review, ingest, preapproved, rules
from mlr.nodes import extract_claims as extract
from mlr.nodes import grounding_gate as grounding
from mlr.nodes import route as routing
from mlr.nodes import severity as severity_node
from mlr.nodes.editor import editor
from mlr.state import ReviewState

# The four specialist checks, in the order they appear in the audit timeline.
#
# ``substantiation`` covers what a separate ``reference_support`` check used to: both asked their
# question of the same retrieved passages and both tenants graded them identically, so the split
# bought two strong-model calls and a stream of near-duplicate findings for the editor to merge.
CORE_CHECKS: tuple[str, ...] = (
    "substantiation",
    "off_label",
    "fair_balance",
    "safety_prominence",
)

RULE_BRANCHES: tuple[str, ...] = ("deterministic_rules", "llm_rules", "reference_doc_review")

# Nodes a stepped run pauses before.
#
# ``human_review`` is excluded because it already interrupts on its own: pausing before it as well would
# make the demo stop twice in the same place for two different reasons.
#
# ``finalize`` is excluded because pausing there shows nothing -- it only records the outcome -- and
# because it would leave a resumed run sitting at "running" after the reviewer had already decided,
# needing one more unexplained click before it read "complete".
STEPPABLE_NODES: tuple[str, ...] = (
    "ingest",
    "extract_claims",
    "preapproved_filter",
    "grounding_gate",
    "severity",
    "editor",
    "confidence_gate",
    "route",
)

# Node names in pipeline order, for the UI timeline and for tests that assert the shape.
PIPELINE: tuple[str, ...] = (
    "ingest",
    "extract_claims",
    "preapproved_filter",
    *CORE_CHECKS,
    *RULE_BRANCHES,
    "grounding_gate",
    "severity",
    "editor",
    "confidence_gate",
    "route",
    "human_review",
    "finalize",
)


def fan_out(state: ReviewState) -> list[Send]:
    """Dispatch the parallel branches.

    A branch is skipped when it has nothing to do -- no claims, or no rules of that kind -- rather
    than being invoked to return an empty list. That keeps the audit timeline honest: an absent
    ``llm_rules`` event means the tenant had no LLM rules in scope, which is different from a check
    that ran and found nothing.
    """
    claims = state.get("claims") or []
    applied = state.get("rules_applied") or []
    payload = dict(state)

    sends: list[Send] = []

    for name in CORE_CHECKS:
        # The two document-level checks run even with no claims: a piece with no extractable claims
        # can still be missing its safety information.
        document_level = name in ("fair_balance", "safety_prominence")
        if claims or document_level:
            sends.append(Send(name, payload))

    kinds = {a.kind for a in applied}
    if "deterministic" in kinds:
        sends.append(Send("deterministic_rules", payload))
    if "llm" in kinds:
        sends.append(Send("llm_rules", payload))

    # One reviewer per reference document, not one prompt holding all of them.
    for rule in applied:
        if rule.kind == "reference_doc":
            sends.append(Send("reference_doc_review", {**payload, "_rule_id": rule.id}))

    # A piece with nothing to check still has to reach a human, so the pipeline must not stall.
    return sends or [Send("grounding_gate", payload)]


def build(*, fake: Any = None) -> StateGraph:
    """Build the graph. ``fake`` is threaded into the LLM nodes for offline runs.

    Passing the fake explicitly rather than reading it from a global keeps the graph a pure function
    of its inputs: two tests can run with different scripted responses in the same process, and an
    eval replay cannot accidentally pick up another run's cassette.
    """

    def _llm(fn: Any) -> Any:
        def node(state: ReviewState) -> dict[str, Any]:
            return fn(state, fake=fake)

        node.__name__ = getattr(fn, "__name__", "node")
        return node

    builder = StateGraph(ReviewState)

    def add(name: str, fn: Any) -> None:
        """Register a node, wrapped in the state tracer.

        Every node goes through here, which is why tracing lives at this seam rather than as print
        calls inside fifteen node functions: it is uniform, it is off unless MLR_TRACE is set, and it
        cannot be half-removed later. See mlr/trace.py.
        """
        builder.add_node(name, trace.traced(name, fn))

    add("ingest", ingest.ingest)
    add("extract_claims", _llm(extract.extract_claims))
    add("preapproved_filter", _llm(preapproved.preapproved_filter))

    add("substantiation", _llm(checks.substantiation))
    add("off_label", _llm(checks.off_label))
    add("fair_balance", _llm(checks.fair_balance))
    add("safety_prominence", _llm(checks.safety_prominence))

    add("deterministic_rules", rules.deterministic_rules)
    add("llm_rules", _llm(rules.llm_rules))
    add("reference_doc_review", _llm(rules.reference_doc_review))

    add("grounding_gate", grounding.grounding_gate)
    add("severity", severity_node.severity)
    add("editor", editor)
    add("confidence_gate", routing.confidence_gate)
    add("route", routing.route)
    add("human_review", human_review.human_review)
    add("finalize", finalize.finalize)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "extract_claims")
    builder.add_edge("extract_claims", "preapproved_filter")

    # The fan-out. Every branch converges on the grounding gate, which LangGraph runs once all of
    # them have finished -- so grounding sees the complete finding set, never a partial one.
    builder.add_conditional_edges(
        "preapproved_filter",
        fan_out,
        [*CORE_CHECKS, *RULE_BRANCHES, "grounding_gate"],
    )
    for branch in (*CORE_CHECKS, *RULE_BRANCHES):
        builder.add_edge(branch, "grounding_gate")

    builder.add_edge("grounding_gate", "severity")
    builder.add_edge("severity", "editor")
    builder.add_edge("editor", "confidence_gate")
    builder.add_edge("confidence_gate", "route")
    builder.add_edge("route", "human_review")
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)

    return builder


@contextmanager
def checkpointer() -> Iterator[Any]:
    """A checkpointer that knows how to reconstruct this use case's state types.

    Wraps ``platform_core.runtime.checkpointer`` so a caller cannot forget to declare them. Without the
    declaration, every Pydantic model in state returns from a checkpoint as a plain dict and a resumed
    run fails inside a node rather than at the point of the mistake.
    """
    from mlr.state import CHECKPOINT_TYPES
    from platform_core.runtime import checkpointer as base_checkpointer

    with base_checkpointer(allowed_types=CHECKPOINT_TYPES) as saver:
        yield saver


def compile_with(saver: Any, *, fake: Any = None, step: bool = False) -> Any:
    """Compile the graph with a checkpointer. The checkpointer is what makes the human step work.

    ``step=True`` pauses before every node, so a caller can advance the run one node at a time and read
    the state in between. That exists for the demo: watching a review proceed node by node, with the
    findings accumulating, is the clearest way to show what the pipeline actually does -- and far more
    convincing than a spinner followed by a finished report.

    Trade-off: stepping writes a checkpoint per node instead of per superstep, so it is slower and
    produces a longer history. That is the right trade for a demo and the wrong one for production,
    which is why it is opt-in rather than the default.
    """
    from platform_core.runtime import compile_graph

    interrupt_before = list(STEPPABLE_NODES) if step else None
    return compile_graph(build(fake=fake), saver=saver, interrupt_before=interrupt_before)
