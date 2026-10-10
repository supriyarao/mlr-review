"""Wire the MLR review graph.

Shape:

    ingest -> extract_claims -> preapproved_filter
        -> [ 3 specialist checks | house_rules ]
        -> consolidate -> route
        -> human_review (interrupt) -> finalize

Three things about the structure are worth explaining.

**Which tenant rules apply is decided in ``ingest``, not in a node of its own.** Selection is a pure
function of the submission metadata ingest resolves, it calls no model, and it has to happen once
before the fan-out: three rule branches writing ``rules_applied`` concurrently would collide on a key
with no reducer. It also belongs beside ``policy_version``, which hashes the same active rule set.
Trade-off: one node now does two things, in exchange for a pipeline a reader can follow without being
told what a rulebook picker is.

**The fan-out is one conditional edge returning a list of ``Send``.** Four branches run concurrently
and each appends to ``findings`` through its reducer. The tenant's rules are one branch, not three:
inside ``house_rules`` the deterministic rules run first and free, then the judged ones, then one
call per reference document -- never several codes of practice in one prompt, because a reviewer
prompt carrying two unrelated documents does measurably worse at citing the right clause from either.

**Everything after the fan-out is sequential and deterministic.** Grounding, severity, dedupe, the
confidence threshold and routing are all plain code, in a fixed order -- the first four as stages
of ``consolidate``, then ``route``. That is what makes a decision reconstructable: given the findings the checks produced, the
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
from mlr.nodes import checks, consolidate, finalize, human_review, ingest, preapproved, rules
from mlr.nodes import extract_claims as extract
from mlr.nodes import route as routing
from mlr.prompts.checks import CHECKS
from mlr.state import ReviewState
from platform_core import decisions

# The three specialist checks, in the order they appear in the audit timeline.
#
# ``substantiation`` covers what a separate ``reference_support`` check used to: both asked their
# question of the same retrieved passages and both tenants graded them identically, so the split
# bought two strong-model calls and a stream of near-duplicate findings for the editor to merge.
CORE_CHECKS: tuple[str, ...] = (
    "substantiation",
    "off_label",
    "presentation",
)

# One branch for the tenant's own rules: the yes-or-no ones, the judged ones, and the reference
# documents, in that order inside a single node. Three boxes in a timeline for "the company's
# rules" told a reader nothing the finding's own citation does not.
RULE_BRANCHES: tuple[str, ...] = ("house_rules",)

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
    "consolidate",
    "route",
)

# Node names in pipeline order, for the UI timeline and for tests that assert the shape.
PIPELINE: tuple[str, ...] = (
    "ingest",
    "extract_claims",
    "preapproved_filter",
    *CORE_CHECKS,
    *RULE_BRANCHES,
    "consolidate",
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
        # Which checks survive a piece with no extractable claims is the check's own property, read
        # off its prompt rather than restated here -- a second copy of that list is a thing that
        # drifts. A piece with nothing extractable can still be missing its safety information.
        if claims or CHECKS[name].runs_without_claims:
            sends.append(Send(name, payload))

    if applied:
        sends.append(Send("house_rules", payload))

    # A piece with nothing to check still has to reach a human, so the pipeline must not stall.
    return sends or [Send("consolidate", payload)]


def build(*, fake: Any = None, decider: Any = None) -> StateGraph:
    """Build the graph. ``fake`` and ``decider`` are threaded into the model nodes for offline runs.

    Passing them explicitly rather than reading them from a global keeps the graph a pure function
    of its inputs: two tests can run with different scripted responses in the same process, and an
    eval replay cannot accidentally pick up another run's cassette.

    Two kinds of model node, two wrappers. ``_llm`` nodes send a prompt to a chat model and parse
    what comes back; ``_decide`` nodes ask a decision model a set of typed questions. Keeping the
    distinction visible here means a reader can see which nodes generate and which ones only choose,
    without opening either file.
    """

    def _llm(fn: Any) -> Any:
        def node(state: ReviewState) -> dict[str, Any]:
            return fn(state, fake=fake)

        node.__name__ = getattr(fn, "__name__", "node")
        return node

    def _decide(fn: Any) -> Any:
        # No decider supplied and no live backend configured means an offline run that was not given
        # scripted answers. Default to a recorder-less fake that matches nothing, so the run
        # continues with every claim treated as new. The alternative -- letting the node fail closed
        # on each claim -- reaches the same result but fills the audit trail with errors that say
        # nothing except "this was an offline run".
        resolved = decider
        if resolved is None and decisions.backend() == "fake":
            resolved = decisions.FakeDecider(default_answer={"type": "noul", "noul": 0.0})

        def node(state: ReviewState) -> dict[str, Any]:
            return fn(state, decider=resolved)

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
    add("preapproved_filter", _decide(preapproved.preapproved_filter))

    add("substantiation", _llm(checks.substantiation))
    add("off_label", _llm(checks.off_label))
    add("presentation", _llm(checks.presentation))

    add("house_rules", _llm(rules.house_rules))

    add("consolidate", consolidate.consolidate)
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
        [*CORE_CHECKS, *RULE_BRANCHES, "consolidate"],
    )
    for branch in (*CORE_CHECKS, *RULE_BRANCHES):
        builder.add_edge(branch, "consolidate")

    builder.add_edge("consolidate", "route")
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


def compile_with(saver: Any, *, fake: Any = None, decider: Any = None, step: bool = False) -> Any:
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
    return compile_graph(
        build(fake=fake, decider=decider), saver=saver, interrupt_before=interrupt_before
    )
