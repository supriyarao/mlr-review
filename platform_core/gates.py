"""Validation gates: a checkpoint in a graph that can divert the flow.

Why this exists: an agent that produces output and stops is not reviewable. A gate is a named,
deterministic predicate over graph state that either passes or names a route to take instead --
so "this output was not trustworthy enough to continue normally" becomes a recorded decision with
a destination, rather than a silently swallowed warning.

Deterministic code, by design. A gate must never call a model: if the thing deciding whether to
trust model output is itself model output, the check has no independent value. Gates here do
string comparison, arithmetic, and set membership, and nothing else.

Note what a gate deliberately does *not* do: it does not drop or rewrite the items that failed.
A failing item stays in state, flagged, and the route carries it to a human. Silently discarding
a finding the system could not verify would be the single worst behaviour in a review system --
it would convert "I am not sure" into "there is no problem".

Gates know nothing about the use case: this module never mentions a claim, a finding, or a
severity. It sees opaque items and a predicate supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class GateResult(BaseModel):
    """The outcome of one gate, recorded in state and in the audit log.

    ``details`` carries a sentence written for a human reviewer, not a developer: it shows up in
    the UI as the reason a review landed on their desk.
    """

    model_config = ConfigDict(extra="forbid")

    gate: str
    passed: bool
    # Human-readable explanation. Always populated, including on a pass, so the audit trail can
    # show that a gate ran and what it concluded rather than only recording failures.
    details: str = ""
    # Identifiers of the items that failed, e.g. finding ids. Lets the UI highlight exactly what
    # tripped the gate instead of just naming the gate.
    failed_items: list[str] = Field(default_factory=list)
    # Where to go when this gate fails. None means "do not divert, just record the failure" --
    # used by advisory gates that annotate without changing the route.
    on_fail_route: str | None = None
    # Machine-readable reason the route layer matches tenant escalation rules against.
    reason_code: str = ""

    def summary(self) -> str:
        verdict = "passed" if self.passed else "failed"
        return f"{self.gate}: {verdict}" + (f" ({self.details})" if self.details else "")


@runtime_checkable
class Gate(Protocol):
    """Anything that can inspect state and return a :class:`GateResult`.

    A Protocol rather than a base class so a plain function, a class, or a partial can all be a
    gate without inheriting anything.
    """

    name: str

    def check(self, state: Any) -> GateResult: ...


class PredicateGate:
    """A gate over a list of items pulled from state, each judged by a predicate.

    Covers the common shape -- "every finding must have a verbatim citation", "every finding must
    clear the tenant's confidence threshold" -- without each one re-implementing iteration,
    id extraction, and result assembly.

    Args:
        name: Gate name, recorded in state and the audit log.
        items: Pulls the items to judge out of graph state.
        predicate: True if an item is acceptable.
        item_id: How to identify an item in ``failed_items``.
        on_fail_route: Route to take when any item fails.
        reason_code: Reason the routing layer matches tenant rules against.
        describe: Builds the human-readable ``details`` from the failures.
    """

    def __init__(
        self,
        *,
        name: str,
        items: Callable[[Any], Sequence[Any]],
        predicate: Callable[[Any], bool],
        item_id: Callable[[Any], str],
        on_fail_route: str | None = None,
        reason_code: str = "",
        describe: Callable[[Sequence[Any]], str] | None = None,
    ) -> None:
        self.name = name
        self._items = items
        self._predicate = predicate
        self._item_id = item_id
        self._on_fail_route = on_fail_route
        self._reason_code = reason_code
        self._describe = describe

    def check(self, state: Any) -> GateResult:
        items = list(self._items(state))
        failures = [item for item in items if not self._predicate(item)]

        if not failures:
            return GateResult(
                gate=self.name,
                passed=True,
                details=f"all {len(items)} item(s) passed" if items else "nothing to check",
                reason_code=self._reason_code,
            )

        details = (
            self._describe(failures)
            if self._describe
            else f"{len(failures)} of {len(items)} item(s) failed"
        )
        return GateResult(
            gate=self.name,
            passed=False,
            details=details,
            failed_items=[self._item_id(item) for item in failures],
            on_fail_route=self._on_fail_route,
            reason_code=self._reason_code,
        )


def run_gates(gates: Sequence[Gate], state: Any) -> list[GateResult]:
    """Run every gate and collect the results.

    Trade-off: all gates run even once one has failed, rather than short-circuiting. A reviewer
    who is handed an escalation wants the complete list of what was wrong, not just the first
    problem found -- and gates are cheap deterministic checks, so there is nothing to save by
    stopping early.
    """
    return [gate.check(state) for gate in gates]


def failed_routes(results: Sequence[GateResult]) -> list[str]:
    """Distinct routes requested by failing gates, in order encountered."""
    routes: list[str] = []
    for r in results:
        if not r.passed and r.on_fail_route and r.on_fail_route not in routes:
            routes.append(r.on_fail_route)
    return routes


def failure_reasons(results: Sequence[GateResult]) -> list[str]:
    """Reason codes from failing gates, for matching against tenant escalation rules."""
    reasons: list[str] = []
    for r in results:
        if not r.passed and r.reason_code and r.reason_code not in reasons:
            reasons.append(r.reason_code)
    return reasons


def all_passed(results: Sequence[GateResult]) -> bool:
    return all(r.passed for r in results)
