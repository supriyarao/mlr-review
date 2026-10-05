"""Shared plumbing for review nodes: auditing, timing, and failing safely.

Why this exists: every node has to record an audit event carrying the conditions that produced its
output, and every node has to decide what happens when it fails. Repeating that in fifteen nodes
means fifteen chances to forget the policy version or to let an exception kill a run.

The failure policy is the important part, and it is deliberate: **a node that fails records the
failure and lets the run continue to a human.** A crash loses the work done so far and tells the
reviewer nothing; a partial review that escalates with "the off-label check could not complete" is
worth having. The one thing a failure must never do is look like a clean result.

Deterministic code.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from platform_core import audit
from platform_core.llm import Usage, describe_backend
from platform_core.policy import policy_version
from platform_core.tenancy import load_tenant


@dataclass
class NodeRun:
    """Mutable record of one node execution, filled in by the node as it works."""

    node: str
    run_id: str
    tenant_id: str
    usage: Usage = field(default_factory=Usage)
    prompt_version: str = ""
    output: Any = None
    citations: Any = None
    error: str = ""
    # Set when a structured call needed its retry. Worth seeing in the trail: a node that retries on
    # every call is a prompt problem, not a model problem.
    recovered_from: str = ""

    def record(self, result: Any) -> None:
        """Absorb a :class:`platform_core.llm.StructuredResult`."""
        self.usage.add(
            model=result.usage.model,
            tokens_in=result.usage.tokens_in,
            tokens_out=result.usage.tokens_out,
            latency_ms=result.usage.latency_ms,
        )
        self.prompt_version = result.prompt_version or self.prompt_version
        if result.recovered_from:
            self.recovered_from = result.recovered_from


@contextmanager
def node_run(
    state: dict[str, Any],
    node: str,
    *,
    event_type: str = "node",
    input_value: Any = None,
) -> Iterator[NodeRun]:
    """Time a node, then write its audit event -- whether it succeeded or not.

    The event is written in a ``finally`` block, so a node that raises still leaves a trail. An
    auditor asking "what happened on this run" must not be told "nothing" because the step that
    mattered crashed.
    """
    run = NodeRun(
        node=node,
        run_id=str(state.get("run_id", "")),
        tenant_id=str(state.get("tenant_id", "")),
    )
    started = time.perf_counter()
    try:
        yield run
    except Exception as exc:
        run.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = int((time.perf_counter() - started) * 1000)
        # Trade-off: swallowing an audit write failure is uncomfortable, but the alternative is
        # that a full disk turns a recoverable review into a lost one. A production deployment would
        # alert on this; here the run continues and the gap is visible as a missing event.
        with contextlib.suppress(Exception):  # pragma: no cover
            audit.emit(
                run_id=run.run_id,
                tenant_id=run.tenant_id,
                node=node,
                event_type="error" if run.error else event_type,
                input_value=input_value,
                output=run.output,
                citations=run.citations,
                model=run.usage.model,
                prompt_version=run.prompt_version,
                policy_version=str(state.get("policy_version") or ""),
                label_version=str(state.get("label_version", "")),
                tokens_in=run.usage.tokens_in,
                tokens_out=run.usage.tokens_out,
                cost_usd=round(run.usage.cost_usd, 6),
                # The node's own measured latency, not the model's, so retries show up.
                latency_ms=elapsed or run.usage.latency_ms,
                error=run.error,
            )


def rollup_cost(run_id: str) -> dict[str, Any]:
    """Build the run's cost total by reading the audit log.

    Why derived rather than accumulated through state: ``cost`` is written by one node, so it is a
    plain key with no reducer -- and eight branches run concurrently in the fan-out. Having each
    branch add its own usage to ``cost`` meant eight concurrent writes to a single-value key, which
    LangGraph rejects outright (``InvalidUpdateError: At key 'cost': Can receive only one value per
    step``).

    The alternative was giving ``cost`` an adding reducer. This is better: the audit log already
    records tokens, cost and latency per node, so accumulating a second running total through state
    was duplicating the system of record and inviting the two to disagree. Reading it back once, from
    a node that runs on its own, means there is exactly one answer to "what did this review cost".

    Called from ``route`` and again from ``finalize``, both sequential, so the total is populated
    before a reviewer sees the report and updated once the human step has been recorded.
    """
    from mlr.schemas import RunCost

    summary = audit.run_cost_summary(run_id)
    model_calls = sum(
        1 for event in audit.events_for_run(run_id) if event.model and event.event_type == "node"
    )
    return {
        "cost": RunCost(
            tokens_in=int(summary["tokens_in"]),
            tokens_out=int(summary["tokens_out"]),
            cost_usd=float(summary["cost_usd"]),
            latency_ms=int(summary["latency_ms"]),
            model_calls=model_calls,
        )
    }


def resolve_policy_version(tenant_id: str) -> str:
    """Policy hash for a tenant, tolerating a tenant whose rules are not seeded.

    Returns an empty string rather than raising: a missing policy version degrades the audit trail,
    which is bad, but refusing to review the piece at all is worse. The empty value is visible in the
    trail as a gap.
    """
    try:
        return policy_version(tenant_id)
    except Exception:  # pragma: no cover - depends on local seed state
        return ""


def tenant_backend_provenance(tenant_id: str) -> dict[str, str]:
    """Provenance fields a report carries so it is self-contained once exported."""
    backend = describe_backend()
    try:
        config_version = load_tenant(tenant_id).version
    except Exception:  # pragma: no cover
        config_version = ""
    return {
        "policy_version": resolve_policy_version(tenant_id),
        "config_version": config_version,
        "backend": backend["backend"],
    }
