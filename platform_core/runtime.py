"""Graph runtime: compile, run, suspend at a human, resume, inspect.

Why this exists: the interesting part of this system is that a run *stops in the middle* and waits
for a person, possibly for days, then continues from exactly where it left off. That needs a
checkpointer and a stable thread identity, and it needs a status view that can tell the difference
between "still working", "waiting for a human", and "finished".

Deterministic code. No model involved.

This module is use-case agnostic: it takes a compiled graph and a dict of input state, and knows
nothing about claims or findings. The MLR graph is built in ``mlr/graph.py`` and handed here.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command


# Modules whose types may appear inside a checkpoint. A use case adds its own schema module here (or
# passes its own serializer) rather than relying on a permissive default.
#
# Trade-off: this couples the platform's checkpointer to the name of a use-case module, which is the
# one place that boundary bends. The alternative -- a registration hook the use case calls at import
# time -- is more correct and more indirection than one tuple earns, and the import-boundary test still
# passes because this is a string, not an import.
def data_dir() -> Path:
    d = Path(os.environ.get("MLR_DATA_DIR", "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path() -> Path:
    return data_dir() / "checkpoints.db"


def new_run_id() -> str:
    """A fresh run id, also used as the checkpointer thread id.

    One id for both means an audit event and a checkpoint can always be joined, which is what
    makes the "what did the system know at this step" replay view possible.
    """
    return uuid.uuid4().hex[:12]


def thread_config(run_id: str) -> dict[str, Any]:
    """Checkpointer config addressing one run."""
    return {"configurable": {"thread_id": run_id}}


@contextmanager
def checkpointer(allowed_types: Iterable[type] | None = None) -> Iterator[SqliteSaver]:
    """A SqliteSaver over ``data/checkpoints.db``.

    ``check_same_thread=False`` because the graph runs in a background thread while the API
    answers status polls on another; WAL so those readers never block the writer.

    ``allowed_types`` declares which classes may be reconstructed from a checkpoint. State carries
    Pydantic models, and LangGraph will refuse to deserialise unregistered types in a future version,
    so passing the types makes that explicit now and keeps an unexpected type an error rather than a
    silent downgrade to a dict. Omitting it keeps LangGraph's permissive default, which warns.

    Trade-off: SQLite keeps the demo to a single process with no infrastructure to start, which is
    the right call for something meant to be cloned and run. It is also the first thing to replace
    for real use -- a Postgres checkpointer is the documented next step, and because the graph only
    ever sees this contextmanager, swapping it is a change to this function alone.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    conn = sqlite3.connect(str(checkpoint_path()), check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        serde = (
            JsonPlusSerializer(allowed_msgpack_modules=_allowlist(allowed_types))
            if allowed_types is not None
            else JsonPlusSerializer()
        )
        yield SqliteSaver(conn, serde=serde)
    finally:
        conn.close()


def _allowlist(types: Iterable[type]) -> list[tuple[str, str]]:
    """Build LangGraph's msgpack allowlist from the classes themselves.

    Derived from ``cls.__module__`` and ``cls.__name__`` rather than written out as strings, because a
    hand-maintained list of names would rot silently: renaming a class would not break the build, it
    would make that class come back from a checkpoint as a plain dict, and the failure would surface as
    an ``AttributeError`` inside a node after a resume -- a long way from its cause.

    The use case passes its own types. The platform must not enumerate them: it cannot import them, and
    knowing them is not its job.
    """
    return [(cls.__module__, cls.__name__) for cls in types]


def compile_graph(builder: Any, *, saver: Any, interrupt_before: list[str] | None = None) -> Any:
    """Compile a ``StateGraph`` with checkpointing enabled.

    ``interrupt_before`` names nodes the run should pause before, which is how a caller steps through
    a graph one node at a time. Generic: the platform does not care which nodes a use case considers
    worth pausing at.
    """
    if interrupt_before:
        return builder.compile(checkpointer=saver, interrupt_before=interrupt_before)
    return builder.compile(checkpointer=saver)


def interrupt_payload(state: Any) -> dict[str, Any] | None:
    """Extract the pending interrupt payload from a graph result or state snapshot.

    LangGraph surfaces interrupts differently depending on whether you are looking at an invoke
    result or a state snapshot, so this normalises both into "the thing the human was asked", or
    None when nothing is pending.
    """
    if isinstance(state, dict):
        raw = state.get("__interrupt__")
        if raw:
            first = raw[0] if isinstance(raw, (list, tuple)) else raw
            return getattr(first, "value", first)

    tasks = getattr(state, "tasks", None) or ()
    for task in tasks:
        for intr in getattr(task, "interrupts", None) or ():
            value = getattr(intr, "value", None)
            if value is not None:
                return value
    return None


def start_run(graph: Any, state: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    """Run a graph until it finishes or stops at a human.

    Returns the run id, the final (or suspended) state, and the interrupt payload when the run is
    waiting on a person.
    """
    rid = run_id or new_run_id()
    config = thread_config(rid)
    result = graph.invoke(state, config=config)
    pending = interrupt_payload(result)
    return {
        "run_id": rid,
        "state": result,
        "interrupt": pending,
        "status": "awaiting_human" if pending else "complete",
    }


def resume_run(graph: Any, run_id: str, resume_value: Any) -> dict[str, Any]:
    """Resume a suspended run with a human's answer.

    The resume value is whatever the ``interrupt()`` call should evaluate to -- here, a review
    decision. The graph continues from the checkpoint, so nothing before the human step is re-run
    and nothing is re-charged.
    """
    config = thread_config(run_id)
    result = graph.invoke(Command(resume=resume_value), config=config)
    pending = interrupt_payload(result)
    return {
        "run_id": run_id,
        "state": result,
        "interrupt": pending,
        "status": "awaiting_human" if pending else "complete",
    }


def get_state(graph: Any, run_id: str) -> Any:
    """Current checkpoint snapshot for a run."""
    return graph.get_state(thread_config(run_id))


def get_status(graph: Any, run_id: str) -> dict[str, Any]:
    """Status of a run, for the API's polling endpoint and the live UI timeline."""
    snapshot = get_state(graph, run_id)
    values = getattr(snapshot, "values", None) or {}
    pending = interrupt_payload(snapshot)
    next_nodes = list(getattr(snapshot, "next", ()) or ())

    if pending:
        status = "awaiting_human"
    elif next_nodes:
        status = "running"
    elif values:
        status = "complete"
    else:
        status = "unknown"

    return {
        "run_id": run_id,
        "status": status,
        "next": next_nodes,
        "interrupt": pending,
        "values": values,
    }


def history(graph: Any, run_id: str) -> list[dict[str, Any]]:
    """Checkpoint history, oldest first.

    This is what backs the audit page's replay view: each entry is what the system actually knew
    at that step, rather than a reconstruction after the fact.
    """
    snapshots = list(graph.get_state_history(thread_config(run_id)))
    snapshots.reverse()
    out: list[dict[str, Any]] = []
    for index, snap in enumerate(snapshots):
        config = getattr(snap, "config", {}) or {}
        out.append(
            {
                "step": index,
                "checkpoint_id": (config.get("configurable") or {}).get("checkpoint_id", ""),
                "next": list(getattr(snap, "next", ()) or ()),
                "values": getattr(snap, "values", None) or {},
            }
        )
    return out


def get_audit(run_id: str) -> list[dict[str, Any]]:
    """Ordered audit events for a run, as plain dicts.

    Part of the runtime's public surface alongside :func:`start_run`, :func:`resume_run` and
    :func:`get_status`, so a caller building a status view has one module to import rather than
    needing to know that events live in a different store from checkpoints.

    Thin by design: the audit log is append-only and owns its own schema, so this converts rather
    than interprets.
    """
    from platform_core.audit import events_for_run

    return [event.model_dump(mode="json") for event in events_for_run(run_id)]


def render_mermaid(graph: Any) -> str:
    """Mermaid source for the compiled graph.

    Generated from the compiled graph rather than drawn by hand, so the diagram in the README can
    never drift from the code it documents.
    """
    return graph.get_graph().draw_mermaid()
