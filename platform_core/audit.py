"""Append-only audit log.

Why this exists: the system's output is only useful if a regulator, an auditor, or the reviewer
themselves can reconstruct *why* a decision was made months later. That means recording, per
node, not just the output but the exact conditions that produced it: which model, which prompt
version, which tenant policy, which label version, what it cost, how long it took.

Deterministic code. No model involved.

Design constraints worth noting:

* **Append-only by construction.** This module exposes :func:`emit` and read helpers, and no
  update or delete at all. An audit log you can edit is not an audit log.
* **Inputs are hashed, not stored.** Node inputs can be a whole promotional document; storing
  every copy would bloat the log and duplicate the checkpointer, which already holds full state.
  The hash is enough to prove two runs saw identical input.
* **Human decisions go in the same log as model steps.** The question an auditor asks is "who
  decided this", and splitting machine and human records across two tables makes that harder to
  answer than it needs to be.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, event, text
from sqlmodel import Field, Session, SQLModel, create_engine, select


def data_dir() -> Path:
    """Directory for runtime databases and the vector store."""
    d = Path(os.environ.get("MLR_DATA_DIR", "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class AuditEvent(SQLModel, table=True):
    """One recorded step: a node execution, a gate decision, or a human decision.

    Every field here answers a question an auditor actually asks. ``policy_version`` and
    ``label_version`` are the ones most often missing from systems like this -- without them you
    can show what the system decided but not what rules it was applying at the time.
    """

    __tablename__ = "audit_events"

    id: int | None = Field(default=None, primary_key=True)

    # What this event belongs to.
    run_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    # Monotonic position within a run. Wall-clock timestamps collide when parallel check nodes
    # finish in the same millisecond, so ordering uses this, not ts.
    seq: int
    ts: datetime

    # What happened.
    node: str = Field(index=True)
    # "node" | "gate" | "human" | "error" -- lets the UI group the timeline without parsing names.
    event_type: str = "node"
    # Hash of the node's input, not the input itself. See module docstring.
    input_hash: str
    output_json: str = ""
    # Citations are pulled out of the output rather than left buried in it: "show me the evidence
    # for this finding" is the single most common audit question.
    citations_json: str = ""

    # The exact conditions that produced the output. These are what make a decision reproducible.
    model: str = ""
    prompt_version: str = ""
    policy_version: str = ""  # tenant config content hash
    label_version: str = ""

    # Cost and performance accounting, per step.
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0

    # Set when event_type == "error", so a failed run is still fully auditable.
    error: str = ""


_engine: Engine | None = None
_engine_path: Path | None = None


def _enable_wal(dbapi_connection, _connection_record) -> None:
    """Put SQLite in WAL mode so a reader (the API serving a status poll) never blocks a writer.

    Trade-off: the demo runs the graph in a background thread in the same process as the API, so
    concurrent read/write is routine. WAL makes that safe cheaply. A production deployment would
    use Postgres and this hook would go away.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def get_engine() -> Engine:
    """Process-wide engine for the application database.

    Recreated if ``MLR_DATA_DIR`` changes, which is what lets each test run against its own
    temporary database without leaking state between tests.
    """
    global _engine, _engine_path
    path = data_dir() / "app.db"
    if _engine is None or _engine_path != path:
        engine = create_engine(
            f"sqlite:///{path}",
            # The graph runs in a background thread while the API answers polls on another.
            connect_args={"check_same_thread": False},
        )
        event.listen(engine, "connect", _enable_wal)
        SQLModel.metadata.create_all(engine)
        _engine, _engine_path = engine, path
    return _engine


def reset_engine() -> None:
    """Drop the cached engine. Used by tests when switching data directories."""
    global _engine, _engine_path
    if _engine is not None:
        _engine.dispose()
    _engine, _engine_path = None, None


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session, committed on success and rolled back on error."""
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def hash_input(value: Any) -> str:
    """Stable short hash of a node input.

    Canonical JSON with sorted keys so logically identical inputs hash identically regardless of
    dict ordering; falls back to ``repr`` for anything not JSON-serialisable so that hashing can
    never be the reason a run fails.
    """
    try:
        payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _next_seq(session: Session, run_id: str) -> int:
    """Next position in this run's event sequence.

    Taken inside the caller's transaction. Trade-off: under genuinely concurrent writes for the
    same run this could collide, and the correct fix is a per-run counter row with a row lock. It
    is left simple because events are written at node boundaries, and the consequence of a
    collision is two events sharing a sort position, not lost data.
    """
    current = session.exec(
        select(AuditEvent.seq).where(AuditEvent.run_id == run_id).order_by(AuditEvent.seq.desc())
    ).first()
    return (current or 0) + 1


def emit(
    *,
    run_id: str,
    tenant_id: str,
    node: str,
    event_type: str = "node",
    input_value: Any = None,
    output: Any = None,
    citations: Any = None,
    model: str = "",
    prompt_version: str = "",
    policy_version: str = "",
    label_version: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
    error: str = "",
) -> AuditEvent:
    """Append one event. The only write path in this module.

    Returns the persisted event so a caller can assert on it in a test.
    """
    with session_scope() as session:
        ev = AuditEvent(
            run_id=run_id,
            tenant_id=tenant_id,
            seq=_next_seq(session, run_id),
            ts=datetime.now(UTC),
            node=node,
            event_type=event_type,
            input_hash=hash_input(input_value) if input_value is not None else "",
            output_json=json.dumps(output, default=str) if output is not None else "",
            citations_json=json.dumps(citations, default=str) if citations is not None else "",
            model=model,
            prompt_version=prompt_version,
            policy_version=policy_version,
            label_version=label_version,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            error=error,
        )
        session.add(ev)
        session.flush()
        session.refresh(ev)
        # Detached copy, so callers can read it after the session closes.
        return AuditEvent.model_validate(ev.model_dump())


def events_for_run(run_id: str) -> list[AuditEvent]:
    """Every event for a run, in execution order."""
    with session_scope() as session:
        rows = session.exec(
            select(AuditEvent).where(AuditEvent.run_id == run_id).order_by(AuditEvent.seq)
        ).all()
        return [AuditEvent.model_validate(r.model_dump()) for r in rows]


def events_for_tenant(tenant_id: str, limit: int = 500) -> list[AuditEvent]:
    """Recent events for one tenant, newest first. Tenant-scoped by construction."""
    with session_scope() as session:
        rows = session.exec(
            select(AuditEvent)
            .where(AuditEvent.tenant_id == tenant_id)
            .order_by(AuditEvent.id.desc())
            .limit(limit)
        ).all()
        return [AuditEvent.model_validate(r.model_dump()) for r in rows]


def run_cost_summary(run_id: str) -> dict[str, float | int]:
    """Totals for a run: what it cost, how long it took, how many tokens it used.

    Surfaced in the UI next to the findings, because cost per review is the number that decides
    whether a system like this is deployable at scale.
    """
    events = events_for_run(run_id)
    return {
        "events": len(events),
        "tokens_in": sum(e.tokens_in for e in events),
        "tokens_out": sum(e.tokens_out for e in events),
        "cost_usd": round(sum(e.cost_usd for e in events), 6),
        "latency_ms": sum(e.latency_ms for e in events),
    }


def integrity_check(run_id: str) -> dict[str, Any]:
    """Cheap tamper check: are this run's sequence numbers complete and unique?

    Not cryptographic integrity -- a hash chain would be the real answer, and is noted in the
    README as a production step. This catches the realistic failure, which is a dropped or
    double-written event, and gives the audit UI something concrete to assert.
    """
    events = events_for_run(run_id)
    seqs = [e.seq for e in events]
    expected = list(range(1, len(events) + 1))
    return {
        "run_id": run_id,
        "event_count": len(events),
        "sequence_complete": seqs == expected,
        "duplicate_seqs": sorted({s for s in seqs if seqs.count(s) > 1}),
    }


def journal_mode() -> str:
    """Report SQLite's journal mode, so a test can prove the WAL hook actually ran."""
    with session_scope() as session:
        return session.exec(text("PRAGMA journal_mode")).one()[0]
