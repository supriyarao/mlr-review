"""A lightweight record per review run.

Why this exists when the checkpointer already holds every run's full state: answering "list the runs"
or "what is pending for the legal reviewer" from checkpoints means deserialising every run's entire
state to read three fields. That is fine with four demo runs and wrong as a design.

So this table holds only what a listing or a queue needs -- identity, tenant, status, route, counts --
and the checkpoint remains the source of truth for everything else. Nothing here is a second copy of
the findings.

Deterministic code. No model involved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlmodel import Field, SQLModel, select

from platform_core.audit import get_engine, session_scope


class RunRecord(SQLModel, table=True):
    """Index row for one review run."""

    __tablename__ = "runs"

    run_id: str = Field(primary_key=True)
    tenant_id: str = Field(index=True)
    product_id: str = ""
    label_version: str = ""
    audience: str = "any"
    channel: str = ""
    market: str = ""

    # queued | running | awaiting_human | complete | failed
    status: str = Field(default="queued", index=True)
    # The reviewer tier. Indexed because the queue page filters on it.
    route: str = Field(default="", index=True)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Denormalised counts, so a listing needs no checkpoint read at all.
    blockers: int = 0
    advisories: int = 0
    claims: int = 0
    ungrounded: int = 0

    # Set when status is "failed", so a failed run explains itself in a listing.
    error: str = ""
    # True when the run was started in step mode, so the UI knows to offer an Advance control.
    stepping: bool = False


def _ensure_tables() -> None:
    SQLModel.metadata.create_all(get_engine())


def create(
    run_id: str,
    *,
    tenant_id: str,
    product_id: str,
    label_version: str = "",
    audience: str = "any",
    channel: str = "",
    market: str = "",
    stepping: bool = False,
) -> RunRecord:
    """Register a run as queued, before the graph starts.

    Called synchronously by ``start_run`` so the run exists the moment the caller gets its id. Without
    this, a UI that polls immediately after submitting would get a 404 for the run it just created --
    a race that only shows up on a fast click.
    """
    _ensure_tables()
    record = RunRecord(
        run_id=run_id,
        tenant_id=tenant_id,
        product_id=product_id,
        label_version=label_version,
        audience=audience,
        channel=channel,
        market=market,
        status="queued",
        stepping=stepping,
    )
    with session_scope() as session:
        session.add(record)
    return record


def update(run_id: str, **fields: Any) -> None:
    """Patch a run's index row. Unknown runs are ignored rather than raising.

    Ignored because this is an index, not the record of truth: a status update for a run that was
    cleaned up should not take down the graph that was finishing it.
    """
    _ensure_tables()
    with session_scope() as session:
        record = session.get(RunRecord, run_id)
        if record is None:
            return
        for key, value in fields.items():
            if hasattr(record, key):
                setattr(record, key, value)
        record.updated_at = datetime.now(UTC)
        session.add(record)


def get(run_id: str) -> RunRecord | None:
    _ensure_tables()
    with session_scope() as session:
        record = session.get(RunRecord, run_id)
        return RunRecord.model_validate(record.model_dump()) if record else None


def exists(run_id: str) -> bool:
    return get(run_id) is not None


def list_runs(
    *, tenant_id: str = "", status: str = "", route: str = "", limit: int = 100
) -> list[RunRecord]:
    """Runs newest first, optionally filtered."""
    _ensure_tables()
    with session_scope() as session:
        statement = select(RunRecord).order_by(RunRecord.created_at.desc()).limit(limit)
        if tenant_id:
            statement = statement.where(RunRecord.tenant_id == tenant_id)
        if status:
            statement = statement.where(RunRecord.status == status)
        if route:
            statement = statement.where(RunRecord.route == route)
        return [RunRecord.model_validate(r.model_dump()) for r in session.exec(statement).all()]
