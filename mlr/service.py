"""The service façade the API and UI build against. **This is the contract.**

Signatures and return types are those documented in ``docs/contracts.md`` and have not changed. What
changed is that they now run the real graph instead of returning fixture data.

How a run proceeds:

1. :func:`start_run` registers the run as ``queued`` **synchronously**, so the caller's id is valid the
   instant it is returned, then executes the graph on a worker thread.
2. The graph runs to the human step and suspends. The checkpoint is the source of truth; the ``runs``
   index carries only what a listing needs.
3. :func:`get_status` reads the checkpoint and returns the same :class:`~mlr.schemas.ReviewReport`
   shape at every stage -- a running review has fewer findings, not a different payload.
4. :func:`resume` validates the reviewer's decision and continues from the checkpoint. Nothing before
   the human step is re-run or re-charged.

Step mode (:func:`advance`) exists for the demo: a run started with ``step=True`` pauses before each
node, so a reviewer can watch findings accumulate one stage at a time. That is far more convincing than
a spinner followed by a finished report, and it makes the gates visible.

**With no API key**, set ``MLR_LLM_BACKEND=fake`` and the LLM findings come from a fixture matched to
the golden set (:mod:`mlr.demo`); everything else -- retrieval, grounding against the real corpus,
tenant severity rules, dedupe, routing, deterministic business rules, the interrupt -- runs for real.
:func:`describe_backend_note` returns the sentence the UI should show so it is never ambiguous about
which part was stood in for.
"""

from __future__ import annotations

import contextlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mlr import store
from mlr.graph import STEPPABLE_NODES, checkpointer, compile_with
from mlr.nodes.ingest import build_doc_meta, normalize_document, parse_front_matter
from mlr.schemas import AppliedRule, ReviewDecision, ReviewReport, ReviewTask, RunCost
from mlr.state import initial_state, report_findings
from platform_core import audit, runtime
from platform_core.hitl import parse_decision
from platform_core.llm import backend, describe_backend
from platform_core.tenancy import load_tenant

# The façade runs the real graph. Kept as a flag because the API reports it on /health, and a demo that
# quietly served fixture data would be worse than one that says what it is doing.
IS_STUB: bool = False

# Background execution. A small pool rather than a thread per request: reviews are model-bound, and an
# unbounded pool would let a burst of submissions open dozens of concurrent model connections.
#
# Trade-off: this is in-process, so a restart loses any run that was mid-flight -- though not its work,
# because the checkpoint survives and the run can be resumed. A production deployment would put a real
# queue here; the checkpointer already makes that a change of executor rather than of design.
_MAX_CONCURRENT_RUNS = int(os.environ.get("MLR_MAX_CONCURRENT_RUNS", "4"))
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


class RunNotFoundError(Exception):
    """Raised when a run id is unknown. The API maps this to a 404."""


def _pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=_MAX_CONCURRENT_RUNS, thread_name_prefix="mlr-review"
                )
    return _executor


def describe_backend_note() -> str:
    """One sentence for the UI about where findings come from.

    Returned by the service rather than written in the UI so there is one wording, and so it cannot
    drift out of step with what the backend is actually doing.
    """
    info = describe_backend()
    if info["backend"] == "fake":
        return (
            "Findings from language models are supplied by a fixture, not a model, because no API key "
            "is configured. Everything else runs for real: retrieval, grounding against the corpus, "
            "tenant severity rules, deduplication, routing, and all deterministic business rules."
        )
    if info["backend"] == "ollama":
        return (
            f"Running against a local model ({info['model_strong']}). Free, but slow on CPU and less "
            f"reliable at quoting evidence verbatim, so expect findings to fail the grounding gate "
            f"more often than with a hosted model."
        )
    return f"Running against {info['model_strong']} (strong) and {info['model_fast']} (fast)."


# A read-only graph, shared across status polls.
#
# Reads need a compiled graph only to interpret a checkpoint -- no node executes -- but building and
# compiling one costs about 20ms, and a UI polling twice a second was paying it on every call. That is
# pure waste: the graph's structure never changes.
#
# Safe to share: SqliteSaver holds its own lock and documents that ``check_same_thread=False`` is fine
# because of it, so one long-lived connection serves every reader. Writes (start_run, advance, resume)
# still compile their own, because each binds a per-run scripted model into the nodes.
#
# Keyed on the checkpoint path so the cache invalidates when MLR_DATA_DIR changes, which every test does.
_read_graphs: dict[tuple[str, bool], Any] = {}
_read_saver: Any = None
_read_saver_path: str = ""
_read_lock = threading.Lock()


def _read_graph(step: bool) -> Any:
    """A compiled graph for reading checkpoints. Cached; never used to execute a node."""
    global _read_saver, _read_saver_path

    path = str(runtime.checkpoint_path())
    with _read_lock:
        if _read_saver is None or _read_saver_path != path:
            _reset_read_cache_locked()
            import sqlite3

            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
            from langgraph.checkpoint.sqlite import SqliteSaver

            from mlr.state import CHECKPOINT_TYPES
            from platform_core.runtime import _allowlist

            conn = sqlite3.connect(path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            _read_saver = SqliteSaver(
                conn, serde=JsonPlusSerializer(allowed_msgpack_modules=_allowlist(CHECKPOINT_TYPES))
            )
            _read_saver_path = path

        key = (path, step)
        if key not in _read_graphs:
            _read_graphs[key] = compile_with(_read_saver, step=step)
        return _read_graphs[key]


def _reset_read_cache_locked() -> None:
    """Drop the cached reader. Caller holds ``_read_lock``."""
    global _read_saver, _read_saver_path
    _read_graphs.clear()
    if _read_saver is not None:
        # A closed connection is the desired end state either way.
        with contextlib.suppress(Exception):  # pragma: no cover
            _read_saver.conn.close()
    _read_saver, _read_saver_path = None, ""


def reset_read_cache() -> None:
    """Drop the cached reader. Tests call this when they change the data directory."""
    with _read_lock:
        _reset_read_cache_locked()


def _fake_for(document: str, tenant_id: str) -> Any:
    """A scripted model, when the configured backend has none of its own."""
    if backend() != "fake":
        return None
    from mlr import demo

    return demo.build_fake(document, tenant_id=tenant_id)


def _prepare(document: str, doc_meta: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Normalise the submission the same way ingest will, so matching and layout agree."""
    _, body = parse_front_matter(document)
    text = normalize_document(body)
    meta = build_doc_meta(text)
    if doc_meta:
        meta = {**meta, **doc_meta}
    return text, meta


# ---------------------------------------------------------------------------------------------
# Starting and advancing a run
# ---------------------------------------------------------------------------------------------


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
    step: bool = False,
    block: bool = False,
) -> str:
    """Submit a piece for review and return its run id.

    The run is registered before this returns, so polling immediately is safe. Execution happens on a
    worker thread unless ``block=True``, which tests and the CLI use to get a deterministic result
    without polling.

    ``audience`` and ``channel`` are required inputs rather than inferred from the text, because they
    determine which tenant rules are in scope -- applying a patient-facing disclaimer rule to a piece
    written for clinicians would manufacture false blockers.

    ``step=True`` pauses the run before each node, for :func:`advance`.

    Raises:
        TenantConfigError: if the tenant or the product is unknown. Raised synchronously, so a bad
            submission fails at the API boundary rather than as a failed run the caller has to poll for.
    """
    config = load_tenant(tenant_id)
    product = config.product(product_id)

    rid = run_id or runtime.new_run_id()
    text, meta = _prepare(document, doc_meta)

    store.create(
        rid,
        tenant_id=tenant_id,
        product_id=product_id,
        label_version=label_version or product.label_version,
        audience=audience or "any",
        channel=channel,
        market=market,
        stepping=step,
    )

    state = initial_state(
        run_id=rid,
        tenant_id=tenant_id,
        product_id=product_id,
        label_version=label_version or product.label_version,
        audience=audience or "any",
        channel=channel,
        market=market,
        document=text,
        doc_meta=meta,
    )

    def execute() -> None:
        _execute(rid, dict(state), text=text, tenant_id=tenant_id, step=step)

    if block:
        execute()
    else:
        _pool().submit(execute)
    return rid


def _execute(run_id: str, state: dict[str, Any], *, text: str, tenant_id: str, step: bool) -> None:
    """Run the graph and keep the index row in step with it.

    Any exception is recorded on the run rather than lost in a worker thread. A review that crashed must
    be visible as failed, not silently stuck at ``queued`` -- which a caller polling for a report would
    otherwise wait on forever.
    """
    store.update(run_id, status="running")
    try:
        with checkpointer() as saver:
            graph = compile_with(saver, fake=_fake_for(text, tenant_id), step=step)
            result = runtime.start_run(graph, state, run_id=run_id)
            _sync_index(run_id, result["state"], graph=graph)
    except Exception as exc:  # noqa: BLE001 - the failure belongs on the run, not in the log only
        store.update(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")


def advance(run_id: str) -> ReviewReport:
    """Run the next node of a stepped run, then return the state.

    Raises:
        RunNotFoundError: if the id is unknown.
        ValueError: if the run was not started in step mode, or has nothing left to advance.
    """
    record = store.get(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    if not record.stepping:
        raise ValueError(f"run {run_id} was not started in step mode")
    if record.status == "awaiting_human":
        # A stepped run still stops at the human step, and that stop is not a step to advance past.
        # Advancing it would resume the interrupt with no decision, which fails validation deep inside
        # the node. Refusing here says the right thing: the run is waiting for a person.
        raise ValueError(
            f"run {run_id} is waiting for a reviewer; call resume() with a decision rather than "
            f"advance()"
        )

    with checkpointer() as saver:
        graph = compile_with(
            saver, fake=_fake_for(_document_for(run_id), record.tenant_id), step=True
        )
        snapshot = graph.get_state(runtime.thread_config(run_id))
        if not (snapshot.next or ()):
            raise ValueError(f"run {run_id} has no further steps")
        # Resuming a stepped graph with None runs exactly one node, then pauses before the next.
        result = graph.invoke(None, config=runtime.thread_config(run_id))
        _sync_index(run_id, result, graph=graph)
    return get_status(run_id)


def _document_for(run_id: str) -> str:
    """The submitted text, read back from the checkpoint.

    Needed because a stepped run is advanced by a later HTTP request, by which point the original
    submission is no longer in memory -- and the scripted model has to be primed from the same text the
    run started with, or the demo's findings would not match the piece.
    """
    snapshot = _read_graph(True).get_state(runtime.thread_config(run_id))
    values = getattr(snapshot, "values", None) or {}
    return str(values.get("document") or "")


def _sync_index(run_id: str, state: dict[str, Any], *, graph: Any) -> None:
    """Copy the handful of fields a listing needs out of the checkpoint."""
    findings = report_findings(state)
    snapshot_status = runtime.get_status(graph, run_id)["status"]
    status = str(state.get("status") or "")

    if snapshot_status == "awaiting_human":
        status = "awaiting_human"
    elif snapshot_status == "running":
        status = "running"
    elif status not in ("complete", "failed"):
        status = snapshot_status if snapshot_status != "unknown" else status or "running"

    store.update(
        run_id,
        status=status,
        route=str(state.get("route") or ""),
        blockers=sum(1 for f in findings if f.is_blocker),
        advisories=sum(1 for f in findings if not f.is_blocker),
        claims=len(state.get("claims") or []),
        ungrounded=sum(1 for f in findings if not f.grounded),
        error="; ".join(str(e) for e in state.get("errors") or [])[:500],
    )


# ---------------------------------------------------------------------------------------------
# Reading a run
# ---------------------------------------------------------------------------------------------


def _report_from(run_id: str, record: store.RunRecord, state: dict[str, Any]) -> ReviewReport:
    cost = state.get("cost")
    if isinstance(cost, dict):
        cost = RunCost.model_validate(cost)

    rules = state.get("rules_applied") or []
    applied = [r if isinstance(r, AppliedRule) else AppliedRule.model_validate(r) for r in rules]

    return ReviewReport(
        run_id=run_id,
        tenant_id=record.tenant_id,
        product_id=record.product_id,
        label_version=record.label_version,
        audience=record.audience,
        channel=record.channel,
        market=record.market,
        status=record.status,  # type: ignore[arg-type]
        route=str(state.get("route") or ""),
        escalation=state.get("escalation"),
        findings=report_findings(state),
        gate_results=list(state.get("gate_results") or []),
        rules_applied=applied,
        claims_extracted=len(state.get("claims") or []),
        preapproved_matches=dict(state.get("preapproved_matches") or {}),
        policy_version=str(state.get("policy_version") or ""),
        config_version=_config_version(record.tenant_id),
        backend=describe_backend()["backend"],
        cost=cost or RunCost(),
        error=record.error,
    )


def _config_version(tenant_id: str) -> str:
    try:
        return load_tenant(tenant_id).version
    except Exception:  # pragma: no cover
        return ""


def get_status(run_id: str) -> ReviewReport:
    """Current state of a run, as the report the UI renders.

    The same shape at every stage. ``status`` is the only field whose value gates what is populated, so
    a caller never has to branch on a different payload.

    Raises:
        RunNotFoundError: if the id is unknown.
    """
    record = store.get(run_id)
    if record is None:
        raise RunNotFoundError(run_id)

    if record.status in ("queued", "failed"):
        # "queued": registered but not started. A UI polling straight after submitting should see
        # "queued", not a 404.
        # "failed": the run may have died before any checkpoint was written, or because the
        # checkpointer itself is unavailable -- so reporting the failure must not depend on reading one.
        return _report_from(run_id, record, {})

    values = runtime.get_status(_read_graph(record.stepping), run_id)["values"] or {}
    return _report_from(run_id, record, values)


def get_state(run_id: str) -> dict[str, Any]:
    """The raw checkpoint, plus which nodes are pending.

    For the audit page's "what did the system know at this step" view, and for a stepped demo that wants
    to show the next node before running it.
    """
    record = store.get(run_id)
    if record is None:
        raise RunNotFoundError(run_id)

    status = runtime.get_status(_read_graph(record.stepping), run_id)

    return {
        "run_id": run_id,
        "status": record.status,
        "next": status["next"],
        "stepping": record.stepping,
        "steppable_nodes": list(STEPPABLE_NODES),
        "interrupt": status["interrupt"],
        "values": status["values"],
    }


def get_history(run_id: str) -> list[dict[str, Any]]:
    """Checkpoint history, oldest first.

    Each entry is what the system actually knew at that step, rather than a reconstruction after the
    fact -- which is the difference between a replay view and a plausible story.
    """
    record = store.get(run_id)
    if record is None:
        raise RunNotFoundError(run_id)

    return runtime.history(_read_graph(record.stepping), run_id)


def get_review_task(run_id: str) -> ReviewTask | None:
    """What the reviewer is being asked, when a run is suspended at the human step.

    None when the run is not waiting on anyone.
    """
    record = store.get(run_id)
    if record is None:
        raise RunNotFoundError(run_id)
    if record.status != "awaiting_human":
        return None

    status = runtime.get_status(_read_graph(record.stepping), run_id)

    payload = status.get("interrupt")
    if payload:
        return ReviewTask.model_validate(payload)

    # Suspended but the payload is not in the snapshot: rebuild it from state rather than returning
    # nothing, so a reviewer is never shown an empty queue entry for a run that is genuinely waiting.
    from mlr.nodes.human_review import build_task

    return build_task(status["values"] or {})


def get_queue(tier: str = "") -> list[ReviewTask]:
    """Pending review tasks, optionally for one reviewer tier.

    Tier filtering is how the demo simulates logging in as a senior reviewer rather than a standard one.
    There is no authentication here; a real deployment would derive the tier from the authenticated user
    rather than a query parameter.
    """
    tasks: list[ReviewTask] = []
    for record in store.list_runs(status="awaiting_human"):
        if tier and record.route != tier:
            continue
        task = get_review_task(record.run_id)
        if task is not None:
            tasks.append(task)
    return tasks


def resume(run_id: str, decision: dict[str, Any] | ReviewDecision) -> ReviewReport:
    """Record a human decision and resume the suspended run.

    The decision is validated before the graph sees it: it arrives over HTTP, and an invalid payload
    must fail loudly rather than record an approval nobody gave.

    Resuming continues from the checkpoint, so nothing before the human step is re-run or re-charged.

    Raises:
        RunNotFoundError: if the id is unknown.
        ValidationError: if the decision payload is malformed.
    """
    record = store.get(run_id)
    if record is None:
        raise RunNotFoundError(run_id)

    # Validated here, before anything is written, so a malformed decision changes nothing.
    parsed = parse_decision(decision)

    with checkpointer() as saver:
        graph = compile_with(
            saver, fake=_fake_for(_document_for(run_id), record.tenant_id), step=record.stepping
        )
        result = runtime.resume_run(graph, run_id, parsed.model_dump(mode="json"))
        _sync_index(run_id, result["state"], graph=graph)
    return get_status(run_id)


def get_audit(run_id: str) -> list[dict[str, Any]]:
    """Ordered audit events for a run.

    One event per node, gate and human decision, each carrying the model, prompt version, policy
    version, tokens, cost and latency that produced it. This is what makes a decision defensible months
    later, so it is part of the interface rather than an admin endpoint.
    """
    if not store.exists(run_id):
        raise RunNotFoundError(run_id)
    return runtime.get_audit(run_id)


def get_audit_integrity(run_id: str) -> dict[str, Any]:
    """Is this run's event sequence complete and free of duplicates?"""
    if not store.exists(run_id):
        raise RunNotFoundError(run_id)
    return audit.integrity_check(run_id)


def list_runs(tenant_id: str = "", status: str = "") -> list[ReviewReport]:
    """Recent runs, newest first, optionally filtered.

    Built from the index rather than from checkpoints: a listing needs identity, status, route and
    counts, and deserialising every run's full state to read four fields would make the page slower the
    more it has to show.
    """
    reports: list[ReviewReport] = []
    for record in store.list_runs(tenant_id=tenant_id, status=status):
        reports.append(
            ReviewReport(
                run_id=record.run_id,
                tenant_id=record.tenant_id,
                product_id=record.product_id,
                label_version=record.label_version,
                audience=record.audience,
                channel=record.channel,
                market=record.market,
                status=record.status,  # type: ignore[arg-type]
                route=record.route,
                claims_extracted=record.claims,
                config_version=_config_version(record.tenant_id),
                backend=describe_backend()["backend"],
                error=record.error,
            )
        )
    return reports


def get_graph_mermaid() -> str:
    """Mermaid source for the compiled graph.

    Rendered from the compiled graph rather than drawn by hand, so the diagram in the README and on the
    UI home page cannot drift from the code it documents.
    """
    return runtime.render_mermaid(_read_graph(False))
