"""Review endpoints: submit a piece, watch it, decide it, audit it, export it.

Every handler here is deterministic glue over ``mlr.service``. No model is called from this module;
the graph behind the façade is the only thing that calls one.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile

from api import export, ingest, steps, submissions
from api.deps import known_tenant
from mlr import service
from mlr.schemas import ReviewDecision, ReviewReport, RunStatus
from platform_core.tenancy import TenantConfigError

router = APIRouter(tags=["reviews"])

Audience = Literal["hcp", "patient", "any"]

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _report(run_id: str) -> ReviewReport:
    """The run's report, or a 404. An id that could not be a run id is not looked up at all."""
    if not submissions.is_safe_run_id(run_id):
        raise HTTPException(404, f"no run {run_id!r}")
    return service.get_status(run_id)


def _document(run_id: str) -> str:
    """The text the checks actually read, which is what every finding's quote and offset refer to.

    Taken from the run's own state once ingest has run, because ingest normalises the text (it
    strips emphasis markers, for one) and offsets into the raw upload would point at the wrong
    words. The copy kept at submission is only a fallback for a run that has not got that far.
    """
    normalised = (service.get_state(run_id)["values"] or {}).get("document")
    if normalised:
        return str(normalised)
    stored = submissions.load(run_id)
    return stored["document"] if stored else ""


@router.post("/reviews", status_code=202)
async def submit_review(
    tenant_id: Annotated[str, Form()],
    product_id: Annotated[str, Form()],
    file: Annotated[UploadFile | None, File()] = None,
    text: Annotated[str, Form()] = "",
    audience: Annotated[Audience, Form()] = "any",
    channel: Annotated[str, Form()] = "",
    market: Annotated[str, Form()] = "",
    label_version: Annotated[str, Form()] = "",
) -> dict[str, str]:
    """Submit a piece for review. Returns at once; the review runs in the background.

    ``audience`` and ``channel`` are form fields rather than something read out of the document,
    because they decide which tenant rules are in scope and that has to be a person's statement.
    """
    config = known_tenant(tenant_id)
    try:
        config.product(product_id)
    except TenantConfigError as exc:
        raise HTTPException(422, str(exc)) from exc

    has_file = file is not None and bool(file.filename)
    if has_file == bool(text.strip()):
        raise HTTPException(422, "provide exactly one of: an uploaded file, or text")

    try:
        if has_file:
            filename = file.filename or ""
            document = ingest.extract_text(filename, await file.read())
        else:
            filename = ""
            _, document = ingest.split_front_matter(text)
    except ingest.IngestError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc

    if not document.strip():
        raise HTTPException(422, "the submission contains no text to review")

    run_id = service.start_run(
        tenant_id=tenant_id,
        product_id=product_id,
        document=document,
        label_version=label_version,
        audience=audience,
        channel=channel,
        market=market,
    )
    submissions.save(
        run_id,
        document=document,
        filename=filename,
        tenant_id=tenant_id,
        product_id=product_id,
        audience=audience,
        channel=channel,
        market=market,
    )
    return {"run_id": run_id, "status": "queued"}


@router.get("/reviews")
def list_reviews(
    tenant_id: str = "", status: Annotated[RunStatus | None, Query()] = None
) -> dict[str, Any]:
    """Recent runs, newest first. A summary per run; fetch the run for its findings."""
    reports = service.list_runs(tenant_id=tenant_id, status=status or "")
    runs = [
        {
            "run_id": r.run_id,
            "tenant_id": r.tenant_id,
            "product_id": r.product_id,
            "status": r.status,
            "route": r.route,
            "blockers": len(r.blockers()),
            "advisories": len(r.advisories()),
            "cost_usd": r.cost.cost_usd,
        }
        for r in reports
    ]
    return {"runs": runs, "total": len(runs)}


@router.get("/queue")
def review_queue(tier: str = "") -> dict[str, Any]:
    """Pending human review tasks, optionally for one reviewer tier.

    Trade-off: the tier is a query parameter because the demo has no login. In a real deployment it
    is derived from the authenticated user, and this endpoint would not accept it from the client.
    """
    tasks = service.get_queue(tier)
    return {"tier": tier, "tasks": [t.model_dump(mode="json") for t in tasks]}


@router.get("/reviews/{run_id}")
def get_review(run_id: str) -> ReviewReport:
    """Current state of a run. The same shape at every stage; ``status`` says how far it has got."""
    return _report(run_id)


@router.post("/reviews/{run_id}/decision")
def decide_review(run_id: str, decision: ReviewDecision) -> ReviewReport:
    """Record a reviewer's decision and resume the suspended run.

    Two checks sit in front of the façade because they are about the request, not the review: the
    run must actually be waiting for a person, and the person must be of the tier it was routed to.
    Without the second, a standard reviewer could clear a piece that policy escalated past them.
    """
    _report(run_id)
    task = service.get_review_task(run_id)
    if task is None:
        raise HTTPException(409, f"run {run_id} is not awaiting a human decision")
    if decision.tier != task.tier:
        raise HTTPException(
            403, f"run {run_id} is routed to {task.tier!r}; a {decision.tier!r} cannot decide it"
        )
    return service.resume(run_id, decision)


def integrity_of(run_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Are the sequence numbers of *these* events complete and unique?

    Computed over the events being returned, so the integrity statement describes exactly the trail
    the caller is looking at.
    """
    seqs = [int(e["seq"]) for e in events]
    return {
        "run_id": run_id,
        "event_count": len(events),
        "sequence_complete": seqs == list(range(1, len(events) + 1)),
        "duplicate_seqs": sorted({s for s in seqs if seqs.count(s) > 1}),
    }


@router.get("/reviews/{run_id}/audit")
def get_audit(run_id: str) -> dict[str, Any]:
    """Ordered audit events for a run, with a completeness check over them."""
    _report(run_id)
    events = service.get_audit(run_id)
    return {"run_id": run_id, "events": events, "integrity": integrity_of(run_id, events)}


@router.get("/reviews/{run_id}/steps")
def get_steps(run_id: str) -> dict[str, Any]:
    """The run node by node: each step's status, what it was given and what it produced."""
    _report(run_id)
    return steps.steps(run_id)


@router.get("/reviews/{run_id}/document")
def get_document(run_id: str) -> dict[str, Any]:
    """The text that was submitted, so a finding's quote can be shown highlighted in place."""
    _report(run_id)
    document = _document(run_id)
    if not document:
        raise HTTPException(404, f"no stored document for run {run_id}")
    stored = submissions.load(run_id) or {"run_id": run_id, "filename": ""}
    return {**stored, "document": document}


@router.get("/reviews/{run_id}/report.json")
def export_json(run_id: str) -> dict[str, Any]:
    """The findings export as JSON: one row per finding."""
    return export.export_payload(_report(run_id), _document(run_id))


@router.get("/reviews/{run_id}/report.xlsx")
def export_xlsx(run_id: str) -> Response:
    """The findings export as a spreadsheet, with the same columns as the JSON rows."""
    content = export.export_xlsx(_report(run_id), _document(run_id))
    return Response(
        content,
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="mlr-review-{run_id}.xlsx"'},
    )
