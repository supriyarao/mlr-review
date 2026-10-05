"""Eval endpoints: start a run, list runs, read one, compare with the baseline, sweep thresholds.

Deterministic glue. See ``api/evals.py`` for what is real today and what is behind a seam; while the
runner is not built, the affected responses are the committed contract examples and say so in an
``X-MLR-Stub`` header.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from api import evals, stubs
from api.deps import known_tenant

router = APIRouter(tags=["evals"])

STUB_NAME = "eval_runner"


class EvalRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "offline" replays recorded model responses and is free; "live" calls the configured backend.
    mode: Literal["offline", "live"] = "offline"
    tenant_id: str = ""
    repeats: int = Field(default=1, ge=1, le=10)


class SweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thresholds: list[int] = Field(default_factory=lambda: list(evals.DEFAULT_THRESHOLDS))
    eval_run_id: str = ""


def _mark_stub(response: Response) -> None:
    response.headers[stubs.STUB_HEADER] = STUB_NAME


@router.post("/evals/run", status_code=202)
def start_eval(
    body: EvalRunRequest, background: BackgroundTasks, response: Response
) -> dict[str, Any]:
    """Start an eval over the golden set. Returns at once; poll ``GET /evals/runs/{id}``."""
    if body.tenant_id:
        known_tenant(body.tenant_id)
    if evals.is_stub():
        _mark_stub(response)
        return {**stubs.example("POST_evals_run.json"), "mode": body.mode}

    eval_run_id = evals.new_job(body.mode)
    background.add_task(
        evals.execute,
        eval_run_id,
        mode=body.mode,
        tenant_id=body.tenant_id,
        repeats=body.repeats,
    )
    return {"eval_run_id": eval_run_id, "mode": body.mode, "status": "queued"}


@router.get("/evals/runs")
def list_evals(response: Response) -> dict[str, Any]:
    """Stored eval runs, newest first."""
    results = evals.stored_results()
    if not results and evals.is_stub():
        _mark_stub(response)
        return stubs.example("GET_evals_runs.json")
    return {"runs": [evals.summary(r) for r in results]}


@router.get("/evals/baseline")
def get_baseline() -> dict[str, Any]:
    """The committed baseline every run is compared against."""
    base = evals.baseline()
    if base is None:
        raise HTTPException(404, "no baseline has been committed yet")
    return evals.detail(base)


@router.get("/evals/runs/{eval_run_id}")
def get_eval(eval_run_id: str, response: Response) -> dict[str, Any]:
    """One eval run: provenance, metrics and per-case results."""
    result = evals.find_result(eval_run_id)
    if result is not None:
        return evals.detail(result)

    state = evals.job(eval_run_id)
    if state is not None:
        return evals.pending_detail(eval_run_id, state)

    if evals.is_stub():
        example = stubs.example("GET_evals_runs_{id}.json")
        if example["eval_run_id"] == eval_run_id:
            _mark_stub(response)
            return {**example, "status": "complete", "error": ""}
    raise HTTPException(404, f"no eval run {eval_run_id!r}")


@router.get("/evals/runs/{eval_run_id}/comparison")
def compare_eval(eval_run_id: str) -> dict[str, Any]:
    """How one run moved against the baseline: guarded metrics, and the cases that changed."""
    result = evals.find_result(eval_run_id)
    if result is None:
        raise HTTPException(404, f"no stored result for eval run {eval_run_id!r}")
    return evals.comparison(result)


@router.post("/evals/sweep")
def sweep_thresholds(body: SweepRequest, response: Response) -> dict[str, Any]:
    """Blocker recall against false-escalation rate across confidence thresholds.

    Recomputed from stored findings, with no model calls, so the curve isolates the effect of the
    threshold from run-to-run model variation.
    """
    if not body.thresholds or any(not 0 <= t <= 100 for t in body.thresholds):
        raise HTTPException(422, "thresholds must be a non-empty list of values from 0 to 100")
    if evals.is_stub():
        _mark_stub(response)
        return stubs.example("POST_evals_sweep.json")

    swept = evals.sweep(body.thresholds, body.eval_run_id)
    if swept is None:
        raise HTTPException(404, "no stored eval result to sweep; run an eval first")
    return swept
