"""The FastAPI application: the one HTTP surface in front of the platform.

Deterministic code. No model is called from the API; it validates requests, calls the service
façade (``mlr.service``) and the platform's registries, and shapes the responses documented in
``docs/contracts.md``.

Run with ``make api``. Interactive docs are at ``/docs``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api import evals
from api.export import DISCLAIMER
from api.routes import evals as eval_routes
from api.routes import prompts, reviews, rules, samples, tenants
from mlr import service
from platform_core.rules.registry import RuleStorageError
from platform_core.tenancy import TenantConfigError

app = FastAPI(
    title="MLR Review Agent",
    version="0.1.0",
    summary="Multi-tenant agentic review of pharmaceutical promotional content.",
    description=(
        f"**{DISCLAIMER}**\n\n"
        "The system recommends; a human decides. No endpoint approves content: every run ends "
        "at a reviewer, and `POST /reviews/{run_id}/decision` is the only way a run completes."
    ),
)

app.include_router(reviews.router)
app.include_router(tenants.router)
app.include_router(rules.router)
app.include_router(eval_routes.router)
app.include_router(samples.router)
app.include_router(prompts.router)


# --- Errors --------------------------------------------------------------------------------------
# Domain exceptions are mapped here, once, so no handler needs its own try/except to turn "not
# found" into a 404 -- and so a new endpoint cannot forget to.


@app.exception_handler(service.RunNotFoundError)
def _run_not_found(_request: Request, exc: service.RunNotFoundError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(TenantConfigError)
def _tenant_problem(_request: Request, exc: TenantConfigError) -> JSONResponse:
    """An unknown tenant is a 404 raised before any config is loaded (``api.deps.known_tenant``),
    so an error from the loader itself means a configured tenant's file is broken: a server fault."""
    return JSONResponse({"detail": str(exc)}, status_code=500)


@app.exception_handler(RuleStorageError)
def _rule_problem(_request: Request, exc: RuleStorageError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=404)


@app.exception_handler(ValidationError)
def _invalid_payload(_request: Request, exc: ValidationError) -> JSONResponse:
    """A payload the domain models rejected after FastAPI's own request validation passed -- for
    example rule content nested inside a request body. Same status as any other invalid request."""
    return JSONResponse({"detail": json.loads(exc.json(include_url=False))}, status_code=422)


# --- Service-level endpoints ---------------------------------------------------------------------


@app.get("/health", tags=["service"])
def health() -> dict[str, Any]:
    """Liveness, plus which components are answering with example data.

    The stub flags are here so the UI can say so on screen. A demo that quietly showed fixture data
    as if it were a real review would be worse than one that is upfront about it.
    """
    return {
        "status": "ok",
        "backend": os.environ.get("MLR_LLM_BACKEND", ""),
        "stubs": {"review_service": service.IS_STUB, "eval_runner": evals.is_stub()},
        "disclaimer": DISCLAIMER,
    }


@app.get("/graph", tags=["service"])
def graph() -> dict[str, str]:
    """Mermaid source of the review graph, rendered from the compiled graph rather than drawn."""
    return {"format": "mermaid", "source": service.get_graph_mermaid()}
