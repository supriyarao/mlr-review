"""The prompts the model-backed nodes use, so the UI can show a reader exactly what a model is asked.

Deterministic code. Read-only: prompts are versioned files, changed in a commit, never over HTTP.
The text returned is the fixed instruction part of each prompt. The piece under review, the claims
and the evidence are filled in at run time and are not shown here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from mlr.prompts import checks, extract_claims, preapproved
from platform_core.rules import prompting

router = APIRouter(tags=["prompts"])


def _entry(node: str, version: str, tier: str, file: str, *parts: str) -> dict[str, Any]:
    return {
        "node": node,
        "version": version,
        "tier": tier,
        "file": file,
        "text": "\n\n".join(p.strip() for p in parts if p.strip()),
    }


def all_prompts() -> list[dict[str, Any]]:
    entries = [
        _entry(
            "extract_claims",
            extract_claims.PROMPT_VERSION,
            "strong",
            "mlr/prompts/extract_claims.py",
            extract_claims.SYSTEM,
            extract_claims.TEMPLATE,
        ),
        _entry(
            "preapproved_filter",
            preapproved.PROMPT_VERSION,
            "fast",
            "mlr/prompts/preapproved.py",
            preapproved.SYSTEM,
            preapproved.TEMPLATE,
        ),
    ]
    for check in checks.CHECKS.values():
        entries.append(
            _entry(
                check.name,
                check.version,
                check.tier,
                "mlr/prompts/checks.py",
                check.system,
                check.question,
                # The standing rules as this check actually receives them: the citation half
                # differs depending on whether the check is given evidence or measurements.
                checks._SHARED_OPEN
                + (checks._SHARED_EVIDENCE if check.needs_evidence else checks._SHARED_MEASUREMENT)
                + checks._SHARED_CLOSE,
            )
        )
    # Both rule nodes share the platform's rule-checking instructions; the tenant's own rule text
    # or reference document is added at run time.
    for node, tier in (("llm_rules", "fast"), ("reference_doc_review", "strong")):
        entries.append(
            _entry(
                node,
                prompting.PROMPT_VERSION,
                tier,
                "platform_core/rules/prompting.py",
                prompting._INSTRUCTIONS,
            )
        )
    return entries


@router.get("/prompts")
def list_prompts() -> dict[str, Any]:
    """Every model-backed node's prompt: its version, model tier, source file and instruction text."""
    return {"prompts": all_prompts()}
