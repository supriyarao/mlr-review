"""The four specialist checks, fanned out in parallel.

substantiation, off_label, fair_balance, safety_prominence.

Each is a thin binding over :func:`mlr.nodes.checks.runner.run_check`: what distinguishes them is
their prompt and evidence, not their mechanics, so the mechanics live in one place. The separate
functions exist because LangGraph needs a named callable per node, and because a node name is what
appears in the audit trail and the UI timeline.
"""

from __future__ import annotations

from typing import Any

from mlr.nodes.checks.runner import run_check


def substantiation(state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """Is each claim supported by the evidence at the strength stated, and does the reference it
    cites report the claimed endpoint, population and timepoint? Strong model."""
    return run_check("substantiation", state, fake=fake)


def off_label(state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """Explicit or implied use outside the approved label. Strong model."""
    return run_check("off_label", state, fake=fake)


def fair_balance(state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """Is risk information comparable in prominence to the benefit claims? Fast model."""
    return run_check("fair_balance", state, fake=fake)


def safety_prominence(state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """Is required safety information present and not buried? Fast model, reads doc_meta."""
    return run_check("safety_prominence", state, fake=fake)


__all__ = [
    "fair_balance",
    "off_label",
    "run_check",
    "safety_prominence",
    "substantiation",
]
