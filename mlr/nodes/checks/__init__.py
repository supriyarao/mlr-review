"""The three specialist checks, fanned out in parallel.

substantiation, off_label, presentation.

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


def presentation(state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """How the piece presents risk: is the safety information there and readable, and is it given
    weight comparable to the benefit claims? Fast model, reads doc_meta, calls two validators.

    One node, two concerns. Findings carry ``safety_prominence`` or ``fair_balance`` as their
    check_type so a tenant still grades them apart -- Acme treats imbalance as editorial and a
    missing ISI as blocking, which is the whole of golden cases 007 and 008.
    """
    return run_check("presentation", state, fake=fake)


__all__ = [
    "off_label",
    "presentation",
    "run_check",
    "substantiation",
]
