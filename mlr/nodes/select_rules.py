"""Select the tenant business rules in force for this submission.

Why it is a node rather than a lookup inside each check: which rules apply is a decision with
consequences, so it is recorded. An auditor asking "why did this piece not get flagged for the
missing disclaimer" needs the answer "that rule is patient-facing only and this was an HCP piece" --
which requires knowing what the selection was at the time, not what it would be today.

**Deterministic code. No model involved.** Selection is policy: tenant, status, effective date and
scope. Nothing here is a judgement call.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from mlr.nodes.base import node_run
from mlr.schemas import AppliedRule
from platform_core.rules.registry import RuleRegistry
from platform_core.rules.schema import Rule


def applicable_rules(state: dict[str, Any], *, on: date | None = None) -> list[Rule]:
    """The rules in force for this submission, in id order.

    Four filters, all of which must pass: the rule belongs to this tenant, it is active, it is inside
    its effective window, and its scope covers this piece.
    """
    registry = RuleRegistry(str(state["tenant_id"]))
    return registry.applicable_rules(
        audience=str(state.get("audience") or "any"),
        channel=str(state.get("channel") or ""),
        product_id=str(state.get("product_id") or ""),
        market=str(state.get("market") or ""),
        on=on,
    )


def select_rules(state: dict[str, Any]) -> dict[str, Any]:
    """Record which rules this run will apply."""
    with node_run(
        state,
        "select_rules",
        input_value={
            "audience": state.get("audience"),
            "channel": state.get("channel"),
            "product_id": state.get("product_id"),
        },
    ) as run:
        try:
            rules = applicable_rules(state)
        except Exception as exc:
            # A registry that cannot be read means no business rules are applied. That is a real
            # reduction in review coverage, so it is recorded as an error and surfaced to the
            # reviewer rather than looking like a tenant with no rules.
            run.error = f"{type(exc).__name__}: {exc}"
            return {
                "rules_applied": [],
                "errors": [f"select_rules: could not load tenant rules ({exc})"],
            }

        applied = [
            AppliedRule(id=r.id, version=r.version, kind=r.kind, severity=r.severity, title=r.title)
            for r in rules
        ]
        run.output = {
            "selected": [f"{r.id} v{r.version}" for r in rules],
            "by_kind": {
                kind: sum(1 for r in rules if r.kind == kind)
                for kind in ("deterministic", "llm", "reference_doc")
            },
        }
        return {"rules_applied": applied}
