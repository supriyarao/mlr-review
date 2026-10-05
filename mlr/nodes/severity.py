"""Assign each finding its final severity from tenant policy.

**Deterministic code. No model involved.** Severity is policy, not science: the same overstatement is
a blocker for one client and an advisory for another, and that difference belongs in a YAML file a
compliance owner can edit, not in a branch in this module. The rules are read from
``tenant.severity_rules``; this node only applies them.

One case needs explaining. A finding produced by a **tenant business rule keeps that rule's own
severity.** Re-deriving it from ``check_type`` would run it through the tenant's generic mapping,
where ``business_rule`` falls to the catch-all -- so a blocker the tenant explicitly wrote would be
silently downgraded to advisory. The tenant already stated the severity when they authored the rule;
overriding that would be the system second-guessing its own configuration.
"""

from __future__ import annotations

from typing import Any

from mlr.nodes.base import node_run
from mlr.schemas import Finding
from platform_core.tenancy import SeverityRule, TenantConfig, load_tenant


def resolve_severity(finding: Finding, config: TenantConfig) -> tuple[str, str]:
    """Return ``(severity, rule_id)`` for one finding.

    Matching order is the tenant's YAML order, so a tenant controls precedence by arranging their
    file -- the most specific rule first. A ``check_type`` of ``"*"`` matches anything, which is how a
    tenant supplies a catch-all.
    """
    # A rule finding carries the tenant's own decision about its own rule.
    if finding.rule_id:
        return finding.severity_proposed, f"rule:{finding.rule_id}"

    exact: SeverityRule | None = None
    wildcard: SeverityRule | None = None
    for rule in config.severity_rules:
        if rule.check_type == finding.check_type and exact is None:
            exact = rule
        elif rule.check_type == "*" and wildcard is None:
            wildcard = rule

    chosen = exact or wildcard
    if chosen is None:
        # No applicable rule. Defaults to advisory and routes to a human rather than being dropped:
        # an unclassified observation is still an observation.
        return "advisory", "unmapped"
    return chosen.severity, chosen.rule_id


def severity(state: dict[str, Any]) -> dict[str, Any]:
    """Apply the tenant's severity rules to every finding."""
    with node_run(state, "severity") as run:
        from mlr.state import report_findings

        findings = report_findings(state)
        if not findings:
            run.output = {"findings": 0}
            return {}

        config = load_tenant(str(state["tenant_id"]))
        graded: list[Finding] = []
        changed = 0

        for finding in findings:
            level, rule_id = resolve_severity(finding, config)
            if level != finding.severity_proposed:
                changed += 1
            graded.append(
                finding.model_copy(update={"severity_final": level, "severity_rule_id": rule_id})
            )

        run.output = {
            "findings": len(graded),
            "blockers": sum(1 for f in graded if f.is_blocker),
            "advisories": sum(1 for f in graded if not f.is_blocker),
            # Visible on purpose: this is the tenant overriding what a check proposed, and it is one
            # of the most interesting things in the audit trail.
            "severity_changed_from_proposal": changed,
            "tenant_config_version": config.version,
        }
        return {"final_findings": graded}
