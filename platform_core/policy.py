"""Composite policy version: the single value that identifies "the rules in force".

Why this exists, and why it is not just the tenant config hash: a decision is only defensible if you
can say which policy produced it. Policy now has two parts -- the tenant's configuration (thresholds,
severity mappings, escalation tiers) and the tenant's business rules, which compliance owners edit
through a web form.

If ``policy_version`` stayed the config hash alone, editing a rule would not move it. Two runs with
materially different rules in force would then claim identical provenance in the audit log, and an
auditor comparing them would be misled by the system's own records. That is the failure this module
exists to prevent.

Deterministic code. No model involved.
"""

from __future__ import annotations

import hashlib

from platform_core.rules.registry import RuleRegistry
from platform_core.tenancy import load_tenant


def policy_version(tenant_id: str) -> str:
    """Hash identifying the complete policy in force for a tenant right now.

    Composed from the tenant config hash and the active rule set hash. Written into every audit
    event, so any decision can be traced to the exact configuration *and* the exact rule versions
    that produced it.
    """
    config_hash = load_tenant(tenant_id).config_hash()
    rules_hash = RuleRegistry(tenant_id).active_set_hash()
    combined = f"{tenant_id}:{config_hash}:{rules_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def policy_parts(tenant_id: str) -> dict[str, str]:
    """The components behind :func:`policy_version`.

    Surfaced in the audit UI so "the policy changed" can be narrowed to *which half* changed --
    a config edit and a rule edit need different conversations.
    """
    config = load_tenant(tenant_id)
    registry = RuleRegistry(tenant_id)
    return {
        "tenant_id": tenant_id,
        "policy_version": policy_version(tenant_id),
        "config_version": config.version,
        "config_hash": config.config_hash(),
        "rules_hash": registry.active_set_hash(),
        "active_rule_count": str(len(registry.active_rules())),
    }
