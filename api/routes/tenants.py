"""Tenant endpoints: who is configured, and what policy each one runs under.

Deterministic code. Read-only: a tenant's configuration is a reviewed YAML file, not something
edited over HTTP. (Business rules are editable, and live in ``api/routes/rules.py``.)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from api.deps import known_tenant
from platform_core import tenancy
from platform_core.rules.registry import RuleRegistry
from platform_core.tenancy import TenantConfig

router = APIRouter(tags=["tenants"])


def tenant_payload(config: TenantConfig, *, detail: bool) -> dict[str, Any]:
    """A tenant as the API shows it. ``config_hash`` is included in both forms because it is what
    every audit event records: it lets a reader tie a decision to the policy that produced it."""
    payload: dict[str, Any] = {
        "tenant_id": config.tenant_id,
        "name": config.name,
        "version": config.version,
        "confidence_threshold": config.confidence_threshold,
        "products": [p.model_dump(mode="json") for p in config.products],
        "config_hash": config.config_hash(),
    }
    if detail:
        payload["severity_rules"] = [r.model_dump(mode="json") for r in config.severity_rules]
        payload["escalation"] = config.escalation.model_dump(mode="json")
        payload["policy_flags"] = config.policy_flags
        payload["active_rule_count"] = len(RuleRegistry(config.tenant_id).active_rules())
    return payload


@router.get("/tenants")
def list_tenants() -> dict[str, Any]:
    """Every configured tenant, in summary."""
    return {"tenants": [tenant_payload(c, detail=False) for c in tenancy.load_all_tenants()]}


@router.get("/tenants/{tenant_id}")
def get_tenant(tenant_id: str) -> dict[str, Any]:
    """One tenant's full policy: severity mapping, escalation tiers, thresholds and switches."""
    return tenant_payload(known_tenant(tenant_id), detail=True)
