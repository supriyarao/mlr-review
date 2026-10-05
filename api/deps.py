"""Small lookups shared by the route modules."""

from __future__ import annotations

from fastapi import HTTPException

from platform_core import tenancy
from platform_core.tenancy import TenantConfig


def known_tenant(tenant_id: str) -> TenantConfig:
    """The tenant's config, or a 404.

    Checked against the list of configured tenants *before* anything is loaded: the id comes from a
    URL and the loader turns it into a directory name, so an id that is not on the list must never
    reach the filesystem.
    """
    if tenant_id not in tenancy.tenant_ids():
        raise HTTPException(404, f"no tenant {tenant_id!r}")
    return tenancy.load_tenant(tenant_id)
