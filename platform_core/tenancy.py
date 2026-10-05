"""Tenant configuration: who the client is, and what their policy says.

Why this exists: a multi-tenant review platform has to behave differently per client without
growing a branch per client. Every tenant-specific decision the graph makes -- what counts as a
blocker, how confident a finding must be, who an escalation goes to -- is read from a YAML file
validated here, never from an ``if tenant_id == ...`` in node code.

Deterministic code. No model involved.

The other half of the contract is auditability: each config carries a ``version``, and
:meth:`TenantConfig.config_hash` is a content hash of the whole config. That hash is written into
every audit event, so months later an auditor can establish exactly which policy produced a
given decision -- even if the YAML has since been edited.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_TENANT_DIR = Path(__file__).resolve().parent.parent / "corpus" / "tenants"


class TenantConfigError(Exception):
    """Raised when a tenant config is missing, malformed, or fails validation."""


class Product(BaseModel):
    """A product a tenant may submit promotional content about.

    ``label_version`` matters for retrieval: evidence is only valid against the label version it
    was approved under, so it is part of the retrieval filter rather than free-text metadata.
    """

    model_config = ConfigDict(extra="forbid")

    product_id: str
    name: str
    label_version: str
    indication: str


class SeverityRule(BaseModel):
    """Maps a finding's check type (and optionally a policy tag) to a severity.

    Severity is policy, not science: the same overstatement can be a blocker for one client and
    an advisory for another. Keeping these as data is what makes that true in practice.
    """

    model_config = ConfigDict(extra="forbid")

    # The check that produced the finding, or "*" to match any check.
    check_type: str = "*"
    # Optional extra condition the use case can set on a finding, e.g. "comparative_claim".
    when_tag: str | None = None
    severity: Literal["blocker", "advisory"]
    # Human-readable rule id, quoted back to the reviewer and recorded in the audit log so a
    # decision can always be traced to the rule that drove it.
    rule_id: str
    rationale: str = ""


class EscalationRule(BaseModel):
    """Routes an escalation to a reviewer tier based on why it escalated.

    Two tenants deliberately differ here: one sends everything serious to a single senior tier,
    the other splits off-label to a medical director and comparative claims to legal.
    """

    model_config = ConfigDict(extra="forbid")

    # Reason emitted by the use case, e.g. "off_label", "comparative_claim", "ungrounded".
    when_reason: str
    tier: str


class EscalationPolicy(BaseModel):
    """Where reviews go. There is always a human at the end of every path."""

    model_config = ConfigDict(extra="forbid")

    # Where a clean or advisory-only piece goes. Never "auto-approve": even a review with no
    # findings is routed to a person.
    standard_tier: str = "standard_reviewer"
    # Fallback tier when no specific rule matches an escalation reason.
    default_escalation_tier: str
    rules: list[EscalationRule] = Field(default_factory=list)

    def tier_for(self, reasons: list[str]) -> str:
        """Pick the tier for a set of escalation reasons.

        First matching rule wins, in config order, so a tenant controls precedence by ordering
        their YAML. Falls back to ``default_escalation_tier``.
        """
        for rule in self.rules:
            if rule.when_reason in reasons:
                return rule.tier
        return self.default_escalation_tier


class TenantConfig(BaseModel):
    """A single tenant's identity and policy."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    name: str
    # Bumped by hand when policy changes. Distinct from config_hash(): the version is the
    # tenant's own label for a policy generation, the hash is tamper-evident proof of content.
    version: str
    # Findings below this confidence are flagged low_confidence and force escalation. The
    # safety/cycle-time dial -- the eval harness sweeps it to show the trade-off curve.
    confidence_threshold: int = Field(ge=0, le=100)
    products: list[Product]
    severity_rules: list[SeverityRule]
    escalation: EscalationPolicy
    # Open-ended switches the use case interprets. Deliberately untyped at the platform layer:
    # "comparative claims need head-to-head data" is an MLR concept, and platform_core must not
    # know what an MLR claim is. The use case validates the keys it cares about.
    policy_flags: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tenant_id")
    @classmethod
    def _tenant_id_is_safe_as_an_identifier(cls, v: str) -> str:
        """tenant_id becomes part of a vector-store collection name and a filesystem path.

        Restricting it to a conservative character set is a cheap way to keep a config file from
        turning into a path-traversal or collection-name-collision problem.
        """
        if not v or not all(c.isalnum() or c == "_" for c in v):
            raise ValueError(f"tenant_id must be alphanumeric/underscore, got {v!r}")
        return v

    def product(self, product_id: str) -> Product:
        for p in self.products:
            if p.product_id == product_id:
                return p
        known = ", ".join(p.product_id for p in self.products) or "none"
        raise TenantConfigError(
            f"tenant {self.tenant_id!r} has no product {product_id!r} (known: {known})"
        )

    def config_hash(self) -> str:
        """Content hash of the whole config, recorded in every audit event.

        Canonical JSON (sorted keys) so the hash is stable across YAML key reordering and
        formatting changes -- it should only move when policy actually moves.
        """
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def tenant_dir() -> Path:
    """Root directory holding one subdirectory per tenant. Overridable via ``MLR_TENANT_DIR``.

    Layout, one directory per tenant::

        corpus/tenants/acme_pharma/
            tenant.yaml          # the TenantConfig validated by this module
            rules/*.yaml         # business rules, owned by the tenant's compliance team
            reference_docs/*.md  # codes of practice, style guides cited by reference_doc rules

    A directory rather than a single file because a tenant owns more than a config: rules and the
    documents those rules cite are tenant content too, and keeping them together is what makes
    "onboard a tenant" a matter of adding a folder.
    """
    override = os.environ.get("MLR_TENANT_DIR")
    return Path(override) if override else DEFAULT_TENANT_DIR


def tenant_path(tenant_id: str) -> Path:
    """Directory for one tenant."""
    return tenant_dir() / tenant_id


def tenant_config_path(tenant_id: str) -> Path:
    """The tenant's config file."""
    return tenant_path(tenant_id) / "tenant.yaml"


def tenant_rules_dir(tenant_id: str) -> Path:
    """Where this tenant's rule YAML files live. Seeded into SQLite by scripts/seed.py."""
    return tenant_path(tenant_id) / "rules"


def tenant_reference_docs_dir(tenant_id: str) -> Path:
    """Where documents cited by this tenant's reference_doc rules live."""
    return tenant_path(tenant_id) / "reference_docs"


def load_tenant_from_path(path: Path) -> TenantConfig:
    """Load and validate one tenant config file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TenantConfigError(f"no tenant config at {path}") from exc
    except yaml.YAMLError as exc:
        raise TenantConfigError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise TenantConfigError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")

    try:
        return TenantConfig.model_validate(raw)
    except Exception as exc:
        # Surfaced as a config error rather than a Pydantic traceback: this is the error a
        # platform operator sees when onboarding a tenant, so it should read as a config problem.
        raise TenantConfigError(f"{path} failed validation: {exc}") from exc


@cache
def load_tenant(tenant_id: str) -> TenantConfig:
    """Load one tenant by id.

    Cached because it is read on nearly every node call and the file does not change inside a
    run. Call :func:`clear_tenant_cache` after editing configs (the test suite and the UI do).
    """
    config = load_tenant_from_path(tenant_config_path(tenant_id))
    if config.tenant_id != tenant_id:
        raise TenantConfigError(
            f"tenant_id mismatch: {tenant_id}/tenant.yaml declares {config.tenant_id!r}"
        )
    return config


def tenant_ids() -> list[str]:
    """Ids of every configured tenant, sorted.

    A directory counts as a tenant only if it actually holds a ``tenant.yaml``, so a stray folder
    (``.DS_Store``, an editor backup, a half-finished onboarding) is skipped rather than raising.
    """
    root = tenant_dir()
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "tenant.yaml").is_file())


def load_all_tenants() -> list[TenantConfig]:
    """Every configured tenant, sorted by id for stable display and test ordering."""
    return [load_tenant(tenant_id) for tenant_id in tenant_ids()]


def clear_tenant_cache() -> None:
    """Drop the config cache. Call after changing tenant files."""
    load_tenant.cache_clear()
