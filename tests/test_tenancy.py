"""Tenant configuration: validation, policy lookup, and the audit hash."""

from __future__ import annotations

import pytest

from platform_core import tenancy
from tests.conftest import minimal_tenant


def test_loads_a_valid_tenant(write_tenant):
    config = write_tenant(minimal_tenant())
    assert config.tenant_id == "acme_pharma"
    assert config.confidence_threshold == 70
    assert config.product("zelvoran").indication.endswith("adults")


def test_unknown_product_names_what_is_available(write_tenant):
    config = write_tenant(minimal_tenant())
    with pytest.raises(tenancy.TenantConfigError, match="known: zelvoran"):
        config.product("nexapril")


def test_unknown_field_is_rejected(write_tenant):
    """extra="forbid" so a typo in a policy key fails at load, not silently at runtime."""
    with pytest.raises(tenancy.TenantConfigError, match="failed validation"):
        write_tenant(minimal_tenant(confidence_treshold=70))  # deliberate typo


def test_confidence_threshold_must_be_a_percentage(write_tenant):
    with pytest.raises(tenancy.TenantConfigError):
        write_tenant(minimal_tenant(confidence_threshold=140))


def test_tenant_id_must_be_safe_as_an_identifier(write_tenant):
    """tenant_id becomes a collection name and a path component."""
    with pytest.raises(tenancy.TenantConfigError):
        write_tenant(minimal_tenant(tenant_id="../escape"))


def test_filename_must_match_declared_tenant_id(tenant_dir):
    import yaml

    (tenant_dir / "borealis_bio.yaml").write_text(
        yaml.safe_dump(minimal_tenant("acme_pharma")), encoding="utf-8"
    )
    tenancy.clear_tenant_cache()
    with pytest.raises(tenancy.TenantConfigError, match="tenant_id mismatch"):
        tenancy.load_tenant("borealis_bio")


def test_missing_config_is_a_config_error(tenant_dir):
    with pytest.raises(tenancy.TenantConfigError, match="no tenant config"):
        tenancy.load_tenant("does_not_exist")


def test_malformed_yaml_is_a_config_error(tenant_dir):
    (tenant_dir / "broken.yaml").write_text("name: [unclosed", encoding="utf-8")
    tenancy.clear_tenant_cache()
    with pytest.raises(tenancy.TenantConfigError, match="not valid YAML"):
        tenancy.load_tenant("broken")


def test_config_hash_is_stable_across_formatting(write_tenant, tenant_dir):
    """The hash must only move when policy moves, not when the YAML is reformatted."""
    first = write_tenant(minimal_tenant()).config_hash()

    # Same content, keys written in a different order.
    import yaml

    data = minimal_tenant()
    reordered = dict(reversed(list(data.items())))
    (tenant_dir / "acme_pharma.yaml").write_text(
        yaml.safe_dump(reordered, sort_keys=False), encoding="utf-8"
    )
    tenancy.clear_tenant_cache()
    assert tenancy.load_tenant("acme_pharma").config_hash() == first


def test_config_hash_changes_when_policy_changes(write_tenant):
    before = write_tenant(minimal_tenant()).config_hash()
    after = write_tenant(minimal_tenant(confidence_threshold=80)).config_hash()
    assert before != after, "a policy change must be visible in the audit hash"


def test_escalation_routes_by_reason_in_config_order(write_tenant):
    """Two tenants differ here on purpose; precedence is the config author's choice."""
    config = write_tenant(
        minimal_tenant(
            escalation={
                "standard_tier": "standard_reviewer",
                "default_escalation_tier": "senior_regulatory",
                "rules": [
                    {"when_reason": "off_label", "tier": "medical_director"},
                    {"when_reason": "comparative_claim", "tier": "legal"},
                ],
            }
        )
    )
    assert config.escalation.tier_for(["off_label"]) == "medical_director"
    assert config.escalation.tier_for(["comparative_claim"]) == "legal"
    # First matching rule wins, in config order.
    assert config.escalation.tier_for(["off_label", "comparative_claim"]) == "medical_director"
    # Unmatched reasons fall back rather than dropping the escalation.
    assert config.escalation.tier_for(["ungrounded"]) == "senior_regulatory"


def test_escalation_never_falls_through_to_nothing(write_tenant):
    config = write_tenant(minimal_tenant())
    assert config.escalation.tier_for([]) == "senior_regulatory"
    assert config.escalation.standard_tier == "standard_reviewer"


def test_policy_flags_stay_opaque_to_the_platform(write_tenant):
    """MLR-specific policy is data the platform carries but does not interpret."""
    config = write_tenant(
        minimal_tenant(policy_flags={"require_head_to_head_for_superiority": True})
    )
    assert config.policy_flags["require_head_to_head_for_superiority"] is True


def test_load_all_tenants_is_sorted(write_tenant):
    write_tenant(minimal_tenant("borealis_bio"))
    write_tenant(minimal_tenant("acme_pharma"))
    assert [t.tenant_id for t in tenancy.load_all_tenants()] == ["acme_pharma", "borealis_bio"]
