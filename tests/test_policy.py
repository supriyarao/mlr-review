"""The composite policy version.

A decision is only defensible if you can say which policy produced it. Policy has two halves now --
tenant configuration and tenant business rules -- so the hash has to cover both. These tests exist
because the failure mode is silent: if editing a rule did not move policy_version, the audit log
would claim two materially different runs had identical provenance, and an auditor comparing them
would be misled by the system's own records.
"""

from __future__ import annotations

from datetime import date

from platform_core import policy
from platform_core.rules.registry import RuleRegistry
from platform_core.rules.schema import Rule
from tests.conftest import minimal_tenant


def a_rule(rule_id: str = "ACME-BAN-001", **overrides) -> Rule:
    data = {
        "id": rule_id,
        "tenant_id": "acme_pharma",
        "title": "Prohibited efficacy language",
        "description": "Promotional copy must not claim the product cures the condition.",
        "kind": "deterministic",
        "severity": "blocker",
        "params": {"type": "banned_terms", "terms": ["cure"]},
        "examples": [
            {"text": "Zelvoran can cure psoriasis.", "expected": "fail"},
            {"text": "Zelvoran treats plaque psoriasis.", "expected": "pass"},
        ],
        "effective_from": date(2026, 1, 1),
    }
    data.update(overrides)
    return Rule.model_validate(data)


def test_policy_version_is_stable_when_nothing_changes(write_tenant):
    write_tenant(minimal_tenant())
    assert policy.policy_version("acme_pharma") == policy.policy_version("acme_pharma")


def test_editing_the_tenant_config_moves_the_policy_version(write_tenant):
    write_tenant(minimal_tenant())
    before = policy.policy_version("acme_pharma")
    write_tenant(minimal_tenant(confidence_threshold=80))
    assert policy.policy_version("acme_pharma") != before


def test_activating_a_rule_moves_the_policy_version(write_tenant):
    """The whole reason this module exists."""
    write_tenant(minimal_tenant())
    before = policy.policy_version("acme_pharma")

    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(a_rule())
    # A draft is not in force, so it must not change the policy in force.
    assert policy.policy_version("acme_pharma") == before

    registry.activate("ACME-BAN-001", 1)
    assert policy.policy_version("acme_pharma") != before


def test_editing_an_active_rule_moves_the_policy_version(write_tenant):
    write_tenant(minimal_tenant())
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(a_rule())
    registry.activate("ACME-BAN-001", 1)
    before = policy.policy_version("acme_pharma")

    registry.save_new_version(a_rule(description="Copy must not claim a cure, in any wording."))
    registry.activate("ACME-BAN-001", 2)
    assert policy.policy_version("acme_pharma") != before


def test_retiring_a_rule_moves_the_policy_version(write_tenant):
    write_tenant(minimal_tenant())
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(a_rule())
    registry.activate("ACME-BAN-001", 1)
    before = policy.policy_version("acme_pharma")

    registry.retire("ACME-BAN-001", 1)
    assert policy.policy_version("acme_pharma") != before


def test_two_tenants_have_different_policy_versions(write_tenant):
    """Even with identical configs: the tenant id is part of the hash."""
    write_tenant(minimal_tenant("acme_pharma"))
    write_tenant(minimal_tenant("borealis_bio"))
    assert policy.policy_version("acme_pharma") != policy.policy_version("borealis_bio")


def test_one_tenants_rule_change_does_not_move_the_others_policy(write_tenant):
    write_tenant(minimal_tenant("acme_pharma"))
    write_tenant(minimal_tenant("borealis_bio"))
    borealis_before = policy.policy_version("borealis_bio")

    acme = RuleRegistry("acme_pharma")
    acme.save_new_version(a_rule())
    acme.activate("ACME-BAN-001", 1)

    assert policy.policy_version("borealis_bio") == borealis_before


def test_policy_parts_say_which_half_changed(write_tenant):
    """'The policy changed' is not actionable; a config edit and a rule edit differ."""
    write_tenant(minimal_tenant())
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(a_rule())
    registry.activate("ACME-BAN-001", 1)

    before = policy.policy_parts("acme_pharma")
    assert before["active_rule_count"] == "1"
    assert before["config_version"] == "2026.1"

    registry.save_new_version(a_rule("ACME-BAN-002", title="Second prohibited term list"))
    registry.activate("ACME-BAN-002", 1)
    after = policy.policy_parts("acme_pharma")

    # The rules half moved; the config half did not.
    assert after["rules_hash"] != before["rules_hash"]
    assert after["config_hash"] == before["config_hash"]
    assert after["active_rule_count"] == "2"
