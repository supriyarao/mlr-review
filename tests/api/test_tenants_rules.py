"""Tenant and business-rule endpoints.

Tenants are read straight from the shipped configs, so those responses must equal the contract
examples. Rules are created during the tests, so they are checked for *shape* against the examples
and for behaviour directly: versioning, the two activation gates, and the audit trail.
"""

from __future__ import annotations

import pytest

from platform_core import audit
from platform_core.rules.registry import RuleRegistry
from tests.api.conftest import assert_same_shape, draft_rule, fixture

ACME = "/tenants/acme_pharma"


def _create(client, **overrides):
    response = client.post(
        f"{ACME}/rules",
        json={"rule": draft_rule(**overrides), "author": "r.owner", "change_note": "first draft"},
    )
    assert response.status_code == 201, response.text
    return response.json()


# --- Tenants -------------------------------------------------------------------------------------


def test_the_tenant_list_matches_the_contract_example(client):
    assert client.get("/tenants").json() == fixture("GET_tenants.json")


def test_a_tenant_matches_the_contract_example(client, seeded_rules):
    assert client.get(ACME).json() == fixture("GET_tenants_{id}.json")


def test_the_two_tenants_really_do_differ(client, seeded_rules):
    """The multi-tenancy claim, at the API: same endpoint, materially different policy."""
    acme = client.get(ACME).json()
    borealis = client.get("/tenants/borealis_bio").json()
    assert acme["confidence_threshold"] != borealis["confidence_threshold"]
    assert acme["config_hash"] != borealis["config_hash"]

    def severity_of(tenant, check):
        return next(r["severity"] for r in tenant["severity_rules"] if r["check_type"] == check)

    assert severity_of(acme, "fair_balance") == "advisory"
    assert severity_of(borealis, "fair_balance") == "blocker"


@pytest.mark.parametrize("tenant_id", ["nobody", "..", "acme_pharma%2F..%2Fborealis_bio"])
def test_an_unknown_tenant_is_a_404_and_never_reaches_the_filesystem(client, tenant_id):
    for suffix in ("", "/rules", "/rules/ACME-BAN-001"):
        assert client.get(f"/tenants/{tenant_id}{suffix}").status_code == 404


# --- Reading rules -------------------------------------------------------------------------------


def test_the_rule_list_has_the_contract_shape(client, seeded_rules):
    body = client.get(f"{ACME}/rules").json()
    assert_same_shape(body, fixture("GET_tenants_{id}_rules.json"))
    assert body["total"] == len(body["rules"]) == 8
    assert {r["tenant_id"] for r in body["rules"]} == {"acme_pharma"}


def test_the_rule_list_filters_by_kind_and_status(client, seeded_rules):
    deterministic = client.get(f"{ACME}/rules", params={"kind": "deterministic"}).json()["rules"]
    assert deterministic and {r["kind"] for r in deterministic} == {"deterministic"}
    assert client.get(f"{ACME}/rules", params={"status": "draft"}).json()["total"] == 0


def test_a_rule_matches_the_contract_example_with_its_history(client, seeded_rules):
    body = client.get(f"{ACME}/rules/ACME-BAN-001").json()
    expected = fixture("GET_tenants_{id}_rules_{rule_id}.json")
    assert body["rule"] == expected["rule"]
    assert_same_shape(body["history"], expected["history"])
    assert [h["version"] for h in body["history"]] == [1]


def test_one_tenant_cannot_read_another_tenants_rule(client, seeded_rules):
    assert client.get(f"{ACME}/rules/BOR-BAN-001").status_code == 404
    assert client.get("/tenants/borealis_bio/rules/BOR-BAN-001").status_code == 200


# --- Authoring -----------------------------------------------------------------------------------


def test_creating_a_rule_stores_a_draft_and_reports_validation(client, seeded_rules):
    body = _create(client)
    assert_same_shape(body, fixture("POST_tenants_{id}_rules.json"))
    assert (body["rule"]["version"], body["rule"]["status"]) == (1, "draft")
    assert body["validation"]["can_activate"] is True
    assert body["validation"]["accuracy"] == 1.0
    # A draft is not in force: the active count is unchanged.
    assert client.get(ACME).json()["active_rule_count"] == 8


def test_a_structurally_invalid_rule_is_not_stored(client, seeded_rules):
    response = client.post(f"{ACME}/rules", json={"rule": draft_rule(examples=[])})
    assert response.status_code == 422
    assert response.json()["detail"]["validation"]["can_save"] is False
    assert client.get(f"{ACME}/rules/ACME-BAN-003").status_code == 404


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": "borealis_bio"},  # a rule for someone else
        {"severity": "catastrophic"},  # not a severity
        {"effective_to": "2025-01-01"},  # ends before it starts
    ],
)
def test_a_malformed_or_cross_tenant_rule_is_rejected(client, seeded_rules, overrides):
    response = client.post(f"{ACME}/rules", json={"rule": draft_rule(**overrides)})
    assert response.status_code == 422
    assert RuleRegistry("acme_pharma").latest_version_number("ACME-BAN-003") == 0
    assert RuleRegistry("borealis_bio").latest_version_number("ACME-BAN-003") == 0


def test_creating_a_rule_that_exists_is_a_conflict(client, seeded_rules):
    _create(client)
    assert client.post(f"{ACME}/rules", json={"rule": draft_rule()}).status_code == 409


def test_an_edit_is_a_new_version_and_the_active_one_stays_in_force(client, seeded_rules):
    """Rules are never overwritten: last month's decision must still cite what it was made under."""
    current = client.get(f"{ACME}/rules/ACME-BAN-001").json()["rule"]
    edited = {**current, "title": "Prohibited efficacy, safety and hype language"}

    response = client.put(f"{ACME}/rules/ACME-BAN-001", json={"rule": edited, "author": "r.owner"})
    assert response.status_code == 200
    assert (response.json()["rule"]["version"], response.json()["rule"]["status"]) == (2, "draft")

    body = client.get(f"{ACME}/rules/ACME-BAN-001").json()
    assert (body["rule"]["version"], body["rule"]["title"]) == (1, current["title"])
    assert [(h["version"], h["status"]) for h in body["history"]] == [(1, "active"), (2, "draft")]

    v2 = client.get(f"{ACME}/rules/ACME-BAN-001", params={"version": 2}).json()["rule"]
    assert v2["title"] == edited["title"]


def test_an_edit_cannot_rename_the_rule_or_target_a_missing_one(client, seeded_rules):
    current = client.get(f"{ACME}/rules/ACME-BAN-001").json()["rule"]
    renamed = client.put(f"{ACME}/rules/ACME-BAN-001", json={"rule": {**current, "id": "OTHER-1"}})
    assert renamed.status_code == 422
    assert client.put(f"{ACME}/rules/ACME-NOPE", json={"rule": draft_rule()}).status_code == 404


# --- Testing a rule ------------------------------------------------------------------------------


def test_testing_an_unsaved_rule_reports_violations_and_saves_nothing(client, seeded_rules):
    response = client.post(
        f"{ACME}/rules/ACME-BAN-003/test",
        json={"rule": draft_rule(), "text": "A genuine breakthrough for adults with psoriasis."},
    )
    assert response.status_code == 200
    body = response.json()
    assert_same_shape(body, fixture("POST_tenants_{id}_rules_{rule_id}_test.json"))
    assert [v["quote"] for v in body["violations"]] == ["breakthrough"]
    assert body["saved"] is False
    assert client.get(f"{ACME}/rules/ACME-BAN-003").status_code == 404


def test_testing_a_stored_rule_against_a_sample_piece(client, seeded_rules):
    body = client.post(
        f"{ACME}/rules/ACME-REQ-001/test", json={"sample": "zelvoran-hcp-subgroup-claim"}
    ).json()
    assert body["source"] == "sample:zelvoran-hcp-subgroup-claim"
    assert body["rule_id"] == "ACME-REQ-001"
    assert isinstance(body["violations"], list)


def test_a_rule_that_needs_a_model_cannot_be_tested_offline(client, seeded_rules):
    response = client.post(f"{ACME}/rules/ACME-LLM-001/test", json={"text": "Some copy."})
    assert response.status_code == 422
    assert "needs a model" in response.json()["detail"]


def test_a_rule_test_needs_exactly_one_source_of_text(client, seeded_rules):
    for body in ({}, {"text": "x", "sample": "zelvoran-hcp-subgroup-claim"}):
        assert client.post(f"{ACME}/rules/ACME-BAN-001/test", json=body).status_code == 422
    missing = client.post(f"{ACME}/rules/ACME-BAN-001/test", json={"sample": "no-such-piece"})
    assert missing.status_code == 404


# --- Activating and retiring ---------------------------------------------------------------------


def test_activating_a_rule_changes_the_policy_version_and_is_audited(client, seeded_rules):
    _create(client)
    response = client.post(
        f"{ACME}/rules/ACME-BAN-003/activate", json={"actor": "r.owner", "note": "approved by RA"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert_same_shape(body, fixture("POST_tenants_{id}_rules_{rule_id}_activate.json"))
    assert body["status"] == "active"
    assert body["policy_version_before"] != body["policy_version_after"]
    assert client.get(ACME).json()["active_rule_count"] == 9

    # No runner and no baseline in this test, so the check must say it did not run -- not pass.
    assert body["regression_check"]["ran"] is False

    events = audit.events_for_run("policy:acme_pharma")
    assert [e.node for e in events] == ["rule_save", "rule_activate"]
    assert events[-1].event_type == "human"
    assert events[-1].policy_version == body["policy_version_after"]
    assert '"actor": "r.owner"' in events[-1].output_json


def test_a_rule_that_fails_its_own_examples_cannot_be_activated(client, seeded_rules):
    wrong = [
        {"text": "Zelvoran achieved PASI-75 in 68.4% of adults.", "expected": "fail"},
        {"text": "Zelvoran is a breakthrough in psoriasis care.", "expected": "pass"},
    ]
    _create(client, examples=wrong)

    response = client.post(f"{ACME}/rules/ACME-BAN-003/activate", json={})
    assert response.status_code == 409
    assert response.json()["detail"]["validation"]["can_activate"] is False
    assert client.get(f"{ACME}/rules/ACME-BAN-003").json()["rule"]["status"] == "draft"

    # Forcing it through needs a recorded reason, and then works.
    assert (
        client.post(f"{ACME}/rules/ACME-BAN-003/activate", json={"force": True}).status_code == 409
    )
    forced = client.post(
        f"{ACME}/rules/ACME-BAN-003/activate", json={"force": True, "note": "RA sign-off 12/10"}
    )
    assert forced.status_code == 200


def test_an_activation_that_costs_blocker_recall_is_rolled_back(client, seeded_rules, monkeypatch):
    """A tenant's policy change must not be able to reduce blocker recall silently."""
    from api import evals

    movement = {"metric": "blocker_recall.recall", "baseline": 1.0, "current": 0.5, "delta": -0.5}
    monkeypatch.setattr(
        evals,
        "regression_check",
        lambda _tenant: {"ran": True, "regressed": True, "movements": [movement]},
    )
    current = client.get(f"{ACME}/rules/ACME-BAN-001").json()["rule"]
    client.put(f"{ACME}/rules/ACME-BAN-001", json={"rule": {**current, "title": "Edited title"}})
    before = client.get(ACME).json()

    refused = client.post(f"{ACME}/rules/ACME-BAN-001/activate", json={"version": 2})
    assert refused.status_code == 409
    assert refused.json()["detail"]["regression_check"]["movements"] == [movement]
    # Version 1 is back in force, and the tenant's rule count is what it was.
    assert client.get(f"{ACME}/rules/ACME-BAN-001").json()["rule"]["version"] == 1
    assert client.get(ACME).json()["active_rule_count"] == before["active_rule_count"]
    assert audit.events_for_run("policy:acme_pharma")[-1].node == "rule_activate_refused"

    # An override needs a reason, and the reason is recorded in the response and the audit log.
    no_reason = client.post(
        f"{ACME}/rules/ACME-BAN-001/activate", json={"version": 2, "override_regression": True}
    )
    assert no_reason.status_code == 422
    overridden = client.post(
        f"{ACME}/rules/ACME-BAN-001/activate",
        json={"version": 2, "override_regression": True, "override_reason": "golden case is stale"},
    )
    assert overridden.status_code == 200
    check = overridden.json()["regression_check"]
    assert (check["overridden"], check["override_reason"]) == (True, "golden case is stale")
    assert "golden case is stale" in audit.events_for_run("policy:acme_pharma")[-1].output_json


def test_a_new_rule_that_regresses_is_not_left_in_force(client, seeded_rules, monkeypatch):
    from api import evals

    monkeypatch.setattr(
        evals, "regression_check", lambda _t: {"ran": True, "regressed": True, "movements": []}
    )
    _create(client)
    assert client.post(f"{ACME}/rules/ACME-BAN-003/activate", json={}).status_code == 409
    assert client.get(f"{ACME}/rules/ACME-BAN-003").json()["rule"]["status"] != "active"
    assert client.get(ACME).json()["active_rule_count"] == 8


def test_retiring_a_rule_takes_it_out_of_force_but_keeps_it_readable(client, seeded_rules):
    response = client.post(f"{ACME}/rules/ACME-TM-001/retire", json={"note": "superseded"})
    assert response.status_code == 200
    body = response.json()
    assert_same_shape(body, fixture("POST_tenants_{id}_rules_{rule_id}_retire.json"))
    assert body["status"] == "retired"
    assert client.get(ACME).json()["active_rule_count"] == 7
    assert client.get(f"{ACME}/rules/ACME-TM-001").json()["rule"]["status"] == "retired"


def test_activating_or_retiring_a_missing_rule_or_version_is_a_404(client, seeded_rules):
    assert client.post(f"{ACME}/rules/ACME-NOPE/activate", json={}).status_code == 404
    assert client.post(f"{ACME}/rules/ACME-NOPE/retire", json={}).status_code == 404
    assert (
        client.post(f"{ACME}/rules/ACME-BAN-001/activate", json={"version": 9}).status_code == 404
    )


def test_export_writes_the_active_rules_back_to_yaml(client, tenants_copy, seeded_rules):
    _create(client)
    client.post(f"{ACME}/rules/ACME-BAN-003/activate", json={})

    body = client.post(f"{ACME}/rules/export").json()
    assert_same_shape(body, fixture("POST_tenants_{id}_rules_export.json"))
    assert len(body["written"]) == 9
    exported = tenants_copy / "acme_pharma" / "rules" / "acme-ban-003.yaml"
    assert exported.is_file() and "breakthrough" in exported.read_text(encoding="utf-8")
