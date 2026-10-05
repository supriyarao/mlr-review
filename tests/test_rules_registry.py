"""Rule storage, versioning, scope selection, validation, and the policy hash."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
import yaml

from platform_core.rules import evals as rule_evals
from platform_core.rules.registry import (
    RuleActivationError,
    RuleRegistry,
    RuleStorageError,
)
from platform_core.rules.schema import Rule, RuleScope
from platform_core.rules.validation import validate_rule


def rule_data(rule_id: str = "ACME-BAN-001", tenant: str = "acme_pharma", **overrides) -> dict:
    data = {
        "id": rule_id,
        "tenant_id": tenant,
        "title": "Prohibited efficacy language",
        "description": "Promotional copy must not claim the product cures the condition.",
        "kind": "deterministic",
        "severity": "blocker",
        "params": {"type": "banned_terms", "terms": ["cure"]},
        "examples": [
            {"text": "Zelvoran can cure psoriasis.", "expected": "fail"},
            {"text": "Zelvoran treats plaque psoriasis in adults.", "expected": "pass"},
        ],
        "owner": "Regulatory Affairs",
        "effective_from": "2026-01-01",
    }
    data.update(overrides)
    return data


def make_rule(rule_id: str = "ACME-BAN-001", **overrides) -> Rule:
    return Rule.model_validate(rule_data(rule_id, **overrides))


# --- Versioning ---------------------------------------------------------------------------------


def test_a_registry_cannot_exist_without_a_tenant():
    for bad in ("", "   "):
        with pytest.raises(RuleStorageError, match="non-empty tenant_id"):
            RuleRegistry(bad)


def test_the_first_saved_rule_is_version_one_and_a_draft():
    """New rules must not take effect until someone activates them."""
    stored = RuleRegistry("acme_pharma").save_new_version(make_rule())
    assert stored.version == 1
    assert stored.status == "draft"


def test_an_edit_creates_a_new_version_and_never_overwrites():
    """If you can silently change the rule behind last month's decision, it is not defensible."""
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule())
    registry.save_new_version(make_rule(description="Copy must not claim a cure, in any wording."))

    versions = registry.all_versions("ACME-BAN-001")
    assert [v.version for v in versions] == [1, 2]
    assert versions[0].description.endswith("cures the condition.")
    assert versions[1].description.endswith("in any wording.")


def test_a_client_cannot_overwrite_history_by_resubmitting_a_version_number():
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule())
    # Version is assigned by the registry, so a hostile or buggy client cannot force version 1.
    stored = registry.save_new_version(
        make_rule(version=1, description="A different rule entirely.")
    )
    assert stored.version == 2
    assert len(registry.all_versions("ACME-BAN-001")) == 2


def test_activating_a_version_retires_the_previously_active_one():
    """Exactly one version active, or 'which rule produced this finding' has two answers."""
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule())
    registry.activate("ACME-BAN-001", 1)
    registry.save_new_version(make_rule(description="Revised wording of the cure prohibition."))
    registry.activate("ACME-BAN-001", 2)

    statuses = {v.version: v.status for v in registry.all_versions("ACME-BAN-001")}
    assert statuses == {1: "retired", 2: "active"}
    assert registry.get("ACME-BAN-001").version == 2


def test_a_retired_rule_stays_readable_for_auditing_past_decisions():
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule())
    registry.activate("ACME-BAN-001", 1)
    registry.retire("ACME-BAN-001", 1, note="superseded by policy review")

    assert registry.active_rules() == []
    retired = registry.get("ACME-BAN-001", version=1)
    assert retired.status == "retired"
    assert retired.description  # still fully readable


def test_history_is_available_for_the_version_diff_view():
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule(), created_by="a.author", change_note="initial")
    registry.save_new_version(
        make_rule(description="Revised wording of the cure prohibition."),
        created_by="b.author",
        change_note="tightened wording",
    )
    history = registry.history("ACME-BAN-001")
    assert [h["version"] for h in history] == [1, 2]
    assert history[1]["created_by"] == "b.author"
    assert history[1]["change_note"] == "tightened wording"


def test_an_unknown_rule_is_a_storage_error():
    with pytest.raises(RuleStorageError, match="no rule"):
        RuleRegistry("acme_pharma").get("NOPE-001")


def test_an_unknown_version_is_a_storage_error():
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule())
    with pytest.raises(RuleStorageError, match="no version 7"):
        registry.get("ACME-BAN-001", version=7)


# --- Tenant isolation ---------------------------------------------------------------------------


def test_a_registry_refuses_to_store_another_tenants_rule():
    with pytest.raises(RuleStorageError, match="refusing to store"):
        RuleRegistry("acme_pharma").save_new_version(make_rule(tenant="borealis_bio"))


def test_one_tenants_rules_are_invisible_to_another():
    """Acme's banned-term list must never apply to a Borealis review."""
    acme = RuleRegistry("acme_pharma")
    borealis = RuleRegistry("borealis_bio")
    acme.save_new_version(make_rule())
    acme.activate("ACME-BAN-001", 1)

    assert [r.id for r in acme.active_rules()] == ["ACME-BAN-001"]
    assert borealis.active_rules() == []
    assert borealis.applicable_rules() == []


def test_two_tenants_may_use_the_same_rule_id_without_colliding():
    """Rule ids are unique within a tenant, not globally."""
    acme = RuleRegistry("acme_pharma")
    borealis = RuleRegistry("borealis_bio")
    acme.save_new_version(make_rule("BAN-001", description="Acme's version of the cure rule."))
    borealis.save_new_version(
        make_rule("BAN-001", tenant="borealis_bio", description="Borealis's own cure rule wording.")
    )
    assert "Acme" in acme.get("BAN-001").description
    assert "Borealis" in borealis.get("BAN-001").description


# --- Scope and effective dates ------------------------------------------------------------------


def test_scope_restricts_a_rule_to_its_audience():
    """Same requirement, patient-facing only for one tenant -- this is what makes tenants differ."""
    scope = RuleScope(audience="patient")
    assert scope.applies_to(audience="patient") is True
    assert scope.applies_to(audience="hcp") is False


def test_an_unscoped_rule_applies_to_every_audience():
    scope = RuleScope()
    assert scope.applies_to(audience="patient") is True
    assert scope.applies_to(audience="hcp") is True


def test_an_unknown_submission_audience_only_picks_up_unscoped_rules():
    """Applying a patient-facing rule to a possibly-clinical piece would create false blockers."""
    assert RuleScope(audience="patient").applies_to(audience="any") is False
    assert RuleScope().applies_to(audience="any") is True


def test_scope_filters_on_channel_product_and_market():
    scope = RuleScope(channels=["email"], product_ids=["zelvoran"], markets=["us"])
    assert scope.applies_to(channel="email", product_id="zelvoran", market="us") is True
    assert scope.applies_to(channel="print", product_id="zelvoran", market="us") is False
    assert scope.applies_to(channel="email", product_id="nexapril", market="us") is False
    assert scope.applies_to(channel="email", product_id="zelvoran", market="eu") is False


def test_only_active_rules_are_applied():
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule())
    assert registry.applicable_rules() == []  # still a draft
    registry.activate("ACME-BAN-001", 1)
    assert [r.id for r in registry.applicable_rules()] == ["ACME-BAN-001"]


def test_a_rule_authored_ahead_of_its_effective_date_does_not_apply_yet():
    """A new code of practice can be loaded before it takes effect and start on its own."""
    future = date.today() + timedelta(days=30)
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule(effective_from=future.isoformat()))
    registry.activate("ACME-BAN-001", 1)

    assert registry.applicable_rules(on=date.today()) == []
    assert [r.id for r in registry.applicable_rules(on=future)] == ["ACME-BAN-001"]


def test_an_expired_rule_stops_applying():
    yesterday = date.today() - timedelta(days=1)
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(
        make_rule(effective_from="2026-01-01", effective_to=yesterday.isoformat())
    )
    registry.activate("ACME-BAN-001", 1)
    assert registry.applicable_rules() == []


def test_an_effective_window_cannot_end_before_it_starts():
    with pytest.raises(ValueError, match="cannot precede"):
        make_rule(effective_from="2026-06-01", effective_to="2026-01-01")


def test_list_rules_shows_one_entry_per_rule_not_per_version():
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule())
    registry.activate("ACME-BAN-001", 1)
    registry.save_new_version(make_rule(description="A newer draft wording of the rule."))
    listed = registry.list_rules()
    assert len(listed) == 1
    # The active version is what the table shows, not the newer draft.
    assert listed[0].version == 1
    assert listed[0].status == "active"


def test_list_rules_filters_by_status_and_kind():
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule("ACME-BAN-001"))
    registry.activate("ACME-BAN-001", 1)
    registry.save_new_version(
        make_rule(
            "ACME-LLM-001",
            kind="llm",
            params={},
            description="Patient materials must not imply results are typical.",
        )
    )
    assert [r.id for r in registry.list_rules(status="active")] == ["ACME-BAN-001"]
    assert [r.id for r in registry.list_rules(kind="llm")] == ["ACME-LLM-001"]


# --- Validation ---------------------------------------------------------------------------------


def test_a_rule_whose_examples_behave_as_labelled_can_be_activated():
    report = validate_rule(make_rule())
    assert report.can_save is True
    assert report.can_activate is True
    assert report.examples_passed is True
    assert report.accuracy() == 1.0


def test_a_rule_whose_examples_contradict_their_labels_saves_as_draft_but_cannot_activate():
    """Saving stays permissive -- losing half-written work is worse than storing something wrong."""
    broken = make_rule(
        examples=[
            # Labelled "fail" but the rule will not fire: no banned term present.
            {"text": "Zelvoran treats plaque psoriasis.", "expected": "fail"},
            {"text": "Zelvoran is well tolerated.", "expected": "pass"},
        ]
    )
    report = validate_rule(broken)
    assert report.can_save is True
    assert report.can_activate is False
    assert report.accuracy() == 0.5


def test_a_rule_needs_both_a_passing_and_a_failing_example():
    """Each direction catches a different mistake: never fires, versus always fires."""
    only_fail = validate_rule(make_rule(examples=[{"text": "A cure.", "expected": "fail"}]))
    assert only_fail.can_save is False
    assert any("expected: pass" in e for e in only_fail.errors)

    only_pass = validate_rule(
        make_rule(examples=[{"text": "Treats psoriasis.", "expected": "pass"}])
    )
    assert any("expected: fail" in e for e in only_pass.errors)


def test_validation_compiles_regexes_at_author_time():
    report = validate_rule(
        make_rule(params={"type": "banned_terms", "regex": True, "terms": ["(unclosed"]})
    )
    assert report.can_save is False
    assert any("not a valid pattern" in e for e in report.errors)


def test_an_unknown_deterministic_type_is_rejected_by_the_schema():
    with pytest.raises(ValueError, match="params.type"):
        make_rule(params={"type": "vibes_check"})


def test_a_placeholder_description_is_rejected():
    """The description becomes the citation a reviewer reads."""
    with pytest.raises(ValueError, match="citable text"):
        make_rule(description="TODO")


def test_a_reference_doc_rule_needs_its_document_to_exist(tmp_path):
    rule = make_rule(
        "ACME-DOC-001",
        kind="reference_doc",
        params={"doc_path": "promotional_code.md"},
        description="Content must comply with the Acme Brand and Promotional Code.",
    )
    missing = validate_rule(rule, reference_docs_dir=tmp_path)
    assert missing.can_save is False
    assert any("does not exist" in e for e in missing.errors)

    (tmp_path / "promotional_code.md").write_text(
        "## 1. Claims\nAll claims must be consistent with the approved label.\n", encoding="utf-8"
    )
    present = validate_rule(rule, reference_docs_dir=tmp_path)
    assert present.can_save is True


def test_a_cited_clause_must_appear_in_the_document(tmp_path):
    (tmp_path / "code.md").write_text(
        "## 1. Claims\nAll claims must be consistent with the approved label.\n", encoding="utf-8"
    )
    rule = make_rule(
        "ACME-DOC-001",
        kind="reference_doc",
        description="Content must comply with the Acme Brand and Promotional Code.",
        params={"doc_path": "code.md", "clauses": ["No clause like this exists in the document"]},
    )
    report = validate_rule(rule, reference_docs_dir=tmp_path)
    assert any("does not appear" in e for e in report.errors)


def test_a_doc_path_cannot_escape_the_tenants_own_folder(tmp_path):
    rule = make_rule(
        "ACME-DOC-001",
        kind="reference_doc",
        description="Content must comply with the Acme Brand and Promotional Code.",
        params={"doc_path": "../../../etc/passwd"},
    )
    report = validate_rule(rule, reference_docs_dir=tmp_path)
    assert any("escapes the tenant" in e for e in report.errors)


def test_an_llm_rule_is_activatable_on_structure_but_warns_about_the_gap():
    """Honest: an llm rule's examples cannot be verified offline, and the author is told so."""
    report = validate_rule(
        make_rule(
            "ACME-LLM-001",
            kind="llm",
            params={},
            description="Patient materials must not imply results are typical unless the label says so.",
        )
    )
    assert report.can_activate is True
    assert report.examples_executed == []
    assert any("cannot be verified without a model call" in w for w in report.warnings)
    assert "not executable offline" in report.summary()


# --- Per-rule accuracy --------------------------------------------------------------------------


def test_per_rule_accuracy_requires_deterministic_rules_to_be_perfect():
    good = make_rule()
    broken = make_rule(
        "ACME-BAN-002",
        examples=[
            {"text": "No banned word here.", "expected": "fail"},
            {"text": "Treats psoriasis.", "expected": "pass"},
        ],
    )
    summary = rule_evals.summarize(rule_evals.score_rules([good, broken]))
    assert summary["deterministic"]["all_perfect"] is False
    assert summary["deterministic"]["imperfect"][0]["rule_id"] == "ACME-BAN-002"


def test_llm_rules_are_reported_separately_from_deterministic_accuracy():
    """A broken regex must not hide behind an LLM rule's inherent fuzziness."""
    rows = rule_evals.score_rules(
        [
            make_rule(),
            make_rule(
                "ACME-LLM-001",
                kind="llm",
                params={},
                description="Patient materials must not imply results are typical.",
            ),
        ]
    )
    summary = rule_evals.summarize(rows)
    assert summary["deterministic"]["scored"] == 1
    assert summary["deterministic"]["all_perfect"] is True
    assert summary["not_self_testable"] == [{"rule_id": "ACME-LLM-001", "kind": "llm"}]


# --- YAML seed and export -----------------------------------------------------------------------


def test_seeding_loads_and_activates_rules_from_yaml(tmp_path):
    (tmp_path / "ban-001.yaml").write_text(yaml.safe_dump(rule_data()), encoding="utf-8")
    loaded = RuleRegistry("acme_pharma").seed_from_yaml(tmp_path)
    assert [r.id for r in loaded] == ["ACME-BAN-001"]
    assert loaded[0].status == "active"


def test_reseeding_identical_content_does_not_inflate_version_history(tmp_path):
    """`make seed` run twice must not look like two policy changes."""
    (tmp_path / "ban-001.yaml").write_text(yaml.safe_dump(rule_data()), encoding="utf-8")
    registry = RuleRegistry("acme_pharma")
    registry.seed_from_yaml(tmp_path)
    registry.seed_from_yaml(tmp_path)
    assert [v.version for v in registry.all_versions("ACME-BAN-001")] == [1]


def test_reseeding_changed_content_does_create_a_new_version(tmp_path):
    path = tmp_path / "ban-001.yaml"
    path.write_text(yaml.safe_dump(rule_data()), encoding="utf-8")
    registry = RuleRegistry("acme_pharma")
    registry.seed_from_yaml(tmp_path)

    path.write_text(
        yaml.safe_dump(rule_data(description="Copy must not claim a cure, in any wording.")),
        encoding="utf-8",
    )
    registry.seed_from_yaml(tmp_path)
    assert [v.version for v in registry.all_versions("ACME-BAN-001")] == [1, 2]
    assert registry.get("ACME-BAN-001").version == 2


def test_seeding_refuses_a_file_that_declares_another_tenant(tmp_path):
    (tmp_path / "x.yaml").write_text(
        yaml.safe_dump(rule_data(tenant="borealis_bio")), encoding="utf-8"
    )
    with pytest.raises(RuleStorageError, match="declares tenant"):
        RuleRegistry("acme_pharma").seed_from_yaml(tmp_path)


def test_export_writes_the_active_set_back_for_committing(tmp_path):
    """A compliance owner edits in the UI, exports, and the diff is reviewable in git."""
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule())
    registry.activate("ACME-BAN-001", 1)

    written = registry.export_to_yaml(tmp_path)
    assert [p.name for p in written] == ["acme-ban-001.yaml"]
    exported = yaml.safe_load(written[0].read_text(encoding="utf-8"))
    assert exported["id"] == "ACME-BAN-001"
    # Version and status belong to the registry, not to the shipped rule content.
    assert "version" not in exported
    assert "status" not in exported


def test_an_exported_rule_round_trips_through_seeding(tmp_path):
    source = RuleRegistry("acme_pharma")
    source.save_new_version(make_rule())
    source.activate("ACME-BAN-001", 1)
    source.export_to_yaml(tmp_path)

    reseeded = RuleRegistry("acme_pharma").seed_from_yaml(tmp_path)
    assert reseeded[0].id == "ACME-BAN-001"
    assert reseeded[0].status == "active"


def test_seeding_an_empty_directory_is_not_an_error(tmp_path):
    assert RuleRegistry("acme_pharma").seed_from_yaml(tmp_path / "nope") == []


# --- The applied set ----------------------------------------------------------------------------


def test_the_applied_set_records_what_a_run_actually_used():
    registry = RuleRegistry("acme_pharma")
    registry.save_new_version(make_rule())
    registry.activate("ACME-BAN-001", 1)
    applied = registry.applied_set(registry.applicable_rules())
    assert applied == [
        {"id": "ACME-BAN-001", "version": 1, "kind": "deterministic", "severity": "blocker"}
    ]


# --- Activation is gated, on every path that reaches it ------------------------------------------


def test_a_rule_that_fails_validation_cannot_be_activated():
    """The bar lives in activate(), not in each caller.

    Activation is reachable from seeding, the API and the UI. Enforcing the rule in each of those
    means it only holds where someone remembered -- which is exactly what went wrong: seed_from_yaml
    activated every rule file without validating, so a blocker whose own examples contradicted it
    went live and `make seed` reported success.
    """
    registry = RuleRegistry("acme_pharma")
    broken = make_rule(
        examples=[
            {"text": "No banned term in this sentence.", "expected": "fail"},
            {"text": "Zelvoran treats plaque psoriasis.", "expected": "pass"},
        ]
    )
    stored = registry.save_new_version(broken)

    with pytest.raises(RuleActivationError, match="cannot be activated"):
        registry.activate(stored.id, stored.version)
    # And it really is still inactive, not merely reported as refused.
    assert registry.get(stored.id, version=stored.version).status == "draft"
    assert registry.active_rules() == []


def test_a_failed_activation_can_be_overridden_but_only_with_a_reason():
    """A rule can legitimately need to go live against the checker's advice.

    An override with no recorded reason is indistinguishable from a mistake six months later, so the
    note is required rather than optional.
    """
    registry = RuleRegistry("acme_pharma")
    stored = registry.save_new_version(
        make_rule(
            examples=[
                {"text": "No banned term here.", "expected": "fail"},
                {"text": "Zelvoran treats psoriasis.", "expected": "pass"},
            ]
        )
    )
    with pytest.raises(RuleActivationError, match="requires a note"):
        registry.activate(stored.id, stored.version, force=True)

    activated = registry.activate(
        stored.id, stored.version, force=True, note="Accepted by Regulatory pending rewording."
    )
    assert activated.status == "active"
    assert "Accepted by Regulatory" in registry.history(stored.id)[-1]["change_note"]


def test_a_valid_rule_activates_normally():
    registry = RuleRegistry("acme_pharma")
    stored = registry.save_new_version(make_rule())
    assert registry.activate(stored.id, stored.version).status == "active"


def test_seeding_leaves_an_invalid_rule_as_a_draft_and_records_why(tmp_path):
    """Authored work is kept, but a rule the checker would refuse never takes effect."""
    good = rule_data("ACME-BAN-001")
    bad = rule_data(
        "ACME-BAN-002",
        examples=[
            {"text": "No banned term in this sentence.", "expected": "fail"},
            {"text": "Zelvoran treats psoriasis.", "expected": "pass"},
        ],
    )
    (tmp_path / "good.yaml").write_text(yaml.safe_dump(good), encoding="utf-8")
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")

    registry = RuleRegistry("acme_pharma")
    registry.seed_from_yaml(tmp_path)

    assert [r.id for r in registry.active_rules()] == ["ACME-BAN-001"]
    assert registry.get("ACME-BAN-002").status == "draft"
    assert len(registry.rejected) == 1
    assert "ACME-BAN-002" in registry.rejected[0][0]


def test_a_reference_doc_rule_cannot_be_activated_without_its_document(tmp_path):
    """Seeding must pass the reference docs directory, or validation cannot check the clauses."""
    registry = RuleRegistry("acme_pharma")
    stored = registry.save_new_version(
        make_rule(
            "ACME-DOC-001",
            kind="reference_doc",
            description="Content must comply with the Acme Brand and Promotional Code.",
            params={"doc_path": "missing-code.md"},
        )
    )
    with pytest.raises(RuleActivationError):
        registry.activate(stored.id, stored.version, reference_docs_dir=tmp_path)

    (tmp_path / "missing-code.md").write_text(
        "## 1. Claims\nClaims match the label.\n", encoding="utf-8"
    )
    assert (
        registry.activate(stored.id, stored.version, reference_docs_dir=tmp_path).status == "active"
    )
