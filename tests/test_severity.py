"""How a tenant's severity rules grade a finding.

Worth testing directly rather than only through a graph run: this is where a tenant's policy
actually takes effect, and a rule that quietly never applies looks exactly like a rule that applies
and agrees with the default.
"""

from __future__ import annotations

from mlr.nodes.severity import resolve_severity
from mlr.schemas import Finding
from platform_core.tenancy import load_tenant


def _finding(check_type: str, tags: list[str] | None = None) -> Finding:
    return Finding(id="f1", check_type=check_type, issue="x", tags=tags or [])


def test_a_tagged_rule_beats_an_untagged_one_for_the_same_check_type():
    """``when_tag`` is how a tenant grades a subset of one check differently.

    It was on the SeverityRule schema from the start and read by nothing: a tenant could write one,
    seeding accepted it, and it silently never applied. Promoting a product in a market where it
    has no authorisation is an off_label finding, but it cannot be fixed by rewording the way an
    ordinary off-label claim can, so it earns its own rule.
    """
    config = load_tenant("acme_pharma")

    plain = resolve_severity(_finding("off_label"), config)
    tagged = resolve_severity(_finding("off_label", ["unapproved_market"]), config)

    assert plain[1] == "ACME-SEV-OFFLABEL"
    assert tagged[1] == "ACME-SEV-UNAPPROVED"
    assert plain[1] != tagged[1], "the tag has to change which rule grades the finding"


def test_a_tag_does_not_leak_into_another_check_type():
    """A tagged rule scoped to off_label must not grade a substantiation finding.

    Without this the tag would be a back door into every check at once -- a model attaching it to
    the wrong finding would silently change that finding's severity.
    """
    config = load_tenant("acme_pharma")
    graded = resolve_severity(_finding("substantiation", ["unapproved_market"]), config)
    assert graded[1] == "ACME-SEV-SUBSTANTIATION"


def test_an_unknown_tag_falls_back_to_the_ordinary_rule():
    """A tag nobody configured must not change anything."""
    config = load_tenant("acme_pharma")
    assert (
        resolve_severity(_finding("off_label", ["not_a_real_tag"]), config)[1]
        == "ACME-SEV-OFFLABEL"
    )


def test_both_tenants_grade_an_unapproved_market_on_its_own_rule():
    """Both tenants treat it as blocking, by their own rule rather than the generic fallback."""
    for tenant_id, expected in (
        ("acme_pharma", "ACME-SEV-UNAPPROVED"),
        ("borealis_bio", "BOR-SEV-UNAPPROVED"),
    ):
        severity, rule_id = resolve_severity(
            _finding("off_label", ["unapproved_market"]), load_tenant(tenant_id)
        )
        assert (severity, rule_id) == ("blocker", expected)
