"""MLR eval metrics: the matching rules, and the two numbers that must be read together."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlr import eval_metrics
from platform_core.evals.schema import CaseResult, GoldenCase, load_cases

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"


def case(**overrides) -> GoldenCase:
    data = {
        "id": "c1",
        "title": "t",
        "why_hard": "Long enough to explain what this case tests that the others do not.",
        "tenant_id": "acme_pharma",
        "product_id": "zelvoran",
        "promo": "Zelvoran reduces risk in adults.",
        "expected_findings": [
            {
                "check_type": "substantiation",
                "claim_contains": "reduces risk",
                "severity": "blocker",
            }
        ],
        "expected_route": "senior_regulatory",
    }
    data.update(overrides)
    return GoldenCase.model_validate(data)


def finding(**overrides) -> dict:
    data = {
        "id": "f1",
        "check_type": "substantiation",
        "quote": "Zelvoran reduces risk",
        "issue": "Overstated.",
        "severity": "blocker",
        "confidence": 90,
        "grounded": True,
        "grounded_by": "evidence",
        "rule_id": "",
    }
    data.update(overrides)
    return data


def result(**overrides) -> CaseResult:
    data = {
        "case_id": "c1",
        "tenant_id": "acme_pharma",
        "passed": True,
        "expected_route": "senior_regulatory",
        "actual_route": "senior_regulatory",
        "route_correct": True,
    }
    data.update(overrides)
    return CaseResult.model_validate(data)


# --- Matching ------------------------------------------------------------------------------------


def test_a_finding_matches_on_check_type_and_claim_text():
    c = case()
    assert eval_metrics.finding_matches(c.expected_findings[0], finding(), c)


def test_the_wrong_check_type_does_not_match():
    """Finding the right problem under the wrong check is not a pass: the fix differs."""
    c = case()
    assert not eval_metrics.finding_matches(
        c.expected_findings[0], finding(check_type="off_label"), c
    )


def test_claim_text_is_matched_against_the_issue_as_well_as_the_quote():
    """A check may quote a shorter span while describing the same problem.

    Requiring the phrase in the quote alone would measure transcription rather than detection.
    """
    c = case()
    hit = finding(quote="reduces", issue="The claim that Zelvoran reduces risk is unsupported.")
    assert eval_metrics.finding_matches(c.expected_findings[0], hit, c)


def test_matching_is_case_insensitive():
    c = case(
        expected_findings=[
            {
                "check_type": "substantiation",
                "claim_contains": "REDUCES RISK",
                "severity": "blocker",
            }
        ]
    )
    assert eval_metrics.finding_matches(c.expected_findings[0], finding(), c)


def test_one_finding_cannot_satisfy_two_expectations():
    """Otherwise one vague finding could claim credit for several distinct problems."""
    c = case(
        expected_findings=[
            {"check_type": "substantiation", "claim_contains": "reduces", "severity": "blocker"},
            {"check_type": "substantiation", "claim_contains": "risk", "severity": "blocker"},
        ]
    )
    matches = eval_metrics.match_findings(c, [finding()])
    assert [m.matched for m in matches] == [True, False]


def test_a_severity_disagreement_is_a_match_with_a_mismatch_flag():
    """Detection and calibration are different failures and need different fixes."""
    c = case()
    matches = eval_metrics.match_findings(c, [finding(severity="advisory")])
    assert matches[0].matched is True
    assert matches[0].severity_mismatch is True
    assert matches[0].actual_severity == "advisory"


# --- False alarms --------------------------------------------------------------------------------


def test_a_finding_quoting_forbidden_text_is_a_false_alarm():
    c = case(must_not_flag=["Zelvoran reduces risk"])
    assert eval_metrics.false_alarms(c, [finding()]) == ["Zelvoran reduces risk"]


def test_a_short_quote_inside_a_forbidden_sentence_is_not_a_false_alarm():
    """Regression guard for a bug that made the precision metric the least honest number in the set.

    The trademark rule's finding quotes the single word "Zelvoran". Matching any shared substring in
    either direction reported that as a false alarm against every must_not_flag sentence containing the
    brand name -- which was most of them, on four otherwise-passing cases.
    """
    c = case(must_not_flag=["Zelvoran can lower your ability to fight infections"])
    assert eval_metrics.false_alarms(c, [finding(quote="Zelvoran")]) == []


def test_a_substantial_fragment_of_forbidden_text_is_still_a_false_alarm():
    c = case(must_not_flag=["Zelvoran can lower your ability to fight infections"])
    fragment = finding(quote="Zelvoran can lower your ability to fight")
    assert eval_metrics.false_alarms(c, [fragment]) == [c.must_not_flag[0]]


def test_an_absence_finding_cannot_be_a_false_alarm():
    """A finding with no quote is not about any particular text."""
    c = case(must_not_flag=["Zelvoran reduces risk"])
    assert eval_metrics.false_alarms(c, [finding(quote="")]) == []


# --- The headline pair ---------------------------------------------------------------------------


def test_blocker_recall_names_what_was_missed():
    """A missed blocker is the thing to go and look at, so counting it is not enough."""
    c = case()
    hit = result(matches=eval_metrics.match_findings(c, []))
    recall = eval_metrics.blocker_recall([hit])
    assert recall == {
        "expected": 1,
        "matched": 0,
        "recall": 0.0,
        "missed": [
            {
                "case_id": "c1",
                "check_type": "substantiation",
                "claim_contains": "reduces risk",
            }
        ],
    }


def test_blocker_recall_ignores_advisory_expectations():
    c = case(
        expected_findings=[
            {
                "check_type": "substantiation",
                "claim_contains": "reduces risk",
                "severity": "advisory",
            }
        ]
    )
    matches = eval_metrics.match_findings(c, [])
    assert eval_metrics.blocker_recall([result(matches=matches)])["expected"] == 0
    assert eval_metrics.advisory_recall([result(matches=matches)])["expected"] == 1


def test_false_escalation_counts_only_cases_that_should_have_stayed_standard():
    """The counterweight to recall: a system that escalates everything is useless."""
    cases = [
        result(
            case_id="clean", expected_route="standard_reviewer", actual_route="senior_regulatory"
        ),
        result(
            case_id="also_clean",
            expected_route="standard_reviewer",
            actual_route="standard_reviewer",
        ),
        result(
            case_id="blocker", expected_route="senior_regulatory", actual_route="senior_regulatory"
        ),
    ]
    rate = eval_metrics.false_escalation_rate(cases)
    assert rate["standard_expected"] == 2
    assert rate["escalated"] == 1
    assert rate["rate"] == 0.5
    assert rate["cases"][0]["case_id"] == "clean"


def test_grounding_rate_is_broken_down_by_mode():
    """Three different strengths of claim; one aggregate would mix them."""
    cases = [
        result(
            findings=[
                finding(id="a", grounded=True, grounded_by="evidence"),
                finding(id="b", grounded=True, grounded_by="rule"),
                finding(id="c", grounded=False, grounded_by="none"),
            ]
        )
    ]
    rate = eval_metrics.grounding_rate(cases)
    assert rate["findings"] == 3
    assert rate["grounded"] == 2
    assert rate["by_mode"] == {"evidence": 1, "none": 1, "rule": 1}


def test_expected_rules_are_checked_separately_from_findings():
    """'The right rule fired' and 'the finding reads correctly' are different failures."""
    c = case(expected_rule_ids=["ACME-BAN-001", "ACME-TM-001"])
    fired = eval_metrics.expected_rules_fired(c, [finding(rule_id="ACME-BAN-001")])
    assert fired["missing"] == ["ACME-TM-001"]
    assert fired["all_fired"] is False


def test_the_regression_gate_guards_blocker_recall():
    assert eval_metrics.GUARD_METRICS == [("blocker_recall", "recall")]


# --- The shipped golden set ----------------------------------------------------------------------


def test_the_golden_set_covers_both_tenants_and_every_check():
    cases = load_cases(GOLDEN_DIR)
    assert len(cases) >= 13, "the spec calls for at least 13 cases"
    assert {c.tenant_id for c in cases} == {"acme_pharma", "borealis_bio"}
    checks = {f.check_type for c in cases for f in c.expected_findings}
    assert {
        "substantiation",
        "off_label",
        "fair_balance",
        "safety_prominence",
        "business_rule",
    } <= checks


def test_the_golden_set_has_clean_cases_to_measure_false_alarms():
    """Without them, false escalation rate has no denominator and recall can be gamed."""
    cases = load_cases(GOLDEN_DIR)
    clean = [c for c in cases if not c.expected_findings]
    assert len(clean) >= 2
    assert all(c.expected_route == "standard_reviewer" for c in clean)
    assert all(c.must_not_flag for c in clean)


def test_every_golden_case_explains_why_it_is_hard():
    for c in load_cases(GOLDEN_DIR):
        assert len(c.why_hard.strip()) >= 150, f"{c.id} does not explain what makes it hard"


def test_the_paired_tenant_cases_use_the_same_piece():
    """Cases 007 and 008 are the multi-tenancy demonstration: identical input, different outcome."""
    cases = {c.id: c for c in load_cases(GOLDEN_DIR)}
    acme = cases["007-same-piece-passes-for-acme"]
    borealis = cases["008-same-piece-blocked-for-borealis"]
    assert acme.promo_path == borealis.promo_path
    assert acme.tenant_id != borealis.tenant_id
    assert acme.expected_route == "standard_reviewer"
    assert borealis.expected_route != "standard_reviewer"


def test_a_case_expecting_a_specialist_tier_exists():
    """Proves escalation routing is data: Borealis sends comparative claims to legal."""
    cases = load_cases(GOLDEN_DIR)
    assert any(c.expected_route == "legal" for c in cases)


def test_the_committed_baseline_declares_what_it_measures():
    """An offline baseline must never be mistaken for a model measurement."""
    from platform_core.evals.harness import load_result

    baseline = REPO_ROOT / "evals" / "baselines" / "latest.json"
    if not baseline.is_file():
        pytest.skip("no baseline committed yet")

    result_obj = load_result(baseline)
    assert result_obj.cases, "a baseline with no cases guards nothing"
    assert result_obj.tenant_config_hashes, "a baseline must record the policy that produced it"
    assert result_obj.prompt_versions, "a baseline must record the prompts that produced it"
    if result_obj.mode == "offline":
        assert result_obj.notes.startswith("PROVISIONAL"), (
            "an offline baseline must say that its recall figure is not a model measurement"
        )
