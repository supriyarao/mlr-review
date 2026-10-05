"""The eval harness and its metrics, exercised with a stub use case.

The harness must be use-case agnostic, so these tests give it a toy runner and scorer rather than
the MLR graph. If a change to MLR breaks these, the harness has leaked domain knowledge.
"""

from __future__ import annotations

import json

import pytest
import yaml
from pydantic import ValidationError

from platform_core.evals import harness, judge, metrics
from platform_core.evals.schema import (
    CaseResult,
    EvalResult,
    ExpectedFinding,
    FindingMatch,
    GoldenCase,
    load_case,
    load_cases,
)


def case(case_id: str = "c1", **overrides) -> GoldenCase:
    data = {
        "id": case_id,
        "title": "Overstated subgroup result",
        "why_hard": "Label shows benefit in a subgroup; the promo generalises it.",
        "tenant_id": "acme_pharma",
        "product_id": "zelvoran",
        "label_version": "v1.0",
        "promo": "Zelvoran reduces risk.",
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


def result(case_id: str = "c1", *, passed: bool = True, **overrides) -> CaseResult:
    data = {
        "case_id": case_id,
        "tenant_id": "acme_pharma",
        "passed": passed,
        "expected_route": "senior_regulatory",
        "actual_route": "senior_regulatory",
        "route_correct": True,
    }
    data.update(overrides)
    return CaseResult.model_validate(data)


def matched(severity: str = "blocker", *, hit: bool = True, mismatch: bool = False) -> FindingMatch:
    return FindingMatch(
        expected=ExpectedFinding(
            check_type="substantiation", claim_contains="reduces risk", severity=severity
        ),
        matched=hit,
        severity_mismatch=mismatch,
        actual_severity="advisory" if mismatch else severity,
    )


# --- Case loading -------------------------------------------------------------------------------


def test_a_case_must_say_what_makes_it_hard():
    """Without why_hard an eval set fills up with cases that pass trivially."""
    with pytest.raises(ValidationError):
        GoldenCase.model_validate({"id": "c1", "title": "t", "tenant_id": "a", "product_id": "p"})


def test_cases_load_from_yaml_sorted_and_filterable(tmp_path):
    for cid, tenant in [("c2", "borealis_bio"), ("c1", "acme_pharma")]:
        (tmp_path / f"{cid}.yaml").write_text(
            yaml.safe_dump(case(cid, tenant_id=tenant).model_dump(mode="json")), encoding="utf-8"
        )
    assert [c.id for c in load_cases(tmp_path)] == ["c1", "c2"]
    assert [c.id for c in load_cases(tmp_path, tenant_id="borealis_bio")] == ["c2"]


def test_promo_can_come_from_a_file(tmp_path):
    (tmp_path / "promo.md").write_text("Headline copy.", encoding="utf-8")
    c = case(promo="", promo_path="promo.md")
    assert c.promo_text(tmp_path) == "Headline copy."


def test_a_case_with_no_promo_at_all_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="neither promo nor promo_path"):
        case(promo="", promo_path="").promo_text(tmp_path)


def test_a_typo_in_a_case_field_fails_at_load(tmp_path):
    (tmp_path / "c1.yaml").write_text(
        yaml.safe_dump({**case().model_dump(mode="json"), "expected_rout": "x"}), encoding="utf-8"
    )
    with pytest.raises(ValidationError):
        load_case(tmp_path / "c1.yaml")


# --- Metrics ------------------------------------------------------------------------------------


def test_recall_counts_only_the_requested_severity():
    cases = [
        result("c1", matches=[matched("blocker", hit=True), matched("advisory", hit=False)]),
        result("c2", matches=[matched("blocker", hit=False)]),
    ]
    blockers = metrics.recall_by_class(cases, "blocker")
    assert blockers == {"expected": 2, "matched": 1, "recall": 0.5}
    assert metrics.recall_by_class(cases, "advisory")["recall"] == 0.0


def test_a_case_with_nothing_to_find_cannot_fail_on_recall():
    """Otherwise the clean-piece case would drag the aggregate down for passing perfectly."""
    assert metrics.recall_by_class([result("clean", matches=[])])["recall"] == 1.0


def test_severity_mismatches_are_reported_apart_from_misses():
    """Different fixes: a mismatch is policy calibration, a miss is detection."""
    cases = [result("c1", matches=[matched("blocker", hit=True, mismatch=True)])]
    assert metrics.recall_by_class(cases, "blocker")["recall"] == 1.0
    mismatches = metrics.severity_mismatches(cases)
    assert mismatches["count"] == 1
    assert mismatches["rows"][0]["expected"] == "blocker"
    assert mismatches["rows"][0]["actual"] == "advisory"


def test_false_alarms_are_counted_unambiguously():
    """must_not_flag is an explicit assertion, unlike an merely unexpected finding."""
    cases = [result("clean", passed=False, false_alarms=["standard dosing statement"])]
    assert metrics.false_alarms(cases)["count"] == 1


def test_finding_precision_separates_accounted_from_unexpected():
    cases = [
        result(
            "c1",
            findings=[{"id": "f1"}, {"id": "f2"}, {"id": "f3"}],
            unexpected_findings=[{"id": "f3"}],
        )
    ]
    p = metrics.finding_precision(cases)
    assert p == {"produced": 3, "accounted_for": 2, "unexpected": 1, "precision": 0.6667}


def test_routing_accuracy_ignores_cases_with_no_expected_route():
    cases = [
        result("c1", route_correct=True),
        result("c2", route_correct=False),
        result("c3", expected_route="", route_correct=False),
    ]
    assert metrics.routing_accuracy(cases) == {"scored": 2, "correct": 1, "accuracy": 0.5}


def test_latency_reports_p95_not_just_the_mean():
    """A pipeline that is usually fast and occasionally very slow is a different product."""
    cases = [result(f"c{i}", latency_ms=ms) for i, ms in enumerate([100, 120, 130, 140, 9000])]
    stats = metrics.cost_and_latency(cases)
    assert stats["latency_p50_ms"] == 130
    assert stats["latency_p95_ms"] == 9000
    assert stats["latency_mean_ms"] > stats["latency_p50_ms"]


def test_percentile_handles_empty_and_single_values():
    assert metrics.percentile([], 95) == 0.0
    assert metrics.percentile([42.0], 95) == 42.0


def test_cost_is_reported_per_run_as_well_as_in_total():
    cases = [result("c1", cost_usd=0.01), result("c2", cost_usd=0.03)]
    stats = metrics.cost_and_latency(cases)
    assert stats["cost_usd_total"] == 0.04
    assert stats["cost_usd_per_run"] == 0.02


def test_per_tenant_breakdown_surfaces_a_single_tenant_failure():
    """An aggregate can hide that the system works for one policy and not another."""
    cases = [
        result("c1", tenant_id="acme_pharma", passed=True),
        result("c2", tenant_id="borealis_bio", passed=False),
    ]
    grouped = metrics.per_group(cases, "tenant_id")
    assert grouped["acme_pharma"]["pass_rate"]["pass_rate"] == 1.0
    assert grouped["borealis_bio"]["pass_rate"]["pass_rate"] == 0.0


def test_mean_and_std_reports_spread_for_repeated_runs():
    """Models are nondeterministic; a point estimate with no spread invites false regressions."""
    assert metrics.mean_and_std([0.8, 0.9, 0.85])["mean"] == 0.85
    assert metrics.mean_and_std([0.9])["std"] == 0.0
    assert metrics.mean_and_std([])["n"] == 0


# --- Harness ------------------------------------------------------------------------------------


def test_the_harness_runs_a_golden_set_with_an_injected_runner_and_scorer():
    """It must know nothing about findings; the use case supplies both functions."""
    cases = [case("c1"), case("c2")]

    def runner(c: GoldenCase) -> dict:
        return {"route": "senior_regulatory", "findings": [{"id": "f1"}]}

    def scorer(c: GoldenCase, raw: dict) -> CaseResult:
        return result(c.case_id if hasattr(c, "case_id") else c.id, findings=raw["findings"])

    out = harness.run_eval(cases, runner=runner, scorer=scorer, mode="offline")
    assert len(out.cases) == 2
    assert out.metrics["pass_rate"]["pass_rate"] == 1.0


def test_one_broken_case_does_not_destroy_the_other_results():
    cases = [case("ok"), case("boom")]

    def runner(c: GoldenCase) -> dict:
        if c.id == "boom":
            raise RuntimeError("model unavailable")
        return {}

    out = harness.run_eval(cases, runner=runner, scorer=lambda c, raw: result(c.id), mode="offline")
    by_id = {c.case_id: c for c in out.cases}
    assert by_id["ok"].passed is True
    assert by_id["boom"].passed is False
    assert "model unavailable" in by_id["boom"].error


def test_a_scoring_failure_is_recorded_on_the_case():
    def scorer(c: GoldenCase, raw: dict) -> CaseResult:
        raise ValueError("bad shape")

    out = harness.run_eval([case("c1")], runner=lambda c: {}, scorer=scorer, mode="offline")
    assert "scoring failed" in out.cases[0].error


def test_repeats_keep_the_worst_outcome():
    """'Catches this blocker four times out of five' is a miss, not a partial pass."""
    attempts = iter([True, False, True])

    def scorer(c: GoldenCase, raw: dict) -> CaseResult:
        return result(c.id, passed=next(attempts))

    out = harness.run_eval(
        [case("c1")], runner=lambda c: {}, scorer=scorer, repeats=3, mode="offline"
    )
    assert out.cases[0].passed is False
    assert out.repeats == 3


def test_a_metrics_hook_adds_use_case_specific_numbers():
    out = harness.run_eval(
        [case("c1")],
        runner=lambda c: {},
        scorer=lambda c, raw: result(c.id),
        metrics_hook=lambda cases: {"blocker_recall": 0.9, "false_escalation_rate": 0.1},
        mode="offline",
    )
    assert out.metrics["blocker_recall"] == 0.9
    # Generic metrics are still present alongside.
    assert "routing" in out.metrics


def test_provenance_is_recorded_so_a_number_can_be_explained():
    out = harness.run_eval(
        [case("c1")],
        runner=lambda c: {},
        scorer=lambda c, raw: result(c.id),
        mode="live",
        provenance={
            "backend": "anthropic",
            "model_strong": "claude-sonnet-5",
            "model_fast": "claude-haiku-4-5",
            "prompt_versions": {"substantiation": "sub-v1"},
            "tenant_config_hashes": {"acme_pharma": "abc123"},
        },
    )
    assert out.backend == "anthropic"
    assert out.model_strong == "claude-sonnet-5"
    assert out.prompt_versions["substantiation"] == "sub-v1"
    assert out.tenant_config_hashes["acme_pharma"] == "abc123"
    assert out.mode == "live"


def test_results_round_trip_through_json(tmp_path):
    out = harness.run_eval(
        [case("c1")], runner=lambda c: {}, scorer=lambda c, raw: result(c.id), mode="offline"
    )
    path = harness.save_result(out, directory=tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["mode"] == "offline"
    assert harness.load_result(path).eval_run_id == out.eval_run_id


# --- Regression gate ----------------------------------------------------------------------------


def _eval_with(blocker_recall: float, *, cases: list[CaseResult] | None = None) -> EvalResult:
    return EvalResult(
        eval_run_id="e1",
        started_at="2026-01-01T00:00:00Z",
        metrics={"blocker_recall": {"recall": blocker_recall}},
        cases=cases or [result("c1")],
    )


def test_a_drop_in_the_guarded_metric_is_a_regression():
    """A tenant's policy change must not silently cost blocker recall."""
    diff = harness.compare_to_baseline(
        _eval_with(0.80), _eval_with(0.90), guard_metrics=[("blocker_recall", "recall")]
    )
    assert diff["regressed"] is True
    assert diff["regressions"][0]["delta"] == pytest.approx(-0.10)


def test_an_improvement_is_not_a_regression():
    diff = harness.compare_to_baseline(
        _eval_with(0.95), _eval_with(0.90), guard_metrics=[("blocker_recall", "recall")]
    )
    assert diff["regressed"] is False
    assert diff["movements"][0]["delta"] == pytest.approx(0.05)


def test_an_unchanged_metric_is_not_a_regression():
    diff = harness.compare_to_baseline(
        _eval_with(0.90), _eval_with(0.90), guard_metrics=[("blocker_recall", "recall")]
    )
    assert diff["regressed"] is False


def test_tolerance_allows_for_model_nondeterminism():
    diff = harness.compare_to_baseline(
        _eval_with(0.89),
        _eval_with(0.90),
        guard_metrics=[("blocker_recall", "recall")],
        tolerance=0.02,
    )
    assert diff["regressed"] is False


def test_case_level_diff_names_what_broke():
    before = _eval_with(0.9, cases=[result("c1", passed=True), result("c2", passed=True)])
    after = _eval_with(0.8, cases=[result("c1", passed=True), result("c2", passed=False)])
    changes = {row["case_id"]: row["change"] for row in harness.diff_cases(after, before)}
    assert changes == {"c2": "broke"}


def test_case_level_diff_reports_new_and_missing_cases():
    before = _eval_with(0.9, cases=[result("c1"), result("gone")])
    after = _eval_with(0.9, cases=[result("c1"), result("brand_new")])
    changes = {row["case_id"]: row["change"] for row in harness.diff_cases(after, before)}
    assert changes["brand_new"] == "new"
    assert changes["gone"] == "missing"


# --- Judge --------------------------------------------------------------------------------------


def test_judge_parses_a_verdict_with_its_reason():
    v = judge.parse_verdict(
        '{"supports": true, "confidence": 90, "reason": "The passage states the endpoint."}',
        key="supports",
    )
    assert v.verdict is True
    assert v.confidence == 90
    assert v.reason.startswith("The passage")
    assert v.usable


def test_judge_tolerates_prose_and_code_fences_around_the_json():
    """A judge that fails on formatting makes the metric about formatting."""
    wrapped = 'Sure!\n```json\n{"supports": false, "confidence": 40, "reason": "Different endpoint."}\n```'
    assert judge.parse_verdict(wrapped, key="supports").verdict is False


def test_an_unparseable_judge_response_is_flagged_not_silently_false():
    """A judge failure would otherwise look identical to a real finding of unfaithfulness."""
    v = judge.parse_verdict("I cannot answer that.", key="supports")
    assert v.usable is False
    assert "no JSON object" in v.parse_error


def test_a_judge_response_missing_the_verdict_field_is_unusable():
    v = judge.parse_verdict('{"confidence": 90}', key="supports")
    assert v.usable is False
    assert "missing" in v.parse_error


def test_judge_sampling_is_deterministic():
    """So the offline eval judges the same items every time and the baseline reproduces."""
    items = list(range(100))
    assert judge.sample_for_judging(items, limit=5) == judge.sample_for_judging(items, limit=5)
    assert len(judge.sample_for_judging(items, limit=5)) == 5
    assert judge.sample_for_judging(items, limit=0) == items


def test_judge_summary_keeps_unusable_verdicts_visible():
    """A judge that silently started failing should show a shrinking denominator, not a drift."""
    verdicts = [
        judge.JudgeVerdict(verdict=True, confidence=90),
        judge.JudgeVerdict(verdict=False, confidence=60, reason="different endpoint"),
        judge.JudgeVerdict(verdict=False, parse_error="no JSON object"),
    ]
    summary = judge.summarize(verdicts)
    assert summary == {
        "judged": 3,
        "usable": 2,
        "unusable": 1,
        "positive": 1,
        "rate": 0.5,
        "mean_confidence": 75.0,
        "failures": [{"reason": "different endpoint", "confidence": 60}],
    }


def test_judge_prompts_confine_the_question():
    """The judge must not be asked whether a finding is correct -- only whether a citation fits."""
    prompt = judge.faithfulness_prompt(issue="overstates benefit", passage="in a subgroup")
    # The prompt is line-wrapped in the source, so compare whitespace-normalised.
    flat = " ".join(prompt.split())
    assert "Do not judge whether the issue is a real problem" in flat
    assert "do not use any knowledge beyond the passage shown" in flat
    assert "overstates benefit" in flat
