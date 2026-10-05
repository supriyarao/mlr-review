"""Gates: deterministic checks that divert the flow instead of swallowing a problem."""

from __future__ import annotations

from dataclasses import dataclass

from platform_core.gates import (
    Gate,
    GateResult,
    PredicateGate,
    all_passed,
    failed_routes,
    failure_reasons,
    run_gates,
)


@dataclass
class Item:
    """Stand-in for a finding. The platform layer sees opaque items, never MLR concepts."""

    id: str
    grounded: bool = True
    confidence: int = 90


def grounding_gate() -> PredicateGate:
    return PredicateGate(
        name="grounding",
        items=lambda s: s["items"],
        predicate=lambda i: i.grounded,
        item_id=lambda i: i.id,
        on_fail_route="senior_regulatory",
        reason_code="ungrounded",
        describe=lambda fails: f"{len(fails)} finding(s) could not be verified against a source",
    )


def confidence_gate(threshold: int) -> PredicateGate:
    return PredicateGate(
        name="confidence",
        items=lambda s: s["items"],
        predicate=lambda i: i.confidence >= threshold,
        item_id=lambda i: i.id,
        on_fail_route="senior_regulatory",
        reason_code="low_confidence",
    )


def test_a_passing_gate_still_records_that_it_ran():
    """A pass must be auditable too, not just a failure."""
    result = grounding_gate().check({"items": [Item("f1"), Item("f2")]})
    assert result.passed is True
    assert result.details == "all 2 item(s) passed"
    assert result.failed_items == []


def test_nothing_to_check_is_a_pass_with_an_honest_reason():
    result = grounding_gate().check({"items": []})
    assert result.passed is True
    assert result.details == "nothing to check"


def test_a_failing_gate_names_the_items_and_the_route():
    result = grounding_gate().check({"items": [Item("f1"), Item("f2", grounded=False)]})
    assert result.passed is False
    assert result.failed_items == ["f2"]
    assert result.on_fail_route == "senior_regulatory"
    assert result.reason_code == "ungrounded"
    assert "could not be verified" in result.details


def test_failing_items_are_not_dropped_from_state():
    """The worst possible behaviour would be turning 'I cannot verify this' into silence."""
    items = [Item("f1"), Item("f2", grounded=False)]
    state = {"items": items}
    grounding_gate().check(state)
    assert state["items"] == items, "a gate must flag and route, never discard"


def test_confidence_gate_respects_the_tenant_threshold():
    items = [Item("f1", confidence=75)]
    # Same finding, two tenant policies, two outcomes.
    assert confidence_gate(70).check({"items": items}).passed is True
    assert confidence_gate(80).check({"items": items}).passed is False


def test_threshold_is_inclusive_at_the_boundary():
    assert confidence_gate(70).check({"items": [Item("f1", confidence=70)]}).passed is True


def test_run_gates_does_not_short_circuit():
    """A reviewer needs the complete list of problems, not just the first one found."""
    state = {"items": [Item("f1", grounded=False, confidence=10)]}
    results = run_gates([grounding_gate(), confidence_gate(70)], state)
    assert len(results) == 2
    assert [r.gate for r in results] == ["grounding", "confidence"]
    assert not all_passed(results)


def test_routes_and_reasons_are_deduplicated_in_order():
    state = {"items": [Item("f1", grounded=False, confidence=10)]}
    results = run_gates([grounding_gate(), confidence_gate(70)], state)
    assert failed_routes(results) == ["senior_regulatory"]
    assert failure_reasons(results) == ["ungrounded", "low_confidence"]


def test_a_gate_with_no_route_records_without_diverting():
    """Advisory gates annotate the run; they do not change where it goes."""
    advisory = PredicateGate(
        name="style",
        items=lambda s: s["items"],
        predicate=lambda i: False,
        item_id=lambda i: i.id,
    )
    results = [advisory.check({"items": [Item("f1")]})]
    assert not all_passed(results)
    assert failed_routes(results) == []


def test_predicate_gate_satisfies_the_gate_protocol():
    assert isinstance(grounding_gate(), Gate)


def test_gate_result_summary_reads_for_a_human():
    passed = GateResult(gate="grounding", passed=True, details="all 3 item(s) passed")
    failed = GateResult(gate="grounding", passed=False, details="1 finding unverified")
    assert passed.summary() == "grounding: passed (all 3 item(s) passed)"
    assert failed.summary() == "grounding: failed (1 finding unverified)"
