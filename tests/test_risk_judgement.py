"""The graded risk judgement, and the routing reason a tenant may opt into.

The property that matters most here is that this signal is **additive**: a tenant that configures
nothing must route exactly as it did before the node existed. Several tests exist only to pin that.
"""

from __future__ import annotations

from typing import Any

from mlr.nodes import risk_judgement as node
from mlr.nodes import route as routing
from mlr.prompts import risk_judgement as prompt
from platform_core import decisions


def _score(level: str, *, confidence: float = 0.95) -> decisions.FakeDecider:
    """A decider that answers the risk question with ``level`` as the most likely."""
    index = prompt.LEVELS.index(level)
    return decisions.FakeDecider(
        scripted={
            prompt.QUESTION_NAME: {
                "type": "score",
                "score": float(index),
                "confidence": confidence,
                "legend": {str(i): name for i, name in enumerate(prompt.LEVELS)},
                "probabilities": {
                    str(i): (1.0 if i == index else 0.0) for i in range(len(prompt.LEVELS))
                },
            }
        }
    )


def _state(tenant: str = "acme_pharma", **over: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": f"risk-{tenant}",
        "tenant_id": tenant,
        "product_id": "zelvoran",
        "document": "Powerful relief. No safety information anywhere in this piece.",
        "final_findings": [],
        "findings": [],
        "gate_results": [],
        "claims": [],
        "errors": [],
    }
    state.update(over)
    return state


def test_the_level_recorded_is_the_one_the_model_chose():
    out = node.risk_judgement(_state(), decider=_score("serious"))
    assert out["risk_level"] == "serious"
    assert out["risk_score"] == 2.0


def test_an_empty_document_is_not_judged():
    out = node.risk_judgement(_state(document=""), decider=_score("serious"))
    assert out == {"risk_level": "", "risk_score": 0.0}


def test_a_failed_judgement_removes_the_signal_rather_than_the_run():
    """Additive means its absence costs nothing. The run must route as if the node never ran."""

    class _Exploding(decisions.FakeDecider):
        def respond(self, **_kwargs: Any) -> dict[str, Any]:
            raise decisions.DecisionError("endpoint unavailable")

    out = node.risk_judgement(_state(), decider=_Exploding())
    assert out == {"risk_level": "", "risk_score": 0.0}


def test_the_question_grades_rather_than_asking_yes_no():
    q = prompt.build_questions()[prompt.QUESTION_NAME]
    assert q["type"] == "score"
    assert q["criteria"] == list(prompt.LEVELS)
    # Ordered lowest first; routing compares by index, so the order is load-bearing.
    assert prompt.LEVELS[0] == "routine" and prompt.LEVELS[-1] == "serious"


def test_the_piece_is_truncated_but_says_so():
    state = prompt.build_state(document="x" * (prompt.MAX_DOCUMENT_CHARS + 500), findings=[])
    assert len(state["piece"]) == prompt.MAX_DOCUMENT_CHARS
    assert state["truncated"] is True


def test_findings_travel_as_counts_not_arguments():
    """Handing over full findings would invite the model to re-judge them, not judge the piece."""
    state = prompt.build_state(
        document="a piece",
        findings=[
            {
                "check_type": "substantiation",
                "severity_final": "blocker",
                "quote": "q1",
                "explanation": "a long argument that must not be sent",
            },
            {"check_type": "off_label", "severity_final": "advisory", "quote": "q2"},
        ],
    )
    summary = state["findings_so_far"]
    assert summary == {
        "total": 2,
        "blocking": 1,
        "by_check": {"substantiation": 1, "off_label": 1},
        "quotes": ["q1", "q2"],
    }
    assert "must not be sent" not in str(state)


# --- the routing half -------------------------------------------------------------------------


def _reasons(state: dict[str, Any]) -> list[str]:
    return [r.code for r in routing._escalation_reasons(state, [])]


def test_a_tenant_that_configures_nothing_is_unaffected():
    """Acme sets no flag. The signal must be invisible to it, at every level."""
    for level in prompt.LEVELS:
        codes = _reasons(_state("acme_pharma", risk_level=level))
        assert "high_risk_judgement" not in codes, f"acme escalated on {level!r}"


def test_a_tenant_that_opts_in_escalates_at_its_configured_level():
    """Borealis sets `escalate_on_risk_judgement: serious`."""
    assert "high_risk_judgement" in _reasons(_state("borealis_bio", risk_level="serious"))


def test_below_the_configured_level_does_not_escalate():
    for level in ("routine", "worth a second look"):
        codes = _reasons(_state("borealis_bio", risk_level=level))
        assert "high_risk_judgement" not in codes, f"borealis escalated on {level!r}"


def test_a_missing_judgement_escalates_nobody():
    assert "high_risk_judgement" not in _reasons(_state("borealis_bio", risk_level=""))


def test_an_unknown_level_is_ignored_rather_than_guessed():
    """A level the question does not define cannot be compared, so it must not route."""
    assert "high_risk_judgement" not in _reasons(_state("borealis_bio", risk_level="catastrophic"))


def test_the_reason_says_it_is_a_judgement_not_a_finding():
    """A reviewer should be able to tell this apart from 'this claim has no evidence'."""
    reason = next(
        r
        for r in routing._escalation_reasons(_state("borealis_bio", risk_level="serious"), [])
        if r.code == "high_risk_judgement"
    )
    assert "judgement about the piece as a whole" in reason.explanation
    assert reason.items == []


def test_the_same_level_routes_differently_per_tenant():
    """The whole point: risk appetite is configuration, not code."""
    serious = {"risk_level": "serious"}
    assert "high_risk_judgement" not in _reasons(_state("acme_pharma", **serious))
    assert "high_risk_judgement" in _reasons(_state("borealis_bio", **serious))
