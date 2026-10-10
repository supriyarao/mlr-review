"""The preapproved_filter node on the decisions path.

This file exists because the old chat-shaped path had a hole in it: ``tests/fake_responses.py``
carried an ``EQUIVALENT`` script, but nothing ever asserted that a pre-approved claim was actually
matched. The node could have stopped matching entirely and every graph test would still have passed.
"""

from __future__ import annotations

from typing import Any

import pytest

from mlr.nodes import preapproved
from mlr.schemas import Claim
from platform_core import decisions


class _Candidate:
    """One retrieved pre-approved claim. Only the two fields the node reads."""

    def __init__(self, source_id: str, text: str) -> None:
        self.source_id = source_id
        self.text = text


class _Retriever:
    """Stands in for TenantRetriever so these tests need no vector store."""

    def __init__(self, candidates: list[_Candidate]) -> None:
        self.candidates = candidates
        self.calls = 0

    def search(self, _query: str, **kwargs: Any) -> list[_Candidate]:
        self.calls += 1
        assert kwargs["doc_type"] == "preapproved_claim", (
            "searching anything but the cleared-claim library would match the underlying study "
            "and report a brand new claim as already approved"
        )
        return self.candidates[: kwargs.get("k", 3)]


def _state(**over: Any) -> dict[str, Any]:
    state = {
        "run_id": "test-run",
        "tenant_id": "acme_pharma",
        "product_id": "zelvoran",
        "claims": [Claim(id="c1", original_text="68.4% achieved PASI-75", normalized_claim="")],
    }
    state.update(over)
    return state


@pytest.fixture
def three_candidates(monkeypatch):
    candidates = [
        _Candidate("acme_pharma:preapproved_claim:ACME-PAC-001", "first approved claim"),
        _Candidate("acme_pharma:preapproved_claim:ACME-PAC-002", "second approved claim"),
        _Candidate("acme_pharma:preapproved_claim:ACME-PAC-003", "third approved claim"),
    ]
    retriever = _Retriever(candidates)
    monkeypatch.setattr(preapproved, "TenantRetriever", lambda _tenant: retriever)
    return retriever


def _decider(answers: dict[str, float]) -> decisions.FakeDecider:
    return decisions.FakeDecider(
        scripted={name: {"type": "noul", "noul": p} for name, p in answers.items()},
        default_answer={"type": "noul", "noul": 0.0},
    )


def test_a_confident_match_marks_the_claim_as_already_approved(three_candidates):
    result = preapproved.preapproved_filter(_state(), decider=_decider({"candidate_1": 0.99}))
    assert result["preapproved_matches"] == {"c1": "ACME-PAC-002"}


def test_the_best_candidate_wins_not_the_first_over_the_line(three_candidates):
    """The old path stopped at the first acceptable candidate. With every answer in hand there is
    no reason to settle for a worse match that happened to rank higher."""
    result = preapproved.preapproved_filter(
        _state(),
        decider=_decider({"candidate_0": 0.88, "candidate_1": 0.97, "candidate_2": 0.90}),
    )
    assert result["preapproved_matches"] == {"c1": "ACME-PAC-002"}


def test_a_tie_goes_to_the_better_ranked_candidate(three_candidates):
    result = preapproved.preapproved_filter(
        _state(), decider=_decider({"candidate_0": 0.95, "candidate_2": 0.95})
    )
    assert result["preapproved_matches"] == {"c1": "ACME-PAC-001"}


def test_below_the_threshold_is_not_a_match(three_candidates):
    """A false match skips substantiation on a claim nobody cleared. That is the error to avoid."""
    just_under = preapproved.EQUIVALENCE_THRESHOLD - 0.01
    result = preapproved.preapproved_filter(_state(), decider=_decider({"candidate_0": just_under}))
    assert result["preapproved_matches"] == {}


def test_exactly_at_the_threshold_matches(three_candidates):
    result = preapproved.preapproved_filter(
        _state(), decider=_decider({"candidate_0": preapproved.EQUIVALENCE_THRESHOLD})
    )
    assert result["preapproved_matches"] == {"c1": "ACME-PAC-001"}


def test_one_request_covers_every_candidate(three_candidates):
    """The reason this node moved: three candidates used to cost three model calls."""
    decider = _decider({"candidate_0": 0.1})
    preapproved.preapproved_filter(_state(), decider=decider)
    assert len(decider.calls) == 1
    assert decider.calls[0] == {
        "node": "preapproved_filter",
        "version": "preapproved-v2",
    }


def test_a_failed_decision_leaves_the_claim_to_substantiation(three_candidates):
    """Fails closed. The cost of a missed match is an unnecessary check, never a skipped one."""

    class _Exploding(decisions.FakeDecider):
        def respond(self, **_kwargs: Any) -> dict[str, Any]:
            raise decisions.DecisionError("endpoint unavailable")

    result = preapproved.preapproved_filter(_state(), decider=_Exploding())
    assert result["preapproved_matches"] == {}


def test_no_retrieved_candidates_asks_nothing(monkeypatch):
    retriever = _Retriever([])
    monkeypatch.setattr(preapproved, "TenantRetriever", lambda _tenant: retriever)
    decider = _decider({})
    result = preapproved.preapproved_filter(_state(), decider=decider)
    assert result["preapproved_matches"] == {}
    assert decider.calls == [], "no candidates means there is nothing to ask about"


def test_no_claims_skips_retrieval_entirely(monkeypatch):
    called = False

    def _never(_tenant: str) -> Any:
        nonlocal called
        called = True
        raise AssertionError("retrieval must not run when there are no claims")

    monkeypatch.setattr(preapproved, "TenantRetriever", _never)
    result = preapproved.preapproved_filter(_state(claims=[]), decider=_decider({}))
    assert result["preapproved_matches"] == {}
    assert called is False
