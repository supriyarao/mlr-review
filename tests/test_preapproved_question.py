"""The preapproved-v2 typed question: shape, naming, and that it still asks v1's question."""

from __future__ import annotations

from mlr.prompts import preapproved


def test_the_two_versions_coexist_during_the_migration():
    assert preapproved.PROMPT_VERSION == "preapproved-v1"
    assert preapproved.QUESTION_VERSION == "preapproved-v2"


def test_the_submitted_claim_is_sent_once_for_every_candidate():
    """The saving is structural: the claim is the expensive part, the questions are not."""
    state = preapproved.build_state(
        submitted="  Nearly 6 in 10 adults reached PASI-90.  ",
        candidates=["58.1% with prior biologic exposure achieved PASI-90.", "44.2% overall."],
    )
    assert state["submitted_claim"] == "Nearly 6 in 10 adults reached PASI-90."
    assert list(state["approved_claims"]) == ["candidate_0", "candidate_1"]
    assert state["approved_claims"]["candidate_1"] == "44.2% overall."


def test_one_question_per_candidate_each_naming_its_own_candidate():
    questions = preapproved.build_questions(count=3)
    assert list(questions) == ["candidate_0", "candidate_1", "candidate_2"]
    for index, (name, question) in enumerate(questions.items()):
        assert question["type"] == "noul"
        # The question has to say which approved claim it is about. Order is not a contract.
        assert f"approved_claims.candidate_{index}" in question["instructions"]
        assert name == f"candidate_{index}"


def test_the_question_carries_the_equivalence_criteria():
    """Without criteria, 'the same claim' is answered against the model's own idea of sameness."""
    question = preapproved.build_questions(count=1)["candidate_0"]
    assert question["criteria"]["true"] == preapproved.TRUE_MEANS
    assert question["criteria"]["false"] == preapproved.FALSE_MEANS


def test_v2_asks_the_same_question_as_v1():
    """The transport changed, not the test. Otherwise the eval cannot attribute a difference.

    Compared with whitespace flattened: v1's template is hard-wrapped, so "the same endpoint" spans
    a newline there and would not match as a literal substring.
    """
    v1 = " ".join(preapproved.TEMPLATE.split())
    for phrase in ("the same population", "the same endpoint", "the same timepoint"):
        assert phrase in preapproved.TRUE_MEANS
        assert phrase in v1
    for phrase in ("a wider population", "a stronger result", "a dropped comparator"):
        assert phrase in preapproved.FALSE_MEANS
        assert phrase in v1


def test_no_candidates_means_no_questions():
    assert preapproved.build_questions(count=0) == {}
    assert preapproved.build_state(submitted="x", candidates=[])["approved_claims"] == {}
