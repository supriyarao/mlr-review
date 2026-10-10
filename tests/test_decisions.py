"""The decisions client: backend selection, typed answers, offline replay, cost accounting."""

from __future__ import annotations

import json

import pytest

from platform_core import decisions


def test_the_backend_defaults_to_fake(monkeypatch):
    """Opposite of the chat factory's default, and deliberately so.

    A decision call costs a fraction of a cent, so an accidental live one would never show up on a
    bill. That is the argument for defaulting off, not against it.
    """
    monkeypatch.delenv("MLR_DECISIONS_BACKEND", raising=False)
    assert decisions.backend() == "fake"


def test_an_unknown_backend_is_rejected_by_name(monkeypatch):
    monkeypatch.setenv("MLR_DECISIONS_BACKEND", "jev")
    with pytest.raises(decisions.DecisionConfigError, match="unknown MLR_DECISIONS_BACKEND"):
        decisions.backend()


def test_the_model_alias_is_configuration(monkeypatch):
    monkeypatch.delenv("MLR_DECISIONS_MODEL", raising=False)
    assert decisions.model_id() == "jev-latest"
    monkeypatch.setenv("MLR_DECISIONS_MODEL", "jev-preview")
    assert decisions.model_id() == "jev-preview"


def test_a_noul_question_carries_its_criteria():
    q = decisions.noul("Same claim?", true_means="identical meaning", false_means="broader")
    assert q == {
        "type": "noul",
        "instructions": "Same claim?",
        "criteria": {"true": "identical meaning", "false": "broader"},
    }
    # Criteria are optional, and absent rather than empty when not given.
    assert "criteria" not in decisions.noul("Same claim?")


def test_choice_and_score_questions_match_the_wire_shape():
    assert decisions.choice("Which?", {"a": "first", "b": "second"}) == {
        "type": "choice",
        "instructions": "Which?",
        "criteria": {"a": "first", "b": "second"},
    }
    assert decisions.score("How urgent?", ["can wait", "today"]) == {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["can wait", "today"],
    }


def test_a_scripted_noul_comes_back_as_a_probability():
    fake = decisions.FakeDecider(scripted={"same": {"type": "noul", "noul": 0.93}})
    result = decisions.decide(
        state="a claim",
        questions={"same": decisions.noul("Same claim?")},
        node="preapproved_filter",
        version="preapproved-v2",
        fake=fake,
    )
    assert result["same"].probability == 0.93
    assert result["same"].kind == "noul"
    assert fake.calls == [{"node": "preapproved_filter", "version": "preapproved-v2"}]


def test_is_yes_needs_an_explicit_threshold_to_be_meaningful():
    answer = decisions.Answer(name="same", kind="noul", probability=0.62)
    assert answer.is_yes(0.5) is True
    assert answer.is_yes(0.8) is False


def test_is_yes_refuses_a_question_that_was_not_yes_no():
    answer = decisions.Answer(name="dept", kind="choice", chosen="billing")
    with pytest.raises(decisions.DecisionError, match="only meaningful for a noul"):
        answer.is_yes()


def test_several_questions_are_answered_in_one_call():
    """The reason to prefer this over a chat call per question: one request, N answers."""
    fake = decisions.FakeDecider(default_answer={"type": "noul", "noul": 0.1})
    result = decisions.decide(
        state="a claim",
        questions={
            "cand_0": decisions.noul("Matches approved claim 0?"),
            "cand_1": decisions.noul("Matches approved claim 1?"),
            "cand_2": decisions.noul("Matches approved claim 2?"),
        },
        node="preapproved_filter",
        version="preapproved-v2",
        fake=fake,
    )
    assert sorted(result.answers) == ["cand_0", "cand_1", "cand_2"]


def test_an_unusable_answer_kind_is_surfaced_not_skipped():
    """A refusal must not read downstream as 'the question was never asked'."""
    fake = decisions.FakeDecider(scripted={"same": {"type": "refusal"}})
    with pytest.raises(decisions.DecisionError, match="unusable answer of kind 'refusal'"):
        decisions.decide(
            state="a claim",
            questions={"same": decisions.noul("Same?")},
            node="n",
            version="v1",
            fake=fake,
        )


def test_asking_for_a_missing_answer_names_what_came_back():
    fake = decisions.FakeDecider(scripted={"same": {"type": "noul", "noul": 1.0}})
    result = decisions.decide(
        state="x",
        questions={"same": decisions.noul("Same?")},
        node="n",
        version="v1",
        fake=fake,
    )
    with pytest.raises(decisions.DecisionError, match="no answer named 'other'"):
        result["other"]


def test_cost_is_input_only():
    """Output tokens are counted for the audit log and charged at nothing."""
    result = decisions.DecisionResult(answers={}, tokens_in=1_000_000, tokens_out=500_000)
    assert result.cost_usd == pytest.approx(0.042)


def test_no_questions_is_a_caller_error():
    with pytest.raises(decisions.DecisionError, match="called with no questions"):
        decisions.decide(state="x", questions={}, node="n", version="v1")


def test_an_oversized_request_fails_here_not_at_the_provider():
    fake = decisions.FakeDecider(default_answer={"type": "noul", "noul": 0.0})
    with pytest.raises(decisions.DecisionError, match="over the 64000 budget"):
        decisions.decide(
            state="x" * 400_000,
            questions={"same": decisions.noul("Same?")},
            node="n",
            version="v1",
            fake=fake,
        )


def test_the_fake_backend_insists_on_a_fake(monkeypatch):
    monkeypatch.setenv("MLR_DECISIONS_BACKEND", "fake")
    with pytest.raises(decisions.DecisionConfigError, match="needs a FakeDecider"):
        decisions.decide(
            state="x", questions={"same": decisions.noul("Same?")}, node="n", version="v1"
        )


def test_the_live_backend_without_a_key_fails_before_the_network(monkeypatch):
    monkeypatch.setenv("MLR_DECISIONS_BACKEND", "typesafe")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(decisions.DecisionConfigError, match="TYPESAFE_API_KEY is not set"):
        decisions.decide(
            state="x", questions={"same": decisions.noul("Same?")}, node="n", version="v1"
        )


def test_record_then_replay_reproduces_the_live_answer(tmp_path, monkeypatch):
    """The contract that makes a paid run reusable: record once, replay forever, no key."""
    monkeypatch.setenv("MLR_CASSETTE_DIR", str(tmp_path))
    payload_questions = {"same": decisions.noul("Same claim?")}
    live_body = {
        "model": "jev-1.13.0",
        "answers": {"same": {"type": "noul", "noul": 0.87}},
        "usage": {"input_tokens": 360, "output_tokens": 39},
    }

    recorder = decisions.FakeDecider(record=True)
    monkeypatch.setenv("MLR_DECISIONS_BACKEND", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(decisions, "_post", lambda payload: live_body)

    live = decisions.decide(
        state="a claim",
        questions=payload_questions,
        node="preapproved_filter",
        version="preapproved-v2",
        fake=recorder,
    )
    assert live["same"].probability == 0.87
    assert live.tokens_in == 360
    assert live.cost_usd == pytest.approx(360 / 1_000_000 * 0.042)
    assert list(tmp_path.glob("*.json")), "a live run with a recorder recorded nothing"

    monkeypatch.setenv("MLR_DECISIONS_BACKEND", "fake")
    replayed = decisions.decide(
        state="a claim",
        questions=payload_questions,
        node="preapproved_filter",
        version="preapproved-v2",
        fake=decisions.FakeDecider(mode="replay"),
    )
    assert replayed["same"].probability == 0.87


def test_changing_the_question_version_invalidates_its_recording(tmp_path, monkeypatch):
    """Otherwise an edited question would replay the answer to a different one."""
    monkeypatch.setenv("MLR_CASSETTE_DIR", str(tmp_path))
    monkeypatch.setenv("MLR_DECISIONS_BACKEND", "fake")
    questions = {"same": decisions.noul("Same claim?")}

    decisions.FakeDecider(record=True).save(
        node="preapproved_filter",
        version="preapproved-v2",
        payload={"state": "a claim", "model": "jev-latest", "questions": questions},
        response={"model": "jev-1.13.0", "answers": {"same": {"type": "noul", "noul": 1.0}}},
    )

    with pytest.raises(decisions.DecisionConfigError, match="no recorded decision"):
        decisions.decide(
            state="a claim",
            questions=questions,
            node="preapproved_filter",
            version="preapproved-v3",
            fake=decisions.FakeDecider(mode="replay"),
        )


def test_a_cassette_is_scrubbed_before_it_is_written(tmp_path, monkeypatch):
    """This repo is public. A recording must never carry anything key-shaped."""
    monkeypatch.setenv("MLR_CASSETTE_DIR", str(tmp_path))
    decisions.FakeDecider(record=True).save(
        node="n",
        version="v1",
        payload={"state": "x", "model": "jev-latest", "questions": {}},
        response={"answers": {}, "leaked": "sk-ant-api03-DEADBEEFDEADBEEFDEADBEEF"},
    )
    written = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "DEADBEEF" not in written
    assert json.loads(written)["node"] == "n"
