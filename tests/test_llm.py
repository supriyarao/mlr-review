"""The model factory: tier indirection, cost accounting, FakeLLM determinism, secret scrubbing."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from platform_core import llm


def test_the_graph_asks_for_a_tier_not_a_model(monkeypatch):
    """Cost tiering is configuration. Changing a tier's model must not touch node code."""
    monkeypatch.setenv("MLR_LLM_BACKEND", "anthropic")
    monkeypatch.setenv("MLR_MODEL_STRONG", "claude-opus-5")
    monkeypatch.setenv("MLR_MODEL_FAST", "claude-haiku-4-5")
    assert llm.model_id("strong") == "claude-opus-5"
    assert llm.model_id("fast") == "claude-haiku-4-5"


def test_each_backend_has_sensible_defaults(monkeypatch):
    monkeypatch.delenv("MLR_MODEL_STRONG", raising=False)
    monkeypatch.delenv("MLR_MODEL_FAST", raising=False)
    for backend, strong in [
        ("anthropic", "claude-sonnet-5"),
        # Deliberately the small model, and deliberately the same as the fast tier: on a local CPU
        # backend "strong" means slower rather than better, and at 7b a review did not finish.
        # See the comment on DEFAULT_MODELS.
        ("ollama", "qwen2.5:1.5b"),
        ("fake", "fake-strong"),
    ]:
        monkeypatch.setenv("MLR_LLM_BACKEND", backend)
        assert llm.model_id("strong") == strong


def test_the_ollama_tiers_are_the_same_model_on_purpose(monkeypatch):
    """Pinned so a well-meaning change back to a bigger strong model is a conscious one.

    Tiering is real on a hosted backend and a fiction locally. Four fan-out branches queue on one
    ollama process, so a larger strong model does not buy quality per unit time -- it stops the
    review finishing at all.
    """
    monkeypatch.delenv("MLR_MODEL_STRONG", raising=False)
    monkeypatch.delenv("MLR_MODEL_FAST", raising=False)
    monkeypatch.setenv("MLR_LLM_BACKEND", "ollama")
    assert llm.model_id("strong") == llm.model_id("fast")


def test_the_model_tier_can_still_be_overridden(monkeypatch):
    """Anyone who wants the larger local model can have it, and wait for it."""
    monkeypatch.setenv("MLR_LLM_BACKEND", "ollama")
    monkeypatch.setenv("MLR_MODEL_STRONG", "qwen2.5:7b")
    assert llm.model_id("strong") == "qwen2.5:7b"


def test_an_unknown_backend_fails_loudly(monkeypatch):
    monkeypatch.setenv("MLR_LLM_BACKEND", "gpt-something")
    with pytest.raises(llm.LLMConfigError, match="unknown MLR_LLM_BACKEND"):
        llm.backend()


def test_anthropic_backend_refuses_to_start_without_a_key(monkeypatch):
    """The error must name the free alternatives, not just complain."""
    monkeypatch.setenv("MLR_LLM_BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.LLMConfigError, match="MLR_LLM_BACKEND=ollama"):
        llm.get_chat_model("strong")


def test_cost_is_computed_per_call():
    # 1M input tokens on Sonnet 5 is $2.00; 1M output is $10.00.
    assert llm.cost_usd("claude-sonnet-5", 1_000_000, 0) == pytest.approx(2.00)
    assert llm.cost_usd("claude-sonnet-5", 0, 1_000_000) == pytest.approx(10.00)
    assert llm.cost_usd("claude-sonnet-5", 1000, 500) == pytest.approx(0.002 + 0.005)


def test_local_and_unknown_models_cost_nothing():
    """A free local model must not inflate the cost figures the demo reports."""
    assert llm.cost_usd("qwen2.5:7b", 100_000, 50_000) == 0.0
    assert llm.cost_usd("fake-strong", 100_000, 50_000) == 0.0


def test_usage_accumulates_across_calls():
    usage = llm.Usage()
    usage.add(model="claude-sonnet-5", tokens_in=1000, tokens_out=500, latency_ms=800)
    usage.add(model="claude-sonnet-5", tokens_in=2000, tokens_out=300, latency_ms=400)
    assert usage.calls == 2
    assert usage.tokens_in == 3000
    assert usage.latency_ms == 1200
    assert usage.as_audit_fields()["cost_usd"] == pytest.approx(0.014, abs=1e-6)


def test_missing_token_counts_do_not_fail_a_run():
    """Local models often report no usage. Accounting gaps must not break the review."""
    from langchain_core.messages import AIMessage

    usage = llm.usage_from_message(AIMessage(content="hi"), model="qwen2.5:7b", latency_ms=10)
    assert usage.tokens_in == 0
    assert usage.cost_usd == 0.0
    assert usage.calls == 1


def test_token_counts_are_read_from_provider_metadata():
    from langchain_core.messages import AIMessage

    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
    )
    usage = llm.usage_from_message(msg, model="claude-sonnet-5", latency_ms=50)
    assert (usage.tokens_in, usage.tokens_out) == (120, 30)


# --- FakeLLM ------------------------------------------------------------------------------------


def test_scripted_responses_are_keyed_by_node():
    fake = llm.FakeLLM(scripted={"extract_claims": ['{"claims": []}']}, default_response="{}")
    assert (
        fake.respond(node="extract_claims", prompt_version="v1", messages="p") == '{"claims": []}'
    )
    assert fake.respond(node="off_label", prompt_version="v1", messages="p") == "{}"


def test_a_node_can_be_given_a_different_answer_on_retry():
    """Needed to exercise the schema-failure retry path: bad output, then good output."""
    fake = llm.FakeLLM(scripted={"extract_claims": ["not json", '{"claims": []}']})
    first = fake.respond(node="extract_claims", prompt_version="v1", messages="p")
    second = fake.respond(node="extract_claims", prompt_version="v1", messages="p")
    assert first == "not json"
    assert second == '{"claims": []}'
    # Once exhausted the last response repeats rather than raising.
    assert (
        fake.respond(node="extract_claims", prompt_version="v1", messages="p") == '{"claims": []}'
    )


def test_fake_llm_records_which_nodes_called_it():
    fake = llm.FakeLLM()
    fake.respond(node="substantiation", prompt_version="sub-v1", messages="p")
    assert fake.calls == [{"node": "substantiation", "prompt_version": "sub-v1"}]


def test_replay_is_deterministic_for_identical_prompts(tmp_path, monkeypatch):
    monkeypatch.setenv("MLR_CASSETTE_DIR", str(tmp_path))
    recorder = llm.FakeLLM()
    recorder.save(
        node="off_label",
        prompt_version="off-v1",
        messages="is this off label?",
        response='{"findings": []}',
    )

    player = llm.FakeLLM(mode="replay")
    for _ in range(3):
        assert (
            player.respond(node="off_label", prompt_version="off-v1", messages="is this off label?")
            == '{"findings": []}'
        )


def test_changing_a_prompt_version_invalidates_its_recording(tmp_path, monkeypatch):
    """Otherwise an edited prompt would silently replay answers to a different question."""
    monkeypatch.setenv("MLR_CASSETTE_DIR", str(tmp_path))
    llm.FakeLLM().save(node="off_label", prompt_version="off-v1", messages="q", response="old")

    player = llm.FakeLLM(mode="replay")
    assert player.respond(node="off_label", prompt_version="off-v1", messages="q") == "old"
    with pytest.raises(llm.LLMConfigError, match="no recorded response"):
        player.respond(node="off_label", prompt_version="off-v2", messages="q")


def test_changing_the_prompt_content_invalidates_its_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("MLR_CASSETTE_DIR", str(tmp_path))
    llm.FakeLLM().save(node="off_label", prompt_version="v1", messages="question A", response="x")
    with pytest.raises(llm.LLMConfigError):
        llm.FakeLLM(mode="replay").respond(
            node="off_label", prompt_version="v1", messages="question B"
        )


def test_a_missing_recording_explains_how_to_make_one(tmp_path, monkeypatch):
    monkeypatch.setenv("MLR_CASSETTE_DIR", str(tmp_path))
    with pytest.raises(llm.LLMConfigError, match="Record cassettes with a live backend"):
        llm.FakeLLM(mode="replay").respond(node="n", prompt_version="v1", messages="q")


def test_cassette_keys_are_stable_across_equivalent_message_shapes():
    a = llm.cassette_key(node="n", prompt_version="v1", messages="hello")
    b = llm.cassette_key(node="n", prompt_version="v1", messages=["hello"])
    assert a == b


# --- Secret scrubbing ---------------------------------------------------------------------------


def test_credential_shaped_strings_are_scrubbed():
    """Cassettes are committed to a public repo, so this runs on the way in."""
    dirty = (
        "Call failed. x-api-key: sk-ant-api03-ABCdef123456789 and "
        "Authorization: Bearer abc.def.ghi plus AKIAIOSFODNN7EXAMPLE"
    )
    clean = llm.scrub_secrets(dirty)
    assert "sk-ant-api03" not in clean
    assert "AKIAIOSFODNN7EXAMPLE" not in clean
    assert "abc.def.ghi" not in clean
    assert "[REDACTED]" in clean


def test_saving_a_cassette_scrubs_before_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLR_CASSETTE_DIR", str(tmp_path))
    path = llm.FakeLLM().save(
        node="n",
        prompt_version="v1",
        messages="q",
        response="leaked sk-ant-api03-SHOULDNOTAPPEAR here",
    )
    written = path.read_text(encoding="utf-8")
    assert "SHOULDNOTAPPEAR" not in written
    assert json.loads(written)["response"] == "leaked [REDACTED] here"


def test_describe_backend_reports_what_actually_ran(monkeypatch):
    """Recorded in eval results, so a number can be tied to the models that produced it."""
    monkeypatch.setenv("MLR_LLM_BACKEND", "ollama")
    monkeypatch.setenv("MLR_MODEL_STRONG", "qwen2.5:7b")
    monkeypatch.setenv("MLR_MODEL_FAST", "qwen2.5:7b")
    assert llm.describe_backend() == {
        "backend": "ollama",
        "model_strong": "qwen2.5:7b",
        "model_fast": "qwen2.5:7b",
    }


class _Verdict(BaseModel):
    ok: bool


def test_a_live_run_with_a_recorder_writes_a_cassette(tmp_path, monkeypatch):
    """The whole point of `--record`: one paid run has to leave something replayable behind.

    Regression test. Recording was gated on a `record` argument to ``call_structured`` that no
    caller ever passed, and which could not have worked if it had -- saving needs the ``FakeLLM``
    that does the writing, and live runs passed ``fake=None``. So `--record` cost money and
    produced an empty evals/cassettes/. Nothing caught it because no test called
    ``call_structured`` at all.
    """
    monkeypatch.setenv("MLR_CASSETTE_DIR", str(tmp_path))
    monkeypatch.setenv("MLR_LLM_BACKEND", "anthropic")

    def fake_invoke(tier, *, prompt, system, schema):
        return '{"ok": true}', llm.Usage()

    monkeypatch.setattr(llm, "_invoke_text", fake_invoke)

    recorder = llm.FakeLLM(record=True)
    result = llm.call_structured(
        "strong",
        node="off_label",
        prompt="is this off label?",
        schema=_Verdict,
        prompt_version="off-v1",
        fake=recorder,
    )

    assert result.parsed.ok is True
    assert list(tmp_path.glob("*.json")), "a live run with a recorder recorded nothing"

    # And the recording has to be the thing replay will serve back.
    player = llm.FakeLLM(mode="replay")
    assert json.loads(
        player.respond(node="off_label", prompt_version="off-v1", messages="is this off label?")
    ) == {"ok": True}


def test_a_live_run_without_a_recorder_writes_nothing(tmp_path, monkeypatch):
    """`make eval` is documented as recording nothing. Keep it that way."""
    monkeypatch.setenv("MLR_CASSETTE_DIR", str(tmp_path))
    monkeypatch.setenv("MLR_LLM_BACKEND", "anthropic")
    monkeypatch.setattr(
        llm, "_invoke_text", lambda tier, *, prompt, system, schema: ('{"ok": true}', llm.Usage())
    )

    llm.call_structured(
        "strong",
        node="off_label",
        prompt="q",
        schema=_Verdict,
        prompt_version="off-v1",
        fake=None,
    )
    assert not list(tmp_path.glob("*.json"))
