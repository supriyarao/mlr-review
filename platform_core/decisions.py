"""Typed decisions: ask a set of fixed questions about some state, get typed answers back.

A decision model is not a chat model, so this is a module of its own rather than a fourth backend
behind :mod:`platform_core.llm`. The difference is the shape of the call, not the vendor: a chat
model takes a prompt and returns text that has to be parsed and validated, while a decision model
takes *declared questions* and returns the answer already typed -- a probability, a chosen option, a
score. There is no JSON to coax out of it and nothing to retry on a schema failure, so the whole
retry-and-validate apparatus in ``call_structured`` has no job here.

Why it earns its place: a yes/no question answered by a chat model costs a full generative call and
comes back as prose that may or may not parse. The same question here is one number.

Backends, chosen by ``MLR_DECISIONS_BACKEND``:

``fake``      tests, CI and the offline eval. Scripted or replayed. No network.
``typesafe``  TypeSafe's System One endpoint, model ``jev-latest``.

Use-case agnostic on purpose: nothing in this module knows what a claim or a finding is.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from platform_core.llm import cassette_dir, cassette_key, cost_usd, scrub_secrets

# Question kinds, named as the wire protocol names them.
#
# Trade-off: these are TypeSafe's names, not neutral ones invented here. A neutral vocabulary would
# need a translation layer in both directions, and the first thing a reader would do is look up which
# neutral name meant which real one. Borrowing the wire's words costs a rename if a second provider
# disagrees; inventing our own costs a mapping table today and still would.
QuestionKind = Literal["noul", "choice", "score"]

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# Prices live in llm.py's COST_PER_MTOK, keyed by model id, alongside the chat models. One table, so
# a price can never be right in one place and stale in the other. Output is billed at zero there.

# The endpoint's documented budget. Enforced locally so an oversized request fails here, with a
# message naming the limit, rather than as a 422 from the provider.
MAX_TOTAL_TOKENS = 64_000
# Rough chars-per-token used only for the guard above. Deliberately pessimistic.
CHARS_PER_TOKEN = 3


class DecisionError(RuntimeError):
    """A decision call could not be completed. Callers decide what a missing answer means."""


class DecisionConfigError(DecisionError):
    """Backend or credentials are not usable. Raised before any network call."""


def backend() -> str:
    """Which decisions backend is active. ``MLR_DECISIONS_BACKEND``, defaulting to ``fake``.

    Defaults to ``fake`` rather than to the live backend, which is the opposite of
    :func:`platform_core.llm.backend`. A decision call is cheap enough that an accidental live call
    would not be noticed on a bill, which is exactly why the default must not be live.
    """
    value = os.environ.get("MLR_DECISIONS_BACKEND", "fake").strip().lower()
    if value not in ("fake", "typesafe"):
        raise DecisionConfigError(
            f"unknown MLR_DECISIONS_BACKEND {value!r}; expected 'fake' or 'typesafe'"
        )
    return value


def model_id() -> str:
    """The model alias to send. ``MLR_DECISIONS_MODEL``, defaulting to ``jev-latest``."""
    return os.environ.get("MLR_DECISIONS_MODEL", "").strip() or DEFAULT_MODEL


def noul(instructions: str, *, true_means: str = "", false_means: str = "") -> dict[str, Any]:
    """A yes/no question. The answer is a probability, not a boolean.

    ``true_means`` / ``false_means`` fill the provider's optional ``criteria`` block. Supplying them
    is worth the characters: they are what stop "is this the same claim" being answered against the
    model's own idea of sameness.
    """
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true_means or false_means:
        question["criteria"] = {"true": true_means, "false": false_means}
    return question


def choice(instructions: str, options: dict[str, str]) -> dict[str, Any]:
    """Pick one of ``options`` (value -> what it means). The answer carries a confidence."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: str, levels: list[str]) -> dict[str, Any]:
    """Rate against ordered ``levels``, lowest first. Two to ten levels."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


@dataclass
class Answer:
    """One typed answer. Exactly one of the three value fields is meaningful."""

    name: str
    kind: str
    # noul: probability that the answer is yes, 0..1.
    probability: float = 0.0
    # choice: the chosen option value.
    chosen: str = ""
    # score: the probability-weighted value.
    value: float = 0.0
    # choice and score only. noul carries its own certainty in the probability itself.
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    # score only: which level each index means, e.g. {"0": "routine", "2": "serious"}. The score
    # scale is zero-based, so a three-level question answers somewhere in 0.0..2.0.
    legend: dict[str, str] = field(default_factory=dict)

    def level(self) -> str:
        """The most likely level of a ``score``, by name.

        The argmax of the probabilities, not the rounded ``value``. ``value`` is a
        probability-weighted average, so a question genuinely torn between "routine" and "serious"
        averages to the middle level the model never actually chose. Rounding it would invent a
        verdict; taking the most likely level reports one the model really assigned, and the number
        it assigned stays available as ``value`` for anyone who wants it.
        """
        if self.kind != "score":
            raise DecisionError(
                f"{self.name}: level() is only meaningful for a score, not {self.kind}"
            )
        if not self.probabilities:
            return ""
        index = max(self.probabilities, key=lambda k: self.probabilities[k])
        return self.legend.get(index, index)

    def is_yes(self, threshold: float = 0.5) -> bool:
        """Collapse a ``noul`` to a boolean at an explicit threshold.

        The threshold is a parameter with no default worth trusting: 0.5 is the coin-flip line, not a
        policy. Callers that care should pass their own, and the point of a decision model is that
        they *can* -- a chat model's ``true`` gives them nothing to tune.
        """
        if self.kind != "noul":
            raise DecisionError(
                f"{self.name}: is_yes() is only meaningful for a noul, not {self.kind}"
            )
        return self.probability >= threshold


@dataclass
class DecisionResult:
    """The answers to one request, plus what the call cost."""

    answers: dict[str, Answer]
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0

    @property
    def cost_usd(self) -> float:
        """Input-only billing. Output tokens are recorded for the audit log and charged at zero.

        Falls back to the configured alias when the response did not name a model, so a cost is
        still attributed rather than silently becoming zero.
        """
        return cost_usd(self.model or model_id(), self.tokens_in, self.tokens_out)

    def __getitem__(self, name: str) -> Answer:
        try:
            return self.answers[name]
        except KeyError as exc:
            raise DecisionError(f"no answer named {name!r}; got {sorted(self.answers)}") from exc


@dataclass
class FakeDecider:
    """Offline stand-in for the decisions endpoint. Mirrors :class:`platform_core.llm.FakeLLM`.

    Two modes, both deterministic, plus a write-only recording mode:

    * **scripted** -- answers supplied by a test, keyed by question name, with a default.
    * **replay** -- answers read from the cassette directory, keyed by node, version and request.

    ``record=True`` on a *live* run saves each response as a cassette. Gated on this object rather
    than on a flag passed alongside it, for the reason written on ``FakeLLM.record``: a flag and the
    thing that does the writing are two sources of truth, and the one that cannot fire silently
    records nothing.
    """

    # question name -> the raw answer dict the provider would have returned
    scripted: dict[str, dict[str, Any]] = field(default_factory=dict)
    default_answer: dict[str, Any] | None = None
    mode: Literal["scripted", "replay"] = "scripted"
    record: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)

    def respond(self, *, node: str, version: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return the whole response body for one request."""
        self.calls.append({"node": node, "version": version})

        if self.mode == "replay":
            key = cassette_key(node=node, prompt_version=version, messages=_canonical(payload))
            path = cassette_dir() / f"{key}.json"
            if not path.exists():
                raise DecisionConfigError(
                    f"no recorded decision for {key!r}. Record with a live backend "
                    f"(MLR_DECISIONS_BACKEND=typesafe and a recorder) before replaying."
                )
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        answers: dict[str, Any] = {}
        for name in payload.get("questions", {}):
            scripted = self.scripted.get(name, self.default_answer)
            if scripted is None:
                raise DecisionConfigError(
                    f"FakeDecider has no scripted answer for {name!r} and no default_answer"
                )
            answers[name] = dict(scripted)
        return {"model": "fake-decider", "answers": answers, "usage": {}}

    def save(
        self, *, node: str, version: str, payload: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Write one response to a cassette, scrubbed of anything credential-shaped."""
        key = cassette_key(node=node, prompt_version=version, messages=_canonical(payload))
        path = cassette_dir() / f"{key}.json"
        body = json.loads(scrub_secrets(json.dumps(response)))
        path.write_text(
            json.dumps({"node": node, "version": version, "response": body}, indent=2),
            encoding="utf-8",
        )


def _canonical(payload: dict[str, Any]) -> str:
    """The request as a stable string, so a cassette key does not move with dict ordering."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _parse(body: dict[str, Any]) -> dict[str, Answer]:
    """Turn the provider's answers map into typed answers.

    Unknown answer kinds are surfaced rather than skipped. A ``refusal`` is the documented case, and
    silently dropping it would read downstream as "the question was not asked".
    """
    answers: dict[str, Answer] = {}
    for name, raw in (body.get("answers") or {}).items():
        kind = str(raw.get("type") or "")
        if kind not in ("noul", "choice", "score"):
            raise DecisionError(f"{name}: unusable answer of kind {kind!r}")
        answers[name] = Answer(
            name=name,
            kind=kind,
            probability=float(raw.get("noul") or 0.0),
            chosen=str(raw.get("choice") or ""),
            value=float(raw.get("score") or 0.0),
            confidence=float(raw.get("confidence") or 0.0),
            probabilities={str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()},
            legend={str(k): str(v) for k, v in (raw.get("legend") or {}).items()},
        )
    return answers


def decide(
    *,
    state: Any,
    questions: dict[str, dict[str, Any]],
    node: str,
    version: str,
    fake: FakeDecider | None = None,
) -> DecisionResult:
    """Ask ``questions`` about ``state`` and return typed answers.

    **A model decision**, but not a generative one: the questions are declared by the caller and the
    answer is a number, so there is no output to validate and no retry on a malformed shape. The one
    retry that does exist is for a rate limit, which is a condition of the transport rather than of
    the answer.

    ``node`` and ``version`` are recorded with every cassette and are what make a question change
    invalidate its recordings, exactly as a prompt version does for a chat call.
    """
    if not questions:
        raise DecisionError(f"{node}: decide() called with no questions")

    payload = {"state": state, "model": model_id(), "questions": questions}

    budget = len(_canonical(payload)) // CHARS_PER_TOKEN
    if budget > MAX_TOTAL_TOKENS:
        raise DecisionError(
            f"{node}: request is roughly {budget} tokens, over the {MAX_TOTAL_TOKENS} budget"
        )

    name = backend()
    started = time.perf_counter()

    if name == "fake":
        if fake is None:
            raise DecisionConfigError(
                "the fake decisions backend needs a FakeDecider instance; pass fake=..."
            )
        body = fake.respond(node=node, version=version, payload=payload)
    else:
        body = _post(payload)
        if fake is not None and fake.record:
            fake.save(node=node, version=version, payload=payload, response=body)

    usage = body.get("usage") or {}
    return DecisionResult(
        answers=_parse(body),
        model=str(body.get("model") or ""),
        tokens_in=int(usage.get("input_tokens") or 0),
        tokens_out=int(usage.get("output_tokens") or 0),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def _post(payload: dict[str, Any]) -> dict[str, Any]:
    """One HTTP call to the live endpoint, with a single backoff retry on a rate limit.

    Trade-off: one retry, not an escalating loop. The caller of a decision is on a path a reviewer is
    waiting on, and a second 429 means the service is saturated rather than briefly busy -- at which
    point failing and letting the node fall back is better than holding the review open.
    """
    import httpx

    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise DecisionConfigError(
            "TYPESAFE_API_KEY is not set. Set it, or use MLR_DECISIONS_BACKEND=fake."
        )

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in (1, 2):
        try:
            response = httpx.post(ENDPOINT, json=payload, headers=headers, timeout=30.0)
        except httpx.HTTPError as exc:
            raise DecisionError(f"decisions request failed: {type(exc).__name__}: {exc}") from exc

        if response.status_code in (429, 529) and attempt == 1:
            time.sleep(1.0)
            continue
        if response.status_code == 401:
            raise DecisionConfigError(
                "decisions endpoint rejected the key (401). Check TYPESAFE_API_KEY."
            )
        if response.status_code >= 400:
            # The body carries the provider's reason; truncated so a 422 listing every question
            # does not fill the audit log.
            raise DecisionError(
                f"decisions endpoint returned {response.status_code}: {response.text[:300]}"
            )
        return dict(response.json())

    raise DecisionError("decisions endpoint stayed rate limited after a retry")
