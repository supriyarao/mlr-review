"""Model factory: one interface, three interchangeable backends.

Why this exists: the graph should ask for a *capability tier* -- "I need the strong model here,
the cheap one is fine there" -- and never name a model. That keeps cost tiering a configuration
decision instead of something buried in node code, and it makes the whole system runnable in
three very different situations without touching the graph:

======== ================================================================
backend  used for
======== ================================================================
fake     tests, CI, and ``make eval-offline``. Scripted or replayed
         responses, no network, no API key. Forced on in tests/conftest.
ollama   local development. Free, runs on the developer's machine.
anthropic the live demo and the committed eval baseline.
======== ================================================================

Deterministic code (the factory itself). What it returns is what makes LLM calls.

Also here: token and cost accounting. Cost per review is the number that decides whether a
system like this is deployable, so it is measured per call and recorded in the audit log rather
than estimated afterwards.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage

Tier = Literal["strong", "fast"]

# USD per 1M tokens, by model id. Local models are free; that is the point of having them.
# Trade-off: a hard-coded table goes stale when prices change. The alternative -- querying a
# pricing endpoint at runtime -- adds a network dependency to cost accounting, which would then
# fail in exactly the offline mode this system is designed to support.
COST_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Decision models (see platform_core/decisions.py). Not chat models, but they bill per token and
    # their usage flows through the same Usage accumulator, so their prices belong in the same table
    # rather than in a second one that could disagree with it. Output is free, hence the 0.0.
    "jev-latest": (0.042, 0.0),
    "jev-1.13.0": (0.042, 0.0),
}

DEFAULT_MODELS: dict[str, dict[Tier, str]] = {
    "anthropic": {"strong": "claude-sonnet-5", "fast": "claude-haiku-4-5"},
    # Both tiers are the small model here, and the tier distinction is deliberately a fiction on
    # this backend. "Strong" on a local CPU does not mean better, it means slower: measured on this
    # corpus, the same 8.7k-character check prompt takes 192s on qwen2.5:7b and 55s on qwen2.5:1.5b,
    # and the 1.5b held the response schema without a retry.
    #
    # What makes that decisive is not the per-call number but the fan-out. Four branches run
    # concurrently, which is right against a hosted API -- four simultaneous requests -- and exactly
    # wrong against one local process: they queue on the same model and gain nothing. At 7b a review
    # had not finished after half an hour, with the fan-out still going. A review that never
    # finishes teaches a reader nothing, so the default is the one that completes.
    #
    # Trade-off: a 1.5b model finds less and quotes evidence worse, so expect more findings that
    # fail the grounding gate. That fails safe -- an ungrounded finding escalates -- so the cost
    # lands as extra human review rather than as a missed problem. Someone who wants the larger
    # model can set MLR_MODEL_STRONG=qwen2.5:7b and should expect roughly four times the wall clock.
    "ollama": {"strong": "qwen2.5:1.5b", "fast": "qwen2.5:1.5b"},
    "fake": {"strong": "fake-strong", "fast": "fake-fast"},
}

# Anything that looks like a credential, scrubbed before a response is written to a cassette.
# Cassettes are committed to a public repo, so this runs on the way in, not as a review step.
_SECRET_PATTERNS = [
    # Provider-specific key shapes.
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # "Bearer <token>" -- the keyword is followed by whitespace, not a separator, so this needs
    # its own pattern rather than being folded into the name:value rule below.
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    # name: value / name=value, for any credential-ish field name.
    re.compile(
        r"(?i)\b(?:authorization|x-api-key|api[_-]?key|access[_-]?token|token|secret|password)"
        r"\b\s*[:=]\s*\S+"
    ),
]


class LLMConfigError(Exception):
    """Raised when the configured backend cannot be constructed."""


def backend() -> str:
    """Which backend is active. ``MLR_LLM_BACKEND``, defaulting to the free local one."""
    value = os.environ.get("MLR_LLM_BACKEND", "ollama").strip().lower()
    if value not in DEFAULT_MODELS:
        raise LLMConfigError(
            f"unknown MLR_LLM_BACKEND {value!r}; expected one of {sorted(DEFAULT_MODELS)}"
        )
    return value


def model_id(tier: Tier) -> str:
    """Concrete model for a tier, from env, falling back to the backend's default."""
    env_var = "MLR_MODEL_STRONG" if tier == "strong" else "MLR_MODEL_FAST"
    configured = os.environ.get(env_var, "").strip()
    return configured or DEFAULT_MODELS[backend()][tier]


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Cost of one call. Unknown and local models cost nothing."""
    rate_in, rate_out = COST_PER_MTOK.get(model, (0.0, 0.0))
    return (tokens_in / 1_000_000) * rate_in + (tokens_out / 1_000_000) * rate_out


def scrub_secrets(text: str) -> str:
    """Replace anything credential-shaped with a placeholder.

    Applied to recorded model responses before they are written to disk, because cassettes are
    committed to a public repository.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


@dataclass
class Usage:
    """Token, cost and latency accounting for one or more calls."""

    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    calls: int = 0

    def add(self, *, model: str, tokens_in: int, tokens_out: int, latency_ms: int) -> None:
        self.model = model or self.model
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += cost_usd(model, tokens_in, tokens_out)
        self.latency_ms += latency_ms
        self.calls += 1

    def as_audit_fields(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": self.latency_ms,
        }


def usage_from_message(message: BaseMessage, *, model: str, latency_ms: int) -> Usage:
    """Extract token counts from a response, whichever backend produced it.

    Providers report usage in different places, and local models often do not report it at all.
    Missing counts become zero rather than an error: a run must not fail because accounting data
    was absent.
    """
    meta = getattr(message, "usage_metadata", None) or {}
    tokens_in = int(meta.get("input_tokens") or 0)
    tokens_out = int(meta.get("output_tokens") or 0)

    if not (tokens_in or tokens_out):
        raw = getattr(message, "response_metadata", None) or {}
        for key in ("usage", "token_usage", "eval_count"):
            block = raw.get(key)
            if isinstance(block, dict):
                tokens_in = int(block.get("input_tokens") or block.get("prompt_tokens") or 0)
                tokens_out = int(block.get("output_tokens") or block.get("completion_tokens") or 0)
                break

    usage = Usage()
    usage.add(model=model, tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms)
    return usage


def cassette_dir() -> Path:
    """Where recorded responses live. Committed, so the offline eval is reproducible."""
    override = os.environ.get("MLR_CASSETTE_DIR")
    path = (
        Path(override)
        if override
        else Path(__file__).resolve().parent.parent / "evals" / "cassettes"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def cassette_key(*, node: str, prompt_version: str, messages: Any) -> str:
    """Deterministic key for a model call.

    Keyed on the node, its prompt version, and the rendered prompt content -- so changing a prompt
    invalidates its recordings rather than silently replaying answers to a different question.
    """
    payload = json.dumps(
        {"node": node, "prompt_version": prompt_version, "messages": _render(messages)},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{node}__{prompt_version}__{digest}" if node else digest


def _render(messages: Any) -> list[str]:
    """Flatten whatever a caller passes into a stable list of strings for hashing."""
    if isinstance(messages, str):
        return [messages]
    if isinstance(messages, BaseMessage):
        return [str(messages.content)]
    if isinstance(messages, (list, tuple)):
        return [part for item in messages for part in _render(item)]
    return [str(messages)]


@dataclass
class FakeLLM:
    """Offline stand-in for a chat model.

    Two modes, both deterministic:

    * **scripted** -- responses supplied by a test, keyed by node name, with a default fallback.
      Used by unit tests that need a specific model answer to exercise a branch.
    * **replay** -- responses read from ``evals/cassettes/``, keyed by node, prompt version and
      prompt content. This is what makes ``make eval-offline`` reproduce the committed baseline
      exactly, and what lets someone who cloned the repo run the evals with no API key.

    Not a ``BaseChatModel`` subclass: the only surface the graph needs is "send messages, get
    structured output back", and implementing that directly keeps the fake small enough to read.
    The real backends are wrapped in :class:`StructuredCaller` to present the same surface.
    """

    scripted: dict[str, list[str]] = field(default_factory=dict)
    default_response: str = "{}"
    mode: Literal["scripted", "replay"] = "scripted"
    # Write-only use: set on a *live* run so each validated response is saved as a cassette.
    # Backend dispatch keys off the backend name, not off whether a fake is present, so a recorder
    # passed into a live run never answers a call -- it only writes what the real model returned.
    record: bool = False
    _cursor: dict[str, int] = field(default_factory=dict, init=False)
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)

    def respond(self, *, node: str, prompt_version: str, messages: Any) -> str:
        """Return the response text for one call."""
        self.calls.append({"node": node, "prompt_version": prompt_version})

        if self.mode == "replay":
            return self._replay(node=node, prompt_version=prompt_version, messages=messages)

        queue = self.scripted.get(node)
        if not queue:
            return self.default_response
        # Walk the queue so a node called repeatedly (e.g. a retry after a schema failure) can be
        # given a different answer the second time. The last entry repeats once exhausted.
        index = min(self._cursor.get(node, 0), len(queue) - 1)
        self._cursor[node] = index + 1
        return queue[index]

    def _replay(self, *, node: str, prompt_version: str, messages: Any) -> str:
        key = cassette_key(node=node, prompt_version=prompt_version, messages=messages)
        path = cassette_dir() / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        raise LLMConfigError(
            f"no recorded response for {key!r}. Record cassettes with a live backend "
            f"(`make eval` with MLR_LLM_BACKEND=anthropic) before running offline."
        )

    def save(self, *, node: str, prompt_version: str, messages: Any, response: str) -> Path:
        """Write a response to a cassette, scrubbed of anything credential-shaped."""
        key = cassette_key(node=node, prompt_version=prompt_version, messages=messages)
        path = cassette_dir() / f"{key}.json"
        path.write_text(
            json.dumps(
                {
                    "node": node,
                    "prompt_version": prompt_version,
                    "response": scrub_secrets(response),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


def get_chat_model(tier: Tier) -> Any:
    """Construct the raw chat model for a tier from the active backend.

    Most callers want :func:`get_llm` instead; this is the seam that knows provider specifics.
    """
    name = backend()
    model = model_id(tier)

    if name == "fake":
        # GenericFakeChatModel satisfies code paths that insist on a real LangChain model
        # (notably .with_structured_output); FakeLLM handles the response content itself.
        return GenericFakeChatModel(messages=iter([AIMessage(content="{}")]))

    if name == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            # Keep the whole prompt in context: a check prompt carries a label extract plus
            # retrieved evidence, and silent truncation would look like a model mistake.
            num_ctx=32768,
            temperature=0,
            # A local model has no server-side deadline, so without this a wedged call blocks the
            # review forever and the run sits at "running" with nothing to show. Generous, because
            # three minutes for a check prompt on CPU is normal here, not a fault.
            timeout=float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "600")),
        )

    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMConfigError(
                "MLR_LLM_BACKEND=anthropic needs ANTHROPIC_API_KEY. Set it in .env, or use "
                "MLR_LLM_BACKEND=ollama (free, local) or MLR_LLM_BACKEND=fake (offline)."
            )
        # An organisation-level key is not scoped to a workspace, and the API then requires the
        # workspace to be named explicitly -- without it every call fails with a 400 that says so.
        # Supported here rather than insisting on a workspace-scoped key, because which kind of key
        # someone can issue is their organisation's decision, not ours.
        headers = {}
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        if workspace:
            headers["anthropic-workspace-id"] = workspace

        # Note: these models reject temperature/top_p/top_k and thinking.budget_tokens with a
        # 400, so none of them are set here. Reasoning depth is controlled per node instead.
        return ChatAnthropic(
            model=model,
            max_tokens=8192,
            timeout=120,
            stop=None,
            **({"default_headers": headers} if headers else {}),
        )

    raise LLMConfigError(f"unsupported backend {name!r}")


def describe_backend() -> dict[str, str]:
    """Backend and model ids, shown in the UI and recorded in eval results."""
    return {
        "backend": backend(),
        "model_strong": model_id("strong"),
        "model_fast": model_id("fast"),
    }


# ---------------------------------------------------------------------------------------------
# Structured calls
# ---------------------------------------------------------------------------------------------


# HTTP statuses that mean the request itself is wrong, so sending it again cannot help: a bad or
# unscoped key, a malformed request, a model the account cannot reach. Transient statuses (429,
# 5xx) are deliberately absent -- the provider SDKs already retry those with backoff.
_NON_RETRYABLE_STATUSES = {400, 401, 403, 404}


def _request_fault(exc: BaseException) -> str:
    """The provider's own message when a call failed for a reason a retry cannot fix.

    Returns "" for anything that might be a model-output problem.

    This exists because the retry below used to treat every exception as a schema failure, and a
    400 then surfaced as "model output failed validation twice" -- which sent a reader to look at
    the prompt when the actual message was "this API key is not scoped to a workspace". The cost
    was a wasted second call and a misleading diagnosis, which is the worse half.

    Detected by status code and class name rather than by importing a provider's exception types:
    ``platform_core`` must not need to know which backend is configured to report its errors.
    """
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    name = type(exc).__name__
    looks_like_request = any(
        token in name for token in ("InvalidRequest", "Authentication", "Permission", "NotFound")
    )
    if status in _NON_RETRYABLE_STATUSES or looks_like_request:
        return f"{name}: {exc}"
    return ""


class LLMRequestError(Exception):
    """A call the provider rejected outright. Not a model-output problem, so never retried."""


class SchemaFailure(Exception):
    """A model could not produce output matching the requested schema, after a retry.

    Deliberately an exception rather than a sentinel value: a node that cannot parse its own output
    has not produced a reviewable finding, and the correct response is to record the failure and
    escalate to a human -- not to continue with an empty result that reads as "nothing wrong".
    """

    def __init__(self, message: str, *, node: str, attempts: int, last_error: str = "") -> None:
        super().__init__(message)
        self.node = node
        self.attempts = attempts
        self.last_error = last_error


@dataclass
class StructuredResult:
    """A parsed model response, with everything needed to audit the call that produced it."""

    parsed: Any
    usage: Usage
    attempts: int
    # The validation error from the first attempt, when a retry was needed. Kept because a model
    # that needs a retry on every call is a prompt problem worth seeing in the audit log.
    recovered_from: str = ""
    prompt_version: str = ""
    node: str = ""


_RETRY_SUFFIX = """

Your previous response could not be parsed. The error was:

{error}

Return only valid JSON matching the requested schema. Do not include any text outside the JSON."""


def _validate(schema: type, text: str) -> Any:
    """Parse model output into a Pydantic model, tolerating a code fence or surrounding prose.

    Being lenient about packaging while strict about content is deliberate: a model that returns the
    right fields inside a ```json block has done the task, and failing it for formatting would make
    the schema gate a test of punctuation. A response whose *fields* are wrong still fails.
    """
    import json as _json
    import re as _re

    raw = text.strip()
    if not raw:
        raise ValueError("empty response")

    candidate = raw
    fence = _re.search(r"```(?:json)?\s*(.*?)```", raw, _re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    else:
        # Fall back to the outermost JSON object or array in the response.
        match = _re.search(r"[\[{].*[\]}]", raw, _re.DOTALL)
        if match:
            candidate = match.group(0)

    data = _json.loads(candidate)
    return schema.model_validate(data)


def call_structured(
    tier: Tier,
    *,
    node: str,
    prompt: str,
    schema: type,
    prompt_version: str,
    fake: FakeLLM | None = None,
    system: str = "",
) -> StructuredResult:
    """Call a model and return output validated against ``schema``.

    This is the schema gate. One retry on a validation failure, with the error appended to the
    prompt, then :class:`SchemaFailure`. Why exactly one: a model that cannot produce the shape twice
    in a row will usually not produce it on a fifth attempt either, and each retry costs real money
    and latency on a path a reviewer is waiting on. Escalating a schema failure to a human is a
    better outcome than a long retry loop that eventually returns something plausible.

    The same text path is used for every backend, so a cassette recorded against Claude replays
    identically under ``FakeLLM``. Provider-native structured output is used where available to
    improve first-attempt success, but what gets recorded is always the validated JSON.
    """
    backend_name = backend()
    model = model_id(tier)
    usage = Usage()
    attempt_prompt = prompt
    recovered_from = ""

    for attempt in (1, 2):
        start = time.perf_counter()
        try:
            if backend_name == "fake":
                if fake is None:
                    raise LLMConfigError(
                        "the fake backend needs a FakeLLM instance; pass fake=... "
                        "(tests use the fake_llm fixture)"
                    )
                text = fake.respond(
                    node=node, prompt_version=prompt_version, messages=attempt_prompt
                )
                latency = int((time.perf_counter() - start) * 1000)
                usage.add(model=model, tokens_in=0, tokens_out=0, latency_ms=latency)
            else:
                text, call_usage = _invoke_text(
                    tier, prompt=attempt_prompt, system=system, schema=schema
                )
                latency = int((time.perf_counter() - start) * 1000)
                usage.add(
                    model=model,
                    tokens_in=call_usage.tokens_in,
                    tokens_out=call_usage.tokens_out,
                    latency_ms=latency,
                )

            parsed = _validate(schema, text)

        except Exception as exc:  # noqa: BLE001 - sorted into request fault vs schema failure
            fault = _request_fault(exc)
            if fault:
                # Retrying cannot fix the request, and reporting it as a schema failure would hide
                # the provider's own explanation of what is wrong.
                raise LLMRequestError(
                    f"{node}: the {backend_name} backend rejected the request -- {fault}"
                ) from exc
            if attempt == 1:
                recovered_from = f"{type(exc).__name__}: {exc}"
                attempt_prompt = prompt + _RETRY_SUFFIX.format(error=recovered_from)
                continue
            raise SchemaFailure(
                f"{node}: model output failed validation twice",
                node=node,
                attempts=2,
                last_error=f"{type(exc).__name__}: {exc}",
            ) from exc

        if fake is not None and fake.record:
            # Record the validated JSON, not the raw text: replay then produces exactly what the
            # live run produced, and a cassette can never hold output that would fail validation.
            # Gated on the recorder rather than on a separate `record` argument, because a flag and
            # the object that does the writing are two sources of truth: the argument spent a while
            # being passed as True on live runs where `fake` was None, which silently recorded
            # nothing. One switch, on the thing that owns `save`.
            fake.save(
                node=node,
                prompt_version=prompt_version,
                messages=prompt,
                response=parsed.model_dump_json(),
            )

        return StructuredResult(
            parsed=parsed,
            usage=usage,
            attempts=attempt,
            recovered_from=recovered_from,
            prompt_version=prompt_version,
            node=node,
        )

    raise AssertionError("unreachable")  # pragma: no cover


def _invoke_text(tier: Tier, *, prompt: str, system: str, schema: type) -> tuple[str, Usage]:
    """Invoke a live model and return its response as JSON text, plus token usage.

    Provider-native structured output is requested where the backend supports it -- Ollama does
    JSON-schema-constrained decoding, which makes schema-valid output near-certain even from a small
    model -- but the result is normalised back to text so the parse, the retry and the cassette all
    behave identically across backends.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    chat = get_chat_model(tier)
    messages: list[Any] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    structured = None
    try:
        if backend() == "ollama":
            structured = chat.with_structured_output(schema, method="json_schema", include_raw=True)
        else:
            structured = chat.with_structured_output(schema, include_raw=True)
    except (NotImplementedError, TypeError, ValueError):
        # Trade-off: if a backend cannot do structured output we still ask for JSON in the prompt and
        # parse it ourselves. Less reliable per call, but it keeps the system runnable on a model
        # that lacks the feature rather than failing outright.
        structured = None

    if structured is not None:
        result = structured.invoke(messages)
        raw = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else result
        error = result.get("parsing_error") if isinstance(result, dict) else None

        usage = (
            usage_from_message(raw, model=model_id(tier), latency_ms=0)
            if raw is not None
            else Usage()
        )
        if parsed is not None:
            return parsed.model_dump_json(), usage
        if error is not None:
            raise ValueError(f"provider structured output failed: {error}")
        if raw is not None:
            return str(raw.content), usage

    message = chat.invoke(messages)
    return str(message.content), usage_from_message(message, model=model_id(tier), latency_ms=0)
