"""Calling code the platform did not write, and recording that it was called.

Deterministic code. No model decisions here; a tool may itself be anything.

**Why this exists.** A company adopting this platform already owns validation code -- often years of
it, already qualified, already argued over with their regulatory function. Asking them to re-express
that as prompts would be both a waste and a regression: a number their own code computed is evidence
an auditor can check, and a number a model estimated is not. So the platform calls their code and
hands the result to the model as input, rather than replacing it.

A tool here is a named, versioned function with a declared input and output schema. Three properties
make it worth having a registry rather than just importing the function:

* **The version is recorded.** A tool's verdict lands in the audit trail beside the prompt version
  and the policy version, so "why did this run decide that" has one more answer available.
* **A failure is a result, not an exception.** A tool that raises returns ``ok=False`` with the
  error, and the calling node decides what to do. A review must not die because a helper did.
* **The caller is a seam.** Today the nodes invoke tools directly, because the tools they use have
  to run on every piece. Letting a model choose would be a change of caller, not of design.

The registry holds no MLR concepts. A tool that knows what a claim is belongs in ``mlr/tools/``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolError(Exception):
    """Raised for a registry problem -- an unknown tool, or a duplicate registration."""


class ToolResult(BaseModel):
    """What a tool call produced, successful or not.

    Carries the tool's identity and version with the data, because this object is what gets written
    to the audit trail and shown to the model. A result that cannot say which version of which tool
    produced it is not evidence.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str
    version: str
    ok: bool = True
    # Whatever the tool returned, already JSON-serialisable.
    data: dict[str, Any] = Field(default_factory=dict)
    # Set when ok is False. The node decides whether that is fatal; usually it is not.
    error: str = ""
    latency_ms: int = 0

    def label(self) -> str:
        return f"{self.tool} v{self.version}"


class ToolSpec(BaseModel):
    """A registered tool: what it is called, what it does, and what it returns."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    version: str
    # One sentence, written for whoever reads the audit trail -- and the text a model would be shown
    # if a tool were ever offered for the model to choose.
    description: str
    # Called with the payload, returns a JSON-serialisable dict.
    fn: Callable[..., dict[str, Any]]
    # Keys the payload must carry. Checked before the call so a missing input is a clear error
    # rather than a KeyError from inside somebody else's code.
    requires: tuple[str, ...] = ()


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
    """Add a tool. Refuses to shadow an existing name unless asked, so two registrations of the
    same name are a loud error rather than a silent last-one-wins."""
    if spec.name in _REGISTRY and not replace:
        raise ToolError(f"a tool named {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> ToolSpec:
    if name not in _REGISTRY:
        raise ToolError(f"no tool named {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered() -> list[ToolSpec]:
    """Every tool, by name. The API serves this so a reviewer can see what the system can call."""
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def call(name: str, payload: dict[str, Any]) -> ToolResult:
    """Run a tool and return its result. **Never raises** for anything the tool did.

    A tool is somebody else's code. It can be slow, it can be wrong, and it can throw -- and none of
    those are reasons for a review to fail, because the model can still judge the piece without it
    and the reviewer still sees the finding. So a failure comes back as ``ok=False`` with the error
    text, which the node puts in front of the model as "this measurement was unavailable".

    Trade-off: that means a silently broken tool degrades reviews instead of stopping them. The
    protection is that the failure is in the audit trail for every affected run, not swallowed --
    one broken tool is visible as a pattern the moment anyone looks.
    """
    started = time.perf_counter()
    try:
        spec = get(name)
    except ToolError as exc:
        return ToolResult(tool=name, version="", ok=False, error=str(exc))

    missing = [key for key in spec.requires if key not in payload]
    if missing:
        return ToolResult(
            tool=spec.name,
            version=spec.version,
            ok=False,
            error=f"missing required input(s): {', '.join(missing)}",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    try:
        data = spec.fn(**payload)
    except Exception as exc:
        return ToolResult(
            tool=spec.name,
            version=spec.version,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    return ToolResult(
        tool=spec.name,
        version=spec.version,
        ok=True,
        data=dict(data or {}),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
