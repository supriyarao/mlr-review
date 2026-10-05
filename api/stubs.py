"""Example payloads served while a component the API depends on has not been built.

Why this exists: the API and UI are built in parallel with the graph and the eval runner. Until
those land, the endpoints in front of them answer with the committed contract examples from
``tests/fixtures/api/`` -- the same files the contract tests pin -- so the UI can be built and
demonstrated against real shapes.

Every response served from here carries an ``X-MLR-Stub`` header, and ``GET /health`` lists which
components are stubbed. A demo that quietly served example data would be worse than one that says
so. Delete this module when nothing imports it.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "api"

STUB_HEADER = "X-MLR-Stub"


@cache
def _read(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def example(name: str) -> dict[str, Any]:
    """A fresh copy of one committed example response."""
    return json.loads(_read(name))
