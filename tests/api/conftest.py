"""Shared fixtures for the API tests.

Everything runs through FastAPI's ``TestClient`` against the real app: no network, no model, and
(via the autouse ``isolated_data_dir`` in the root conftest) a fresh data directory per test.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.test_service import seeded  # noqa: F401  (the shipped corpus, indexed for this test)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "api"
SAMPLE = REPO_ROOT / "corpus" / "promo" / "zelvoran-hcp-subgroup-claim.md"

# Keys whose value is a free-form mapping: the example shows one instance, not a schema.
FREE_FORM = frozenset(
    {"params", "metrics", "tenant_config_hashes", "policy_flags", "doc_meta", "per_tenant"}
)


def fixture(name: str) -> Any:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def assert_same_shape(actual: Any, expected: Any, path: str = "$") -> None:
    """Assert ``actual`` has every key the contract example has, with the same JSON types.

    Used where a response is real data rather than the example itself (a rule created during the
    test has its own id and timestamp), so equality is the wrong check but the shape still is the
    contract. Extra keys are allowed -- adding an optional field is a safe change -- but a missing
    or retyped one is exactly the drift the fixtures exist to catch.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected an object, got {type(actual).__name__}"
        for key, value in expected.items():
            assert key in actual, f"{path}: missing key {key!r}"
            if key not in FREE_FORM:
                assert_same_shape(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list, got {type(actual).__name__}"
        if expected:
            for index, item in enumerate(actual):
                assert_same_shape(item, expected[0], f"{path}[{index}]")
    elif isinstance(expected, bool):
        assert isinstance(actual, bool), f"{path}: expected a boolean, got {actual!r}"
    elif isinstance(expected, int | float):
        assert isinstance(actual, int | float) and not isinstance(actual, bool), (
            f"{path}: expected a number, got {actual!r}"
        )
    elif isinstance(expected, str):
        assert isinstance(actual, str), f"{path}: expected a string, got {actual!r}"
    # A null in the example says the field is optional; any value is acceptable.


@pytest.fixture
def client() -> TestClient:
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def isolated_evals(tmp_path, monkeypatch):
    """Point the eval store at an empty temp directory and forget any in-flight jobs, so a test
    never reads the developer's own results or another test's run."""
    from api import evals

    monkeypatch.setattr(evals, "RESULTS_DIR", tmp_path / "eval-results")
    monkeypatch.setattr(evals, "BASELINE_PATH", tmp_path / "eval-baselines" / "latest.json")
    evals._jobs.clear()
    yield
    evals._jobs.clear()


@pytest.fixture
def tenants_copy(tmp_path, monkeypatch) -> Path:
    """A private copy of the tenant directory, for tests that write rules back to YAML."""
    from platform_core import tenancy

    target = tmp_path / "tenants"
    shutil.copytree(REPO_ROOT / "corpus" / "tenants", target)
    monkeypatch.setenv("MLR_TENANT_DIR", str(target))
    tenancy.clear_tenant_cache()
    yield target
    tenancy.clear_tenant_cache()


@pytest.fixture
def seeded_rules(isolated_data_dir) -> None:
    """Load both tenants' shipped rules into the registry, as ``make seed`` does."""
    from platform_core import tenancy
    from scripts.seed import seed_rules

    for tenant_id in tenancy.tenant_ids():
        _, refused = seed_rules(tenant_id)
        assert not refused, f"shipped rules refused activation: {refused}"


@pytest.fixture
def real_runs(monkeypatch, seeded):  # noqa: F811
    """Reviews run the real graph, offline, and finish before the request returns.

    The service normally runs a review on a worker thread and the client polls. A test wants the
    finished run, so submissions are made blocking here; nothing else about the path changes.
    """
    from mlr import service

    start = service.start_run
    monkeypatch.setattr(service, "start_run", lambda **kwargs: start(**{**kwargs, "block": True}))


@pytest.fixture
def submitted(client, real_runs) -> str:
    """Submit the sample piece and return its run id. The run stops at the human step."""
    response = client.post(
        "/reviews",
        data={
            "tenant_id": "acme_pharma",
            "product_id": "zelvoran",
            "audience": "hcp",
            "channel": "sales_aid",
            "market": "us",
        },
        files={"file": (SAMPLE.name, SAMPLE.read_bytes(), "text/markdown")},
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def draft_rule(**overrides: Any) -> dict[str, Any]:
    """A valid deterministic rule that is not part of the shipped set."""
    rule = {
        "id": "ACME-BAN-003",
        "tenant_id": "acme_pharma",
        "title": "Prohibited hype language",
        "description": (
            "Promotional material must not describe an Acme product as a breakthrough or as "
            "revolutionary, because neither term has a defined clinical meaning."
        ),
        "kind": "deterministic",
        "severity": "blocker",
        "check_type": "business_rule",
        "params": {"type": "banned_terms", "terms": ["breakthrough", "revolutionary"]},
        "examples": [
            {"text": "Zelvoran is a breakthrough in psoriasis care.", "expected": "fail"},
            {"text": "Zelvoran achieved PASI-75 in 68.4% of adults.", "expected": "pass"},
        ],
        "owner": "Regulatory Affairs",
        "effective_from": "2026-01-01",
        "source_ref": "Acme Brand and Promotional Code, clause 9",
    }
    rule.update(overrides)
    return rule
