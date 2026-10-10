"""Shared test configuration.

The whole suite runs offline. Forcing the ``fake`` model backend here means a test can never
accidentally reach the network or depend on a local Ollama server, whatever the developer's
``.env`` happens to say.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Set before any application module is imported, so module-level config reads see it.
os.environ["MLR_LLM_BACKEND"] = "fake"
os.environ.pop("ANTHROPIC_API_KEY", None)
# Same contract for the decisions endpoint: the test suite must never reach a network, and a
# developer with MLR_DECISIONS_BACKEND=typesafe in their .env must not be able to make `make test`
# spend money. Both the backend and the key are removed, so a miss fails loudly rather than dialling
# out.
os.environ["MLR_DECISIONS_BACKEND"] = "fake"
os.environ.pop("TYPESAFE_API_KEY", None)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Give every test its own data directory.

    Autouse because the alternative -- remembering to request it -- means one forgotten fixture
    writes a test's SQLite rows into the developer's real ``data/`` and the next run sees them.
    """
    from mlr import service
    from platform_core import audit

    monkeypatch.setenv("MLR_DATA_DIR", str(tmp_path / "data"))
    audit.reset_engine()
    # The service caches a checkpoint reader keyed on the data directory; drop it so a test cannot
    # read another test's checkpoints.
    service.reset_read_cache()
    yield tmp_path / "data"
    audit.reset_engine()
    service.reset_read_cache()


@pytest.fixture
def tenant_dir(tmp_path, monkeypatch):
    """An empty tenant-config directory this test owns."""
    from platform_core import tenancy

    d = tmp_path / "tenants"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLR_TENANT_DIR", str(d))
    tenancy.clear_tenant_cache()
    yield d
    tenancy.clear_tenant_cache()


@pytest.fixture
def write_tenant(tenant_dir):
    """Write a tenant config file, returning its loaded form."""
    import yaml

    from platform_core import tenancy

    def _write(config: dict) -> tenancy.TenantConfig:
        directory = tenant_dir / config["tenant_id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tenant.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        tenancy.clear_tenant_cache()
        return tenancy.load_tenant(config["tenant_id"])

    return _write


def minimal_tenant(tenant_id: str = "acme_pharma", **overrides) -> dict:
    """A valid minimal tenant config, for tests that only care about one field."""
    config = {
        "tenant_id": tenant_id,
        "name": "Acme Pharma",
        "version": "2026.1",
        "confidence_threshold": 70,
        "products": [
            {
                "product_id": "zelvoran",
                "name": "Zelvoran",
                "label_version": "v1.0",
                "indication": "moderate-to-severe plaque psoriasis in adults",
            }
        ],
        "severity_rules": [
            {
                "check_type": "off_label",
                "severity": "blocker",
                "rule_id": "OFF-LABEL-1",
                "rationale": "Promotion outside the approved label is never acceptable.",
            }
        ],
        "escalation": {
            "standard_tier": "standard_reviewer",
            "default_escalation_tier": "senior_regulatory",
            "rules": [],
        },
        "policy_flags": {},
    }
    config.update(overrides)
    return config
