"""The scaffold has to produce a tenant that really works, not one that merely parses.

Why this is worth a test: the first version of ``scripts/new_tenant.py`` wrote a starter ISI rule
whose own examples did not behave as labelled. Everything about it looked right -- the YAML was
valid, the file said ``status: active`` -- and seeding quietly refused to activate it, leaving the
tenant with a rule that never fired. Nothing failed. The only thing that caught it was scaffolding
a throwaway tenant by hand and reading the seed output.

So these tests do what that session did: scaffold into a temp directory, then put the result
through the same validation the platform uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from platform_core.rules.schema import Rule
from platform_core.rules.validation import validate_rule
from platform_core.tenancy import load_tenant_from_path
from scripts.new_tenant import scaffold

TENANT_ID = "scaffold_test"
PRODUCT_ID = "testium"


@pytest.fixture
def scaffolded(tmp_path: Path) -> Path:
    scaffold(TENANT_ID, "Scaffold Test", PRODUCT_ID, "Testium", root=tmp_path)
    return tmp_path


def test_the_scaffolded_config_validates(scaffolded: Path) -> None:
    config = load_tenant_from_path(scaffolded / "corpus" / "tenants" / TENANT_ID / "tenant.yaml")
    assert config.tenant_id == TENANT_ID
    assert [p.product_id for p in config.products] == [PRODUCT_ID]


def test_every_scaffolded_rule_can_actually_activate(scaffolded: Path) -> None:
    """A rule that says ``active`` but fails its own examples is seeded as a draft and never fires.

    The YAML keeps saying active, so this is invisible unless the examples are executed.
    """
    tenant_root = scaffolded / "corpus" / "tenants" / TENANT_ID
    rule_files = sorted((tenant_root / "rules").glob("*.yaml"))
    assert rule_files, "the scaffold should write at least one starter rule"

    for path in rule_files:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        report = validate_rule(
            Rule.model_validate(raw), reference_docs_dir=tenant_root / "reference_docs"
        )
        assert report.can_activate, f"{path.name}: {report.summary()}"


def test_the_studies_are_named_so_seeding_will_find_them(scaffolded: Path) -> None:
    """Retrieval matches study files by the prefix the product declares.

    A mismatch here does not fail: it returns no evidence, so every claim comes back unverifiable
    and it reads as a model problem. This was a hardcoded dict in scripts/seed.py.
    """
    config = load_tenant_from_path(scaffolded / "corpus" / "tenants" / TENANT_ID / "tenant.yaml")
    prefix = config.products[0].studies_prefix()
    found = sorted(p.name for p in (scaffolded / "corpus" / "references").glob(f"{prefix}-*.md"))
    assert len(found) == 2, f"expected two studies matching {prefix}-*.md, found {found}"


def test_the_scaffold_does_not_overwrite_existing_files(scaffolded: Path) -> None:
    """Re-running it must be safe: the point is to fill the stubs in, not to lose that work."""
    label = scaffolded / "corpus" / "labels" / f"{PRODUCT_ID}-v1.0.md"
    label.write_text("# My real work\n", encoding="utf-8")

    scaffold(TENANT_ID, "Scaffold Test", PRODUCT_ID, "Testium", root=scaffolded)

    assert label.read_text(encoding="utf-8") == "# My real work\n"
