"""Scaffold sanity checks: the packages import, and the repo layout is what the docs claim."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_packages_import():
    import api
    import mlr
    import platform_core
    import platform_core.evals

    for module in (platform_core, platform_core.evals, mlr, api):
        assert module.__doc__, f"{module.__name__} should document its purpose"


def test_expected_layout_exists():
    for relative in [
        "platform_core",
        "platform_core/evals",
        "mlr/nodes/checks",
        "mlr/prompts",
        "api",
        "ui/pages",
        "corpus/tenants",
        "corpus/labels",
        "corpus/references",
        "corpus/preapproved_claims",
        "corpus/promo",
        "evals/golden",
        "evals/baselines",
        "evals/cassettes",
        "scripts",
    ]:
        assert (REPO_ROOT / relative).is_dir(), f"missing directory: {relative}"


def test_private_build_spec_is_never_committed():
    """PROMPT.md is a private spec. If it exists it must be ignored by git."""
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "PROMPT.md" in gitignore


def test_env_example_has_no_filled_in_secrets():
    """.env.example is committed, so every key-like variable must be left empty."""
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if any(token in name.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            assert value == "", f"{name} must be empty in .env.example, found a value"
