"""Scaffold sanity checks: the packages import, and the repo layout is what the docs claim."""

import contextlib
import os
import shutil
import subprocess
import tempfile
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


def test_the_demo_entry_points_use_a_backend_that_actually_runs():
    """Regression guard for a trap that cost real debugging time.

    ``make api`` and ``make demo`` set no backend, there is no ``.env`` in a fresh clone, and the code
    default was ``ollama`` -- so the demo quietly ran against a local model at roughly two minutes per
    call. A review makes about nine, so it sat at "running" for a quarter of an hour, which looks
    exactly like a stuck run. The UI was polling correctly the whole time.

    This asserted on the literal text ``MLR_LLM_BACKEND ?= fake`` until that line grew a lookup
    into ``.env``. The behaviour stayed correct and the assertion did not, so the test failed for
    six commits without anyone noticing -- it only shows up in a full run. It now asks make what
    the value actually is, in a directory with no ``.env``, which is the thing worth guaranteeing.
    """
    makefile = REPO_ROOT / "Makefile"
    assert "export MLR_LLM_BACKEND" in makefile.read_text(encoding="utf-8"), (
        "the setting has to reach the subprocesses"
    )

    # MLR_LLM_BACKEND has to come out of the environment, and that is the whole subtlety: the
    # Makefile assigns it with `?=`, which defers to an already-set variable, and conftest.py sets
    # it to "fake" for every test. Leave it in and make never consults the fallback, so this test
    # passes whatever the Makefile says -- which is exactly what the first version of it did.
    env = {k: v for k, v in os.environ.items() if k != "MLR_LLM_BACKEND"}

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(makefile, Path(tmp) / "Makefile")
        printed = subprocess.run(
            ["make", "-n", "demo"],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert "backend: fake" in printed.stdout, (
        "with no .env, the demo targets must fall back to a backend that runs without a key; "
        f"make printed: {printed.stdout.strip()[:200]!r}"
    )

    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "MLR_LLM_BACKEND=fake" in env_example, (
        "`make setup` copies .env.example, so it decides the default for anyone following the README"
    )


def test_a_slow_backend_warns_at_submission(monkeypatch):
    """A fifteen-minute review must announce itself rather than look stuck."""
    import warnings

    from mlr import service
    from platform_core.tenancy import TenantConfigError

    monkeypatch.setenv("MLR_LLM_BACKEND", "ollama")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # The tenant lookup fails first in this stripped-down environment; the warning is what matters.
        with contextlib.suppress(TenantConfigError, Exception):
            service.start_run(
                tenant_id="acme_pharma", product_id="zelvoran", document="copy", block=True
            )
    messages = " ".join(str(w.message) for w in caught)
    assert "two minutes per call" in messages or "ollama" in messages.lower()
