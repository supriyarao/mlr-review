"""The platform/use-case boundary, enforced mechanically.

``platform_core`` is meant to be reusable across use cases. That claim is only credible if it
is checked, so this test parses the AST of every module under ``platform_core/`` and fails if
any of them imports the MLR use case or a delivery surface.

Trade-off: an AST scan catches static imports but not a dynamic ``importlib.import_module("mlr")``.
A runtime import hook would close that gap at the cost of complicating the test setup; a static
check is proportionate here because the rule exists to shape how humans write code, not to
defend against an adversary.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# platform_core is the reusable layer, so it may not depend on anything above it.
FORBIDDEN_FOR_PLATFORM = {"mlr", "api", "ui"}

# The UI is a thin HTTP client by design: it must not reach into the graph or the database.
FORBIDDEN_FOR_UI = {"mlr", "platform_core", "langgraph"}


def _python_files(package: str) -> list[Path]:
    return sorted(p for p in (REPO_ROOT / package).rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Return the top-level package name of every import in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        # node.module is None for a relative import ("from . import x"), and level > 0
        # means relative; either way it is in-package and therefore fine.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_platform_core_has_files_to_check():
    """Guard against the boundary test silently passing because it found nothing."""
    assert _python_files("platform_core"), "no modules found under platform_core/"


def test_platform_core_does_not_import_the_use_case():
    violations = []
    for path in _python_files("platform_core"):
        for root in sorted(_imported_roots(path) & FORBIDDEN_FOR_PLATFORM):
            violations.append(f"{path.relative_to(REPO_ROOT)} imports '{root}'")

    assert not violations, (
        "platform_core must stay reusable and may not depend on the MLR use case "
        "or a delivery surface:\n  " + "\n  ".join(violations)
    )


def test_ui_talks_to_the_api_over_http_only():
    """The UI is a separate process in production, so it may not import the graph."""
    violations = []
    for path in _python_files("ui"):
        for root in sorted(_imported_roots(path) & FORBIDDEN_FOR_UI):
            violations.append(f"{path.relative_to(REPO_ROOT)} imports '{root}'")

    assert not violations, (
        "ui/ must reach the system over HTTP, never by importing it:\n  " + "\n  ".join(violations)
    )
