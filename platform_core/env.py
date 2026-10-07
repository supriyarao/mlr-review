"""Load ``.env`` once, at a process entry point.

``make setup`` copies ``.env.example`` to ``.env`` and ``.env.example`` documents six variables,
so the repo has always told a reader that file is how you configure the system. Nothing read it.
Every variable in it did nothing, silently -- a key set there was simply never used, and the only
symptom was the fake backend answering as though no key existed.

**Precedence is shell, then file, then code default.** ``override=False`` is the whole point: a
variable someone exported deliberately for one command must beat a file they set up last week.
Reversing that makes ``MLR_LLM_BACKEND=fake make demo`` quietly not work, which is the same class
of bug as the one this module fixes.

Called explicitly from each entry point rather than on import of some widely-imported module,
because reading a file is a side effect and a side effect at import time runs in tests, in tooling
and in whatever imports the package next. ``tests/conftest.py`` deliberately does not call it: the
suite must pass with no API key and no ``.env``, so it forces the fake backend itself.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(*, path: Path | None = None) -> bool:
    """Load ``.env`` into the environment without overriding what is already set.

    Returns whether a file was found, so an entry point can say so. Missing is normal and not an
    error: the fake backend needs no configuration at all.
    """
    env_path = path or REPO_ROOT / ".env"
    if not env_path.is_file():
        return False

    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        # Parsed by hand rather than failing: a missing optional parser must not stop the app
        # starting, and the format this file uses is KEY=value.
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return True

    load_dotenv(env_path, override=False)
    return True
