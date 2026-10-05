"""Shared test configuration.

The whole suite runs offline. Forcing the ``fake`` model backend here means a test can never
accidentally reach the network or depend on a local Ollama server, whatever the developer's
``.env`` happens to say.
"""

import os

import pytest

# Set before any application module is imported, so module-level config reads see it.
os.environ["MLR_LLM_BACKEND"] = "fake"
os.environ.pop("ANTHROPIC_API_KEY", None)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the application's data directory at a per-test temporary path.

    Keeps SQLite databases and vector stores out of the developer's real ``data/``.
    """
    monkeypatch.setenv("MLR_DATA_DIR", str(tmp_path))
    return tmp_path
