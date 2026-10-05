"""The submitted text of each run, kept so a finding can be shown in place.

Deterministic code. No model involved.

Why the API keeps its own copy: a reviewer has to see the offending quote highlighted in the piece,
and the export needs the heading each finding sits under. The service façade returns a
``ReviewReport``, which deliberately carries findings and not the document, so the surface that
received the upload is the one that can show it back.

Trade-off: this duplicates the ``document`` the graph holds in its checkpoint. A façade accessor
would remove the copy; until one exists, a second copy written once at submission and never
modified is the smaller risk than reaching past the façade into graph state.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from platform_core.runtime import data_dir

# Run ids arrive in URLs and are used to build a filename, so they are restricted to characters
# that cannot leave the directory.
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_safe_run_id(run_id: str) -> bool:
    return bool(_SAFE_RUN_ID.match(run_id or ""))


def _path(run_id: str) -> Path:
    return data_dir() / "submissions" / f"{run_id}.json"


def save(run_id: str, *, document: str, filename: str, **meta: Any) -> None:
    """Record what was submitted for a run."""
    if not is_safe_run_id(run_id):
        raise ValueError(f"unsafe run id: {run_id!r}")
    path = _path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "filename": filename,
        "document": document,
        "submitted_at": datetime.now(UTC).isoformat(),
        **meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load(run_id: str) -> dict[str, Any] | None:
    """The stored submission, or None if this API instance never received one for the run."""
    if not is_safe_run_id(run_id):
        return None
    path = _path(run_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
