"""The synthetic promotional pieces that ship with the repo, for the submission form.

Deterministic code. Served over HTTP so the UI does not read the corpus from disk: the UI is a
separate process and, in a real deployment, a separate machine.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from api.ingest import split_front_matter

router = APIRouter(tags=["samples"])

PROMO_DIR = Path(__file__).resolve().parent.parent.parent / "corpus" / "promo"

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


def _load(path: Path) -> dict[str, Any]:
    meta, text = split_front_matter(path.read_text(encoding="utf-8"))
    return {
        "name": path.stem,
        "title": str(meta.get("title", path.stem)),
        "product_id": str(meta.get("product_id", "")),
        "label_version": str(meta.get("label_version", "")),
        "audience": str(meta.get("audience", "any")),
        "channel": str(meta.get("channel", "")),
        "notes": str(meta.get("notes", "")),
        "text": text,
    }


def load_sample(name: str) -> dict[str, Any]:
    """One sample by name, or a 404. The name is pattern-checked before it becomes a path."""
    path = PROMO_DIR / f"{name}.md"
    if not _SAFE_NAME.match(name) or not path.is_file():
        raise HTTPException(404, f"no sample {name!r}")
    return _load(path)


@router.get("/samples")
def list_samples() -> dict[str, Any]:
    """Every sample piece, without its text."""
    samples = [_load(p) for p in sorted(PROMO_DIR.glob("*.md"))]
    return {"samples": [{k: v for k, v in s.items() if k != "text"} for s in samples]}


@router.get("/samples/{name}")
def get_sample(name: str) -> dict[str, Any]:
    """One sample piece with its text, front matter removed."""
    return load_sample(name)
