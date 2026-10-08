"""The synthetic promotional pieces that ship with the repo, for the submission form.

Deterministic code. Served over HTTP so the UI does not read the corpus from disk: the UI is a
separate process and, in a real deployment, a separate machine.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.ingest import split_front_matter

router = APIRouter(tags=["samples"])

PROMO_DIR = Path(__file__).resolve().parent.parent.parent / "corpus" / "promo"
IMAGE_DIR = PROMO_DIR / "images"

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")
# Image filenames are the one part of a sample's front matter that becomes a path, so they are
# pattern-checked on the way out as well as resolved inside IMAGE_DIR on the way back in.
_SAFE_IMAGE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,80}\.(png|jpg|jpeg)$")


def _load(path: Path) -> dict[str, Any]:
    meta, text = split_front_matter(path.read_text(encoding="utf-8"))

    # `media` says what the piece *is*, not how this version of the pipeline reads it. All five
    # Declasal ads are images; today every one of them is reviewed from a committed transcript, so
    # the UI can show the artwork without waiting on image ingest to land.
    image = str(meta.get("image", ""))
    image_name = Path(image).name if image else ""
    image_url = (
        f"/samples/{path.stem}/image" if image_name and _SAFE_IMAGE.match(image_name) else ""
    )

    return {
        "name": path.stem,
        "title": str(meta.get("title", path.stem)),
        "product_id": str(meta.get("product_id", "")),
        "label_version": str(meta.get("label_version", "")),
        "audience": str(meta.get("audience", "any")),
        "channel": str(meta.get("channel", "")),
        "market": str(meta.get("market", "")),
        "category": str(meta.get("category", "")),
        "media": str(meta.get("media", "text")),
        "image_url": image_url,
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


@router.get("/samples/{name}/image")
def get_sample_image(name: str) -> FileResponse:
    """The artwork for an image sample.

    Served rather than inlined because these are ~100-400KB PNGs: base64 in a JSON list would make
    the sample index several megabytes for a page that only needs thumbnails.
    """
    sample = load_sample(name)
    if not sample["image_url"]:
        raise HTTPException(404, f"sample {name!r} has no image")

    meta, _ = split_front_matter((PROMO_DIR / f"{name}.md").read_text(encoding="utf-8"))
    path = (IMAGE_DIR / Path(str(meta.get("image", ""))).name).resolve()
    # Belt and braces: the name was pattern-checked in _load, and the resolved path must still sit
    # inside the image directory. Front matter is corpus content, but it is content a tenant edits.
    if not str(path).startswith(str(IMAGE_DIR.resolve())) or not path.is_file():
        raise HTTPException(404, f"no image for sample {name!r}")
    return FileResponse(path, media_type="image/png" if path.suffix == ".png" else "image/jpeg")
