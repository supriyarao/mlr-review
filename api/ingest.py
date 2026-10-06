"""Front matter, and locating text inside a document.

Deterministic code. No model involved.

Extraction itself moved to :mod:`platform_core.documents`. It used to live here, with a second copy
in ``mlr/nodes/ingest.py`` for the graph and the eval harness, and nothing kept the two in step --
so the harness and the API were one PDF change away from reviewing different text from the same
file. ``IngestError`` and ``extract_text`` stay as the names this package's routes already import.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from platform_core.documents import (
    ALLOWED_SUFFIXES,
    MAX_UPLOAD_BYTES,
    TEXT_SUFFIXES,
    DocumentError,
    extract_bytes,
)

__all__ = [
    "ALLOWED_SUFFIXES",
    "MAX_UPLOAD_BYTES",
    "TEXT_SUFFIXES",
    "IngestError",
    "extract_text",
    "locate",
    "section_at",
    "split_front_matter",
]

# The routes catch this by name and read ``status_code`` off it.
IngestError = DocumentError

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Separate a YAML front-matter block from the body.

    The corpus samples carry their audience and channel in front matter. Those are returned to the
    caller as *suggestions* for the submission form; they are never applied silently, because
    audience and channel decide which tenant rules are in scope and must be stated by a person.
    """
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return meta, text[match.end() :].lstrip("\n")


def extract_text(filename: str, data: bytes) -> str:
    """Reviewable text from an uploaded file, with any front matter stripped.

    The extraction is :func:`platform_core.documents.extract_bytes`; what is added here is the front
    matter split, which is an API concern -- the corpus samples carry their audience and channel at
    the top of the file, and those are suggestions for the form, never applied silently.
    """
    _, body = split_front_matter(extract_bytes(filename, data))
    return body


def section_at(document: str, offset: int) -> str:
    """The heading a character offset sits under, or ``""`` before the first heading."""
    current = ""
    for match in _HEADING.finditer(document):
        if match.start() > offset:
            break
        current = match.group(1).strip()
    return current


def locate(document: str, quote: str, char_span: tuple[int, int] | list[int]) -> str:
    """Where in the document a quote sits, for the export's location column.

    An absence finding (no quote) has no location, and that is reported as empty rather than
    guessed. The span is trusted only if the text at that span really is the quote; otherwise the
    quote is searched for, so a stale offset can never put a finding under the wrong heading.
    """
    if not document or not quote.strip():
        return ""
    start, end = int(char_span[0]), int(char_span[1])
    if document[start:end] != quote:
        start = document.find(quote)
        if start < 0:
            return ""
    return section_at(document, start)
