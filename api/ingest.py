"""Turning an upload into the plain text the review runs on, and locating text inside it.

Deterministic code. No model involved.

Why this lives in the API rather than the graph: the service façade takes ``document: str``, so
something in front of it has to turn a PDF or a markdown file into that string. Keeping it here
means the graph never sees bytes, filenames or content types.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# Large enough for any promotional piece, small enough that a mistaken upload fails fast.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
TEXT_SUFFIXES = {".md", ".txt"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)


class IngestError(Exception):
    """An upload that cannot be turned into reviewable text. Carries the HTTP status to return."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


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
    """Extract reviewable text from an uploaded file.

    Trade-off: PDFs are read as plain text with PyMuPDF, which loses layout -- and layout is
    exactly what a prominence or placement finding is about. A production system would parse
    position and font size; here that is called out as a limitation rather than half-built.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise IngestError(
            f"unsupported file type {suffix or '(none)'!r}; upload .pdf, .md or .txt", 415
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError(f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB", 413)

    if suffix == ".pdf":
        import pymupdf

        try:
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                text = "\n\n".join(page.get_text().strip() for page in pdf)
        except Exception as exc:
            raise IngestError(f"could not read {filename!r} as a PDF: {exc}") from exc
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise IngestError(f"{filename!r} is not UTF-8 text") from exc

    _, body = split_front_matter(text)
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
