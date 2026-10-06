"""Turning an uploaded file into reviewable text. One implementation, two entry points.

Deterministic code. No model involved.

Why this is in the platform rather than in ``api/`` or ``mlr/``: bytes-to-text is genuinely use-case
agnostic, and there were previously two copies of it -- one reading uploaded bytes in the API, one
reading a path in the graph, used by the eval harness. Nothing kept them in step, so a PDF change in
one would have left the harness and the API reviewing subtly different text from the same file. The
eval baseline is only meaningful if the harness sees what the API sees.

**Headings in a PDF.** A PDF carries no ``#`` markers, and everything downstream -- the section map,
the fair-balance numbers, the placement rules -- is built by looking for them. Plain text extraction
therefore produced a document with no structure at all, and the checks that read that structure
either judged on zeros or reported a confident "required section is missing" about a piece whose
safety information was present and correctly placed.

So extraction recovers the headings instead: PyMuPDF exposes the size and weight of every span, a
heading is a short line set larger than the body text, and the distinct heading sizes rank into
levels. Those are re-emitted as ``#``/``##``/``###`` so the rest of the pipeline is unchanged -- a
PDF becomes a markdown document and every existing rule works on it as written.

**What this still cannot see.** Font size is structure, not layout. Two columns, a safety block in
4pt grey beside a full-page image, a heading set in colour rather than in size -- none of that is
visible here, and a piece whose prominence problem is purely visual will still read as clean. That
needs layout-aware or multimodal parsing. The honest half of the fix is that when no structure is
found at all, the checks now say so rather than inventing a verdict: see
``mlr/nodes/checks/runner.py:_layout_block`` and ``rules/deterministic.py:check_required_section``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Large enough for any promotional piece, small enough that a mistaken upload fails fast.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
TEXT_SUFFIXES = {".md", ".txt"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}

# How much larger than the body text a line must be set to count as a heading.
#
# Small on purpose. Measured against a real 31-page industry code of practice, the body ran at 8pt
# and the three heading levels at 9, 14 and 18 -- so a generous threshold would have thrown away the
# whole third level. A tenth of a point of noise is worth tolerating to keep 9pt-over-8pt.
HEADING_SIZE_DELTA = 0.2

# A heading is a label, not a paragraph. Guards against a large-type pull quote or a cover strapline
# being promoted to a section of its own.
MAX_HEADING_CHARS = 120

# Markdown has six heading levels; a document with more distinct heading sizes than that flattens
# into the last one rather than emitting "#######", which no parser reads as a heading.
#
# The level itself is cosmetic here. ``mlr/nodes/ingest.py:build_doc_meta`` keys its section map on
# the heading *text* and never reads the level, so every heading is an equal section boundary. An
# earlier version of this module dropped sizes that carried too little of the document, to stop
# cover art claiming the top levels -- it bought nothing but tidier markdown, and it cost real
# headings: "Efficacy" is eight characters, and in a one-page leaflet that is under any threshold
# worth setting. Being a heading is what matters; which level it lands on is not.
MAX_HEADING_LEVEL = 6

# A line that is only digits, dots and spaces. Reference lists and clause numbering in a left margin
# extract as their own lines, in their own size, and are numbering rather than headings.
_NUMBER_ONLY = re.compile(r"^[\d.\s]+$")


class DocumentError(Exception):
    """A file that cannot be turned into reviewable text. Carries the HTTP status to return."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def _lines_with_style(pdf: Any) -> list[tuple[int, int, float, bool, str]]:
    """Every non-empty line of a PDF as ``(page, block, size, bold, text)``.

    One line is one visual line, with its spans joined: a heading whose first word is in a different
    span -- a drop cap, a coloured first letter -- is still one line, and taking the largest span's
    size keeps it at heading size.

    The block index is carried because it is the only reliable signal for a *wrapped* heading. Lines
    in one block are one paragraph that happened to wrap; lines in different blocks are different
    things set the same way. A table of contents is a column of same-size lines, each its own block,
    and joining those produces one absurd heading that swallows the page.
    """
    out: list[tuple[int, int, float, bool, str]] = []
    for page_no, page in enumerate(pdf, start=1):
        for block_no, block in enumerate(page.get_text("dict").get("blocks", [])):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                size = round(max(float(s.get("size", 0)) for s in spans), 1)
                bold = any(
                    "bold" in str(s.get("font", "")).lower()
                    or "semibold" in str(s.get("font", "")).lower()
                    for s in spans
                )
                out.append((page_no, block_no, size, bold, text))
    return out


def _size_volumes(lines: list[tuple[int, int, float, bool, str]]) -> dict[float, int]:
    """How many characters are set at each size."""
    volume: dict[float, int] = {}
    for _page, _block, size, _bold, text in lines:
        volume[size] = volume.get(size, 0) + len(text)
    return volume


def _body_size(lines: list[tuple[int, int, float, bool, str]]) -> float:
    """The size the body text is set in: the size carrying the most characters.

    By volume of text, not by number of lines. A document whose body is one long paragraph and whose
    headings are many short lines would otherwise elect a heading size as the body.
    """
    volume = _size_volumes(lines)
    if not volume:
        return 0.0
    return max(volume.items(), key=lambda kv: kv[1])[0]


def pdf_to_markdown(data: bytes) -> str:
    """PDF bytes to markdown, recovering headings from font size.

    Returns plain text with no headings when the document has no size variation to read -- a scanned
    page, or a piece set entirely in one size. That is the honest outcome: no structure was found,
    and the checks downstream are written to say so rather than to treat it as an empty document.
    """
    import pymupdf

    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            lines = _lines_with_style(pdf)
    except Exception as exc:
        raise DocumentError(f"could not read the file as a PDF: {exc}") from exc

    if not lines:
        return ""

    body = _body_size(lines)

    # Every size set above the body text is a heading size; largest is level 1. Anything past the
    # sixth distinct size shares the last level, since markdown has no seventh.
    sizes = sorted(
        {size for _p, _b, size, _bold, _t in lines if size > body + HEADING_SIZE_DELTA},
        reverse=True,
    )
    level_of = {size: min(i + 1, MAX_HEADING_LEVEL) for i, size in enumerate(sizes)}

    out: list[str] = []
    page_now = lines[0][0]
    i = 0
    while i < len(lines):
        page, block, size, _bold, text = lines[i]
        if page != page_now:
            out.append("")
            page_now = page

        level = level_of.get(size)
        is_heading = (
            level is not None and len(text) <= MAX_HEADING_CHARS and not _NUMBER_ONLY.match(text)
        )
        if not is_heading:
            out.append(text)
            i += 1
            continue

        # A heading too long for its column wraps onto the next line at the same size, inside the
        # same block. Joining those matters: left apart, "Standards of Promotional" and "Information"
        # become two sections, and the second one owns all the text belonging to the first. The
        # block boundary is what stops this swallowing a contents page, where each entry is set the
        # same way but is a block of its own.
        parts = [text]
        j = i + 1
        while j < len(lines) and lines[j][:3] == (page, block, size):
            following = lines[j][4]
            if _NUMBER_ONLY.match(following) or len(following) > MAX_HEADING_CHARS:
                break
            parts.append(following)
            j += 1

        out.append("")
        out.append(f"{'#' * level} {' '.join(parts)}")
        out.append("")
        i = j

    # Collapse the runs of blank lines the heading padding leaves behind.
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_bytes(filename: str, data: bytes) -> str:
    """Reviewable text from an uploaded file's bytes. The API's entry point."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise DocumentError(
            f"unsupported file type {suffix or '(none)'!r}; upload .pdf, .md or .txt", 415
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentError(f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB", 413)

    if suffix == ".pdf":
        try:
            return pdf_to_markdown(data)
        except DocumentError as exc:
            raise DocumentError(f"could not read {filename!r} as a PDF: {exc}") from exc
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentError(f"{filename!r} is not UTF-8 text") from exc


def extract_path(path: Path) -> str:
    """Reviewable text from a file on disk. The graph's and the eval harness's entry point.

    Goes through :func:`extract_bytes` so the two entry points cannot drift: the harness and the API
    read the same corpus file into the same string, which is what makes the eval baseline describe
    the API's behaviour rather than its own.
    """
    path = Path(path)
    return extract_bytes(path.name, path.read_bytes())
