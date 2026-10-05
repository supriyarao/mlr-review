"""Ingest: turn a submitted file into the text the rest of the pipeline reasons about.

Why it exists, and why it is more than "read the file": every downstream guarantee depends on this
node's output being the *exact* string the checks saw. Findings quote from it, and the grounding gate
verifies those quotes against it. If ingest and the checks disagree about what the document is -- by
one stray markdown asterisk -- then correct findings get rejected as ungrounded.

**Deterministic code. No model involved.**

Three jobs, each learned the hard way:

1. **Strip front matter and comments.** Submission metadata is not promotional copy. When it was left
   inline, the trademark rule matched the lower-case product id in an HTML comment and reported a
   violation on correctly marked pieces.
2. **Normalise markdown emphasis.** A model quoting "did not assess PASI-90" cannot match a source
   storing "did **not** assess PASI-90", so emphasis markers inside sentences are removed before
   anything quotes from the text.
3. **Map the layout.** Placement rules and the fair-balance check need to know where the safety
   information sits relative to the first efficacy claim, and how much space each gets. That is
   structure, not language, so it is computed here rather than asked of a model.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from mlr.nodes.base import node_run, resolve_policy_version, tenant_backend_provenance

# Markdown constructs stripped from the reviewable text.
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_UNDERSCORE_BOLD = re.compile(r"(?<!\w)__(.+?)__(?!\w)", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# Section names that count as safety information or as efficacy content, however a tenant titles
# them. Matched on the normalised heading slug.
_SAFETY_HINTS = ("safety", "isi", "important_safety_information", "warnings", "risk")
_EFFICACY_HINTS = ("efficacy", "results", "what_", "clinical")


def slugify(heading: str) -> str:
    """Normalise a heading into a stable section key."""
    return re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML front matter from the body.

    Returns ``({}, text)`` when there is none, so a plain .txt or .md file works unchanged.
    """
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        # Malformed front matter is not worth failing a review over; treat it as absent and let the
        # caller's explicit arguments stand.
        return {}, text[match.end() :]
    return (meta if isinstance(meta, dict) else {}), text[match.end() :]


def normalize_document(text: str) -> str:
    """Produce the exact string the checks will quote from.

    Emphasis markers are removed rather than escaped, and comments are dropped entirely. Whitespace
    is left alone: character offsets have to stay meaningful, and collapsing runs would move every
    span after the first blank line.
    """
    body = _HTML_COMMENT.sub("", text)
    body = _BOLD.sub(r"\1", body)
    body = _UNDERSCORE_BOLD.sub(r"\1", body)
    return body.lstrip("\n")


def build_doc_meta(document: str) -> dict[str, Any]:
    """Section offsets and the layout hints the placement checks need.

    ``sections`` maps a slug to ``{"start", "end"}``. Two synthetic keys are added -- ``efficacy``
    and ``important_safety_information`` -- pointing at whichever real section matches, so a rule can
    be written against a stable name rather than against whatever a tenant titled its heading.
    """
    sections: dict[str, dict[str, int]] = {}
    headings = list(_HEADING.finditer(document))

    for i, match in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(document)
        sections[slugify(match.group(2))] = {"start": match.start(), "end": end}

    def first_matching(hints: tuple[str, ...]) -> str | None:
        for name in sections:
            if any(hint in name for hint in hints):
                return name
        return None

    safety_key = first_matching(_SAFETY_HINTS)
    efficacy_key = first_matching(_EFFICACY_HINTS)

    if safety_key and "important_safety_information" not in sections:
        sections["important_safety_information"] = sections[safety_key]
    if efficacy_key and "efficacy" not in sections:
        sections["efficacy"] = sections[efficacy_key]

    safety = sections.get("important_safety_information")
    efficacy = sections.get("efficacy")

    meta: dict[str, Any] = {
        "sections": sections,
        "section_order": [slugify(m.group(2)) for m in headings],
        "document_chars": len(document),
        "safety_section": safety_key or "",
        "efficacy_section": efficacy_key or "",
        "safety_block_chars": (safety["end"] - safety["start"]) if safety else 0,
        "efficacy_block_chars": (efficacy["end"] - efficacy["start"]) if efficacy else 0,
    }

    # The two numbers the fair-balance and placement checks actually turn on. Computed here so both
    # the deterministic rules and the LLM check reason about the same figures.
    if safety and efficacy:
        meta["safety_distance_chars"] = abs(safety["start"] - efficacy["start"])
        meta["safety_after_efficacy"] = safety["start"] > efficacy["start"]
    if meta["efficacy_block_chars"]:
        meta["safety_to_efficacy_ratio"] = round(
            meta["safety_block_chars"] / meta["efficacy_block_chars"], 3
        )
    # Is the safety block pushed below the footnotes or references? The "present but buried" case.
    for marker in ("footnotes", "references"):
        if marker in sections and safety:
            meta[f"safety_after_{marker}"] = safety["start"] > sections[marker]["start"]

    return meta


def extract_text(path: Path) -> str:
    """Read a submission. Text for .md/.txt, PyMuPDF text extraction for .pdf.

    Trade-off: PDF handling is text-only. Real promotional material is laid out visually -- a safety
    block in 4pt type beside a full-page image is a fair-balance problem that no text extraction can
    see -- so a production system needs layout-aware or multimodal parsing. That is noted as a
    limitation rather than pretended away, and the layout hints here come from markdown structure,
    which is honest about what it can know.
    """
    if path.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            return "\n\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8")


def ingest(state: dict[str, Any]) -> dict[str, Any]:
    """Normalise the document and map its layout.

    Also resolves the policy version for the run, which every later audit event carries: pinning it
    once at the start means a rule edited mid-review cannot make the trail self-contradictory.
    """
    with node_run(state, "ingest", input_value={"chars": len(state.get("document", ""))}) as run:
        raw = state.get("document") or ""
        front_matter, body = parse_front_matter(raw)
        document = normalize_document(body)
        doc_meta = build_doc_meta(document)

        # Explicit arguments win over front matter: the API caller chose these deliberately, and a
        # file's own metadata is the submitter's hint, not the reviewer's instruction.
        resolved = {
            key: state.get(key) or str(front_matter.get(key, "") or "")
            for key in ("product_id", "label_version", "audience", "channel", "market")
        }
        resolved["audience"] = resolved["audience"] or "any"

        tenant_id = str(state.get("tenant_id", ""))
        provenance = tenant_backend_provenance(tenant_id)

        # Caller-supplied doc_meta wins, so an eval case can pin the layout and a parsing change is
        # not mistaken for a model regression.
        if state.get("doc_meta"):
            doc_meta = {**doc_meta, **state["doc_meta"]}

        run.output = {
            "document_chars": len(document),
            "sections": list(doc_meta["sections"]),
            "safety_block_chars": doc_meta["safety_block_chars"],
            "efficacy_block_chars": doc_meta["efficacy_block_chars"],
        }

        return {
            "document": document,
            "doc_meta": doc_meta,
            "status": "running",
            "policy_version": provenance["policy_version"] or resolve_policy_version(tenant_id),
            **{k: v for k, v in resolved.items() if v},
        }
