"""Extraction: one module, two entry points, and a PDF that keeps its structure.

The thing under test is not "does PyMuPDF work" -- it is the claim the rest of the pipeline rests
on: **a PDF and the same piece as markdown must produce the same section map**. Everything the
fair-balance and placement checks do is computed from that map, so if the two disagree, the review
a reviewer gets depends on which format the submitter happened to use.

PDFs are generated here rather than committed. A generated one states its own font sizes in the
test, so a failure says which size stopped being read as a heading; a binary fixture would only say
that something changed.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pymupdf
import pytest

from api import ingest as api_ingest
from mlr.nodes.ingest import build_doc_meta, extract_text, normalize_document, parse_front_matter
from platform_core.documents import DocumentError, extract_bytes, extract_path, pdf_to_markdown
from platform_core.rules.deterministic import check_required_section
from platform_core.rules.schema import Rule, RuleInput

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def make_pdf(blocks: list[tuple[str, float, bool]]) -> bytes:
    """Render ``(text, size, bold)`` lines to a PDF, one line per entry, paginating as needed."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60.0
    for text, size, bold in blocks:
        if y > 740:
            page = doc.new_page()
            y = 60.0
        page.insert_text((60, y), text, fontsize=size, fontname="hebo" if bold else "helv")
        y += size + 6
    return bytes(doc.tobytes())


def markdown_to_pdf(body: str) -> bytes:
    """Render a markdown promo piece the way a designer would: headings larger and bold."""
    blocks: list[tuple[str, float, bool]] = []
    for line in body.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading:
            blocks.append((heading.group(2), 17.0 - 2 * len(heading.group(1)), True))
        elif line.strip():
            # Wrap long lines the way a real layout would, at body size.
            blocks += [(line[i : i + 95], 9.0, False) for i in range(0, len(line), 95)]
    return make_pdf(blocks)


# --- Heading recovery ----------------------------------------------------------------------------


def test_a_pdf_gets_its_headings_back_from_font_size():
    """The bug this module exists for: a PDF carries no '#', so structure had to be recovered."""
    pdf = make_pdf(
        [
            ("Zelvoran Journal Advertisement", 18.0, True),
            ("Efficacy", 13.0, True),
            ("In a 16-week trial, 68.4% of adults achieved PASI-75.", 9.0, False),
            ("Important Safety Information", 13.0, True),
            ("Zelvoran is contraindicated in active, serious infection.", 9.0, False),
        ]
    )
    markdown = pdf_to_markdown(pdf)

    assert "# Zelvoran Journal Advertisement" in markdown
    assert "## Efficacy" in markdown
    assert "## Important Safety Information" in markdown
    # Largest size is level 1 and the next size down is level 2 -- not everything at one level.
    assert re.search(r"^# Zelvoran", markdown, re.M)
    assert re.search(r"^## Efficacy", markdown, re.M)


def test_a_heading_that_wraps_is_one_heading():
    """Left unjoined, the second half becomes a section that owns the first half's text."""
    pdf = make_pdf(
        [
            ("Standards of Promotional", 18.0, True),
            ("Information", 18.0, True),
            ("Promotion should be capable of substantiation.", 9.0, False),
        ]
    )
    markdown = pdf_to_markdown(pdf)
    assert "# Standards of Promotional Information" in markdown
    assert "# Information" not in markdown


def test_a_short_heading_in_a_long_document_is_still_found():
    """A one-word heading carries almost none of a document's text, and is still a heading.

    This is the case that killed an earlier attempt to filter heading sizes by how much of the
    document they carried: "Efficacy" is eight characters, and no threshold that excludes cover
    decoration also keeps it.
    """
    body = [
        (f"Body sentence number {i}, carrying the bulk of the document.", 9.0, False)
        for i in range(60)
    ]
    pdf = make_pdf([("Efficacy", 13.0, True), *body])
    assert re.search(r"^#+ Efficacy$", pdf_to_markdown(pdf), re.M)


def test_a_pdf_with_no_size_variation_yields_no_headings():
    """The honest outcome. Not an error, and not invented structure."""
    pdf = make_pdf([(f"Every line is set the same way, line {i}.", 10.0, False) for i in range(8)])
    markdown = pdf_to_markdown(pdf)
    assert markdown
    assert "#" not in markdown
    assert build_doc_meta(markdown)["sections"] == {}


# --- The claim the pipeline rests on --------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["zelvoran-hcp-isi-buried", "zelvoran-hcp-journal-ad", "zelvoran-hcp-subgroup-claim"],
)
def test_the_same_piece_as_pdf_and_as_markdown_agree_on_layout(name):
    """A reviewer's verdict must not depend on which format the submitter uploaded."""
    _, body = parse_front_matter((CORPUS / "promo" / f"{name}.md").read_text(encoding="utf-8"))

    direct = build_doc_meta(normalize_document(body))
    via_pdf = build_doc_meta(normalize_document(pdf_to_markdown(markdown_to_pdf(body))))

    assert list(via_pdf["sections"]) == list(direct["sections"])
    assert via_pdf["safety_section"] == direct["safety_section"]
    assert via_pdf["efficacy_section"] == direct["efficacy_section"]
    assert via_pdf["safety_after_efficacy"] == direct["safety_after_efficacy"]
    # Block sizes shift by a few characters because a PDF re-wraps lines; the ratio the
    # fair-balance check actually turns on must still land in the same place.
    assert via_pdf["safety_to_efficacy_ratio"] == pytest.approx(
        direct["safety_to_efficacy_ratio"], abs=0.02
    )


def test_both_entry_points_produce_identical_text():
    """Known issue #3: two extractors is one divergence away from a meaningless eval baseline."""
    path = CORPUS / "promo" / "zelvoran-hcp-journal-ad.md"
    assert extract_path(path) == extract_bytes(path.name, path.read_bytes())


def test_the_api_strips_front_matter_and_the_node_does_not():
    """The one deliberate difference between the two callers, pinned so it stays deliberate."""
    path = CORPUS / "promo" / "zelvoran-hcp-journal-ad.md"
    assert api_ingest.extract_text(path.name, path.read_bytes()) in extract_path(path)
    assert extract_text(path) == extract_path(path)


# --- Rejections ------------------------------------------------------------------------------------


def test_an_unsupported_file_type_is_refused_with_415():
    with pytest.raises(DocumentError) as exc:
        extract_bytes("deck.pptx", b"anything")
    assert exc.value.status_code == 415


def test_an_oversized_upload_is_refused_with_413():
    with pytest.raises(DocumentError) as exc:
        extract_bytes("big.md", b"x" * (5 * 1024 * 1024 + 1))
    assert exc.value.status_code == 413


def test_a_file_that_is_not_a_pdf_fails_as_one():
    with pytest.raises(DocumentError):
        extract_bytes("claims.pdf", b"this is not a pdf at all")


# --- "Could not evaluate" is not "missing" ---------------------------------------------------------


def _isi_rule() -> Rule:
    return Rule(
        id="T-ISI-001",
        tenant_id="acme_pharma",
        title="ISI must be present",
        description="Every piece must carry important safety information.",
        kind="deterministic",
        severity="blocker",
        check_type="safety_prominence",
        params={"type": "required_section", "section": "important_safety_information"},
        effective_from=date(2026, 1, 1),
        status="active",
    )


def test_no_section_structure_reports_that_it_could_not_check():
    """The false blocker this fix exists to kill.

    A PDF whose headings cannot be read has no section map. Saying "the ISI is missing" there is a
    confident claim about a piece that may carry perfect safety information, and a reviewer has no
    way to tell it apart from a real finding.
    """
    violations = check_required_section(_isi_rule(), RuleInput(text="Some copy.", doc_meta={}))

    assert len(violations) == 1
    message = violations[0].message.lower()
    assert "could not" in message
    assert "missing" not in message
    # Low confidence is what routes it through the tenant's confidence gate and escalates it,
    # rather than letting it block the piece on its own.
    assert violations[0].confidence <= 50


def test_a_genuinely_missing_section_still_reports_missing():
    """The fix must not make a real absence quieter, only an unknown one honest."""
    meta = {"sections": {"efficacy": {"start": 0, "end": 40}}}
    violations = check_required_section(_isi_rule(), RuleInput(text="Some copy.", doc_meta=meta))

    assert len(violations) == 1
    assert "missing" in violations[0].message.lower()
    assert violations[0].confidence == 100


def test_a_present_section_passes():
    meta = {
        "sections": {
            "efficacy": {"start": 0, "end": 40},
            "important_safety_information": {"start": 40, "end": 90},
        }
    }
    assert check_required_section(_isi_rule(), RuleInput(text="x" * 90, doc_meta=meta)) == []
