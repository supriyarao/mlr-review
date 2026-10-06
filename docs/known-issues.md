# Known issues

Things found by reading the code that are real but not yet fixed. Recorded rather than quietly
carried, because an undocumented known issue is indistinguishable from a bug nobody noticed.

Each entry says what is wrong, what it costs, and what the fix is. Ordered by how much it would
mislead someone evaluating the system.

---

## 1. Two document-extraction code paths that nothing keeps in step

**Where:** `api/ingest.py:extract_text(filename, data)` and
`mlr/nodes/ingest.py:extract_text(path)`.

**What.** Both turn `.md` / `.txt` / `.pdf` into reviewable text, and both implement it separately:
the API reads uploaded bytes, the node reads a path (used by `scripts/loop.py` and the eval
harness). They agree today. Nothing enforces that they keep agreeing — a PDF handling change in one
would leave the other behind, and the eval harness and the API would then be reviewing subtly
different text from the same file.

**Why it matters.** The eval baseline is only meaningful if the harness sees what the API sees. Two
extractors is one divergence away from a baseline that no longer describes production behaviour.

**Fix.** One extraction module both import. It cannot live in `api/` — `mlr/` must not depend on a
delivery surface — and bytes-to-text is genuinely use-case-agnostic, so it belongs in the platform:
`platform_core/documents.py`, exposing extraction from both bytes and paths, with `api/` and
`mlr/nodes/ingest.py` as its only callers. Add a test asserting both entry points produce identical
text for the same corpus file.

---

## 2. The API stores the document before normalisation, so evidence panels can vanish

**Where:** `api/routes/reviews.py` (the `submissions.save(...)` call) vs `mlr/service.py:_prepare`.

**What.** The submitted text is stored for later display *before* `normalize_document()` strips
markdown emphasis, and `api/ingest.py` never calls that function. Findings quote verbatim from the
normalised text. So a finding whose quote crossed a `**bold**` marker cannot be located in the
stored copy.

**Why it matters.** It fails safe — `ui/view.py:excerpt_html` returns `None` rather than
mis-highlighting — but the reviewer silently loses the quote-in-context panel for exactly the
findings most likely to involve emphasised copy. Character offsets are also shifted, so the stored
text and any `char_span` disagree.

**Fix.** Store the normalised text, or normalise on read. One line, in `api/`.

---

## 3. A PDF has no markdown headings, so `required_section` rules fire on every PDF

**Where:** `mlr/nodes/ingest.py:build_doc_meta` and
`platform_core/rules/deterministic.py:check_required_section`.

**What.** Section mapping is markdown-structural: `build_doc_meta` finds sections with
`^#{1,6}[ \t]+`. PyMuPDF text extraction produces no `#` lines, so for a PDF `doc_meta["sections"]`
comes out empty, the synthetic `efficacy` / `important_safety_information` keys never resolve, and
`safety_block_chars` / `efficacy_block_chars` are `0`.

`check_required_section` reads those sections and, finding nothing, returns
*"Required section 'important_safety_information' is missing."* — so a correctly formatted PDF
collects a false blocker. The fair-balance and safety-prominence checks are handed
`safety block size: 0 characters` for the same reason.

**Why it matters.** This is worse than the documented limitation it sits next to. That text-only
PDF extraction cannot see visual prominence is stated honestly in both `extract_text` docstrings.
Emitting a confident false blocker is a different thing, and a reviewer has no way to tell it apart
from a real one.

**Fix.** Two parts, and they are separable:

1. Make the layout-dependent checks and rules state their own precondition. When
   `doc_meta["sections"]` is empty, a section rule has no evidence either way: that should produce
   an explicit "could not be evaluated — no section structure in this document" outcome that
   escalates, not a blocker and not silence. The failure policy in `mlr/nodes/base.py` already says
   a check that cannot complete must escalate rather than look clean; this is the same principle
   applied to a check whose *input* is missing.
2. Derive headings from PDFs where it is honestly possible — PyMuPDF exposes font size and weight
   per span, so a heading-detection pass is feasible. Layout-aware or multimodal parsing is the real
   answer and is out of scope here; the point is that the gap should be visible, not silent.

Until (1) lands, uploading a PDF is not a safe demo path. Markdown is.
