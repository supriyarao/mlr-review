# Known issues

Things found by reading the code that are real but not yet fixed. Recorded rather than quietly
carried, because an undocumented known issue is indistinguishable from a bug nobody noticed.

Each entry says what is wrong, what it costs, and what the fix is. Ordered by how much it would
mislead someone relying on the system.

---

## 1. The pre-approved claim library is ungoverned, and `policy_version` does not cover it

**Where:** `corpus/preapproved_claims/<tenant_id>.yaml`, `scripts/seed.py:_preapproved_documents`,
`platform_core/policy.py:policy_version`.

**What.** Three related gaps around the same asset.

*It is not part of the policy version.* `policy_version()` is `hash(tenant_id : config_hash :
rules_hash)`. The pre-approved claim library is not in it. But a matched pre-approved claim **skips
substantiation** — so adding a claim to that YAML and re-seeding materially changes review outcomes
while `policy_version` stays byte-identical. Two runs under different cleared-claim libraries assert
identical provenance in the audit log. That is precisely the failure `platform_core/policy.py`
exists to prevent, applied to an input its author did not account for.

*It has no registry.* Business rules got versioning, status transitions, validation and an audit
trail. Pre-approved claims get a YAML file read once by the seeder. `seed.py` indexes each claim as
a retrieval `Document` and **drops `approved_on`, `source` and `notes` at index time**, so after
seeding the claim exists only as a chunk in the tenant's collection. `preapproved_matches` records
`claim_id -> ACME-PAC-001`, and nothing can resolve that id back to its approval date or source
without re-reading the YAML by hand. "Who cleared this claim, when, against which label version" is
an auditor's first question about a skipped substantiation check, and the system cannot answer it.

*It has no API or UI.* No endpoint lists, creates, edits, versions or uploads pre-approved claims,
and no page shows them. The only way to change the library is to edit a file in the repo and re-run
`make seed`. A cleared-claim library is exactly the kind of asset a compliance owner edits
constantly — the same argument that justified the rules registry.

*And its location is inconsistent.* Rules and reference docs live inside `corpus/tenants/<id>/`;
pre-approved claims sit in a separate top-level directory keyed by tenant filename. Leftover from
before the tenant layout became a directory per tenant.

**Fix.** Treat the library the way rules are treated: move it to
`corpus/tenants/<id>/preapproved_claims.yaml`, give it a `PreapprovedClaimRegistry` with an
`active_set_hash()`, fold that hash into `policy_version()`, carry `approved_on` / `source` /
`label_version` through indexing so a match can be explained, and expose read plus upload endpoints.
Versioned editing can wait; the policy hash and the provenance fields cannot, because without them
the audit trail overstates what it knows.

---

## 2. A PDF has no markdown headings, so `required_section` rules fire on every PDF

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

---

## 3. Two document-extraction code paths that nothing keeps in step

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

## 4. The API stores the document before normalisation, so evidence panels can vanish

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

## 5. The citation check never fetches the reference it is checking

**Where:** `mlr/nodes/checks/runner.py:_evidence_query` and `mlr/prompts/checks.py:SUBSTANTIATION`
(failure family B).

**What.** `substantiation` now answers two questions: does the evidence support this claim, and
does the study the claim cites report it. The second question names a specific source — "Reference
3" — and **nothing in the pipeline resolves that marker to a source.**

Two things narrow the gap without closing it. `_study_ids` reads the study identifiers out of the
piece's numbered reference list and expands every substantiation query with them, so retrieval has a
lexical handle on the documents the copy relies on. And `_retrieve` fills the window in two passes,
studies first, so a study is not outranked by a label chunk that merely reads more like a claim.

But retrieval is still a search. Whether ZEL-303 lands in the twelve passages is now *likely* rather
than accidental — it is not guaranteed, because nothing maps marker `3` to source `zel-303`. The
model still has to make that connection itself by reading the piece's References section, which is
visible inside `<piece>` — an inference the prompt asks for but nothing enforces.

**Why it matters.** It fails *silently*. If the cited study still falls outside the retrieval window, the
check does not report "I could not find Reference 3"; it reports nothing, or — worse — judges the
claim against whichever study did surface. Golden case 003 passes today, and its own `why_hard`
names the dependency: *"A retrieval system that scores relevance will rank ZEL-303 highly for this
claim."* True for a six-study corpus. At forty studies it stops being reliable, and the failure
looks like a model regression rather than a retrieval gap.

This predates the merge of `reference_support` into `substantiation` — the old check had the same
hole. It is recorded now precisely because merging makes it *look* addressed when it is not.

**Fix.** Resolve markers at ingest, not at retrieval time. Parse the piece's numbered reference
list in `build_doc_meta`, map each marker to a source id in the tenant collection, and carry the
mapping on the claim. The check then **fetches those sources directly** and lets retrieval fill only
the remaining slots. A marker that resolves to nothing becomes an explicit "cited reference not
found in this tenant's evidence" outcome that escalates — which is a finding a reviewer wants
anyway, and which the system currently cannot produce.
