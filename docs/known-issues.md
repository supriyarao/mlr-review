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

## 2. ~~A PDF has no markdown headings, so `required_section` rules fire on every PDF~~ — RESOLVED

**Fixed 6 Oct 2026**, in `platform_core/documents.py`. Kept here rather than deleted, because the
limitation that remains is narrower than the one that was fixed and the difference matters.

Extraction now recovers headings from font size and weight: PyMuPDF exposes the size of every span,
a heading is a short line set above the body size, and those are re-emitted as `#`/`##`/`###` so the
section map, the fair-balance numbers and the placement rules all work on a PDF unchanged.
Round-tripped against three corpus pieces, a PDF and the same piece as markdown now produce the same
section map and a safety/efficacy ratio within 0.02 — see `tests/test_documents.py`.

The second half of the fix matters more than the first. When no structure can be read at all,
`check_required_section` now reports *"could not be evaluated"* at confidence 20 — which routes
through the tenant's confidence gate and escalates — instead of *"Required section is missing"* at
full confidence. `_layout_block` likewise tells the model the measurements are UNKNOWN rather than
handing it zeros, because a zero is a measurement and "0 characters of safety information" is a
claim nobody made.

**What is still true.** Font size is structure, not layout. Two columns, a safety block in 4pt grey
beside a full-page image, a heading set in colour rather than in size — none of that is visible, and
a piece whose prominence problem is purely visual will still read as clean. That needs layout-aware
or multimodal parsing. The difference from before is that the system no longer manufactures a
confident blocker out of its own blindness.

---

## 3. ~~Two document-extraction code paths that nothing keeps in step~~ — RESOLVED

**Fixed 6 Oct 2026.** Both paths now call `platform_core.documents`: `extract_bytes` for the API's
uploads, `extract_path` for the graph and the eval harness, with the second implemented in terms of
the first so they cannot drift. `api/ingest.py` keeps the `IngestError` and `extract_text` names its
routes already import, and keeps the front-matter split, which is genuinely an API concern.

`tests/test_documents.py::test_both_entry_points_produce_identical_text` asserts the two produce
byte-identical text for the same corpus file, which is what makes the eval baseline describe the
API's behaviour rather than its own.

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
