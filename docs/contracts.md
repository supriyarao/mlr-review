# Contracts for parallel work

This document is the interface between sessions working in separate git worktrees. The graph that
produces this data does not exist yet (Phase 3); everything described here does, with stub
implementations returning schema-valid fixture data. **Build against these shapes and your work will
integrate without changes when the graph lands.**

Everything below is enforced by `tests/test_contracts.py`. If you change a shape, that file fails —
which is the point. A contract nothing checks is a contract that drifts, and drift is only discovered
at integration, which is the expensive moment to discover it.

---

## 1. What each session needs, and where it is

| If you are building | Read | Call | Ignore |
|---|---|---|---|
| The API (`api/main.py`) | §2, §3, §5 | `mlr.service` | the graph entirely |
| The UI (`ui/`) | §5 | the API over HTTP | `mlr.*` and `platform_core.*` — the boundary test forbids importing them |
| The eval harness (`scripts/run_evals.py`) | §2, §4 | `mlr.service`, `platform_core.evals.*` | the API |
| The graph (`mlr/graph.py`, `mlr/nodes/`) | §2, §3 | `platform_core.*` | the API and UI |

`platform_core` must not import from `mlr`, `api` or `ui`, and `ui` must not import `mlr`,
`platform_core` or `langgraph`. Both are enforced by `tests/test_import_boundary.py`, which has been
verified to fail on a real violation rather than passing vacuously.

---

## 2. Domain schemas — `mlr/schemas.py`

Pydantic v2, `extra="forbid"` throughout, so a renamed field fails at the boundary instead of
arriving as `None` three layers later.

### `Claim`

```
id                  str          "c1"
original_text       str          exactly as it appears in the document
normalized_claim    str          self-contained form, used by the checks
location            str          section heading it was found under
char_span           (int, int)   offset in `document`
claim_type          ClaimType    efficacy | safety | indication | dosing | comparative | economic | other
reference_markers   list[str]    reference numbers the copy attached, e.g. ["2"]
```

`original_text` is what the reviewer is shown; `normalized_claim` is what the checks judge. A
headline reading "Nearly 6 in 10" cannot be assessed on its own, so the extractor expands it into
"Nearly 6 in 10 adults treated with Zelvoran achieved PASI-90 at Week 16" — but the finding still
quotes the words that are actually on the page.

### `Citation`

```
source_id     str            retrieval source id, rule label, or reference document path
passage       str            the exact text relied on
char_span     (int, int)     offset inside that source
kind          CitationKind   evidence | rule | doc_clause
title         str            display name, e.g. "ZEL-302 — Prespecified subgroup analysis"
rule_id       str            set when kind is rule or doc_clause
rule_version  int
```

`passage` must appear **verbatim** in the named source, modulo whitespace. The grounding gate verifies
this, which is why a model cannot assert a citation into existence.

### `Finding`

```
id                   str
check_type           CheckType     substantiation | off_label | fair_balance |
                                   safety_prominence | business_rule
claim_id             str           "" for a document-level finding
quote                str           verbatim from `document`; "" when the finding is an absence
char_span            (int, int)
issue                str           written for a reviewer
suggested_fix        str
citations            list[Citation]
severity_proposed    Severity      what the check suggested
severity_final       Severity|None what the tenant's policy decided; None until `severity` runs
severity_rule_id     str           which tenant rule set it
confidence           int           0–100
low_confidence       bool          set by the confidence gate
grounded             bool
grounded_by          GroundedBy    evidence | rule | doc_clause | none
rule_id, rule_version              set when a tenant business rule produced it
model, prompt_version              empty for deterministic findings
merged_from          list[str]     ids the editor merged into this one
```

Properties: `.severity` (final, falling back to proposed), `.is_blocker`, `.is_absence`,
`.from_rule`.

**Two severities, not one.** Keeping both is what lets the audit trail show that Acme treated as
advisory something Borealis blocks. Collapsing them would mean a reviewer is told "advisory" with no
record that the check said otherwise.

**Grounding is a mode, not a flag.** A finding can be grounded in retrieved evidence, in a tenant
rule's own wording, or in a clause of a tenant reference document. These are different strengths of
claim, so a single boolean would make "grounding rate" a number that silently mixes three things.
`grounded_by == "none"` means the finding failed the gate and is escalating — it is **never dropped**.

### `GateResult`

Defined in `platform_core/gates.py` and re-exported from `mlr.schemas`. A gate is a platform concept,
so there is one definition, not two.

```
gate          str
passed        bool
details       str        written for a reviewer; populated on a pass too
failed_items  list[str]  finding ids that tripped it
on_fail_route str|None   None means "record, do not divert"
reason_code   str        matched against tenant escalation rules
```

### `ReviewReport`

The shape the API returns and the UI renders. Fields: identity (`run_id`, `tenant_id`, `product_id`,
`label_version`, `audience`, `channel`, `market`), `status`, `route`, `escalation`, `findings`,
`gate_results`, `rules_applied`, `claims_extracted`, `preapproved_matches`, provenance
(`policy_version`, `config_version`, `backend`), `cost`, `error`.

Helpers: `.blockers()`, `.advisories()`, `.ungrounded()`, `.summary_line()`.

**`status` is the only field that gates what is populated.** A running review returns the same shape
with fewer findings — not a different payload the caller has to branch on.

**There is no approved state.** `route` always names a human. `status == "complete"` means a human
decided, not that the system approved anything.

---

## 3. Review state — `mlr/state.py`

### The one thing that will bite you

```python
findings: Annotated[list[Finding], operator.add]  # append-only
final_findings: list[Finding]  # replaces
```

`findings` has an append reducer because five check nodes and three rule nodes run in parallel and
each returns its own list. **A reducer applies to every write to that key.** So the editor node, whose
job is to dedupe, *cannot* return a replacement for `findings` — it would be appended to the
originals and every finding would silently double. The counts just come out wrong; nothing raises.

So the editor writes `final_findings`, and everything downstream of it — severity, the confidence
gate, routing, the report, the eval scorer — reads `final_findings`. `findings` is kept so the audit
trail can show what the checks said before anything was merged.

`mlr.state.report_findings(state)` does the right thing: prefers `final_findings`, falls back to
`findings` so a run that failed before the editor still reports what was found rather than appearing
clean.

`ACCUMULATED_KEYS` and `REPLACEABLE_KEYS` classify every key; a test asserts the classification is
exhaustive and disjoint.

### Keys

Set at submission: `run_id`, `tenant_id`, `product_id`, `label_version`, `audience`, `channel`,
`market`. **`audience` and `channel` are inputs, not inferred** — they determine which tenant rules
are in scope, and applying a patient-facing disclaimer rule to a clinician piece would manufacture
false blockers.

Then: `document`, `doc_meta` (ingest) · `claims` (extraction) · `preapproved_matches` ·
`rules_applied` · `findings`/`gate_results`/`errors` (accumulated) · `final_findings` (editor) ·
`route`/`escalation` · `human_decision` · `status`/`policy_version`/`cost`.

`doc_meta` shape: `{"sections": {name: {"start": int, "end": int}}, ...}`. Placement rules and the
safety-prominence check read it.

Start a run with `initial_state(...)`, which initialises the accumulators to `[]` so nodes can read
them unconditionally.

---

## 4. Golden cases — `platform_core/evals/schema.py`

One YAML per case in `evals/golden/`. Complete worked example:
**`evals/golden/001-subgroup-overstatement.yaml`** — read it before writing another.

```
id, title                str
why_hard                 str    required; what this case tests that others do not
tenant_id, product_id    str
label_version            str
audience, channel, market str   part of the case: rule scope depends on them
promo | promo_path       str    inline text, or a path relative to the repo root
doc_meta                 dict   supplied explicitly, so a parsing change is not read as a model regression
expected_findings        list   [{check_type, claim_contains, severity, note}]
expected_rule_ids        list   tenant rules expected to fire
expected_route           str
must_not_flag            list   must NOT be flagged
tags                     list
```

**`must_not_flag` is not optional in spirit.** Measuring only what a system should catch rewards
flagging everything; the false-alarm list is what keeps precision honest.

Matching rule: same `check_type`, claim text contains `claim_contains` (case-insensitive), severity
compared separately so a severity disagreement is reported as a calibration problem rather than a
miss.

---

## 5. The service façade — `mlr/service.py`

**Stubbed today.** Every function returns data from `mlr/examples.py`, built through the real models.
`mlr.service.IS_STUB` is `True`; flip it in Phase 3. Signatures and return types will not change.

```python
start_run(*, tenant_id, product_id, document, label_version="", audience="any",
          channel="", market="", doc_meta=None, run_id=None) -> str   # run_id
get_status(run_id)        -> ReviewReport        # raises RunNotFoundError
get_review_task(run_id)   -> ReviewTask | None   # None when not awaiting a human
get_queue(tier="")        -> list[ReviewTask]
resume(run_id, decision)  -> ReviewReport        # validates the decision first
get_audit(run_id)         -> list[dict]
list_runs(tenant_id="", status="") -> list[ReviewReport]
get_graph_mermaid()       -> str
```

The lower-level, graph-agnostic runtime stays in `platform_core/runtime.py`: `start_run(graph, state)`,
`resume_run(graph, run_id, value)`, `get_status(graph, run_id)`, `get_audit(run_id)`, `history(graph,
run_id)`, `render_mermaid(graph)`. Use `mlr.service` unless you are writing the graph.

### API endpoints and example responses

Every endpoint has a committed example under `tests/fixtures/api/`. They are **generated** from the
models by `make fixtures`, and `tests/test_contracts.py` fails if the committed files no longer match
— so a fixture cannot drift from the schema it illustrates. Do not hand-edit them.

| Method and path | Fixture |
|---|---|
| `POST /reviews` | `POST_reviews.json` |
| `GET /reviews/{run_id}` | `GET_reviews_{run_id}.json`, `…__running.json`, `…__complete.json` |
| `GET /reviews` | `GET_reviews.json` |
| `GET /queue?tier=` | `GET_queue.json` |
| `POST /reviews/{run_id}/decision` | `POST_reviews_{run_id}_decision.json` |
| `GET /reviews/{run_id}/audit` | `GET_reviews_{run_id}_audit.json` |
| `GET /reviews/{run_id}/report.json` | `GET_reviews_{run_id}_report.json` |
| `GET /reviews/{run_id}/report.xlsx` | binary; same columns as the JSON `rows` |
| `GET /tenants`, `GET /tenants/{id}` | `GET_tenants.json`, `GET_tenants_{id}.json` |
| `GET /tenants/{id}/rules` | `GET_tenants_{id}_rules.json` |
| `GET /tenants/{id}/rules/{rule_id}` | `GET_tenants_{id}_rules_{rule_id}.json` (with version history) |
| `POST /tenants/{id}/rules` | `POST_tenants_{id}_rules.json` (creates a draft + validation report) |
| `PUT /tenants/{id}/rules/{rule_id}` | same shape as the POST (new version) |
| `POST /tenants/{id}/rules/{rule_id}/test` | `POST_tenants_{id}_rules_{rule_id}_test.json` (saves nothing) |
| `POST /tenants/{id}/rules/{rule_id}/activate` | `…_activate.json` (validation + regression check) |
| `POST /tenants/{id}/rules/{rule_id}/retire` | `…_retire.json` |
| `POST /tenants/{id}/rules/export` | `POST_tenants_{id}_rules_export.json` |
| `POST /evals/run`, `GET /evals/runs`, `GET /evals/runs/{id}` | `POST_evals_run.json`, `GET_evals_runs.json`, `GET_evals_runs_{id}.json` |
| `POST /evals/sweep` | `POST_evals_sweep.json` |
| `GET /graph` | `GET_graph.json` |

The example run is deliberately a hard case, not a happy path: three blockers, a rule finding citing
a document clause, an escalation with reasons, and both citation kinds. A UI built against an empty
success payload falls over on the first real review.

**Activation returns a regression check.** Editing a rule can change blocker recall, so activation
reports `policy_version` before and after plus the comparison against the committed baseline, and can
be refused. See `POST_tenants_{id}_rules_{rule_id}_activate.json`.

---

## 6. Working in a worktree

```bash
git worktree add ../mlr-api -b feat/api
cd ../mlr-api
make setup && make seed      # each worktree needs its own data/ — it is gitignored, not shared
make test
```

`data/` (SQLite databases and the Chroma store) is per-worktree and gitignored, so `make seed` is
required in each. `.worktreeinclude` lists the gitignored files a new worktree needs copied in
(`PROMPT.md`, `.env`).

Model backend: `MLR_LLM_BACKEND=fake` is forced in `tests/conftest.py`, so the suite never needs a
key or a running Ollama. For manual work, `ollama` is the free default and `anthropic` needs a key in
`.env`.

### Rules of engagement

- **Do not edit `mlr/schemas.py`, `mlr/state.py` or `mlr/service.py` signatures** without saying so —
  other sessions are building against them. Adding an optional field is safe; renaming or removing one
  is not.
- **Do not hand-edit `tests/fixtures/api/`.** Change `mlr/examples.py` and run `make fixtures`.
- `make test` and `make lint` must pass before every commit.
- Prompts are versioned: bump `PROMPT_VERSION` when you change one. The version is recorded in the
  audit log and in eval results, so a prompt change must be attributable.

---

## 7. What is real and what is stubbed, today

| Component | State |
|---|---|
| `platform_core/` — tenancy, audit, gates, llm, retrieval, runtime, rules, evals, policy | **real**, 295 tests |
| `corpus/` — labels, references, promo, two tenants, 15 rules | **real**, seeded by `make seed` |
| `mlr/schemas.py`, `mlr/state.py` | **real** |
| `mlr/examples.py` | **real** — canonical example data |
| `mlr/service.py` | **stub** — real signatures, fixture data, `IS_STUB=True` |
| `tests/fixtures/api/` | **real** — generated, drift-checked |
| `evals/golden/` | one complete case; the remaining 12 come in Phase 6 |
| `mlr/graph.py`, `mlr/nodes/` | **not written** — Phase 3 |
| `api/main.py`, `ui/` | **not written** |
