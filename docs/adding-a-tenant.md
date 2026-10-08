# Adding a tenant

Onboarding a client means adding files, never editing code. This is the wrapper for doing that.

```bash
make new-tenant ID=kestrel_pharma PRODUCT=amvora NAME="Kestrel Pharma" PRODUCT_NAME="Amvora"
make check-tenant ID=kestrel_pharma   # what is still a stub
make seed                             # index it
make demo                             # review something
```

`NAME` and `PRODUCT_NAME` are optional; they default to a title-cased version of the ids.

## Why it is two commands

The scaffold writes every file the platform requires, each one **valid but stubbed**. The check
tells you which bodies are still stubs.

That split exists because of how this system fails. A half-built tenant seeds cleanly, starts
cleanly, and then answers *"cannot verify"* to every claim — because retrieval found no evidence to
verify against. Nothing errors. It looks like a model that has gone stupid, and the actual cause is
a missing file. `check-tenant` is there to say so on day one rather than on day three.

So it sorts what it finds into three buckets, and the middle one is the one that matters:

| bucket | meaning |
|---|---|
| **BLOCKING** | seeding or reviewing will fail outright |
| **WILL MISLEAD** | it runs, and you should not trust the answers |
| **WORTH DOING** | real, but not urgent |

It exits non-zero for the first two. It changes nothing — it only reports.

## What gets created

```
corpus/tenants/<id>/tenant.yaml                     policy: severity, escalation, thresholds
corpus/tenants/<id>/rules/*.yaml                    two starter rules
corpus/tenants/<id>/reference_docs/*.md             a code of practice, for clause-grounded findings
corpus/preapproved_claims/<id>.yaml                 cleared claims (empty)
corpus/labels/<product>-v1.0.md                     the approved label
corpus/references/<pre>-101.md, -102.md             study summaries
corpus/promo/<product>-hcp-example.md               something to review
```

Re-running is safe: a file that already exists is kept, not overwritten.

## Fill it in in this order

Each step depends on the one before it, so going out of order means redoing work.

1. **The label.** Every claim is checked against this, so it is the real source of truth. Write it
   first and the rest follows from it.
2. **The studies.** The figures here must match the label. Where they disagree, the reviewer is
   right to flag it — which makes a careless mismatch look like a bug in the system.
3. **`tenant.yaml`.** Search for `TODO`. Two of them actually change behaviour: the `fair_balance`
   severity, and the escalation tiers.
4. **The rules.** Two to start. Add what this client really enforces.
5. **A promo piece**, with a deliberate and reasonably subtle problem in it.

## No clinical content is generated

The scaffold writes structure, not substance — the label and study bodies are headings and TODOs.

That is deliberate. Inventing a label and a set of study results means inventing the evidence that
every claim is then judged against. The review would run, produce findings, and look completely
convincing, while being a check of one invention against another. Worse, the figures would not hang
together — a study whose numbers do not support its own label gives a reviewer real findings for
fictional reasons, and nobody can tell those apart from true positives afterwards.

All content is synthetic and all products are fictional, but it is written by hand, once, with the
numbers made to agree.

## Two things that look fine and are not

**A rule that says `status: active` may not be active.** Seeding runs every deterministic rule
against its own examples and refuses to activate one whose examples do not behave as labelled — it
is stored as a draft instead, so authored work is never lost, but it never fires. The YAML still
says `active`. Only running the examples tells you, which is why `check-tenant` does.

**Studies are found by filename prefix.** A product's study files are matched by the first three
letters of its `product_id` (`amvora` → `amv-101.md`), or by `study_prefix` in `tenant.yaml` if you
want something else. A mismatch does not error — it silently returns no evidence.

## Covering it with the eval gate

A new tenant sits outside the regression gate until a golden case names it. Copy the shape of
`evals/golden/001-subgroup-overstatement.yaml`, then `make eval-baseline` to record it.

Related: [running the demo](running-the-demo.md) · [known issues](known-issues.md)
