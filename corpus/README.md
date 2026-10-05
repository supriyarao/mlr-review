# Synthetic corpus

**Everything in this directory is fictional.** The products, companies, studies, study
identifiers, numbers, labels and promotional pieces were invented for this demonstration. Nothing
here describes a real medicine, and none of it is medical, legal or regulatory advice. It must not
be used to review real promotional material.

The regulatory *concepts* are real — fair balance, off-label promotion, substantiation,
Important Safety Information, pre-approved claims. The content they are applied to is not.

## What is here

| Directory | Contents |
|---|---|
| `labels/` | Label extracts per product and version: indication with population limits, dosing, contraindications, warnings, adverse reactions, and clinical studies with endpoints, subgroups and timepoints |
| `references/` | Study summaries, each stating its population, comparator, endpoint and timepoint explicitly |
| `preapproved_claims/` | Claims a tenant's MLR committee has already cleared, per tenant |
| `promo/` | Promotional pieces for the demo — a mix of clean and problematic |
| `tenants/<id>/` | One directory per tenant: `tenant.yaml` (policy), `rules/` (business rules), `reference_docs/` (codes of practice and style guides those rules cite) |

## The two products

**Zelvoran** (`zelvoran`), for moderate-to-severe plaque psoriasis **in adults** — originated by
Acme Pharma. Label v1.0. Three studies: ZEL-301, ZEL-302, ZEL-303.

**Nexapril** (`nexapril`), for adults with type 2 diabetes and established atherosclerotic
cardiovascular disease — Borealis Bio. Label v2.1. Three studies: NEX-114, NEX-115, NEX-116.

## Why Zelvoran appears under both tenants

Zelvoran is **co-promoted**: Acme originated it and Borealis promotes it in a second market. Each
holds its own indexed copy of the same evidence, which is how affiliate structures actually work.

This exists so the demo can run **the identical promotional piece under two tenants and get two
different outcomes** — and have that difference be caused purely by policy, not by one tenant
lacking the evidence. Borealis requires safety information in the same section as the first efficacy
claim; Acme does not. Same words, same evidence, same model: one routes to a standard reviewer, the
other blocks.

Nexapril is Borealis-only, which keeps a clean cross-tenant isolation trap: a Zelvoran piece
submitted under Acme that makes a cardiovascular claim can only be supported by Nexapril's NEX-114
study, which exists nowhere in Acme's corpus. The correct behaviour is "cannot verify — escalate",
never "supported".

## Designed against the golden set

The corpus and the golden cases in `evals/golden/` were written together, so each hard case is
genuinely provable from the evidence rather than retrofitted onto it. Some deliberate properties:

- ZEL-302 reports a **prespecified subgroup** (prior biologic exposure) with a markedly better
  PASI-90 rate than the overall population — so a promo quoting the subgroup number as a general
  result is a real overstatement, demonstrable from the reference.
- ZEL-303 reports **PASI-75 at week 16**, not PASI-90 at week 12 — so a promo citing it for the
  latter is citing a study that does not measure what is claimed.
- **No head-to-head study exists for either product.** Any superiority claim is unsubstantiated by
  construction.
- **No two-year data exists.** ZEL-303 runs to 52 weeks, so a durability claim beyond that cannot be
  grounded anywhere.
- Zelvoran's label is **adults only**, with no paediatric data, so family or child-directed copy is
  off-label by implication.
