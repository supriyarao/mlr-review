"""Scaffold a new tenant, then tell you exactly what is still missing.

Two commands, and the gap between them is where you drop your own content in::

    make new-tenant ID=cardiva_health PRODUCT=cardivex
    # ...drop your label, studies and promo pieces into the paths it printed...
    make check-tenant ID=cardiva_health      # says what is still missing, fixes nothing
    make seed                                # index it
    make demo                               # review something

Why a scaffold rather than a template directory to copy: a copy of an existing tenant inherits
every decision that tenant made, and the whole point of a second tenant is that it should differ.
What is generated here is the *structure* plus defaults that are deliberately boring, with a
TODO on every line a real client would argue about.

**Nothing here is a model call and nothing here invents clinical content.** A label extract and a
study summary are the two things you have to write yourself, because they are the evidence
everything else is checked against -- generating them would mean the system was checking claims
against text it had made up, which is worse than having no tenant at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SYNTHETIC_NOTE = "> Synthetic content for demonstration. Not a real medicine."


def _tenant_yaml(tenant_id: str, name: str, product_id: str, product_name: str) -> str:
    return f"""# {name} - tenant policy
#
# Fictional. Every value below is a policy decision, not a technical one: a real client's
# compliance team would argue about each line. The defaults here are the cautious ones.

tenant_id: {tenant_id}
name: {name}
version: "2026.1"

# Findings below this confidence are flagged and escalated. Lower accepts more machine
# judgement and reviews faster; higher sends more to a human. Acme uses 70, Borealis 80.
confidence_threshold: 75

products:
  - product_id: {product_id}
    name: {product_name}
    label_version: "1.0"
    indication: TODO - the approved indication, in the label's own words
    # Filename prefix of this product's studies in corpus/references/.
    # Defaults to the first three letters of product_id; set it if yours differ.
    study_prefix: {product_id[:3]}

severity_rules:
  # First match wins, top to bottom. "*" is the fallback and must stay last.
  - check_type: off_label
    severity: blocker
    rule_id: {tenant_id.upper()}-SEV-OFFLABEL
    rationale: Promotion outside the approved population or indication is not correctable by wording.

  - check_type: substantiation
    severity: blocker
    rule_id: {tenant_id.upper()}-SEV-SUBSTANTIATION
    rationale: >-
      An unsupported or overstated efficacy claim changes a prescribing decision, and a citation
      that does not measure the claimed endpoint or population is no citation.

  - check_type: safety_prominence
    severity: blocker
    rule_id: {tenant_id.upper()}-SEV-SAFETY
    rationale: Required safety information must be present for a reader to weigh benefit and risk.

  # TODO - decide this one deliberately. It is the line that most often differs between clients:
  # advisory treats imbalance as an editorial fix, blocker treats it as a substantive defect.
  - check_type: fair_balance
    severity: advisory
    rule_id: {tenant_id.upper()}-SEV-BALANCE
    rationale: Imbalance of emphasis is usually fixable in layout without changing the claim.

  - check_type: "*"
    severity: advisory
    rule_id: {tenant_id.upper()}-SEV-DEFAULT
    rationale: Unclassified observations go to a reviewer as advisory rather than being dropped.

escalation:
  # Even a piece with no findings goes to a person. The system never approves anything.
  standard_tier: standard_reviewer
  default_escalation_tier: senior_regulatory
  rules:
    # TODO - who actually looks at these? One senior tier for everything, or split by discipline?
    - when_reason: off_label
      tier: senior_regulatory
    - when_reason: comparative_claim
      tier: senior_regulatory

policy_flags:
  # A superiority claim needs a head-to-head trial to stand up.
  require_head_to_head_for_superiority: true
  # Whether safety information must sit beside the first efficacy claim, not just be present.
  require_isi_near_first_efficacy_claim: false
  # A pre-approved claim skips the evidence check but is still checked for balance and off-label
  # implication, because approved wording in a new context can still mislead.
  preapproved_claims_skip_substantiation: true
"""


def _banned_rule(tenant_id: str, product_name: str) -> str:
    prefix = tenant_id.split("_")[0].upper()[:4]
    return f"""id: {prefix}-BAN-001
tenant_id: {tenant_id}
title: Prohibited efficacy and safety language
description: >-
  Promotional material must not state or imply that the product cures the condition it treats,
  that a result is guaranteed, or that the product is completely safe or free of risk.
kind: deterministic
severity: blocker
check_type: business_rule
scope:
  audience: any
params:
  type: banned_terms
  terms:
    - cure
    - cures
    - guaranteed
    - completely safe
    - no side effects
  suggested_fix: >-
    Describe the observed effect using the labelled endpoint and timepoint instead.
examples:
  - text: {product_name} can cure the condition.
    expected: fail
    note: States a cure.
  - text: {product_name} reduced events by 22.0% versus placebo at Week 16 [1].
    expected: pass
    note: A measured effect with its comparator and timepoint.
owner: TODO - which team owns this rule
effective_from: "2026-01-01"
status: active
source_ref: TODO - the clause of the client's own code this comes from
"""


def _isi_rule(tenant_id: str) -> str:
    prefix = tenant_id.split("_")[0].upper()[:4]
    return f"""id: {prefix}-ISI-001
tenant_id: {tenant_id}
title: Important Safety Information must be present
description: >-
  Every promotional piece must carry the Important Safety Information for the product so a reader
  can weigh benefit against risk.
kind: deterministic
severity: blocker
check_type: safety_prominence
scope:
  audience: any
params:
  type: required_section
  section: important_safety_information
  # TODO - add `near_section: efficacy` and `max_distance_chars: 600` if this client requires
  # safety information beside the claim rather than merely present. That single addition is what
  # makes the same piece route differently between the two existing tenants.
  suggested_fix: >-
    Add an Important Safety Information section carrying the contraindications, warnings and most
    common adverse reactions.
# A required_section rule reads the section map the extractor built, not the raw text, so each
# example has to carry its own doc_meta. Without it the map is empty, which the checker reports
# as "could not evaluate" rather than pass or fail -- so an example labelled pass does not pass,
# the rule fails its own self-test, and seeding leaves it as a draft that never fires.
examples:
  - text: "Efficacy\\n\\nWorks well.\\n\\nImportant Safety Information\\n\\nContraindicated in X."
    expected: pass
    note: Safety section present.
    doc_meta:
      sections:
        efficacy:
          start: 0
          end: 22
        important_safety_information:
          start: 24
          end: 76
  - text: "Efficacy\\n\\nWorks well."
    expected: fail
    note: No safety section at all.
    doc_meta:
      sections:
        efficacy:
          start: 0
          end: 22
owner: TODO - which team owns this rule
effective_from: "2026-01-01"
status: active
source_ref: TODO
"""


def _code_of_practice(name: str) -> str:
    return f"""# {name} Promotional Code (fictional)

{SYNTHETIC_NOTE}

A reference document the reviewer reads the piece against. Clauses are cited verbatim, and a
clause the model quotes is checked against this file before it becomes a finding -- so the
wording here matters.

## 1. Accuracy

A claim must be capable of substantiation by the approved label or by scientific evidence, at the
strength at which it is stated.

## 2. Subgroups

A result observed in a subgroup must be described as a subgroup result, naming the subgroup. A
subgroup figure must never be presented as the result in the overall population.

## 3. Durability

Claims about how long an effect lasts must not extend beyond the longest timepoint for which data
exist.

## 4. TODO

Add the clauses this client actually enforces. Each one should be something a reviewer could
point at and say "this sentence breaches that clause".
"""


def _label_stub(product_name: str, product_id: str) -> str:
    return f"""# {product_name.upper()} (fictional) - label extract v1.0

{SYNTHETIC_NOTE}

TODO - this file is the evidence everything is checked against. Write it before seeding, and keep
it internally consistent: the figures here are what a claim will be verified against, so an
inconsistency reads as a model failure months later.

## 1. Indications and Usage

{product_name.upper()} is indicated for TODO - the condition, the population, and any limitation
of use. Be explicit about who it is NOT for: the off-label reviewer checks the piece against this
sentence, and "adults" versus "adults and children" is exactly the kind of difference it catches.

## 2. Dosage and Administration

TODO

## 3. Contraindications

TODO

## 4. Warnings and Precautions

TODO

## 5. Adverse Reactions

TODO - the most common ones, with rates.

## 6. Clinical Studies

### 6.1 {product_id[:3].upper()}-101 - TODO primary endpoint

TODO - the endpoint, the timepoint, the figure, the comparator. One subsection per study, and the
numbers must match the study summaries in corpus/references/.

## 7. Limitations of Use

TODO - safety and effectiveness have not been established in [population]. This sentence is what
makes a paediatric claim catchable.
"""


def _study_stub(product_id: str, n: int) -> str:
    code = f"{product_id[:3].upper()}-{100 + n}"
    return f"""# Reference {n} - {code} (fictional study summary)

{SYNTHETIC_NOTE}

Study: {code}. TODO - design, duration, blinding.
Population: TODO - how many, what age range, and state plainly if a group was NOT enrolled.

## Endpoints and timepoints

TODO - exactly what was measured and when. The citation reviewer checks a claim's footnote
against this: if a piece cites this study for an endpoint it never measured, that is the finding.

## Results

TODO - the figures, with their comparator.

## Subgroup analyses

TODO - if any figure applies only to a subgroup, say so here in those words. The commonest real
failure is a subgroup figure presented as a general result, and this section is what catches it.

## Endpoints not assessed

TODO - say what this study did NOT measure. Counter-intuitively this is the most useful section:
it is what lets a reviewer prove a citation does not support a claim.
"""


def _promo_stub(tenant_id: str, product_id: str, product_name: str) -> str:
    return f"""---
audience: hcp
channel: sales_aid
market: us
product_id: {product_id}
tenant_id: {tenant_id}
---

# TODO - the headline claim

## Efficacy

TODO - write a piece with a deliberate, specific problem in it. The useful ones are subtle: a real
figure applied to the wrong population, a real study cited for an endpoint it did not measure,
approved wording placed next to a line that widens it.

A piece where every claim is obviously wrong tests nothing.

## Important Safety Information

TODO - the contraindications, warnings and common adverse reactions.

## References

1. TODO - study code and design.

Please see full Prescribing Information.
"""


def _preapproved(tenant_id: str, product_id: str) -> str:
    return f"""# Claims {tenant_id} has already cleared. A submitted claim matching one of these skips
# the evidence check, because the committee has already done that work.
#
# WARNING, and it is a real one: this file is NOT yet part of policy_version, so adding a claim
# here changes review outcomes while the audit hash stays identical. See docs/known-issues.md #1.
# Treat additions as policy changes until that is fixed.

product_id: {product_id}
label_version: "1.0"

claims: []
  # - id: {tenant_id.split("_")[0].upper()[:4]}-PAC-001
  #   text: TODO - the exact approved wording, character for character.
  #   approved_on: "2026-01-01"
  #   source: TODO - which label section or study supports it
  #   notes: TODO
"""


def scaffold(
    tenant_id: str,
    name: str,
    product_id: str,
    product_name: str,
    *,
    root: Path | None = None,
) -> list[str]:
    """Create every file a tenant needs. Returns what was written, in the order to fill it in.

    ``root`` rebases the whole corpus, which is what lets the test suite scaffold a tenant into a
    temporary directory and assert the result really validates -- without that, the only way to
    test this script is to write into the committed corpus and clean up afterwards.
    """
    base = root or REPO_ROOT
    tenants, labels, references, promo, preapproved = (
        base / "corpus" / part
        for part in ("tenants", "labels", "references", "promo", "preapproved_claims")
    )
    tenant_root = tenants / tenant_id
    written: list[str] = []

    def write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(f"  kept     {path.relative_to(base)} (already there)")
            return
        path.write_text(content, encoding="utf-8")
        written.append(str(path.relative_to(base)))
        print(f"  created  {path.relative_to(base)}")

    write(tenant_root / "tenant.yaml", _tenant_yaml(tenant_id, name, product_id, product_name))
    write(
        tenant_root / "rules" / f"{tenant_id.split('_')[0][:4]}-ban-001.yaml",
        _banned_rule(tenant_id, product_name),
    )
    write(
        tenant_root / "rules" / f"{tenant_id.split('_')[0][:4]}-isi-001.yaml", _isi_rule(tenant_id)
    )
    write(
        tenant_root / "reference_docs" / f"{tenant_id}-promotional-code.md", _code_of_practice(name)
    )
    write(preapproved / f"{tenant_id}.yaml", _preapproved(tenant_id, product_id))
    write(labels / f"{product_id}-v1.0.md", _label_stub(product_name, product_id))
    for n in (1, 2):
        write(references / f"{product_id[:3]}-{100 + n}.md", _study_stub(product_id, n))
    write(promo / f"{product_id}-hcp-example.md", _promo_stub(tenant_id, product_id, product_name))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold a new tenant and its corpus.")
    parser.add_argument("--id", required=True, help="tenant id, alphanumeric and underscores")
    parser.add_argument("--name", default="", help="display name (default: derived from the id)")
    parser.add_argument("--product", required=True, help="product id, lowercase letters")
    parser.add_argument("--product-name", default="", help="display name for the product")
    args = parser.parse_args()

    tenant_id = args.id.strip().lower()
    if not tenant_id or not all(c.isalnum() or c == "_" for c in tenant_id):
        print(f"tenant id must be alphanumeric/underscore, got {tenant_id!r}", file=sys.stderr)
        return 1

    product_id = args.product.strip().lower()
    if not product_id.isalnum():
        print(f"product id must be alphanumeric, got {product_id!r}", file=sys.stderr)
        return 1

    name = args.name or tenant_id.replace("_", " ").title()
    product_name = args.product_name or product_id.title()

    print(f"\nScaffolding {name} ({tenant_id}) with product {product_name} ({product_id}):\n")
    scaffold(tenant_id, name, product_id, product_name)

    print(f"""
Now fill these in, in this order. Each depends on the one before it:

  1. corpus/labels/{product_id}-v1.0.md
       The approved label. Everything is checked against this, so write it first.

  2. corpus/references/{product_id[:3]}-101.md and -102.md
       The studies. The figures must match the label.

  3. corpus/tenants/{tenant_id}/tenant.yaml
       Search for TODO. The fair_balance severity and the escalation tiers are the two
       decisions that actually change behaviour.

  4. corpus/tenants/{tenant_id}/rules/
       Two rules to start. Add the ones this client really enforces.

  5. corpus/promo/{product_id}-hcp-example.md
       A piece with a deliberate, subtle problem in it.

Then:

  make check-tenant ID={tenant_id}     # says what is still missing
  make seed                            # index it
  make demo                            # review something

Optional: add a case to evals/golden/ so the new tenant is covered by the regression gate.
Copy the shape of evals/golden/001-subgroup-overstatement.yaml, and write its why_hard.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
