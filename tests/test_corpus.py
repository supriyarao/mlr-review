"""The synthetic corpus: does it load, and does it still support the hard cases?

Why this file exists: the corpus and the golden set were designed together, and several of the hard
cases depend on precise properties of the content -- that ZEL-303 reports PASI-75 at Week 16 and
*not* PASI-90 at Week 12, that no head-to-head study exists, that no data run past Week 52. Those
properties are easy to destroy with a well-meaning edit to a markdown file, and the damage would show
up much later as an eval case that passes for the wrong reason.

These tests read the corpus directly. They make no model calls and do not need the vector store, so
they run in the normal offline suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from platform_core import tenancy
from platform_core.rules.deterministic import evaluate_all, normalize_whitespace
from platform_core.rules.schema import Rule, RuleInput, Segment
from platform_core.rules.validation import validate_rule

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "corpus"
LABELS = CORPUS / "labels"
REFERENCES = CORPUS / "references"
PROMO = CORPUS / "promo"
PREAPPROVED = CORPUS / "preapproved_claims"

# The corpus ships at a fixed path, so these tests must not use the tenant_dir fixture.
REAL_TENANTS = CORPUS / "tenants"


@pytest.fixture(autouse=True)
def real_corpus(monkeypatch):
    """Point tenancy at the shipped corpus rather than a temporary directory."""
    monkeypatch.setenv("MLR_TENANT_DIR", str(REAL_TENANTS))
    tenancy.clear_tenant_cache()
    yield
    tenancy.clear_tenant_cache()


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def flat(path: Path) -> str:
    return normalize_whitespace(read(path))


def promo(stem: str) -> tuple[dict, str]:
    """Front matter and body of a promotional piece."""
    _, front, body = read(PROMO / f"{stem}.md").split("---\n", 2)
    return yaml.safe_load(front), body.lstrip("\n")


# --- It all loads -------------------------------------------------------------------------------


def test_both_tenants_load_and_validate():
    assert [t.tenant_id for t in tenancy.load_all_tenants()] == ["acme_pharma", "borealis_bio"]


def test_every_product_has_a_label_extract():
    for tenant in tenancy.load_all_tenants():
        for product in tenant.products:
            path = LABELS / f"{product.product_id}-v{product.label_version}.md"
            assert path.is_file(), f"{tenant.tenant_id} references a missing label: {path.name}"


def test_every_promo_piece_has_the_metadata_ingest_needs():
    pieces = sorted(PROMO.glob("*.md"))
    assert len(pieces) >= 5
    for path in pieces:
        front, body = promo(path.stem)
        for field in ("title", "product_id", "label_version", "audience", "channel"):
            assert field in front, f"{path.name} front matter is missing {field}"
        assert front["audience"] in {"hcp", "patient", "any"}
        assert body.strip(), f"{path.name} has no body"


def test_promo_metadata_is_not_part_of_the_reviewable_text():
    """Regression guard for a real false positive.

    Metadata used to live in an HTML comment, which is still document text -- so the trademark rule
    matched the lower-case product id in the comment instead of the brand in the copy, and reported a
    violation on correctly marked pieces.
    """
    for path in sorted(PROMO.glob("*.md")):
        _, body = promo(path.stem)
        assert "audience:" not in body
        assert "<!--" not in body


def test_every_promo_product_belongs_to_some_tenant():
    known = {p.product_id for t in tenancy.load_all_tenants() for p in t.products}
    for path in sorted(PROMO.glob("*.md")):
        front, _ = promo(path.stem)
        assert front["product_id"] in known, f"{path.name} names an unknown product"


def test_preapproved_claim_files_exist_for_both_tenants():
    for tenant in tenancy.load_all_tenants():
        raw = yaml.safe_load(read(PREAPPROVED / f"{tenant.tenant_id}.yaml"))
        assert raw["tenant_id"] == tenant.tenant_id
        assert raw["claims"], f"{tenant.tenant_id} has no pre-approved claims"
        for claim in raw["claims"]:
            assert claim["id"] and claim["text"].strip()


def test_evidence_prose_has_no_inline_emphasis():
    """Regression guard: markdown emphasis inside a sentence breaks verbatim grounding.

    A finding cites a passage, and the grounding gate requires that passage to appear verbatim in a
    source. A model quoting "did not assess PASI-90" cannot match a source storing
    "did **not** assess PASI-90", so a correct citation would be rejected as ungrounded.
    """
    for path in list(LABELS.glob("*.md")) + list(REFERENCES.glob("*.md")):
        body = read(path)
        assert "**" not in body, f"{path.name} contains inline bold, which breaks verbatim quoting"


def test_everything_is_marked_synthetic():
    """A reader must never mistake this for real product information."""
    for path in list(LABELS.glob("*.md")) + list(REFERENCES.glob("*.md")):
        assert "fictional" in read(path).lower(), f"{path.name} is not marked fictional"
    for path in sorted(PROMO.glob("*.md")):
        front, _ = promo(path.stem)
        assert front.get("synthetic") is True, f"{path.name} is not marked synthetic"


# --- The properties the golden cases depend on --------------------------------------------------


def test_zelvoran_is_adults_only_with_no_paediatric_data():
    """Case 2, off-label by implication, needs the limitation stated explicitly."""
    label = flat(LABELS / "zelvoran-v1.0.md")
    assert "moderate-to-severe plaque psoriasis in adults" in label
    assert "not been established in paediatric patients" in label


def test_zel_302_reports_a_subgroup_better_than_the_overall_population():
    """Case 1, overstated subgroup: both numbers must be present and distinguishable."""
    ref = flat(REFERENCES / "zel-302.md")
    assert "44.2%" in ref, "overall PASI-90 result missing"
    assert "58.1%" in ref, "prior-biologic subgroup PASI-90 result missing"
    assert "prespecified subgroup" in ref.lower()
    # The reference must say in words that the subgroup figure is not the overall result, so the
    # overstatement is demonstrable from the citation rather than only from arithmetic.
    assert "applies only to the subgroup of patients with prior biologic exposure" in ref
    assert "The result in the overall randomised population was 44.2%" in ref


def test_zel_303_does_not_measure_what_case_three_claims_it_does():
    """Case 3, reference/endpoint mismatch. If this ever changes, that case silently passes."""
    ref = flat(REFERENCES / "zel-303.md")
    assert "PASI-75" in ref
    assert "Week 16" in ref and "Week 52" in ref
    assert "did not assess PASI-90" in ref
    assert "no assessment at Week 12" in ref.replace(
        "made no assessment at Week 12", "no assessment at Week 12"
    )


def test_no_head_to_head_study_exists_for_either_product():
    """Case 5, superiority claim: unsubstantiated by construction."""
    for name in ("zelvoran-v1.0", "nexapril-v2.1"):
        assert "head-to-head" in flat(LABELS / f"{name}.md")
        assert "No head-to-head" in read(LABELS / f"{name}.md")


def zelvoran_evidence() -> list[Path]:
    return sorted(REFERENCES.glob("zel-*.md")) + [LABELS / "zelvoran-v1.0.md"]


def affirmative_sentences(path: Path, needle: str) -> list[str]:
    """Sentences mentioning ``needle`` that are not denying something.

    Evidence files state what was *not* studied as well as what was, so a plain "needle not in text"
    assertion is defeated by the corpus's own disclaimers. What matters is whether the mention is a
    reported result or a denial, so negated sentences are filtered out.
    """
    negations = ("no ", "not ", "never ", "none ", "cannot ", "did not")
    out = []
    for sentence in re.split(r"(?<=[.!])\s+", flat(path)):
        if needle.lower() not in sentence.lower():
            continue
        lowered = sentence.lower()
        if any(n in lowered for n in negations):
            continue
        out.append(sentence)
    return out


def test_no_zelvoran_data_exist_beyond_week_52():
    """Case 9, ungrounded durability claim. A two-year claim must have nothing to cite.

    Asserted on the actual timepoints the evidence reports, not on the absence of a phrase. An
    earlier version of this test read ``"two years" not in text or "no data" in text``, which could
    never fail, because zel-303 contains "No data on ZELVORAN beyond 52 weeks exist" and so the
    second clause was always true. Two-year data could have been added and this test would have
    stayed green -- defeating the one case it exists to protect.
    """
    assert "No data on ZELVORAN beyond 52 weeks exist" in flat(REFERENCES / "zel-303.md")

    weeks = {
        int(m.group(1))
        for path in zelvoran_evidence()
        for m in re.finditer(r"Week\s+(\d+)", read(path))
    }
    assert weeks, "no timepoints found at all, so this test is not checking anything"
    assert max(weeks) == 52, (
        f"Zelvoran evidence reports a timepoint beyond Week 52: {sorted(weeks)}"
    )

    # And no affirmative result stated in years or in months beyond 12.
    for path in zelvoran_evidence():
        assert affirmative_sentences(path, "two years") == [], path.name
        assert affirmative_sentences(path, "2 years") == [], path.name
        months = {
            int(m.group(1))
            for s in affirmative_sentences(path, "month")
            for m in re.finditer(r"(\d+)\s+months", s)
        }
        assert not {m for m in months if m > 12}, f"{path.name} reports {months} months"


def test_cardiovascular_evidence_exists_only_for_nexapril():
    """Case 10, cross-tenant leakage trap.

    A Zelvoran piece claiming a cardiovascular outcome can only be supported by Nexapril's evidence,
    which belongs to a different tenant. The correct answer is "cannot verify", not "supported".

    As above, an earlier version asserted ``"major adverse cardiovascular events" not in text or
    "No cardiovascular" in text``, which could never fail: every Zelvoran file already contains "No
    cardiovascular outcomes were assessed". Cardiovascular efficacy data could have been added to
    Zelvoran's evidence with this guard still green.
    """
    nex = flat(REFERENCES / "nex-114.md")
    assert "22% relative risk reduction" in nex
    assert "hazard ratio 0.78" in nex

    for path in zelvoran_evidence():
        text = flat(path)
        # A hazard ratio is how a cardiovascular outcome is reported; Zelvoran has none at all.
        assert "hazard ratio" not in text.lower(), f"{path.name} reports a hazard ratio"
        assert "22% relative risk reduction" not in text
        # Every cardiovascular mention must be a denial, never a reported result.
        affirmative = affirmative_sentences(path, "cardiovascular")
        assert affirmative == [], f"{path.name} states a cardiovascular result: {affirmative}"
        assert affirmative_sentences(path, "MACE") == [], path.name


def test_the_co_promoted_product_is_configured_for_both_tenants():
    """Case 6 needs one product both tenants can review, so the difference is purely policy."""
    acme = {p.product_id for p in tenancy.load_tenant("acme_pharma").products}
    borealis = {p.product_id for p in tenancy.load_tenant("borealis_bio").products}
    assert "zelvoran" in acme and "zelvoran" in borealis
    # And one product that is Borealis-only, which is what makes the leakage trap work.
    assert "nexapril" in borealis and "nexapril" not in acme


def test_the_two_tenants_differ_on_the_policies_the_cases_turn_on():
    acme = tenancy.load_tenant("acme_pharma")
    borealis = tenancy.load_tenant("borealis_bio")

    # Confidence threshold: the safety/throughput dial the eval sweep explores.
    assert acme.confidence_threshold == 70
    assert borealis.confidence_threshold == 80

    # Fair balance: advisory for Acme, blocking for Borealis. This drives case 6.
    def severity(config, check):
        return next(r.severity for r in config.severity_rules if r.check_type == check)

    assert severity(acme, "fair_balance") == "advisory"
    assert severity(borealis, "fair_balance") == "blocker"

    # Escalation: one funnel versus split by discipline. This drives case 5's routing.
    assert acme.escalation.tier_for(["comparative_claim"]) == "senior_regulatory"
    assert borealis.escalation.tier_for(["comparative_claim"]) == "legal"
    assert borealis.escalation.tier_for(["off_label"]) == "medical_director"


# --- Authored rules -----------------------------------------------------------------------------


def all_rules() -> list[tuple[str, Rule]]:
    out = []
    for tenant_id in tenancy.tenant_ids():
        for path in sorted(tenancy.tenant_rules_dir(tenant_id).glob("*.yaml")):
            out.append((tenant_id, Rule.model_validate(yaml.safe_load(read(path)))))
    return out


def test_both_tenants_ship_rules_across_all_three_kinds():
    by_tenant: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for tenant_id, rule in all_rules():
        by_tenant.setdefault(tenant_id, set()).add(rule.kind)
        counts[tenant_id] = counts.get(tenant_id, 0) + 1

    for tenant_id, kinds in by_tenant.items():
        assert kinds == {"deterministic", "llm", "reference_doc"}, f"{tenant_id} is missing a kind"
        assert 6 <= counts[tenant_id] <= 8, f"{tenant_id} has {counts[tenant_id]} rules"


def test_every_shipped_rule_validates_and_can_be_activated():
    """A rule that cannot be activated would silently stop being applied."""
    failures = []
    for tenant_id, rule in all_rules():
        report = validate_rule(
            rule, reference_docs_dir=tenancy.tenant_reference_docs_dir(tenant_id)
        )
        if not (report.can_save and report.can_activate):
            failures.append(f"{tenant_id}/{rule.id}: {report.errors or report.summary()}")
    assert not failures, "rules that cannot be activated:\n  " + "\n  ".join(failures)


def test_every_deterministic_rule_scores_perfectly_on_its_own_examples():
    """Plain code over plain text: anything less than 100% is a bug in the rule."""
    for tenant_id, rule in all_rules():
        if rule.kind != "deterministic":
            continue
        report = validate_rule(rule)
        assert report.accuracy() == 1.0, (
            f"{tenant_id}/{rule.id} scores {report.accuracy()}: {report.summary()}"
        )


def test_every_rule_declares_an_owner_and_a_source():
    """An unattributed rule is one nobody can be asked about."""
    for tenant_id, rule in all_rules():
        assert rule.owner.strip(), f"{tenant_id}/{rule.id} has no owner"
        assert rule.source_ref, f"{tenant_id}/{rule.id} cites no source"


def test_reference_doc_rules_cite_clauses_that_really_exist():
    """Validation enforces this, but assert it on the shipped content too.

    A clause that is not in the document would make a finding uncheckable: the reviewer would be
    shown an authority they cannot look up.
    """
    for tenant_id, rule in all_rules():
        if rule.kind != "reference_doc":
            continue
        doc = tenancy.tenant_reference_docs_dir(tenant_id) / rule.params["doc_path"]
        assert doc.is_file(), f"{tenant_id}/{rule.id} cites a missing document"
        document = flat(doc)
        for clause in rule.params.get("clauses", []):
            assert normalize_whitespace(clause) in document, (
                f"{tenant_id}/{rule.id} cites a clause absent from {doc.name}: {clause[:60]!r}"
            )


def test_the_tenants_rule_sets_are_genuinely_different():
    acme = {r.id for t, r in all_rules() if t == "acme_pharma"}
    borealis = {r.id for t, r in all_rules() if t == "borealis_bio"}
    assert acme and borealis
    assert not (acme & borealis), "the two tenants must not share rule ids in this corpus"


# --- Rules against the real promotional pieces --------------------------------------------------


def section_map(body: str) -> dict:
    """Minimal section map from markdown headings.

    Stands in for the ingest node, which does not exist yet. Deliberately simple: these tests are
    about the corpus, not about ingest.
    """
    sections: dict[str, dict[str, int]] = {}
    heads = list(re.finditer(r"^##\s+(.*)$", body, re.MULTILINE))
    for i, match in enumerate(heads):
        name = re.sub(r"[^a-z0-9]+", "_", match.group(1).lower()).strip("_")
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        sections[name] = {"start": match.start(), "end": end}
    for key in list(sections):
        if "efficacy" in key or key.startswith("what"):
            sections.setdefault("efficacy", sections[key])
            break
    return {"sections": sections}


def efficacy_segments(body: str, meta: dict) -> list[Segment]:
    """Numeric sentences inside the efficacy section, standing in for claim extraction."""
    section = meta["sections"].get("efficacy")
    if not section:
        return []
    chunk = body[section["start"] : section["end"]]
    out = []
    for i, sentence in enumerate(re.split(r"(?<=[.!])\s+", chunk)):
        sentence = sentence.strip().lstrip("#*> ").strip()
        if re.search(r"\d", sentence) and re.search(r"PASI|MACE|HbA1c|%", sentence):
            out.append(Segment(id=f"c{i}", text=sentence, char_span=(0, len(sentence))))
    return out


def run_deterministic(tenant_id: str, stem: str):
    from platform_core.rules.registry import RuleRegistry

    front, body = promo(stem)
    meta = section_map(body)
    registry = RuleRegistry(tenant_id)
    for path in sorted(tenancy.tenant_rules_dir(tenant_id).glob("*.yaml")):
        rule = Rule.model_validate(yaml.safe_load(read(path)))
        if rule.kind == "deterministic":
            registry.save_new_version(rule)
            registry.activate(rule.id, registry.latest_version_number(rule.id))

    applicable = [
        r
        for r in registry.applicable_rules(
            audience=front["audience"],
            channel=front["channel"],
            product_id=front["product_id"],
        )
        if r.kind == "deterministic"
    ]
    return evaluate_all(
        applicable,
        RuleInput(
            text=body,
            doc_meta=meta,
            segments=efficacy_segments(body, meta),
            audience=front["audience"],
            channel=front["channel"],
            product_id=front["product_id"],
        ),
    )


def test_the_clean_piece_trips_no_deterministic_rule():
    """The false-alarm check. A system that flags a correct piece trains reviewers to ignore it."""
    violations = run_deterministic("acme_pharma", "zelvoran-hcp-journal-ad")
    assert violations == [], [f"{v.rule_id}: {v.message}" for v in violations]


def test_the_demo_piece_passes_every_currently_active_rule():
    """The demo adds a new banned-term rule and re-runs this piece.

    It has to start clean, or the new blocker would not be visibly caused by the new rule. It
    deliberately contains 'breakthrough' and 'revolutionary', which no active rule yet prohibits.
    """
    violations = run_deterministic("acme_pharma", "zelvoran-hcp-launch-announcement")
    assert violations == [], [f"{v.rule_id}: {v.message}" for v in violations]
    _, body = promo("zelvoran-hcp-launch-announcement")
    assert "breakthrough" in body.lower()
    assert "revolutionary" in body.lower()


def test_the_same_piece_gets_different_rule_outcomes_under_the_two_tenants():
    """The headline multi-tenancy demonstration, and it must be caused purely by policy.

    Identical text, identical evidence, identical model. Acme requires safety information to be
    present; Borealis requires it beside the first efficacy claim.
    """
    acme = run_deterministic("acme_pharma", "zelvoran-hcp-isi-placement")
    borealis = run_deterministic("borealis_bio", "zelvoran-hcp-isi-placement")

    assert [v for v in acme if v.severity == "blocker"] == []
    blockers = [v for v in borealis if v.severity == "blocker"]
    assert len(blockers) == 1
    assert blockers[0].rule_id == "BOR-ISI-001"
    assert "beyond the 600 allowed" in blockers[0].message


def test_the_patient_brochure_trips_the_prohibited_term_and_the_missing_signpost():
    violations = run_deterministic("acme_pharma", "zelvoran-patient-brochure")
    fired = {v.rule_id for v in violations}
    assert "ACME-BAN-001" in fired
    assert "ACME-REQ-001" in fired
    cure = next(v for v in violations if v.rule_id == "ACME-BAN-001")
    assert cure.quote.lower() == "cure"
    assert cure.severity == "blocker"
    # The quote must be findable verbatim in the piece: that is what grounding will check.
    _, body = promo("zelvoran-patient-brochure")
    assert cure.quote in body


def test_a_patient_scoped_rule_does_not_apply_to_an_hcp_piece():
    """Scope is what makes the two tenants' rule sets genuinely different, not just differently worded."""
    hcp = run_deterministic("borealis_bio", "zelvoran-hcp-journal-ad")
    patient = run_deterministic("borealis_bio", "zelvoran-patient-brochure")
    assert "BOR-DISC-001" not in {v.rule_id for v in hcp}
    assert "BOR-DISC-001" in {v.rule_id for v in patient}


def test_unreferenced_claims_are_caught_for_the_tenant_that_requires_markers():
    """Borealis requires a numbered reference marker on every efficacy claim; Acme does not."""
    borealis = run_deterministic("borealis_bio", "zelvoran-hcp-unverifiable-claims")
    quotes = [v.quote for v in borealis if v.rule_id == "BOR-REF-001"]
    assert any("two years" in q for q in quotes)
    assert any("cardiovascular" in q for q in quotes)

    acme = run_deterministic("acme_pharma", "zelvoran-hcp-unverifiable-claims")
    assert "BOR-REF-001" not in {v.rule_id for v in acme}
