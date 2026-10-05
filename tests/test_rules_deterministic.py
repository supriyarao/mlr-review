"""Deterministic rule evaluators: the checks that must never need a model."""

from __future__ import annotations

from datetime import date

import pytest

from platform_core.rules import deterministic
from platform_core.rules.schema import Rule, RuleInput, Segment


def rule(**overrides) -> Rule:
    data = {
        "id": "ACME-BAN-001",
        "tenant_id": "acme_pharma",
        "title": "Prohibited efficacy language",
        "description": "Promotional copy must not claim the product cures the condition.",
        "kind": "deterministic",
        "severity": "blocker",
        "check_type": "business_rule",
        "params": {"type": "banned_terms", "terms": ["cure"]},
        "examples": [],
        "effective_from": date(2026, 1, 1),
    }
    data.update(overrides)
    return Rule.model_validate(data)


def inp(text: str, **overrides) -> RuleInput:
    return RuleInput(text=text, **overrides)


# --- banned_terms -------------------------------------------------------------------------------


def test_a_banned_term_is_found_and_quoted_verbatim():
    """The quote must be exact: the grounding gate looks for it in the submitted piece."""
    violations = deterministic.evaluate(rule(), inp("Zelvoran can cure plaque psoriasis."))
    assert len(violations) == 1
    assert violations[0].quote == "cure"
    start, end = violations[0].char_span
    assert "Zelvoran can cure plaque psoriasis."[start:end] == "cure"


def test_matching_is_case_insensitive():
    assert deterministic.evaluate(rule(), inp("A CURE for psoriasis."))
    assert deterministic.evaluate(rule(), inp("A Cure for psoriasis."))


def test_word_boundaries_prevent_the_classic_false_positive():
    """'cure' must not fire on 'manicure' or 'secure' -- the usual banned-term list bug."""
    for safe in ["Book a manicure.", "A secure portal.", "Procure supplies.", "Curettage."]:
        assert deterministic.evaluate(rule(), inp(safe)) == [], f"false positive on {safe!r}"


def test_every_occurrence_is_reported():
    violations = deterministic.evaluate(rule(), inp("A cure. Truly a cure."))
    assert len(violations) == 2
    assert [v.char_span[0] for v in violations] == sorted(v.char_span[0] for v in violations)


def test_overlapping_configured_terms_do_not_double_report():
    r = rule(params={"type": "banned_terms", "terms": ["cure", "cure"]})
    assert len(deterministic.evaluate(r, inp("A cure."))) == 1


def test_allowed_contexts_exempt_a_legitimate_use():
    """'cure' is banned in copy but legitimate inside a quoted study title."""
    r = rule(
        params={
            "type": "banned_terms",
            "terms": ["cure"],
            # Anything inside double quotes is a quoted title.
            "allowed_contexts": [r'"[^"]*"'],
        }
    )
    assert deterministic.evaluate(r, inp('See "Toward a cure for psoriasis" (2026).')) == []
    # Outside the quotes it still fires.
    violations = deterministic.evaluate(r, inp('Zelvoran is a cure. See "Toward a cure" (2026).'))
    assert len(violations) == 1
    assert violations[0].char_span[0] < 20


def test_regex_terms_are_supported():
    r = rule(params={"type": "banned_terms", "regex": True, "terms": [r"100\s*%\s*safe"]})
    assert deterministic.evaluate(r, inp("It is 100 % safe."))
    assert deterministic.evaluate(r, inp("It is well tolerated.")) == []


def test_a_bad_regex_is_an_evaluation_error():
    r = rule(params={"type": "banned_terms", "regex": True, "terms": ["(unclosed"]})
    with pytest.raises(deterministic.RuleEvaluationError, match="not a valid pattern"):
        deterministic.evaluate(r, inp("text"))


def test_an_empty_term_list_is_an_evaluation_error():
    with pytest.raises(deterministic.RuleEvaluationError, match="non-empty params.terms"):
        deterministic.evaluate(rule(params={"type": "banned_terms", "terms": []}), inp("text"))


def test_the_violation_cites_the_rule_in_the_owners_own_words():
    """A reviewer must be told which rule was broken, as the rule owner wrote it."""
    v = deterministic.evaluate(rule(), inp("A cure."))[0]
    assert v.citation_text == "Promotional copy must not claim the product cures the condition."
    assert v.citation_kind == "rule"
    assert v.citation_source_id == "ACME-BAN-001 v1"
    assert v.rule_id == "ACME-BAN-001"
    assert v.rule_version == 1
    assert v.severity == "blocker"


# --- required_text ------------------------------------------------------------------------------


def disclaimer_rule(**params) -> Rule:
    base = {
        "type": "required_text",
        "text": "Talk to your doctor about whether this is right for you.",
    }
    base.update(params)
    return rule(
        id="BOR-DISC-001",
        tenant_id="borealis_bio",
        title="Patient disclaimer",
        description="Patient-facing materials must carry the approved patient disclaimer verbatim.",
        params=base,
    )


def test_required_text_present_is_a_pass():
    text = "Nexapril helps. Talk to your doctor about whether this is right for you."
    assert deterministic.evaluate(disclaimer_rule(), inp(text)) == []


def test_required_text_survives_reflowing():
    """A correct disclaimer broken across lines is still the disclaimer."""
    text = "Talk to your doctor\n  about whether this\tis right for you."
    assert deterministic.evaluate(disclaimer_rule(), inp(text)) == []


def test_required_text_missing_is_an_absence_with_no_quote():
    """An absence has nothing to quote, which the grounding gate must handle."""
    violations = deterministic.evaluate(disclaimer_rule(), inp("Nexapril helps."))
    assert len(violations) == 1
    assert violations[0].quote == ""
    assert violations[0].is_absence is True
    assert "missing" in violations[0].message
    assert "Talk to your doctor" in violations[0].suggested_fix


def test_exact_match_mode_rejects_reflowed_text():
    r = disclaimer_rule(match="exact")
    assert deterministic.evaluate(
        r, inp("Talk to your doctor\nabout whether this is right for you.")
    )


def test_required_text_can_be_scoped_to_a_section():
    r = disclaimer_rule(section="important_safety_information")
    text = "Benefits here. SAFETY: Talk to your doctor about whether this is right for you."
    meta = {"sections": {"important_safety_information": {"start": 14, "end": len(text)}}}
    assert deterministic.evaluate(r, inp(text, doc_meta=meta)) == []


def test_required_text_in_the_wrong_section_still_fails():
    """Present somewhere is not the same as present where the tenant requires it."""
    r = disclaimer_rule(section="important_safety_information")
    text = "Talk to your doctor about whether this is right for you. SAFETY: see label."
    meta = {"sections": {"important_safety_information": {"start": 57, "end": len(text)}}}
    violations = deterministic.evaluate(r, inp(text, doc_meta=meta))
    assert len(violations) == 1


def test_a_missing_required_section_is_reported_clearly():
    r = disclaimer_rule(section="important_safety_information")
    violations = deterministic.evaluate(r, inp("No safety block at all."))
    assert "not present" in violations[0].message


# --- trademark_first_mention --------------------------------------------------------------------


def tm_rule() -> Rule:
    return rule(
        id="ACME-TM-001",
        severity="advisory",
        title="Trademark on first mention",
        description="The brand name must carry a registered trademark symbol on first mention.",
        params={"type": "trademark_first_mention", "brand": "Zelvoran"},
    )


def test_trademark_present_on_first_mention_is_a_pass():
    assert deterministic.evaluate(tm_rule(), inp("Zelvoran® is indicated for adults.")) == []


def test_trademark_missing_on_first_mention_fires():
    violations = deterministic.evaluate(tm_rule(), inp("Zelvoran is indicated for adults."))
    assert len(violations) == 1
    assert violations[0].quote == "Zelvoran"
    assert violations[0].severity == "advisory"


def test_only_the_first_mention_is_checked():
    """House style marks the brand once and drops it; flagging later mentions would be noise."""
    text = "Zelvoran® is indicated for adults. Zelvoran is taken daily. Ask about Zelvoran."
    assert deterministic.evaluate(tm_rule(), inp(text)) == []


def test_a_piece_that_never_names_the_brand_is_not_a_violation():
    assert deterministic.evaluate(tm_rule(), inp("Ask your doctor about treatment options.")) == []


def test_a_trademark_symbol_is_accepted_too():
    assert deterministic.evaluate(tm_rule(), inp("Zelvoran™ is indicated.")) == []


# --- required_section ---------------------------------------------------------------------------


def section_rule(**params) -> Rule:
    base = {"type": "required_section", "section": "important_safety_information"}
    base.update(params)
    return rule(
        id="BOR-ISI-001",
        tenant_id="borealis_bio",
        title="Safety information placement",
        description="Safety information must appear close to the first efficacy claim.",
        params=base,
    )


def test_a_present_section_passes():
    meta = {"sections": {"important_safety_information": {"start": 10, "end": 200}}}
    assert deterministic.evaluate(section_rule(), inp("x" * 300, doc_meta=meta)) == []


def test_a_missing_section_fires():
    violations = deterministic.evaluate(section_rule(), inp("copy", doc_meta={"sections": {}}))
    assert "missing" in violations[0].message


def test_a_section_placed_too_far_away_fires():
    """'Present but buried' is the realistic failure, not outright absence."""
    r = section_rule(near_section="efficacy", max_distance_chars=400)
    meta = {
        "sections": {
            "efficacy": {"start": 100, "end": 400},
            "important_safety_information": {"start": 5000, "end": 5200},
        }
    }
    violations = deterministic.evaluate(r, inp("x" * 6000, doc_meta=meta))
    assert len(violations) == 1
    assert "beyond the 400 allowed" in violations[0].message


def test_a_section_placed_nearby_passes():
    r = section_rule(near_section="efficacy", max_distance_chars=400)
    meta = {
        "sections": {
            "efficacy": {"start": 100, "end": 300},
            "important_safety_information": {"start": 320, "end": 500},
        }
    }
    assert deterministic.evaluate(r, inp("x" * 600, doc_meta=meta)) == []


def test_ordering_can_be_required():
    r = section_rule(before_section="footnotes")
    meta = {
        "sections": {
            "footnotes": {"start": 100, "end": 200},
            "important_safety_information": {"start": 900, "end": 1000},
        }
    }
    violations = deterministic.evaluate(r, inp("x" * 1100, doc_meta=meta))
    assert "appears after" in violations[0].message


# --- max_claims_without_reference ---------------------------------------------------------------


def ref_rule(**params) -> Rule:
    base = {"type": "max_claims_without_reference"}
    base.update(params)
    return rule(
        id="BOR-REF-001",
        tenant_id="borealis_bio",
        title="Claims need references",
        description="Every efficacy claim must carry a reference marker to its supporting evidence.",
        params=base,
    )


def test_claims_with_reference_markers_pass():
    """Deterministic given the segments; identifying claims happens upstream."""
    data = inp(
        "copy",
        segments=[
            Segment(id="c1", text="Nexapril reduced events by 22 percent [1]."),
            Segment(id="c2", text="Sustained over 36 months (Ref 2)."),
        ],
    )
    assert deterministic.evaluate(ref_rule(), data) == []


def test_a_claim_without_a_reference_marker_fires_and_names_the_segment():
    data = inp(
        "copy",
        segments=[
            Segment(id="c1", text="Nexapril reduced events by 22 percent [1]."),
            Segment(id="c2", text="Nexapril is the obvious first choice."),
        ],
    )
    violations = deterministic.evaluate(ref_rule(), data)
    assert len(violations) == 1
    assert violations[0].segment_id == "c2"
    assert "no reference marker" in violations[0].message


def test_superscript_markers_count_as_references():
    data = inp("copy", segments=[Segment(id="c1", text="Reduced events by 22 percent¹.")])
    assert deterministic.evaluate(ref_rule(), data) == []


def test_a_tenant_can_require_its_own_marker_format():
    r = ref_rule(marker_pattern=r"\[\d+\]")
    data = inp("copy", segments=[Segment(id="c1", text="Reduced events (Ref 2).")])
    assert deterministic.evaluate(r, data)


def test_an_allowance_can_be_configured():
    r = ref_rule(max_allowed=1)
    data = inp("copy", segments=[Segment(id="c1", text="No marker here.")])
    assert deterministic.evaluate(r, data) == []


def test_no_segments_means_nothing_to_check():
    assert deterministic.evaluate(ref_rule(), inp("copy")) == []


# --- the evaluate_all contract ------------------------------------------------------------------


def test_a_misconfigured_rule_fails_closed_rather_than_being_skipped():
    """A broken blocker must still reach a human, not silently stop being checked."""
    broken = rule(id="ACME-BAN-002", params={"type": "banned_terms", "terms": []})
    violations = deterministic.evaluate_all([rule(), broken], inp("A cure."))
    messages = {v.rule_id: v.message for v in violations}
    assert "ACME-BAN-001" in messages
    assert "needs attention" in messages["ACME-BAN-002"]
    # The misconfigured rule keeps its severity, so it escalates like any blocker.
    assert next(v for v in violations if v.rule_id == "ACME-BAN-002").severity == "blocker"
    # And is marked zero-confidence so the editor can tell it apart from a real finding.
    assert next(v for v in violations if v.rule_id == "ACME-BAN-002").confidence == 0


def test_evaluate_refuses_a_non_deterministic_rule():
    llm = rule(kind="llm", params={})
    with pytest.raises(deterministic.RuleEvaluationError, match="not deterministic"):
        deterministic.evaluate(llm, inp("text"))


def test_deterministic_evaluation_makes_no_model_call(monkeypatch):
    """The whole point: these run with no API key, no Ollama, no cost, no latency."""
    import platform_core.llm as llm_module

    def explode(*args, **kwargs):
        raise AssertionError("a deterministic rule must never construct a model")

    monkeypatch.setattr(llm_module, "get_chat_model", explode)
    assert deterministic.evaluate_all([rule(), tm_rule(), ref_rule()], inp("A cure."))
