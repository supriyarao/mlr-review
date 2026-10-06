"""Calling code the platform did not write.

Two things are under test. The registry's contract -- a tool failure must be a *result*, never an
exception, because a review that dies when a helper throws is worse than one that proceeds without
the measurement. And the placeholder validator's behaviour, which stands in for a customer's own
fair-balance code and has to discriminate between a buried ISI and a balanced piece to be worth
demonstrating at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlr.nodes.checks.runner import _format_tools, _run_tools
from mlr.nodes.ingest import build_doc_meta, normalize_document, parse_front_matter
from mlr.tools.fair_balance_validator import validate_fair_balance
from platform_core import tools

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def piece(name: str) -> tuple[str, dict]:
    _, body = parse_front_matter((CORPUS / "promo" / f"{name}.md").read_text(encoding="utf-8"))
    document = normalize_document(body)
    return document, build_doc_meta(document)


# --- The registry -----------------------------------------------------------------------------


def test_a_tool_that_raises_comes_back_as_a_failed_result():
    """The contract. A review must not die because somebody else's code threw."""
    tools.register(
        tools.ToolSpec(
            name="_exploding",
            version="0.0.1",
            description="Always raises.",
            fn=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
        ),
        replace=True,
    )
    result = tools.call("_exploding", {})
    assert result.ok is False
    assert "boom" in result.error
    assert result.version == "0.0.1"


def test_an_unknown_tool_is_a_failed_result_too():
    result = tools.call("_nothing_is_registered_here", {"document": "x"})
    assert result.ok is False
    assert "no tool named" in result.error


def test_a_missing_required_input_is_caught_before_the_tool_runs():
    """A clear error from the registry beats a KeyError from inside the customer's code."""
    result = tools.call("fair_balance_validator", {})
    assert result.ok is False
    assert "document" in result.error


def test_registering_the_same_name_twice_is_refused():
    spec = tools.ToolSpec(name="_dup", version="1", description="x", fn=lambda **_: {})
    tools.register(spec, replace=True)
    with pytest.raises(tools.ToolError, match="already registered"):
        tools.register(spec)


def test_the_result_carries_the_version_that_produced_it():
    """A measurement that cannot say which version of which tool produced it is not evidence."""
    document, meta = piece("zelvoran-hcp-journal-ad")
    result = tools.call("fair_balance_validator", {"document": document, "doc_meta": meta})
    assert result.ok
    assert result.version == "1.0.0"
    assert result.label() == "fair_balance_validator v1.0.0"


# --- The placeholder validator ------------------------------------------------------------------


def test_it_tells_a_buried_isi_apart_from_a_balanced_piece():
    """If it cannot do this, wrapping it demonstrates nothing."""
    buried, buried_meta = piece("zelvoran-hcp-isi-buried")
    clean, clean_meta = piece("zelvoran-hcp-journal-ad")

    bad = validate_fair_balance(document=buried, doc_meta=buried_meta)
    good = validate_fair_balance(document=clean, doc_meta=clean_meta)

    assert "risk_benefit_word_ratio" in bad["failed"]
    assert "risk_onset" in bad["failed"]
    assert good["failed"] == []


def test_a_measurement_that_cannot_be_taken_says_so_rather_than_passing():
    """Same principle as the layout block: an unavailable number is not a zero and not a pass."""
    result = validate_fair_balance(document="Short copy with no structure.", doc_meta={})
    assert "risk_benefit_word_ratio" in result["inconclusive"]
    assert result["checks"]["risk_benefit_word_ratio"]["verdict"] == "inconclusive"
    assert "why" in result["checks"]["risk_benefit_word_ratio"]


def test_the_visual_score_is_openly_unimplemented():
    """Type size and colour are not in extracted text. Returning a number here would be a lie."""
    document, meta = piece("zelvoran-hcp-journal-ad")
    check = validate_fair_balance(document=document, doc_meta=meta)["checks"][
        "visual_prominence_score"
    ]
    assert check["score"] is None
    assert check["verdict"] == "inconclusive"
    assert "not implemented" in check["why"]


def test_it_never_reports_an_overall_pass():
    """The company's code measures. Clearing a piece is a person's job, and silence is not approval."""
    document, meta = piece("zelvoran-hcp-journal-ad")
    result = validate_fair_balance(document=document, doc_meta=meta)
    assert result["failed"] == []
    assert "pass" not in result["summary"].lower()


def test_minimising_language_is_caught_in_the_safety_block_only():
    document = "# Efficacy\n\nIt works only in adults.\n\n# Important Safety Information\n\nSide effects were generally well tolerated and rare.\n"
    result = validate_fair_balance(document=document, doc_meta=build_doc_meta(document))
    found = result["checks"]["minimising_language"]["terms_found"]
    assert "well tolerated" in found
    assert "rare" in found


# --- Reaching the model --------------------------------------------------------------------------


def test_only_the_presentation_node_is_given_the_validators():
    document, meta = piece("zelvoran-hcp-isi-buried")
    state = {"document": document, "doc_meta": meta}
    results = _run_tools("presentation", state, [])
    assert [r.tool for r in results] == [
        "safety_prominence_validator",
        "fair_balance_validator",
    ]
    assert _run_tools("off_label", state, []) == []
    assert _run_tools("substantiation", state, []) == []


def test_the_block_the_model_sees_names_each_tool_and_its_version():
    document, meta = piece("zelvoran-hcp-isi-buried")
    block = _format_tools(_run_tools("presentation", {"document": document, "doc_meta": meta}, []))
    assert '<tool name="fair_balance_validator" version="1.0.0">' in block
    assert '<tool name="safety_prominence_validator" version="1.0.0">' in block
    assert "risk_benefit_word_ratio" in block
    assert "placement" in block


def test_the_safety_validator_catches_a_buried_but_complete_isi():
    """The case that is this concern's whole point: present, complete, and where nobody reads it."""
    from mlr.tools.safety_prominence_validator import validate_safety_prominence

    document, meta = piece("zelvoran-hcp-isi-buried")
    result = validate_safety_prominence(document=document, doc_meta=meta)
    assert "placement" in result["failed"]
    assert result["checks"]["placement"]["after_references"] is True


def test_the_two_concerns_can_disagree_about_the_same_piece():
    """One node, two validators, and they are not measuring the same thing.

    The journal ad is balanced -- the fair-balance validator finds nothing -- which says nothing
    about whether its safety block is complete. Folding the two into one measurement would lose
    exactly this.
    """
    from mlr.tools.safety_prominence_validator import validate_safety_prominence

    document, meta = piece("zelvoran-hcp-journal-ad")
    assert validate_fair_balance(document=document, doc_meta=meta)["failed"] == []
    safety = validate_safety_prominence(document=document, doc_meta=meta)
    assert safety["checks"]["safety_block_present"]["verdict"] == "pass"


def test_a_failed_call_is_shown_to_the_model_rather_than_dropped():
    """Silence would read as "nothing to report", which is the mistake this whole fix is about."""
    block = _format_tools([tools.call("fair_balance_validator", {})])
    assert "UNAVAILABLE" in block
    assert "document" in block


def test_no_tools_means_no_block_at_all():
    assert _format_tools([]) == ""
