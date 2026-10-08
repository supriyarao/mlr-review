"""What counts as grounding a finding in a measurement.

The gate is the thing standing between a model's assertion and a reviewer's screen, so the
rules about what verifies a finding are worth testing directly rather than only through a
graph run.
"""

from __future__ import annotations


def test_a_categorical_measurement_grounds_without_a_figure():
    """A completeness measurement is a list of names, not a number.

    `missing: ["contraindications"]` is as exact a measurement as a ratio, and requiring a digit
    sent every correctly-identified omission to the reviewer marked "could not be verified" -- the
    system reporting itself unsure of the one thing it had measured precisely.
    """
    from mlr.nodes.grounding_gate import _quotes_measurement

    reported = (
        '{"checks": {"required_topics": {"missing": ["contraindications", '
        '"warnings_and_precautions"], "verdict": "fail"}}}'
    )
    # Reordered and re-punctuated relative to the tool's sorted-key JSON, which is how a model
    # actually cites it.
    assert _quotes_measurement(
        '"missing": ["contraindications", "warnings_and_precautions"], "verdict": "fail"', reported
    )


def test_a_measurement_quote_the_tool_never_emitted_does_not_ground():
    """The anti-fabrication property has to survive dropping the figure requirement."""
    from mlr.nodes.grounding_gate import _quotes_measurement

    reported = '{"checks": {"required_topics": {"missing": ["contraindications"]}}}'
    assert not _quotes_measurement('"missing": ["boxed_warning", "hepatotoxicity"]', reported)


def test_a_measurement_quote_too_thin_to_be_meaningful_does_not_ground():
    """A fragment that would appear in any tool's JSON must not ground a finding by coincidence."""
    from mlr.nodes.grounding_gate import _quotes_measurement

    assert not _quotes_measurement('"verdict"', '{"verdict": "fail"}')
