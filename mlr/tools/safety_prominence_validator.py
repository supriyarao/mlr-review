"""**Placeholder** for a company's own safety-information validator, wrapped as a platform tool.

The sibling of :mod:`mlr.tools.fair_balance_validator`, and the second of the two the presentation
check calls. Same reasoning: a company arriving with this platform already owns the code that
decides whether an ISI block is complete, and that code is more auditable than a model's opinion of
the same question.

The two are kept apart rather than folded into one validator because they answer different
questions and a customer would own them separately. Fair balance is *relative* -- is the risk given
comparable weight to the benefit. This one is *absolute* -- is the required safety information
there, complete, and somewhere a reader will meet it. A piece can pass either and fail the other:
one line of risk under a one-line claim is balanced and useless; six paragraphs of complete ISI on
the last page is complete and buried.

Same honesty rules as the sibling. Measurements are computed for real where the text allows it, the
parts that need the artwork say they need the artwork, and nothing here emits a finding -- the
output is input to the model's judgement.
"""

from __future__ import annotations

import re
from typing import Any

from platform_core.tools import ToolSpec, register

VERSION = "1.0.0"

# Safety content an ISI block may be expected to carry, and the words that count as mentioning it.
#
# Which of these are *required* is not decided here. It is read off the approved label for the
# product in hand, because the answer differs by product: Declasal carries a Boxed Warning and
# Zelvoran does not, so requiring every piece to mention one would mark every correct Zelvoran
# piece incomplete. The label is the thing that knows.
_TOPIC_MARKERS: dict[str, tuple[str, ...]] = {
    "boxed_warning": ("boxed warning", "black box", "warning:"),
    "contraindications": ("contraindicat",),
    "warnings_and_precautions": ("warning", "precaution"),
    "adverse_reactions": ("adverse reaction", "adverse event", "side effect"),
}

# How a label section announces each topic. Matched against section headings, not body text, so a
# label that merely mentions the phrase in passing does not make the topic required.
_LABEL_HEADINGS: dict[str, tuple[str, ...]] = {
    "boxed_warning": ("boxed warning",),
    "contraindications": ("contraindication",),
    "warnings_and_precautions": ("warnings and precautions",),
    "adverse_reactions": ("adverse reactions",),
}

# What to require when no label was supplied. The three that essentially every label carries --
# deliberately not boxed_warning, which is the one that varies.
_DEFAULT_REQUIRED = ("contraindications", "warnings_and_precautions", "adverse_reactions")


def required_topics(label_headings: list[str] | None) -> list[str]:
    """Which safety topics this product's label obliges a piece to carry.

    Falls back to the common three when no label is available, rather than requiring nothing: a
    missing label must not quietly turn the completeness check into a pass.
    """
    if not label_headings:
        return list(_DEFAULT_REQUIRED)

    lowered = [h.lower() for h in label_headings]
    found = [
        topic
        for topic, markers in _LABEL_HEADINGS.items()
        if any(marker in heading for heading in lowered for marker in markers)
    ]
    return found or list(_DEFAULT_REQUIRED)


# Signposting a reader needs in order to go and find the rest.
_SIGNPOSTS: dict[str, tuple[str, ...]] = {
    "full_prescribing_information": ("full prescribing information", "prescribing information"),
    "safety_heading": ("important safety information", "safety information"),
}

# Below this, a safety block is a gesture rather than a disclosure. A real validator's figure would
# come from the tenant's own standard; this one is a round number and says so.
MIN_USEFUL_SAFETY_CHARS = 200

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _block(text: str, sections: dict[str, Any], key: str) -> str:
    entry = sections.get(key)
    if not isinstance(entry, dict):
        return ""
    start, end = int(entry.get("start", 0)), int(entry.get("end", 0))
    return text[start:end] if end > start else ""


def validate_safety_prominence(
    *,
    document: str,
    doc_meta: dict[str, Any] | None = None,
    claims: list[dict[str, Any]] | None = None,
    label_headings: list[str] | None = None,
) -> dict[str, Any]:
    """Run the company's safety-information measurements over one piece.

    ``label_headings`` are the section headings of the approved label, which decide what this
    piece is obliged to carry. Omitting them is allowed and falls back to the common three.
    """
    meta = doc_meta or {}
    sections = meta.get("sections") or {}
    safety = _block(document, sections, "important_safety_information")
    lowered = document.lower()

    checks: dict[str, Any] = {}

    # 1. Is there a safety block at all, and is it long enough to be worth reading?
    if not sections:
        checks["safety_block_present"] = {
            "verdict": "inconclusive",
            "why": "no readable section structure, so no block could be located",
        }
    elif not safety:
        checks["safety_block_present"] = {
            "verdict": "fail",
            "why": "the piece has sections, and none of them is safety information",
        }
    else:
        chars = len(safety)
        checks["safety_block_present"] = {
            "chars": chars,
            "words": len(_WORD.findall(safety)),
            "threshold_chars": MIN_USEFUL_SAFETY_CHARS,
            "verdict": "fail" if chars < MIN_USEFUL_SAFETY_CHARS else "pass",
            **(
                {"why": "present, but too short to let a reader weigh the risk"}
                if chars < MIN_USEFUL_SAFETY_CHARS
                else {}
            ),
        }

    # 2. Completeness. Scanned over the whole piece, not just the block: a contraindication stated
    # somewhere sensible but outside the ISI heading is still stated.
    required = required_topics(label_headings)
    missing = [
        topic
        for topic in required
        if not any(marker in lowered for marker in _TOPIC_MARKERS[topic])
    ]
    checks["required_topics"] = {
        "expected": required,
        "missing": missing,
        "source": "approved label" if label_headings else "default (no label supplied)",
        "verdict": "fail" if missing else "pass",
    }

    # 3. Signposting.
    absent = [
        name
        for name, markers in _SIGNPOSTS.items()
        if not any(marker in lowered for marker in markers)
    ]
    checks["signposting"] = {
        "missing": absent,
        "verdict": "fail" if absent else "pass",
    }

    # 4. Burial. The realistic failure: present, complete, and where nobody reads it. These numbers
    # come from ingest's section map, which is why they go inconclusive on an unstructured piece.
    if not sections or not safety:
        checks["placement"] = {
            "verdict": "inconclusive",
            "why": "no safety block located, so placement could not be measured",
        }
    else:
        after_refs = bool(meta.get("safety_after_references"))
        after_notes = bool(meta.get("safety_after_footnotes"))
        distance = meta.get("safety_distance_chars")
        buried = after_refs or after_notes
        checks["placement"] = {
            "after_references": after_refs,
            "after_footnotes": after_notes,
            "chars_from_efficacy": distance,
            "section_order": meta.get("section_order") or [],
            "verdict": "fail" if buried else "pass",
            **(
                {"why": "safety information sits below the references or the footnotes"}
                if buried
                else {}
            ),
        }

    # 5. The measurement that needs the artwork. Type size relative to the body, colour contrast and
    # whether the block falls below the fold are the things a reviewer means by "prominent", and
    # none of them survive text extraction. See platform_core/documents.py.
    checks["legibility"] = {
        "relative_type_size": None,
        "contrast_ratio": None,
        "verdict": "inconclusive",
        "why": (
            "not implemented in this placeholder: type size and contrast require the source "
            "artwork, not extracted text"
        ),
    }

    failed = [name for name, c in checks.items() if c.get("verdict") == "fail"]
    inconclusive = [name for name, c in checks.items() if c.get("verdict") == "inconclusive"]
    return {
        "validator": "acme-isi-completeness-suite (placeholder)",
        "version": VERSION,
        "checks": checks,
        "failed": failed,
        "inconclusive": inconclusive,
        "summary": (
            f"{len(failed)} measurement(s) outside threshold, "
            f"{len(inconclusive)} could not be taken"
        ),
    }


SPEC = register(
    ToolSpec(
        name="safety_prominence_validator",
        version=VERSION,
        description=(
            "The company's own safety-information measurements over a promotional piece: whether a "
            "safety block exists and is long enough to be useful, whether the required topics and "
            "signposting are present, and whether the block is buried below the references or "
            "footnotes. Returns measurements, not findings."
        ),
        fn=validate_safety_prominence,
        requires=("document",),
    ),
    replace=True,
)
