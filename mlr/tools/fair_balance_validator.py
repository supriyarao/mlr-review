"""**Placeholder** for a company's own fair-balance validator, wrapped as a platform tool.

This stands in for code the customer already owns. Most pharmaceutical companies adopting a system
like this arrive with an existing, qualified fair-balance checker -- a script their regulatory
function has argued over, tuned against their own warning-letter history, and signed off. The point
of this module is to show the shape of plugging that in, not to be that code.

**Why wrap it rather than re-prompt it.** A ratio the company's own code computed is a number an
auditor can recompute by hand. The same ratio estimated by a model is not, however well it reads.
So the arithmetic stays arithmetic and the model is asked only the part that needs judgement:
whether, given these measurements and this copy, a reader meets the risk alongside the benefit.

**What is real here and what is stubbed.** The measurements below are computed for real from the
text and the section map, because a tool that returns constants demonstrates nothing and would hide
the integration's actual failure modes. The one deliberately stubbed value is
``visual_prominence_score``: it depends on type size, colour and position, which text extraction
cannot see (``platform_core/documents.py`` says why), so it returns ``None`` and says so. That is
the honest shape -- a real validator reading the source file would fill it in.

**This does not emit findings.** Its output is *input to the model's judgement*. Deterministic
findings have their own channel: a tenant writes them as rules in ``corpus/tenants/*/rules/`` and
the rules engine reports them. Letting this tool also raise findings would double-report the same
problem under two names and leave the editor to merge them.
"""

from __future__ import annotations

import re
from typing import Any

from platform_core.tools import ToolSpec, register

VERSION = "1.0.0"

# Language that softens a risk statement. A real validator's list would be far longer, tenant
# specific, and maintained by the regulatory function rather than living in code.
_MINIMISING = (
    "mild",
    "minor",
    "rare",
    "rarely",
    "uncommon",
    "generally well tolerated",
    "well tolerated",
    "only",
    "simply",
    "merely",
    "no serious",
)

# Words that mark the start of risk content, for a piece with no section map to read.
_RISK_MARKERS = (
    "contraindicat",
    "warning",
    "precaution",
    "adverse",
    "side effect",
    "serious infection",
    "important safety information",
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENTENCE = re.compile(r"[^.!?]+[.!?]")
_VOWEL_RUN = re.compile(r"[aeiouy]+")


def _syllables(word: str) -> int:
    """Rough syllable count: vowel runs, with a silent trailing 'e' discounted."""
    word = word.lower()
    count = len(_VOWEL_RUN.findall(word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def _grade_level(text: str) -> float | None:
    """Flesch-Kincaid grade level.

    Included because it is a real fair-balance concern and not an invented one: patient-facing
    material is expected to be readable, and risk information written three grades above the
    benefit copy is a way of burying it that word counts do not catch.
    """
    words = _WORD.findall(text)
    sentences = _SENTENCE.findall(text)
    if len(words) < 10 or not sentences:
        return None
    syllables = sum(_syllables(w) for w in words)
    grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    return round(grade, 1)


def _block(text: str, sections: dict[str, Any], key: str) -> str:
    entry = sections.get(key)
    if not isinstance(entry, dict):
        return ""
    start, end = int(entry.get("start", 0)), int(entry.get("end", 0))
    return text[start:end] if end > start else ""


def _first_risk_offset(text: str) -> int | None:
    lowered = text.lower()
    hits = [lowered.find(marker) for marker in _RISK_MARKERS]
    found = [h for h in hits if h >= 0]
    return min(found) if found else None


def validate_fair_balance(
    *,
    document: str,
    doc_meta: dict[str, Any] | None = None,
    claims: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the company's fair-balance measurements over one piece.

    Returns a dict of named checks, each with a value and -- where the company's code has an opinion
    -- a verdict. ``inconclusive`` is a first-class outcome: a measurement that could not be taken
    says so, rather than reporting a zero that reads like a finding.
    """
    sections = (doc_meta or {}).get("sections") or {}
    claims = claims or []

    safety = _block(document, sections, "important_safety_information")
    efficacy = _block(document, sections, "efficacy")
    structured = bool(safety and efficacy)

    checks: dict[str, Any] = {}

    # 1. Volume. The oldest fair-balance measure there is, and still the first one asked about.
    if structured:
        safety_words = len(_WORD.findall(safety))
        efficacy_words = len(_WORD.findall(efficacy))
        ratio = round(safety_words / efficacy_words, 3) if efficacy_words else None
        checks["risk_benefit_word_ratio"] = {
            "safety_words": safety_words,
            "benefit_words": efficacy_words,
            "ratio": ratio,
            "threshold": 0.5,
            "verdict": "fail" if ratio is not None and ratio < 0.5 else "pass",
        }
    else:
        checks["risk_benefit_word_ratio"] = {
            "verdict": "inconclusive",
            "why": "no safety or efficacy section could be located in this piece",
        }

    # 2. Readability gap. Risk written well above the benefit copy's level is buried in plain sight.
    safety_grade = _grade_level(safety) if safety else None
    efficacy_grade = _grade_level(efficacy) if efficacy else None
    if safety_grade is not None and efficacy_grade is not None:
        delta = round(safety_grade - efficacy_grade, 1)
        checks["readability_gap"] = {
            "safety_grade_level": safety_grade,
            "benefit_grade_level": efficacy_grade,
            "delta": delta,
            "threshold": 2.0,
            "verdict": "fail" if delta > 2.0 else "pass",
        }
    else:
        checks["readability_gap"] = {
            "verdict": "inconclusive",
            "why": "one or both blocks were too short to score",
        }

    # 3. How far a reader gets through benefit copy before meeting any risk at all.
    first_risk = _first_risk_offset(document)
    if first_risk is None:
        checks["risk_onset"] = {
            "verdict": "fail",
            "why": "no risk language found anywhere in the piece",
        }
    else:
        share = round(first_risk / len(document), 3) if document else None
        checks["risk_onset"] = {
            "first_risk_char": first_risk,
            "share_of_piece_before_first_risk": share,
            "threshold": 0.6,
            "verdict": "fail" if share is not None and share > 0.6 else "pass",
        }

    # 4. Benefit claims a reader passes before the first risk statement.
    if first_risk is not None and claims:
        before = [
            c
            for c in claims
            if isinstance(c.get("char_span"), (list, tuple))
            and len(c["char_span"]) == 2
            and int(c["char_span"][0]) < first_risk
        ]
        checks["unqualified_benefit_claims"] = {
            "claims_before_first_risk": len(before),
            "total_claims": len(claims),
            "threshold": 2,
            "verdict": "fail" if len(before) > 2 else "pass",
        }
    else:
        checks["unqualified_benefit_claims"] = {
            "verdict": "inconclusive",
            "why": "no claim offsets available, or no risk language to measure against",
        }

    # 5. Language that softens the risk statement.
    hits = sorted(
        {term for term in _MINIMISING if re.search(rf"\b{re.escape(term)}\b", safety, re.I)}
    )
    checks["minimising_language"] = {
        "terms_found": hits,
        "verdict": "fail" if hits else ("pass" if safety else "inconclusive"),
        **({} if safety else {"why": "no safety block to scan"}),
    }

    # 6. The one genuinely stubbed measurement. See the module docstring: type size, colour and
    # position are not in the extracted text, so a text-only validator cannot score them and must
    # not pretend to. A real deployment reads the source artwork here.
    checks["visual_prominence_score"] = {
        "score": None,
        "verdict": "inconclusive",
        "why": (
            "not implemented in this placeholder: requires the source artwork, since type size, "
            "colour and position are not recoverable from extracted text"
        ),
    }

    failed = [name for name, c in checks.items() if c.get("verdict") == "fail"]
    inconclusive = [name for name, c in checks.items() if c.get("verdict") == "inconclusive"]
    return {
        "validator": "acme-fair-balance-suite (placeholder)",
        "version": VERSION,
        "document_had_section_structure": structured,
        "checks": checks,
        "failed": failed,
        "inconclusive": inconclusive,
        # Deliberately not "pass". The company's code measures; it does not clear a piece, and
        # saying "pass" here would invite the model to treat silence as approval.
        "summary": (
            f"{len(failed)} measurement(s) outside threshold, "
            f"{len(inconclusive)} could not be taken"
        ),
    }


SPEC = register(
    ToolSpec(
        name="fair_balance_validator",
        version=VERSION,
        description=(
            "The company's own fair-balance measurements over a promotional piece: risk-to-benefit "
            "volume, readability gap, how far in the first risk statement appears, unqualified "
            "benefit claims, and minimising language. Returns measurements, not findings."
        ),
        fn=validate_fair_balance,
        requires=("document",),
    ),
    replace=True,
)
