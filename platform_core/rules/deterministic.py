"""Deterministic rule evaluators: plain code, no model call, ever.

Why these matter more than the LLM rules: a banned-term list is not a judgement call. "Does this
piece contain the word 'cure'?" has one correct answer, it is cheap, it is reproducible, and an
auditor can verify it by reading the rule. Routing that through a model would add cost, latency and
nondeterminism to a question that regex answers perfectly.

So the design rule here is: anything that *can* be deterministic *is*, and the model is reserved for
the genuinely interpretive rules. An auditor can replay every one of these checks by hand.

Each evaluator takes a :class:`Rule` and a :class:`RuleInput` and returns
:class:`RuleViolation` objects. None of them import anything that can make a network call.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from platform_core.rules.schema import Rule, RuleInput, RuleViolation

# Reference markers an efficacy claim might carry: [1], (Ref 3), superscript digits, "see ref. 2".
DEFAULT_REFERENCE_MARKER = (
    r"(\[\d+\]|\(\s*(?:ref|reference)[^)]*\)|[¹²³⁰-⁹]+|\bref(?:erence)?\.?\s*\d+)"
)

# Characters that mark a trademark on first mention.
TRADEMARK_MARKS = ("®", "™")


class RuleEvaluationError(Exception):
    """Raised when a rule's params are malformed in a way validation should have caught."""


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace runs, for comparisons that should survive reflowing."""
    return " ".join(text.split())


def _violation(rule: Rule, **kwargs: Any) -> RuleViolation:
    """Build a violation pre-filled from the rule, so evaluators stay about their own logic."""
    return RuleViolation(
        rule_id=rule.id,
        rule_version=rule.version,
        tenant_id=rule.tenant_id,
        severity=rule.severity,
        check_type=rule.check_type,
        citation_text=rule.description,
        citation_kind="rule",
        citation_source_id=rule.label(),
        **kwargs,
    )


def _compile_term(term: str, *, is_regex: bool) -> re.Pattern[str]:
    """Compile one banned term.

    Plain terms get word boundaries so "cure" does not fire on "manicure" or "secure" -- the single
    most common false-positive in banned-term lists. A term containing whitespace is wrapped at its
    outer edges only.
    """
    pattern = term if is_regex else re.escape(term)
    if not is_regex:
        pattern = rf"\b{pattern}\b"
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise RuleEvaluationError(f"term {term!r} is not a valid pattern: {exc}") from exc


def _allowed_spans(text: str, contexts: Iterable[str]) -> list[tuple[int, int]]:
    """Spans where a banned term is permitted.

    The motivating case: "cure" is banned in promotional copy but legitimate inside a quoted study
    title or a verbatim label excerpt. Expressing that as "ignore hits inside these spans" keeps the
    exemption readable in YAML and auditable -- the alternative, a hand-tuned negative lookahead,
    is unreadable and gets copied wrongly.
    """
    spans: list[tuple[int, int]] = []
    for context in contexts:
        try:
            pattern = re.compile(context, re.IGNORECASE | re.DOTALL)
        except re.error as exc:
            raise RuleEvaluationError(
                f"allowed_context {context!r} is not a valid pattern: {exc}"
            ) from exc
        spans.extend(m.span() for m in pattern.finditer(text))
    return spans


def _inside(span: tuple[int, int], allowed: list[tuple[int, int]]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in allowed)


def check_banned_terms(rule: Rule, data: RuleInput) -> list[RuleViolation]:
    """Terms the tenant forbids in promotional copy.

    params: ``terms`` (list), ``regex`` (bool, treat terms as patterns),
    ``allowed_contexts`` (list of patterns where a hit is exempt).
    """
    terms = [str(t) for t in rule.params.get("terms", []) if str(t).strip()]
    if not terms:
        raise RuleEvaluationError(f"{rule.label()}: banned_terms needs a non-empty params.terms")

    is_regex = bool(rule.params.get("regex", False))
    allowed = _allowed_spans(data.text, rule.params.get("allowed_contexts", []) or [])

    violations: list[RuleViolation] = []
    seen: set[tuple[int, int]] = set()
    for term in terms:
        for match in _compile_term(term, is_regex=is_regex).finditer(data.text):
            span = match.span()
            # One violation per position, even if two configured terms overlap on it.
            if span in seen or _inside(span, allowed):
                continue
            seen.add(span)
            violations.append(
                _violation(
                    rule,
                    quote=match.group(0),
                    char_span=span,
                    message=f"Uses the prohibited term {match.group(0)!r}.",
                    suggested_fix=str(rule.params.get("suggested_fix", "")),
                )
            )
    return sorted(violations, key=lambda v: v.char_span)


def _section_text(data: RuleInput, section: str) -> str | None:
    """Text of a named section, from the use case's doc_meta.

    ``doc_meta["sections"]`` is expected to be a mapping of section name to
    ``{"start": int, "end": int}``. Returns None when the section is absent, which callers treat as
    a different outcome from "present but wrong".
    """
    sections = data.doc_meta.get("sections") or {}
    entry = sections.get(section)
    if not isinstance(entry, dict):
        return None
    start, end = int(entry.get("start", 0)), int(entry.get("end", 0))
    return data.text[start:end] if end > start else None


def check_required_text(rule: Rule, data: RuleInput) -> list[RuleViolation]:
    """Text that must appear -- a mandatory disclaimer, a fair-balance statement.

    params: ``text`` (required), ``section`` (restrict the search to one section),
    ``match`` ("normalized" default, or "exact").

    Compares whitespace-normalised by default, because a disclaimer that is correct but reflowed
    across different line breaks is still the disclaimer. ``match: exact`` is available for a tenant
    who genuinely requires character-for-character reproduction.
    """
    required = str(rule.params.get("text", ""))
    if not required.strip():
        raise RuleEvaluationError(f"{rule.label()}: required_text needs params.text")

    section = str(rule.params.get("section", "") or "")
    if section:
        # Same distinction as check_required_section: no section map at all means the document had
        # no structure this code could read, which is not the same as the section being absent.
        if not (data.doc_meta.get("sections") or {}):
            return [
                _violation(
                    rule,
                    message=(
                        f"Could not check the required text in the {section!r} section: this "
                        f"document has no readable section structure."
                    ),
                    confidence=20,
                )
            ]
        haystack = _section_text(data, section)
        if haystack is None:
            return [
                _violation(
                    rule,
                    message=(
                        f"Required text must appear in the {section!r} section, which is not "
                        f"present in this piece."
                    ),
                )
            ]
    else:
        haystack = data.text

    exact = str(rule.params.get("match", "normalized")) == "exact"
    found = (
        required in haystack
        if exact
        else normalize_whitespace(required) in normalize_whitespace(haystack)
    )
    if found:
        return []

    where = f" in the {section!r} section" if section else ""
    return [
        _violation(
            rule,
            # No quote: this is an absence. The grounding gate treats it accordingly.
            message=f"Required text is missing{where}.",
            suggested_fix=f"Add: {required}",
        )
    ]


def check_trademark_first_mention(rule: Rule, data: RuleInput) -> list[RuleViolation]:
    """Brand name must carry (R) or (TM) on its first mention.

    params: ``brand`` (required), ``window`` (characters after the mention to search, default 3),
    ``generic_name`` (defaults to the brand lower-cased).

    Only the *first* mention is checked. House style almost universally requires the mark once and
    then drops it, so flagging later mentions would produce a finding on every correctly written
    piece -- the kind of false positive that trains reviewers to ignore a whole check.

    The subtlety is telling a brand mention from the generic name. Pharmaceutical copy writes
    "Zelvoran(R) (zelvoran)": the capitalised form is the trademark and needs the symbol, the
    lower-case form is the international non-proprietary name and does not. Two naive approaches
    both fail, and both were tried here:

    * **Case-insensitive** matching finds the generic name inside "Zelvoran(R) (zelvoran)" and
      reports a violation on correctly marked copy.
    * **Case-sensitive** matching misses "ZELVORAN", which labels and quoted safety information use
      constantly -- so an all-caps piece carrying no symbol at all passes silently, and when an
      all-caps mention is followed by a capitalised one the violation is reported against the
      *second* mention, breaking the first-mention semantics outright.

    So: find every mention case-insensitively, discard the ones written in the all-lower-case generic
    form, and check the first remaining one. A piece that only ever uses the generic name is not
    using the trademark and needs no symbol.
    """
    brand = str(rule.params.get("brand", ""))
    if not brand.strip():
        raise RuleEvaluationError(f"{rule.label()}: trademark_first_mention needs params.brand")

    generic = str(rule.params.get("generic_name", "") or brand).lower()
    window = int(rule.params.get("window", 3))
    pattern = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE)

    first_brand_mention = None
    for match in pattern.finditer(data.text):
        # The all-lower-case form is the generic name, not the trademark.
        if match.group(0) == generic:
            continue
        first_brand_mention = match
        break

    if first_brand_mention is None:
        # The brand is never used as a brand, so there is no first mention to mark.
        return []

    trailing = data.text[first_brand_mention.end() : first_brand_mention.end() + max(window, 1)]
    if any(mark in trailing for mark in TRADEMARK_MARKS):
        return []

    return [
        _violation(
            rule,
            quote=first_brand_mention.group(0),
            char_span=first_brand_mention.span(),
            message=f"First mention of {brand} does not carry a trademark symbol.",
            suggested_fix=f"Write {brand}\u00ae on first mention.",
        )
    ]


def check_required_section(rule: Rule, data: RuleInput) -> list[RuleViolation]:
    """A section must be present, and optionally positioned relative to another.

    params: ``section`` (required), ``before_section`` (must start before that one),
    ``max_distance_chars`` (must start within N characters of ``near_section``),
    ``near_section``.

    Position matters because "present but buried" is the realistic failure, not outright absence:
    safety information pushed below the footnotes technically appears, and is still misleading.
    """
    section = str(rule.params.get("section", ""))
    if not section.strip():
        raise RuleEvaluationError(f"{rule.label()}: required_section needs params.section")

    sections = data.doc_meta.get("sections") or {}

    # No section map at all is different from a section that is absent, and the difference matters
    # to the reviewer reading the finding. An empty map means the document had no structure this
    # code could read -- a PDF whose headings are set in colour rather than in size, a scan -- so
    # the rule has no evidence either way. Reporting "the section is missing" there is a confident
    # false blocker, and a reviewer cannot tell it apart from a real one.
    #
    # So it is reported as what it is, at low confidence, which routes it through the tenant's
    # confidence gate and escalates it instead of blocking on it. The failure policy in
    # mlr/nodes/base.py says a check that cannot complete must escalate rather than look clean;
    # this is the same principle for a check whose *input* is missing.
    if not sections:
        return [
            _violation(
                rule,
                message=(
                    f"Could not check for the required section {section!r}: this document has no "
                    f"readable section structure, so the rule could not be evaluated either way."
                ),
                confidence=20,
            )
        ]

    entry = sections.get(section)
    if not isinstance(entry, dict):
        return [_violation(rule, message=f"Required section {section!r} is missing.")]

    start = int(entry.get("start", 0))
    violations: list[RuleViolation] = []

    before = str(rule.params.get("before_section", "") or "")
    if before:
        other = sections.get(before)
        if isinstance(other, dict) and start > int(other.get("start", 0)):
            violations.append(
                _violation(
                    rule,
                    message=(
                        f"Section {section!r} appears after {before!r}; it must come first so a "
                        f"reader meets it before the claim it qualifies."
                    ),
                )
            )

    near = str(rule.params.get("near_section", "") or "")
    max_distance = rule.params.get("max_distance_chars")
    if near and max_distance is not None:
        other = sections.get(near)
        if isinstance(other, dict):
            distance = abs(start - int(other.get("start", 0)))
            if distance > int(max_distance):
                violations.append(
                    _violation(
                        rule,
                        message=(
                            f"Section {section!r} is {distance} characters from {near!r}, beyond "
                            f"the {max_distance} allowed, so it does not read as part of the same "
                            f"message."
                        ),
                    )
                )
    return violations


def check_max_claims_without_reference(rule: Rule, data: RuleInput) -> list[RuleViolation]:
    """Every claim must carry a reference marker.

    params: ``marker_pattern`` (regex, defaults to common citation forms),
    ``max_allowed`` (default 0).

    Deterministic *given* the segments it is handed. Identifying which statements are claims is an
    interpretive step done upstream; checking that each one cites something is not. Splitting it
    this way keeps the expensive judgement in one place and makes the cheap half auditable.
    """
    pattern_source = str(rule.params.get("marker_pattern", "") or DEFAULT_REFERENCE_MARKER)
    try:
        marker = re.compile(pattern_source, re.IGNORECASE)
    except re.error as exc:
        raise RuleEvaluationError(f"{rule.label()}: bad marker_pattern: {exc}") from exc

    unreferenced = [s for s in data.segments if not marker.search(s.text)]
    if len(unreferenced) <= int(rule.params.get("max_allowed", 0)):
        return []

    return [
        _violation(
            rule,
            quote=segment.text,
            char_span=segment.char_span,
            segment_id=segment.id,
            message="Claim carries no reference marker.",
            suggested_fix="Add a reference citation supporting this claim.",
        )
        for segment in unreferenced
    ]


EVALUATORS: dict[str, Callable[[Rule, RuleInput], list[RuleViolation]]] = {
    "banned_terms": check_banned_terms,
    "required_text": check_required_text,
    "trademark_first_mention": check_trademark_first_mention,
    "required_section": check_required_section,
    "max_claims_without_reference": check_max_claims_without_reference,
}


def evaluate(rule: Rule, data: RuleInput) -> list[RuleViolation]:
    """Run one deterministic rule."""
    if rule.kind != "deterministic":
        raise RuleEvaluationError(f"{rule.label()} is kind={rule.kind}, not deterministic")
    evaluator = EVALUATORS.get(str(rule.params.get("type", "")))
    if evaluator is None:
        raise RuleEvaluationError(f"{rule.label()}: unknown deterministic type")
    return evaluator(rule, data)


def evaluate_all(rules: Iterable[Rule], data: RuleInput) -> list[RuleViolation]:
    """Run every deterministic rule in a set.

    A rule whose params are malformed produces a violation describing the misconfiguration rather
    than an exception, so one broken rule cannot take down the review. The violation carries the
    rule's own severity, which means a misconfigured blocker still reaches a human -- failing
    closed rather than silently skipping a check the tenant asked for.
    """
    out: list[RuleViolation] = []
    for rule in rules:
        try:
            out.extend(evaluate(rule, data))
        except RuleEvaluationError as exc:
            out.append(
                _violation(
                    rule,
                    message=f"Rule could not be evaluated and needs attention: {exc}",
                    confidence=0,
                )
            )
    return out
