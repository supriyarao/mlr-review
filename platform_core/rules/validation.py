"""Validating a rule before it can be activated.

Why validation is split from the schema: Pydantic answers "is this well-formed?" This module answers
the harder question, "does this rule actually do what its author says?" -- by running the rule
against its own examples.

The key policy: **a rule whose examples do not behave as labelled can be saved as a draft but cannot
be activated.** Saving has to stay permissive, because half-written rules are a normal part of
authoring and losing work is worse than storing something imperfect. Activation is where the rule
starts affecting real reviews, so that is where the bar belongs.

Deterministic code for deterministic rules. LLM and reference_doc rules cannot be self-tested without
a model call, so their examples are checked for presence and shape only, and the gap is reported
honestly rather than papered over with a pass.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from platform_core.rules import deterministic, prompting
from platform_core.rules.schema import Rule, RuleInput, Segment


class ExampleOutcome(BaseModel):
    """What happened when a rule was run against one of its own examples."""

    model_config = ConfigDict(extra="forbid")

    text: str
    expected: str
    # "pass" (no violation) or "fail" (fired), as observed.
    actual: str = ""
    correct: bool = False
    # Set when the example could not be executed -- notably for llm/reference_doc kinds.
    skipped_reason: str = ""
    detail: str = ""


class ValidationReport(BaseModel):
    """The result of validating one rule.

    ``can_save`` and ``can_activate`` are separate because they answer different questions. Errors
    block both; a failing example blocks only activation.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    rule_version: int
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    examples: list[ExampleOutcome] = Field(default_factory=list)

    @property
    def examples_executed(self) -> list[ExampleOutcome]:
        return [e for e in self.examples if not e.skipped_reason]

    @property
    def examples_passed(self) -> bool:
        executed = self.examples_executed
        return bool(executed) and all(e.correct for e in executed)

    @property
    def can_save(self) -> bool:
        """Structurally sound enough to store as a draft."""
        return not self.errors

    @property
    def can_activate(self) -> bool:
        """Safe to let loose on real reviews.

        A deterministic rule must demonstrably behave as its examples claim. An llm or
        reference_doc rule cannot be self-tested offline, so it is allowed through on structure
        alone -- with a warning saying so, because that is a real reduction in assurance and the
        person clicking Activate deserves to know.
        """
        if self.errors:
            return False
        return not (self.examples_executed and not self.examples_passed)

    def accuracy(self) -> float:
        executed = self.examples_executed
        if not executed:
            return 0.0
        return round(sum(1 for e in executed if e.correct) / len(executed), 4)

    def summary(self) -> str:
        if self.errors:
            return f"{len(self.errors)} error(s): {self.errors[0]}"
        executed = self.examples_executed
        if not executed:
            return "structure valid; examples not executable offline"
        return (
            f"{sum(1 for e in executed if e.correct)}/{len(executed)} examples behave as labelled"
        )


def _check_examples_present(rule: Rule, errors: list[str]) -> None:
    """At least one passing and one failing example.

    Both directions are required because each catches a different mistake: with no failing example
    a rule that never fires looks fine, and with no passing example a rule that fires on everything
    looks fine.
    """
    expected = {e.expected for e in rule.examples}
    if "fail" not in expected:
        errors.append("needs at least one example with expected: fail (a rule that never fires)")
    if "pass" not in expected:
        errors.append("needs at least one example with expected: pass (a rule that always fires)")


def _check_deterministic_params(rule: Rule, errors: list[str]) -> None:
    """Compile every pattern now, so a bad regex fails at author time, not mid-review."""
    params = rule.params
    kind_type = str(params.get("type", ""))

    if kind_type == "banned_terms":
        terms = [str(t) for t in params.get("terms", []) if str(t).strip()]
        if not terms:
            errors.append("banned_terms needs a non-empty params.terms")
        if params.get("regex"):
            for term in terms:
                try:
                    deterministic._compile_term(term, is_regex=True)
                except deterministic.RuleEvaluationError as exc:
                    errors.append(str(exc))
        for context in params.get("allowed_contexts", []) or []:
            try:
                deterministic._allowed_spans("", [context])
            except deterministic.RuleEvaluationError as exc:
                errors.append(str(exc))

    elif kind_type == "required_text" and not str(params.get("text", "")).strip():
        errors.append("required_text needs params.text")

    elif kind_type == "trademark_first_mention" and not str(params.get("brand", "")).strip():
        errors.append("trademark_first_mention needs params.brand")

    elif kind_type == "required_section" and not str(params.get("section", "")).strip():
        errors.append("required_section needs params.section")

    elif kind_type == "max_claims_without_reference":
        pattern = params.get("marker_pattern")
        if pattern:
            import re

            try:
                re.compile(str(pattern))
            except re.error as exc:
                errors.append(f"bad marker_pattern: {exc}")


def _check_reference_doc(rule: Rule, reference_docs_dir: Path | None, errors: list[str]) -> None:
    """The cited document must exist, and any named clauses must appear in it verbatim."""
    doc_path = str(rule.params.get("doc_path", ""))
    if reference_docs_dir is None:
        errors.append("cannot validate reference_doc rule without a reference document directory")
        return

    resolved = (reference_docs_dir / doc_path).resolve()
    # Keep a tenant's doc_path from reaching outside its own folder.
    if not str(resolved).startswith(str(reference_docs_dir.resolve())):
        errors.append(f"doc_path {doc_path!r} escapes the tenant's reference document directory")
        return
    if not resolved.is_file():
        errors.append(f"reference document {doc_path!r} does not exist")
        return

    document = resolved.read_text(encoding="utf-8")
    flat = deterministic.normalize_whitespace(document)
    for clause in rule.params.get("clauses", []) or []:
        if deterministic.normalize_whitespace(str(clause)) not in flat:
            errors.append(f"clause {str(clause)[:60]!r} does not appear in {doc_path}")


def _run_examples(rule: Rule) -> list[ExampleOutcome]:
    """Execute the rule against its own examples, where that is possible offline."""
    outcomes: list[ExampleOutcome] = []
    for example in rule.examples:
        if rule.kind != "deterministic":
            outcomes.append(
                ExampleOutcome(
                    text=example.text,
                    expected=example.expected,
                    skipped_reason=f"{rule.kind} rules need a model call to evaluate",
                )
            )
            continue

        # A claim-level rule needs segments. An example may list them explicitly; otherwise the
        # whole example text is treated as one segment, which is how such examples are written.
        segments = (
            [
                Segment(id=f"example-{i}", text=s, char_span=(0, len(s)))
                for i, s in enumerate(example.segments)
            ]
            if example.segments
            else [Segment(id="example", text=example.text, char_span=(0, len(example.text)))]
        )
        data = RuleInput(text=example.text, segments=segments, doc_meta=dict(example.doc_meta))
        try:
            violations = deterministic.evaluate(rule, data)
        except deterministic.RuleEvaluationError as exc:
            outcomes.append(
                ExampleOutcome(
                    text=example.text,
                    expected=example.expected,
                    actual="error",
                    correct=False,
                    detail=str(exc),
                )
            )
            continue

        actual = "fail" if violations else "pass"
        outcomes.append(
            ExampleOutcome(
                text=example.text,
                expected=example.expected,
                actual=actual,
                correct=actual == example.expected,
                detail=violations[0].message if violations else "",
            )
        )
    return outcomes


def validate_rule(rule: Rule, *, reference_docs_dir: Path | None = None) -> ValidationReport:
    """Validate a rule and run it against its own examples."""
    errors: list[str] = []
    warnings: list[str] = []

    _check_examples_present(rule, errors)

    if rule.kind == "deterministic":
        _check_deterministic_params(rule, errors)
    elif rule.kind == "reference_doc":
        _check_reference_doc(rule, reference_docs_dir, errors)

    # Tenant-authored text that tries to address the model is defanged before it is ever sent, but
    # the author should still be told, because it is far more likely to be a mistake than an attack.
    for field_name, value in (("description", rule.description), ("title", rule.title)):
        if prompting.looks_like_injection(value):
            warnings.append(
                f"{field_name} contains instruction-like text; it will be neutralised before any "
                f"model sees it"
            )

    if rule.kind != "deterministic":
        warnings.append(
            f"{rule.kind} rule: examples cannot be verified without a model call, so activation "
            f"checks structure only"
        )

    return ValidationReport(
        rule_id=rule.id,
        rule_version=rule.version,
        errors=errors,
        warnings=warnings,
        examples=_run_examples(rule),
    )
