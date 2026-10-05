"""Turning rule examples into eval cases.

Why this is worth automating: every rule already ships worked examples, because
:mod:`platform_core.rules.validation` requires them before activation. Those examples are exactly an
eval set -- labelled inputs with expected outcomes, written by the domain expert who owns the rule.
Generating cases from them means a tenant adding a rule also adds test coverage for it, with no
separate authoring step and no chance of the two drifting apart.

The honest limitation, stated here because it belongs in the code and not only in the README: a rule
scoring 100% against its own examples is close to tautological when those examples are what the
author tuned the pattern against. It catches a regex typo, a later edit that breaks an earlier case,
and a rule that silently stopped firing -- all real failures. It is *not* evidence the rule is
correct on copy nobody thought of. Per-rule accuracy should be read as a regression signal, not as a
measure of rule quality.

Deterministic code for deterministic rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from platform_core.rules.schema import Rule
from platform_core.rules.validation import ValidationReport, validate_rule


class RuleAccuracy(BaseModel):
    """How one rule performed against its own examples."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    rule_version: int
    kind: str
    severity: str
    examples: int = 0
    executed: int = 0
    correct: int = 0
    accuracy: float = 0.0
    skipped: int = 0
    failures: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def is_perfect(self) -> bool:
        return self.executed > 0 and self.correct == self.executed


def accuracy_from_report(rule: Rule, report: ValidationReport) -> RuleAccuracy:
    """Summarise a validation report as a per-rule accuracy row."""
    executed = report.examples_executed
    return RuleAccuracy(
        rule_id=rule.id,
        rule_version=rule.version,
        kind=rule.kind,
        severity=rule.severity,
        examples=len(report.examples),
        executed=len(executed),
        correct=sum(1 for e in executed if e.correct),
        accuracy=report.accuracy(),
        skipped=len(report.examples) - len(executed),
        failures=[
            {"text": e.text[:160], "expected": e.expected, "actual": e.actual, "detail": e.detail}
            for e in executed
            if not e.correct
        ],
    )


def score_rules(rules: Sequence[Rule], **validate_kwargs: Any) -> list[RuleAccuracy]:
    """Run every rule against its own examples."""
    return [accuracy_from_report(r, validate_rule(r, **validate_kwargs)) for r in rules]


def summarize(rows: Sequence[RuleAccuracy]) -> dict[str, Any]:
    """Aggregate per-rule accuracy, with deterministic rules called out separately.

    Deterministic rules are reported apart from the rest and expected to be perfect: they are plain
    code over plain text, so anything less than 100% is a bug in the rule, not model variance. Mixing
    them into one average would let a broken regex hide behind an LLM rule's inherent fuzziness.
    """
    deterministic = [r for r in rows if r.kind == "deterministic"]
    executed_deterministic = [r for r in deterministic if r.executed]
    imperfect = [r for r in executed_deterministic if not r.is_perfect]

    return {
        "rules": len(rows),
        "deterministic": {
            "rules": len(deterministic),
            "scored": len(executed_deterministic),
            "all_perfect": not imperfect,
            "imperfect": [
                {"rule_id": r.rule_id, "version": r.rule_version, "accuracy": r.accuracy}
                for r in imperfect
            ],
        },
        "not_self_testable": [
            {"rule_id": r.rule_id, "kind": r.kind} for r in rows if r.kind != "deterministic"
        ],
        "per_rule": [r.model_dump(mode="json") for r in rows],
    }
