"""Eval data shapes, shared by the harness and whatever use case supplies the cases.

Why these exist: an eval set is only useful if it is reviewable by a domain expert who does not
read Python. So a case is a YAML file with an explicit ``why_hard`` field -- if the author cannot
say what makes a case hard, it probably is not, and the set fills up with cases that pass
trivially and measure nothing.

``must_not_flag`` is the other half that eval sets usually omit. Measuring only what a system
should catch rewards flagging everything; a false-alarm list is what keeps precision honest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ExpectedFinding(BaseModel):
    """One finding a case expects.

    Matched loosely on purpose: ``claim_contains`` is a substring rather than an exact quote,
    because requiring a model to reproduce an exact span would make the eval measure
    transcription rather than judgement.
    """

    model_config = ConfigDict(extra="forbid")

    check_type: str
    claim_contains: str
    severity: str = ""
    # Written for a reviewer: why this is wrong, in the vocabulary of someone who has it.
    note: str = ""
    # The plain-English half, and the correction. Stated by the case rather than composed by the
    # oracle, because a composed one can only ever rephrase the note -- and the offline demo runs
    # on the fake backend, so this is the text a reader actually sees there. Optional: a case that
    # omits them still matches, it just shows a weaker explanation.
    explanation: str = ""
    suggested_fix: str = ""


class GoldenCase(BaseModel):
    """One hard test case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    # Required: forces the author to articulate what this case tests that others do not.
    why_hard: str
    tenant_id: str
    product_id: str
    label_version: str = ""

    # Submission metadata. Part of the case because it changes the result: a rule scoped to
    # patient-facing material does not apply to an HCP piece, so a case that omitted these would not
    # be reproducible -- the same text would score differently depending on an unstated default.
    audience: str = "any"
    channel: str = ""
    market: str = ""

    # Inline promo text, or a path relative to the repo root.
    promo: str = ""
    promo_path: str = ""
    doc_meta: dict[str, Any] = Field(default_factory=dict)

    expected_findings: list[ExpectedFinding] = Field(default_factory=list)
    # Tenant business rules expected to fire, by id. Checked separately from expected_findings
    # because "the right rule fired" and "the finding reads correctly" are different failures.
    expected_rule_ids: list[str] = Field(default_factory=list)
    expected_route: str = ""
    # Things that must NOT be flagged. Guards against a system that escalates everything.
    must_not_flag: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    def promo_text(self, repo_root: Path) -> str:
        """The promotional copy under review, inline or loaded from a path."""
        if self.promo.strip():
            return self.promo
        if self.promo_path:
            return (repo_root / self.promo_path).read_text(encoding="utf-8")
        raise ValueError(f"case {self.id!r} has neither promo nor promo_path")


class FindingMatch(BaseModel):
    """How one expected finding fared, so a failure is diagnosable from the result file."""

    model_config = ConfigDict(extra="forbid")

    expected: ExpectedFinding
    matched: bool
    # True when the check and claim matched but the severity did not. Reported separately: a
    # severity disagreement is a policy-calibration problem, a miss is a detection problem.
    severity_mismatch: bool = False
    actual_severity: str = ""
    matched_finding_id: str = ""


class CaseResult(BaseModel):
    """Outcome of running one case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    tenant_id: str
    passed: bool
    matches: list[FindingMatch] = Field(default_factory=list)
    # Findings the system produced that no expected finding accounts for.
    unexpected_findings: list[dict[str, Any]] = Field(default_factory=list)
    # must_not_flag entries that were flagged anyway.
    false_alarms: list[str] = Field(default_factory=list)
    expected_route: str = ""
    actual_route: str = ""
    route_correct: bool = False
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str = ""
    # Everything the system produced, so a result file can be inspected without a re-run.
    findings: list[dict[str, Any]] = Field(default_factory=list)
    gate_results: list[dict[str, Any]] = Field(default_factory=list)


class EvalResult(BaseModel):
    """A whole eval run: the results, the metrics, and the exact conditions that produced them.

    The provenance fields are not decoration. "Blocker recall is 0.9" is meaningless without
    knowing which models, which prompt versions, and which tenant policies produced it -- and a
    regression gate that cannot tell a prompt change from a policy change is not much use.
    """

    model_config = ConfigDict(extra="forbid")

    eval_run_id: str
    started_at: str
    finished_at: str = ""
    mode: str = "offline"
    git_sha: str = ""
    backend: str = ""
    model_strong: str = ""
    model_fast: str = ""
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    tenant_config_hashes: dict[str, str] = Field(default_factory=dict)
    repeats: int = 1
    cases: list[CaseResult] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    per_tenant: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notes: str = ""


def load_case(path: Path) -> GoldenCase:
    """Load and validate one golden case YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return GoldenCase.model_validate(raw)


def load_cases(directory: Path, *, tenant_id: str = "") -> list[GoldenCase]:
    """Load every case in a directory, sorted by id, optionally filtered to one tenant."""
    cases = [load_case(p) for p in sorted(directory.glob("*.yaml"))]
    if tenant_id:
        cases = [c for c in cases if c.tenant_id == tenant_id]
    return sorted(cases, key=lambda c: c.id)
