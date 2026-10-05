"""The eval harness: run any compiled graph over any golden set.

Why this exists as a platform component: the harness is the part most worth reusing. Loading
cases, running them, matching expected against actual, aggregating metrics, writing a provenanced
result file and diffing against a baseline is identical for every use case. What differs is the
shape of a finding and which metric is the headline -- so those arrive as injected functions.

Deterministic code. The graph it runs may call models; the harness itself does not.

Two things here are deliberate and worth defending:

* **A case that raises is a failed case, not a failed run.** One broken case must not destroy the
  other nine results; the error is recorded on the case and the run continues.
* **The sweep re-runs only the gates and the router.** Recomputing routing at five confidence
  thresholds by re-calling the model five times would cost five times as much and introduce
  nondeterminism into what is meant to be a comparison. Findings are produced once and the
  threshold-dependent logic is replayed over them.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from platform_core.evals.metrics import generic_metrics, per_group
from platform_core.evals.schema import CaseResult, EvalResult, GoldenCase

# Runs one case and returns the raw system output. Supplied by the use case because only it knows
# how to turn a case into graph input and what the graph gives back.
CaseRunner = Callable[[GoldenCase], dict[str, Any]]

# Scores one case's raw output against its expectations, producing a CaseResult.
CaseScorer = Callable[[GoldenCase, dict[str, Any]], CaseResult]

# Adds use-case-specific metrics (for MLR: blocker recall, false escalation rate).
MetricsHook = Callable[[Sequence[CaseResult]], dict[str, Any]]


def git_sha() -> str:
    """Short commit sha, recorded in results so a number can be tied to a code state."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def run_eval(
    cases: Sequence[GoldenCase],
    *,
    runner: CaseRunner,
    scorer: CaseScorer,
    metrics_hook: MetricsHook | None = None,
    mode: str = "offline",
    repeats: int = 1,
    provenance: dict[str, Any] | None = None,
) -> EvalResult:
    """Run a golden set and compute metrics.

    With ``repeats > 1`` every case is run that many times and the *worst* outcome is kept. For a
    safety-critical check, "it catches this blocker four times out of five" is a miss, not a pass
    -- averaging would report a nondeterministic failure as a partial success.
    """
    started = datetime.now(UTC)
    results: list[CaseResult] = []

    for case in cases:
        attempts: list[CaseResult] = []
        for _ in range(max(1, repeats)):
            attempts.append(_run_one(case, runner=runner, scorer=scorer))
        # Worst-case selection: a failure in any attempt is the result.
        attempts.sort(key=lambda r: (r.passed, not r.error))
        results.append(attempts[0])

    metrics = generic_metrics(results)
    if metrics_hook:
        metrics.update(metrics_hook(results))

    prov = provenance or {}
    return EvalResult(
        eval_run_id=uuid.uuid4().hex[:12],
        started_at=started.isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        mode=mode,
        git_sha=git_sha(),
        backend=str(prov.get("backend", "")),
        model_strong=str(prov.get("model_strong", "")),
        model_fast=str(prov.get("model_fast", "")),
        prompt_versions=dict(prov.get("prompt_versions", {})),
        tenant_config_hashes=dict(prov.get("tenant_config_hashes", {})),
        repeats=max(1, repeats),
        cases=results,
        metrics=metrics,
        per_tenant=per_group(results, "tenant_id"),
    )


def _run_one(case: GoldenCase, *, runner: CaseRunner, scorer: CaseScorer) -> CaseResult:
    """Run and score a single case, converting any exception into a recorded failure."""
    start = time.perf_counter()
    try:
        raw = runner(case)
    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            tenant_id=case.tenant_id,
            passed=False,
            expected_route=case.expected_route,
            latency_ms=int((time.perf_counter() - start) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )

    elapsed = int((time.perf_counter() - start) * 1000)
    try:
        result = scorer(case, raw)
    except Exception as exc:
        return CaseResult(
            case_id=case.id,
            tenant_id=case.tenant_id,
            passed=False,
            expected_route=case.expected_route,
            latency_ms=elapsed,
            error=f"scoring failed: {type(exc).__name__}: {exc}",
        )

    if not result.latency_ms:
        result.latency_ms = elapsed
    return result


def results_dir() -> Path:
    path = Path("evals/results")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_result(result: EvalResult, *, directory: Path | None = None) -> Path:
    """Write a result to ``evals/results/{timestamp}_{git_sha}.json``."""
    target = directory or results_dir()
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{result.git_sha}" if result.git_sha else ""
    path = target / f"{stamp}{suffix}.json"
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def load_result(path: Path) -> EvalResult:
    return EvalResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


def compare_to_baseline(
    result: EvalResult,
    baseline: EvalResult,
    *,
    guard_metrics: Sequence[tuple[str, str]],
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Diff a run against a committed baseline and decide whether it regressed.

    ``guard_metrics`` names the paths that must not drop, as ``(group, key)`` pairs -- for MLR,
    blocker recall. Everything else is reported as informational movement.

    Why a hard gate on one metric rather than an overall score: a single composite number lets a
    drop in the metric that matters be offset by a gain in one that does not. The point of the gate
    is that a tenant's policy change cannot silently cost blocker recall.
    """
    regressions: list[dict[str, Any]] = []
    movements: list[dict[str, Any]] = []

    for group, key in guard_metrics:
        new = _metric(result.metrics, group, key)
        old = _metric(baseline.metrics, group, key)
        delta = round(new - old, 6)
        row = {"metric": f"{group}.{key}", "baseline": old, "current": new, "delta": delta}
        movements.append(row)
        if delta < -abs(tolerance):
            regressions.append(row)

    return {
        "regressed": bool(regressions),
        "regressions": regressions,
        "movements": movements,
        "baseline_git_sha": baseline.git_sha,
        "baseline_mode": baseline.mode,
        "current_git_sha": result.git_sha,
        "cases_baseline": len(baseline.cases),
        "cases_current": len(result.cases),
    }


def _metric(metrics: dict[str, Any], group: str, key: str) -> float:
    block = metrics.get(group)
    if isinstance(block, dict):
        value = block.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def diff_cases(result: EvalResult, baseline: EvalResult) -> list[dict[str, Any]]:
    """Per-case pass/fail changes against a baseline, so a regression names the case."""
    old = {c.case_id: c for c in baseline.cases}
    rows: list[dict[str, Any]] = []
    for case in result.cases:
        before = old.get(case.case_id)
        if before is None:
            rows.append({"case_id": case.case_id, "change": "new", "passed": case.passed})
        elif before.passed != case.passed:
            rows.append(
                {
                    "case_id": case.case_id,
                    "change": "broke" if before.passed else "fixed",
                    "passed": case.passed,
                }
            )
    for case_id in sorted(set(old) - {c.case_id for c in result.cases}):
        rows.append({"case_id": case_id, "change": "missing", "passed": False})
    return rows
