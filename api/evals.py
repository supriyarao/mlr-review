"""Eval runs as the API sees them: stored results, the baseline, and a seam to the runner.

Deterministic code. No model is called from this module; a live eval calls models inside the runner.

What is real and what is not, today:

* **Reading results is real.** A result is an ``EvalResult`` JSON file written by
  ``platform_core.evals.harness.save_result``; the baseline is ``evals/baselines/latest.json``.
* **Running an eval and the threshold sweep go through a seam**, because the runner
  (``scripts/run_evals.py``) is being written in another session. The seam is two functions, stated
  once here so the dependency is explicit:

      scripts.run_evals.run(mode: str, tenant_id: str, repeats: int) -> EvalResult
      scripts.run_evals.sweep(result: EvalResult, thresholds: list[int]) -> list[dict]

  ``sweep`` returns one ``{"threshold", "blocker_recall", "false_escalation_rate"}`` per threshold,
  recomputed from the findings stored in ``result`` with no model calls.

  While that module is absent, :func:`runner` returns None and the routes answer with the committed
  contract examples, flagged as stubbed.
"""

from __future__ import annotations

import importlib
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from platform_core.evals import harness
from platform_core.evals.schema import EvalResult

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evals" / "results"
BASELINE_PATH = REPO_ROOT / "evals" / "baselines" / "latest.json"

# The metric a policy or prompt change is never allowed to reduce silently.
GUARD_METRICS = [("blocker_recall", "recall")]
DEFAULT_THRESHOLDS = [50, 60, 70, 80, 90]

SWEEP_NOTE = (
    "Routing recomputed from stored findings at each threshold. No model calls: the curve compares "
    "policy settings against one fixed set of findings, so a difference is the threshold's effect "
    "and not model nondeterminism."
)

# Runs started by this process that have not produced a result file yet.
# Trade-off: in memory, so a restart forgets a run that was in flight. The results themselves are
# files and survive; a job table would only add durability for the minute or so a run takes.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def runner() -> Any | None:
    """The eval runner module, or None while it does not provide the seam above."""
    try:
        module = importlib.import_module("scripts.run_evals")
    except ModuleNotFoundError:
        return None
    if callable(getattr(module, "run", None)) and callable(getattr(module, "sweep", None)):
        return module
    return None


def is_stub() -> bool:
    return runner() is None


# --- Reading results -----------------------------------------------------------------------------


def stored_results() -> list[EvalResult]:
    """Every readable result file, newest first. An unreadable file is skipped, not fatal."""
    results: list[EvalResult] = []
    if RESULTS_DIR.is_dir():
        for path in RESULTS_DIR.glob("*.json"):
            try:
                results.append(harness.load_result(path))
            except (ValueError, OSError):
                continue
    return sorted(results, key=lambda r: r.started_at, reverse=True)


def find_result(eval_run_id: str) -> EvalResult | None:
    return next((r for r in stored_results() if r.eval_run_id == eval_run_id), None)


def baseline() -> EvalResult | None:
    if not BASELINE_PATH.is_file():
        return None
    try:
        return harness.load_result(BASELINE_PATH)
    except (ValueError, OSError):
        return None


def _nested(metrics: dict[str, Any], group: str, key: str) -> float:
    block = metrics.get(group)
    value = block.get(key) if isinstance(block, dict) else None
    return float(value) if isinstance(value, int | float) else 0.0


def summary(result: EvalResult) -> dict[str, Any]:
    """One row for the runs list. The two numbers shown are the two that pull against each other."""
    return {
        "eval_run_id": result.eval_run_id,
        "mode": result.mode,
        "started_at": result.started_at,
        "git_sha": result.git_sha,
        "cases": len(result.cases),
        "blocker_recall": _nested(result.metrics, "blocker_recall", "recall"),
        "false_escalation_rate": _nested(result.metrics, "false_escalation_rate", "rate"),
    }


def detail(result: EvalResult, *, status: str = "complete", error: str = "") -> dict[str, Any]:
    """One run in full. The same shape whether the run has finished or is still going, so a client
    polling it has nothing to branch on but ``status``."""
    payload = result.model_dump(mode="json")
    payload["status"] = status
    payload["error"] = error
    return payload


# --- Running -------------------------------------------------------------------------------------


def new_job(mode: str) -> str:
    eval_run_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[eval_run_id] = {
            "mode": mode,
            "status": "queued",
            "error": "",
            "started_at": datetime.now(UTC).isoformat(),
        }
    return eval_run_id


def job(eval_run_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        found = _jobs.get(eval_run_id)
        return dict(found) if found else None


def _update_job(eval_run_id: str, **changes: Any) -> None:
    with _jobs_lock:
        if eval_run_id in _jobs:
            _jobs[eval_run_id].update(changes)


def execute(eval_run_id: str, *, mode: str, tenant_id: str, repeats: int) -> None:
    """Run an eval and store its result. Called from a background task.

    A failure is recorded on the job rather than raised: nothing is waiting on this call, so an
    exception here would vanish into the server log and the client would poll forever.
    """
    module = runner()
    _update_job(eval_run_id, status="running")
    try:
        if module is None:
            raise RuntimeError("the eval runner is not available")
        result: EvalResult = module.run(mode=mode, tenant_id=tenant_id, repeats=repeats)
        # The id handed back to the client is the one the result is stored under.
        result = result.model_copy(update={"eval_run_id": eval_run_id})
        harness.save_result(result, directory=RESULTS_DIR)
        _update_job(eval_run_id, status="complete")
    except Exception as exc:
        _update_job(eval_run_id, status="failed", error=f"{type(exc).__name__}: {exc}")


def pending_detail(eval_run_id: str, state: dict[str, Any]) -> dict[str, Any]:
    placeholder = EvalResult(
        eval_run_id=eval_run_id, started_at=state["started_at"], mode=state["mode"]
    )
    return detail(placeholder, status=state["status"], error=state["error"])


# --- Comparison and the regression gate ----------------------------------------------------------


def comparison(result: EvalResult) -> dict[str, Any]:
    """How a run moved against the committed baseline, and which cases changed."""
    base = baseline()
    if base is None:
        return {"baseline_available": False}
    compared = harness.compare_to_baseline(result, base, guard_metrics=GUARD_METRICS)
    return {
        "baseline_available": True,
        "baseline": summary(base),
        **compared,
        "case_changes": harness.diff_cases(result, base),
    }


def regression_check(tenant_id: str) -> dict[str, Any]:
    """Would the policy now in force cost blocker recall? Used when a rule is activated.

    Replays the offline golden set for the tenant and compares against the baseline. When the
    runner or the baseline is missing the check reports ``ran: False`` rather than a pass: an
    unverified activation and a verified one must not look the same in the audit log.
    """
    module = runner()
    base = baseline()
    if module is None or base is None:
        return {"ran": False, "regressed": False, "movements": []}
    result = module.run(mode="offline", tenant_id=tenant_id, repeats=1)
    compared = harness.compare_to_baseline(result, base, guard_metrics=GUARD_METRICS)
    return {"ran": True, "regressed": compared["regressed"], "movements": compared["movements"]}


def sweep(thresholds: list[int], eval_run_id: str = "") -> dict[str, Any] | None:
    """The safety-versus-throughput curve for one stored run.

    Defaults to the newest stored result, then the baseline. None when there is nothing to sweep.
    """
    module = runner()
    if module is None:
        return None
    if eval_run_id:
        result = find_result(eval_run_id)
    else:
        stored = stored_results()
        result = stored[0] if stored else baseline()
    if result is None:
        return None
    ordered = sorted(set(thresholds))
    return {
        "thresholds": ordered,
        "note": SWEEP_NOTE,
        "points": module.sweep(result, ordered),
        "eval_run_id": result.eval_run_id,
    }
