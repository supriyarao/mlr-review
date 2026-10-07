"""Run the golden set, compute metrics, sweep the confidence threshold, compare to a baseline.

Three modes, because they answer different questions:

``offline``  a synthetic model derived from each case's own expectations (``mlr.eval_oracle``). No
             API key, deterministic, runs in CI. Measures the **orchestration** -- routing, grounding,
             dedupe, severity, deterministic rules, false escalation -- and explicitly *not* whether a
             model would notice anything. Blocker recall for LLM-sourced findings is near-tautological
             here; the deterministic rule findings are the part that is real.
``live``     the configured backend for real. ``--record`` saves cassettes.
``replay``   recorded real responses replayed from ``evals/cassettes/``. Deterministic *and* a real
             model measurement, which is what the committed baseline should eventually be.

The threshold sweep recomputes routing at each confidence threshold **from the stored findings**,
without calling a model again. That is the point: the curve then shows the threshold's effect, not
model nondeterminism, so two points on it are actually comparable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from mlr import eval_metrics, eval_oracle
from mlr.graph import checkpointer, compile_with
from mlr.nodes.ingest import normalize_document, parse_front_matter
from mlr.state import initial_state, report_findings
from platform_core import audit, runtime
from platform_core.env import load_env
from platform_core.evals import harness
from platform_core.evals.schema import CaseResult, EvalResult, GoldenCase, load_cases
from platform_core.llm import FakeLLM, describe_backend
from platform_core.rules.registry import RuleRegistry
from platform_core.tenancy import load_all_tenants, load_tenant

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"
BASELINE = REPO_ROOT / "evals" / "baselines" / "latest.json"
# Passed explicitly: the platform harness must not have to guess where this repo lives, and a
# cwd-relative default would write results wherever the command happened to be run from.
RESULTS_DIR = REPO_ROOT / "evals" / "results"
DEFAULT_THRESHOLDS = [50, 60, 70, 80, 90]


def prepare_document(case: GoldenCase) -> str:
    """The exact text the graph will reason about, matching what ingest produces."""
    _, body = parse_front_matter(case.promo_text(REPO_ROOT))
    return normalize_document(body)


def make_runner(mode: str, *, record: bool = False):
    """Build the per-case runner the harness calls.

    Each case gets a **fresh** run id per invocation. A deterministic id like ``eval-{case.id}`` looks
    tidier and is wrong: the run id is the checkpointer's thread id, so a second eval run would resume
    the first run's completed checkpoint instead of re-running the case. That failed loudly only once a
    serializer change made the stale state deserialize differently -- before that it silently scored
    the previous run's findings, which is the kind of eval bug that invalidates every number it
    produces.
    """
    invocation = uuid4().hex[:8]

    def run(case: GoldenCase) -> dict[str, Any]:
        document = prepare_document(case)
        fake = None
        if mode == "offline":
            fake = FakeLLM(scripted=eval_oracle.build_script(case, document), default_response="{}")
        elif mode == "replay":
            fake = FakeLLM(mode="replay")

        run_id = f"eval-{case.id}-{invocation}"
        with checkpointer() as saver:
            graph = compile_with(saver, fake=fake)
            state = initial_state(
                run_id=run_id,
                tenant_id=case.tenant_id,
                product_id=case.product_id,
                label_version=case.label_version
                or load_tenant(case.tenant_id).product(case.product_id).label_version,
                audience=case.audience,
                channel=case.channel,
                market=case.market,
                document=document,
                doc_meta=dict(case.doc_meta),
            )
            result = runtime.start_run(graph, dict(state), run_id=run_id)

        final = result["state"]
        return {
            "run_id": run_id,
            "findings": [f.model_dump(mode="json") for f in report_findings(final)],
            "gate_results": [g.model_dump(mode="json") for g in final.get("gate_results") or []],
            "route": final.get("route") or "",
            "rules_applied": [r.model_dump(mode="json") for r in final.get("rules_applied") or []],
            "errors": list(final.get("errors") or []),
            "cost": audit.run_cost_summary(run_id),
        }

    return run


def score(case: GoldenCase, raw: dict[str, Any]) -> CaseResult:
    """Turn one run's output into a scored case result."""
    findings = raw["findings"]
    matches = eval_metrics.match_findings(case, findings)
    alarms = eval_metrics.false_alarms(case, findings)
    rules = eval_metrics.expected_rules_fired(case, findings)
    route_ok = bool(case.expected_route) and raw["route"] == case.expected_route
    cost = raw.get("cost") or {}

    # A case passes only if every expectation held: findings found, route correct, the expected rules
    # fired, and nothing on the must-not-flag list was flagged. Partial credit would let a case that
    # escalated for the wrong reason look like a success.
    passed = all(m.matched for m in matches) and route_ok and rules["all_fired"] and not alarms

    return CaseResult(
        case_id=case.id,
        tenant_id=case.tenant_id,
        passed=passed,
        matches=matches,
        unexpected_findings=eval_metrics.unexpected(case, findings),
        false_alarms=alarms,
        expected_route=case.expected_route,
        actual_route=raw["route"],
        route_correct=route_ok,
        latency_ms=int(cost.get("latency_ms") or 0),
        tokens_in=int(cost.get("tokens_in") or 0),
        tokens_out=int(cost.get("tokens_out") or 0),
        cost_usd=float(cost.get("cost_usd") or 0.0),
        error="; ".join(str(e) for e in raw.get("errors") or []),
        findings=findings,
        gate_results=raw["gate_results"],
    )


def sweep(results: list[CaseResult], thresholds: list[int]) -> dict[str, Any]:
    """Recompute routing at each confidence threshold, with no further model calls.

    Only the threshold-dependent part is replayed: which findings count as low-confidence, and whether
    that forces an escalation. Everything else -- the findings themselves, their grounding, their
    severity -- is held fixed, which is what makes two points on the curve comparable.
    """
    points = []
    for threshold in sorted(thresholds):
        reclassified: list[CaseResult] = []
        for case in results:
            low = [f for f in case.findings if int(f.get("confidence") or 0) < threshold]
            blockers = [f for f in case.findings if f.get("severity") == "blocker"]
            ungrounded = [f for f in case.findings if not f.get("grounded")]
            # The same rule the route node applies: anything serious, unverifiable or uncertain goes up.
            escalates = bool(blockers or ungrounded or low or case.error)
            route = "escalated" if escalates else "standard_reviewer"
            reclassified.append(
                case.model_copy(
                    update={
                        "actual_route": route
                        if case.expected_route == "standard_reviewer" or not escalates
                        else case.actual_route,
                        "route_correct": (route == "standard_reviewer")
                        == (case.expected_route == "standard_reviewer"),
                    }
                )
            )

        points.append(
            {
                "threshold": threshold,
                "blocker_recall": eval_metrics.blocker_recall(reclassified)["recall"],
                "false_escalation_rate": eval_metrics.false_escalation_rate(reclassified)["rate"],
                "low_confidence_findings": sum(
                    1
                    for c in results
                    for f in c.findings
                    if int(f.get("confidence") or 0) < threshold
                ),
            }
        )

    return {
        "thresholds": sorted(thresholds),
        "note": (
            "Routing recomputed from stored findings at each threshold. No model calls, so a "
            "difference between points is the threshold's effect and not model nondeterminism."
        ),
        "points": points,
        "tenant_thresholds": {t.tenant_id: t.confidence_threshold for t in load_all_tenants()},
    }


def provenance(mode: str) -> dict[str, Any]:
    backend = describe_backend()
    from mlr.prompts.checks import PROMPT_VERSIONS
    from mlr.prompts.extract_claims import PROMPT_VERSION as EXTRACT_VERSION
    from mlr.prompts.preapproved import PROMPT_VERSION as PREAPPROVED_VERSION
    from platform_core.policy import policy_version

    hashes = {}
    for tenant in load_all_tenants():
        try:
            hashes[tenant.tenant_id] = policy_version(tenant.tenant_id)
        except Exception:  # pragma: no cover
            hashes[tenant.tenant_id] = tenant.config_hash()

    return {
        "backend": "fake" if mode in ("offline", "replay") else backend["backend"],
        "model_strong": backend["model_strong"],
        "model_fast": backend["model_fast"],
        "prompt_versions": {
            **PROMPT_VERSIONS,
            "extract_claims": EXTRACT_VERSION,
            "preapproved_filter": PREAPPROVED_VERSION,
        },
        "tenant_config_hashes": hashes,
    }


def offline_caveat(mode: str) -> str:
    """The warning that travels with every offline result."""
    if mode != "offline":
        return ""
    return (
        "PROVISIONAL - offline mode. Model responses are synthesised from each case's own expected "
        "findings, so blocker recall for LLM-sourced findings is near-tautological and is NOT a "
        "measurement of model quality. What IS measured here: routing accuracy, grounding against the "
        "real corpus, false escalation on the clean cases, dedupe, severity from tenant policy, and "
        "deterministic rule findings (which make no model call and are fully real). Record cassettes "
        "against a live model and re-baseline before quoting any recall figure."
    )


def print_report(result: EvalResult, sweep_data: dict[str, Any] | None) -> None:
    metrics = result.metrics
    blocker = metrics["blocker_recall"]
    escalation = metrics["false_escalation_rate"]

    print(f"\n{'=' * 84}")
    print(f"Eval {result.eval_run_id}   mode={result.mode}   git={result.git_sha or 'n/a'}")
    print(f"{'=' * 84}")
    if result.notes:
        print(f"\n!! {result.notes}\n")

    print(
        f"  blocker recall          {blocker['recall']:.4f}   "
        f"({blocker['matched']}/{blocker['expected']})"
    )
    print(
        f"  false escalation rate   {escalation['rate']:.4f}   "
        f"({escalation['escalated']}/{escalation['standard_expected']} clean cases escalated)"
    )
    print(f"  advisory recall         {metrics['advisory_recall']['recall']:.4f}")
    print(
        f"  routing accuracy        {metrics['routing']['accuracy']:.4f}   "
        f"({metrics['routing']['correct']}/{metrics['routing']['scored']})"
    )
    print(
        f"  grounding rate          {metrics['grounding_rate']['rate']:.4f}   "
        f"{metrics['grounding_rate']['by_mode']}"
    )
    print(
        f"  pass rate               {metrics['pass_rate']['pass_rate']:.4f}   "
        f"({metrics['pass_rate']['passed']}/{metrics['pass_rate']['cases']})"
    )
    print(
        f"  rule findings           {metrics['rule_findings']['rule_findings']} from "
        f"{metrics['rule_findings']['distinct_rules']} rule(s)"
    )

    cost = metrics["cost_and_latency"]
    print(
        f"  cost                    ${cost['cost_usd_total']:.5f} total, "
        f"${cost['cost_usd_per_run']:.5f}/run"
    )
    print(
        f"  latency                 p50 {cost['latency_p50_ms']}ms   p95 {cost['latency_p95_ms']}ms"
    )

    if blocker["missed"]:
        print("\n  MISSED BLOCKERS:")
        for miss in blocker["missed"]:
            print(
                f"    {miss['case_id']:48} {miss['check_type']:18} {miss['claim_contains'][:40]!r}"
            )

    if metrics["false_alarms"]["count"]:
        print("\n  FALSE ALARMS:")
        for row in metrics["false_alarms"]["rows"]:
            print(f"    {row['case_id']:48} flagged {row['flagged'][:50]!r}")

    if metrics["severity_calibration"]["count"]:
        print("\n  SEVERITY MISMATCHES (policy calibration, not detection):")
        for row in metrics["severity_calibration"]["rows"]:
            print(
                f"    {row['case_id']:48} {row['check_type']:18} "
                f"expected {row['expected']} got {row['actual']}"
            )

    print(f"\n  {'case':50} {'pass':>5} {'route':>22}")
    for case in result.cases:
        mark = "ok" if case.passed else "FAIL"
        route = case.actual_route or "(none)"
        flag = "" if case.route_correct else f"  (expected {case.expected_route})"
        print(f"  {case.case_id:50} {mark:>5} {route:>22}{flag}")
        if case.error:
            print(f"    {case.error[:100]}")

    print("\n  per tenant:")
    for tenant_id, block in result.per_tenant.items():
        print(
            f"    {tenant_id:16} pass {block['pass_rate']['passed']}/{block['pass_rate']['cases']}"
            f"   routing {block['routing']['accuracy']:.2f}"
        )

    if sweep_data:
        print(f"\n  confidence threshold sweep ({sweep_data['tenant_thresholds']} in force today)")
        print(
            f"    {'threshold':>10} {'blocker recall':>16} {'false escalation':>18} {'low-conf':>10}"
        )
        for point in sweep_data["points"]:
            print(
                f"    {point['threshold']:>10} {point['blocker_recall']:>16.4f} "
                f"{point['false_escalation_rate']:>18.4f} {point['low_confidence_findings']:>10}"
            )


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description="Run the MLR golden set.")
    parser.add_argument("--mode", choices=("offline", "live", "replay"), default="offline")
    parser.add_argument("--tenant", default="", help="only cases for this tenant")
    parser.add_argument("--case", action="append", default=[], help="case id; repeatable")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--record", action="store_true", help="live mode: save cassettes")
    parser.add_argument(
        "--threshold-sweep",
        default="",
        help="comma-separated thresholds, or 'default' for 50,60,70,80,90",
    )
    parser.add_argument(
        "--compare", default="", help="'baseline' to diff against the committed one"
    )
    parser.add_argument("--save", action="store_true", help="write the result to evals/results/")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    cases = load_cases(GOLDEN_DIR, tenant_id=args.tenant)
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.id in wanted]
    if not cases:
        print("No cases selected.", file=sys.stderr)
        return 1

    # Rules must be in the registry before a run can apply them.
    for tenant in load_all_tenants():
        from platform_core import tenancy

        RuleRegistry(tenant.tenant_id).seed_from_yaml(
            tenancy.tenant_rules_dir(tenant.tenant_id),
            reference_docs_dir=tenancy.tenant_reference_docs_dir(tenant.tenant_id),
        )

    result = harness.run_eval(
        cases,
        runner=make_runner(args.mode, record=args.record),
        scorer=score,
        metrics_hook=eval_metrics.mlr_metrics,
        mode=args.mode,
        repeats=args.repeats,
        provenance=provenance(args.mode),
    )
    result.notes = offline_caveat(args.mode)

    thresholds = (
        DEFAULT_THRESHOLDS
        if args.threshold_sweep == "default"
        else [int(t) for t in args.threshold_sweep.split(",") if t.strip()]
        if args.threshold_sweep
        else []
    )
    sweep_data = sweep(result.cases, thresholds) if thresholds else None
    if sweep_data:
        result.metrics["threshold_sweep"] = sweep_data

    print_report(result, sweep_data)

    if args.repeats > 1:
        recalls = [result.metrics["blocker_recall"]["recall"]]
        print(
            f"\n  repeats={args.repeats}; worst-case outcome kept per case. "
            f"blocker recall {statistics.fmean(recalls):.4f}"
        )

    exit_code = 0
    if args.compare == "baseline":
        if not BASELINE.is_file():
            print(f"\nNo baseline at {BASELINE.relative_to(REPO_ROOT)}; nothing to compare.")
        else:
            baseline = harness.load_result(BASELINE)
            diff = harness.compare_to_baseline(
                result, baseline, guard_metrics=eval_metrics.GUARD_METRICS
            )
            print(
                f"\n  baseline {baseline.eval_run_id} (mode={baseline.mode}, git={baseline.git_sha})"
            )
            for row in diff["movements"]:
                arrow = "=" if row["delta"] == 0 else ("up" if row["delta"] > 0 else "DOWN")
                print(
                    f"    {row['metric']:28} {row['baseline']:.4f} -> {row['current']:.4f}  {arrow}"
                )
            for row in harness.diff_cases(result, baseline):
                print(f"    case {row['case_id']:46} {row['change']}")
            if baseline.mode != result.mode:
                print(
                    f"    !! baseline mode is {baseline.mode}, this run is {result.mode}; "
                    f"these are not comparable measurements"
                )
            if diff["regressed"]:
                print("\n  REGRESSION: blocker recall dropped. Failing.")
                exit_code = 1

    if args.save or args.write_baseline:
        path = harness.save_result(result, directory=RESULTS_DIR)
        print(f"\n  wrote {path.relative_to(REPO_ROOT)}")
        if args.write_baseline:
            BASELINE.parent.mkdir(parents=True, exist_ok=True)
            BASELINE.write_text(
                json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
            )
            print(f"  wrote {BASELINE.relative_to(REPO_ROOT)}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
