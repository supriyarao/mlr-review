"""Run real promotional pieces through the graph against a live model.

The point of this script is to find out whether the prompts work on real content, which no offline
test can tell you: a scripted response proves the graph wires up correctly, not that a model actually
notices a subgroup figure being generalised.

Run with ``make smoke`` (local, free) or ``MLR_LLM_BACKEND=anthropic make smoke`` (Claude).

Prints, per piece: the claims extracted, every finding with its severity, confidence, grounding mode
and citation, the gate verdicts, the route, and what the run cost. Failures are shown rather than
swallowed -- a smoke test that hides a schema failure is worse than no smoke test.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mlr.graph import checkpointer, compile_with
from mlr.nodes.ingest import extract_text
from mlr.state import initial_state, report_findings
from platform_core import audit, runtime
from platform_core.llm import describe_backend
from platform_core.tenancy import load_tenant

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMO = REPO_ROOT / "corpus" / "promo"

# The three pieces that exercise the hardest judgements, plus the clean one as a false-alarm check.
DEFAULT_PIECES = [
    ("zelvoran-hcp-subgroup-claim", "acme_pharma"),
    ("zelvoran-patient-brochure", "acme_pharma"),
    ("zelvoran-hcp-journal-ad", "acme_pharma"),
]

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
RED, AMBER, GREEN, BLUE = "\033[31m", "\033[33m", "\033[32m", "\033[34m"


def front_matter(stem: str) -> dict:
    import yaml

    text = (PROMO / f"{stem}.md").read_text(encoding="utf-8")
    _, front, _ = text.split("---\n", 2)
    return yaml.safe_load(front)


def severity_colour(severity: str) -> str:
    return RED if severity == "blocker" else AMBER


def run_one(stem: str, tenant_id: str, *, verbose: bool) -> dict:
    meta = front_matter(stem)
    config = load_tenant(tenant_id)
    product = config.product(meta["product_id"])

    print(f"\n{BOLD}{'=' * 96}{RESET}")
    print(f"{BOLD}{stem}{RESET}  ->  {config.name}  ({meta['audience']} / {meta['channel']})")
    print(f"{DIM}{meta.get('notes', '')[:180]}{RESET}")
    print(f"{BOLD}{'=' * 96}{RESET}")

    run_id = runtime.new_run_id()
    state = initial_state(
        run_id=run_id,
        tenant_id=tenant_id,
        product_id=meta["product_id"],
        label_version=str(meta.get("label_version") or product.label_version),
        audience=meta["audience"],
        channel=meta["channel"],
        market="us",
        document=extract_text(PROMO / f"{stem}.md"),
    )

    started = time.perf_counter()
    try:
        with checkpointer() as saver:
            graph = compile_with(saver)
            result = runtime.start_run(graph, dict(state), run_id=run_id)
    except Exception as exc:
        print(f"{RED}RUN FAILED: {type(exc).__name__}: {exc}{RESET}")
        return {"stem": stem, "failed": True, "error": str(exc)}

    elapsed = time.perf_counter() - started
    final = result["state"]
    findings = report_findings(final)

    claims = final.get("claims") or []
    print(f"\n{BOLD}Claims extracted:{RESET} {len(claims)}")
    for claim in claims:
        located = "" if claim.char_span != (0, 0) else f"  {RED}[not located in document]{RESET}"
        print(f"  {DIM}{claim.id}{RESET} [{claim.claim_type}] {claim.original_text[:96]}{located}")
        if verbose and claim.normalized_claim != claim.original_text:
            print(f"       {DIM}in full: {claim.normalized_claim[:110]}{RESET}")

    matches = final.get("preapproved_matches") or {}
    if matches:
        print(f"\n{BOLD}Pre-approved matches:{RESET} {matches}")

    applied = final.get("rules_applied") or []
    print(f"\n{BOLD}Rules in force:{RESET} {len(applied)}  {DIM}{[r.id for r in applied]}{RESET}")

    print(f"\n{BOLD}Findings:{RESET} {len(findings)}")
    for f in findings:
        colour = severity_colour(f.severity)
        ground = f"{GREEN}{f.grounded_by}{RESET}" if f.grounded else f"{RED}UNGROUNDED{RESET}"
        rule = f"  {BLUE}[{f.rule_id} v{f.rule_version}]{RESET}" if f.rule_id else ""
        merged = f"  {DIM}(merged {len(f.merged_from) + 1}){RESET}" if f.merged_from else ""
        print(
            f"\n  {colour}{f.severity.upper():8}{RESET} {f.check_type:18} "
            f"conf={f.confidence:3}  grounded={ground}{rule}{merged}"
        )
        print(
            f'    quote: {DIM}"{f.quote[:96]}"{RESET}' if f.quote else f"    {DIM}(absence){RESET}"
        )
        print(f"    issue: {f.issue[:150]}")
        for c in f.citations[:2]:
            print(f'    cites: {DIM}[{c.kind}] {c.source_id} :: "{c.passage[:80]}"{RESET}')
        if f.suggested_fix and verbose:
            print(f"    fix:   {DIM}{f.suggested_fix[:120]}{RESET}")

    print(f"\n{BOLD}Gates:{RESET}")
    for gate in final.get("gate_results") or []:
        mark = f"{GREEN}pass{RESET}" if gate.passed else f"{RED}FAIL{RESET}"
        print(f"  {mark}  {gate.gate:12} {gate.details[:110]}")

    for err in final.get("errors") or []:
        print(f"  {RED}error{RESET} {str(err)[:140]}")

    route = final.get("route") or "(none)"
    escalation = final.get("escalation")
    print(f"\n{BOLD}Route:{RESET} {route}")
    if escalation:
        for reason in escalation.reasons:
            print(f"  {AMBER}{reason.code}{RESET}: {reason.explanation[:120]}")

    # Read from the audit log, which is the system of record for cost.
    summary = audit.run_cost_summary(run_id)
    print(
        f"\n{BOLD}Cost:{RESET} ${summary['cost_usd']:.5f}  "
        f"{summary['tokens_in']} in / {summary['tokens_out']} out  "
        f"{elapsed:.1f}s wall  {summary['events']} audit events"
    )

    return {
        "stem": stem,
        "tenant_id": tenant_id,
        "failed": False,
        "claims": len(claims),
        "findings": len(findings),
        "blockers": sum(1 for f in findings if f.is_blocker),
        "ungrounded": sum(1 for f in findings if not f.grounded),
        "route": route,
        "cost_usd": float(summary["cost_usd"]),
        "seconds": round(elapsed, 1),
        "errors": len(final.get("errors") or []),
        "unlocated_claims": sum(1 for c in (final.get("claims") or []) if c.char_span == (0, 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run promo pieces against a live model.")
    parser.add_argument("--piece", action="append", default=[], help="promo stem; repeatable")
    parser.add_argument("--tenant", default="acme_pharma")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    backend = describe_backend()
    if backend["backend"] == "fake":
        print(
            "MLR_LLM_BACKEND=fake: this script exists to test prompts against a real model.\n"
            "Use the default (ollama, free) or set MLR_LLM_BACKEND=anthropic.",
            file=sys.stderr,
        )
        return 2

    print(
        f"{BOLD}Backend:{RESET} {backend['backend']}  "
        f"strong={backend['model_strong']}  fast={backend['model_fast']}"
    )

    pieces = [(p, args.tenant) for p in args.piece] if args.piece else DEFAULT_PIECES
    results = [run_one(stem, tenant, verbose=args.verbose) for stem, tenant in pieces]

    print(f"\n{BOLD}{'=' * 96}\nSummary\n{'=' * 96}{RESET}")
    header = f"{'piece':34} {'claims':>6} {'find':>5} {'blk':>4} {'ungr':>5} {'route':20} {'$':>8} {'s':>6}"
    print(BOLD + header + RESET)
    for r in results:
        if r["failed"]:
            print(f"{r['stem']:34} {RED}FAILED: {r['error'][:60]}{RESET}")
            continue
        print(
            f"{r['stem']:34} {r['claims']:>6} {r['findings']:>5} {r['blockers']:>4} "
            f"{r['ungrounded']:>5} {r['route']:20} {r['cost_usd']:>8.5f} {r['seconds']:>6.1f}"
        )

    total = sum(r.get("cost_usd", 0.0) for r in results)
    problems = sum(r.get("errors", 0) + r.get("unlocated_claims", 0) for r in results)
    print(f"\ntotal ${total:.5f}   prompt problems to look at: {problems}")
    return 1 if any(r["failed"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
