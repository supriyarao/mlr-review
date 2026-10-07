"""Drive one review through the complete human-in-the-loop cycle, from the command line.

Submit -> watch the graph run -> see it suspend at the human step -> act as the reviewer -> watch it
resume -> read the audit trail. The same calls the API makes, without needing the API or the UI, so the
backend can be exercised and demonstrated on its own.

Run with ``make loop``. With ``MLR_LLM_BACKEND=fake`` it needs no API key.

``--step`` advances one node at a time and prints the state in between, which is the clearest way to see
what each stage contributes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from mlr import service, store, trace
from mlr.nodes.ingest import extract_text
from platform_core.env import load_env
from platform_core.llm import describe_backend

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMO = REPO_ROOT / "corpus" / "promo"

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
RED, AMBER, GREEN, BLUE, CYAN = "\033[31m", "\033[33m", "\033[32m", "\033[34m", "\033[36m"


def rule(title: str = "") -> None:
    print(f"\n{BOLD}{'=' * 94}{RESET}")
    if title:
        print(f"{BOLD}{title}{RESET}")
        print(f"{BOLD}{'=' * 94}{RESET}")


def front_matter(stem: str) -> dict:
    import yaml

    _, front, _ = (PROMO / f"{stem}.md").read_text(encoding="utf-8").split("---\n", 2)
    return yaml.safe_load(front)


def show_report(report, *, show_findings: bool = True) -> None:
    print(f"  status   {BOLD}{report.status}{RESET}")
    print(f"  route    {report.route or DIM + '(not yet routed)' + RESET}")
    print(f"  claims   {report.claims_extracted}")
    print(f"  rules    {len(report.rules_applied)} in force")
    blockers, advisories = len(report.blockers()), len(report.advisories())
    print(
        f"  findings {blockers} blocker, {advisories} advisory, {len(report.ungrounded())} ungrounded"
    )

    if not show_findings:
        return

    for f in report.findings:
        colour = RED if f.is_blocker else AMBER
        ground = f"{GREEN}{f.grounded_by}{RESET}" if f.grounded else f"{RED}UNGROUNDED{RESET}"
        rule_tag = f"  {BLUE}[{f.rule_id} v{f.rule_version}]{RESET}" if f.rule_id else ""
        merged = (
            f"  {DIM}(merged {len(f.merged_from) + 1} findings){RESET}" if f.merged_from else ""
        )
        print(
            f"\n    {colour}{f.severity.upper():8}{RESET} {f.check_type:18} conf={f.confidence:3} "
            f"grounded={ground}{rule_tag}{merged}"
        )
        print(
            f'      quote  {DIM}"{f.quote[:88]}"{RESET}'
            if f.quote
            else f"      {DIM}(absence){RESET}"
        )
        print(f"      issue  {f.issue[:140]}")
        for c in f.citations[:2]:
            print(f'      cites  {DIM}[{c.kind}] {c.source_id[:46]} :: "{c.passage[:62]}"{RESET}')
        if f.severity_rule_id:
            print(f"      graded {DIM}by {f.severity_rule_id}{RESET}")

    if report.gate_results:
        print(f"\n  {BOLD}gates{RESET}")
        for gate in report.gate_results:
            mark = f"{GREEN}pass{RESET}" if gate.passed else f"{RED}FAIL{RESET}"
            print(f"    {mark}  {gate.gate:12} {gate.details[:96]}")

    if report.escalation:
        print(f"\n  {BOLD}escalated to {report.escalation.tier}{RESET}")
        for reason in report.escalation.reasons:
            print(f"    {AMBER}{reason.code}{RESET}  {reason.explanation[:110]}")


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description="Run one review through the full loop.")
    parser.add_argument("--piece", default="zelvoran-hcp-subgroup-claim")
    parser.add_argument("--tenant", default="")
    parser.add_argument("--step", action="store_true", help="advance one node at a time")
    parser.add_argument(
        "--decision", default="request_changes", choices=("approve", "reject", "request_changes")
    )
    parser.add_argument(
        "--disagree",
        action="store_true",
        help="disagree with the first finding, to exercise the eval-candidate flywheel",
    )
    args = parser.parse_args()

    path = PROMO / f"{args.piece}.md"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in PROMO.glob("*.md")))
        print(f"No such piece: {args.piece}\nAvailable: {available}", file=sys.stderr)
        return 1

    meta = front_matter(args.piece)
    tenant_id = args.tenant or (
        "borealis_bio" if meta["product_id"] == "nexapril" else "acme_pharma"
    )

    rule("1. Submit")
    info = describe_backend()
    print(f"  backend  {BOLD}{info['backend']}{RESET}")
    if info["backend"] == "ollama":
        # Easy to hit by accident: the Make target sets MLR_LLM_BACKEND=fake, a direct invocation does
        # not, and a CPU-only local model takes about two minutes per call -- so a nine-call review
        # looks like a hang rather than slow progress.
        print(
            f"  {AMBER}warning{RESET} a local model takes roughly two minutes per call on CPU, so "
            f"this review will take ~15 minutes. Use MLR_LLM_BACKEND=fake for an instant run."
        )
    print(f"  {DIM}{service.describe_backend_note()}{RESET}")
    trace.banner()
    print(f"\n  piece    {args.piece}")
    print(f"  tenant   {tenant_id}")
    print(
        f"  product  {meta['product_id']} v{meta.get('label_version', '')}  "
        f"audience={meta['audience']}  channel={meta['channel']}"
    )
    if service.IS_STUB:
        print(f"\n  {RED}service is still a stub{RESET}")
        return 1

    started = time.perf_counter()
    run_id = service.start_run(
        tenant_id=tenant_id,
        product_id=meta["product_id"],
        document=extract_text(path),
        label_version=str(meta.get("label_version") or ""),
        audience=meta["audience"],
        channel=meta["channel"],
        market="us",
        step=args.step,
        # Blocking in step mode too, so the first pause happens before we print.
        block=True,
    )
    print(f"\n  run_id   {CYAN}{run_id}{RESET}  (registered synchronously, then executed)")

    if args.step:
        rule("2. Step through the graph")
        step = 0
        while True:
            state = service.get_state(run_id)
            pending = state["next"]
            if not pending:
                break
            step += 1
            # Advancing runs the named node plus any nodes after it that are not pause points --
            # the eight parallel branches all run in the step after preapproved_filter.
            print(
                f"\n  {BOLD}step {step}{RESET} -> running {CYAN}{', '.join(pending)}{RESET}"
                f"{DIM} (and any non-pausing nodes that follow){RESET}"
            )
            report = service.advance(run_id)
            findings = len(report.findings)
            print(
                f"    after: {findings} finding(s), route={report.route or '-'}, "
                f"status={report.status}"
            )
            if report.status == "awaiting_human":
                break
    else:
        rule("2. Run")
        print(f"  {DIM}graph executed to the human step{RESET}")

    rule("3. The review, as the system produced it")
    report = service.get_status(run_id)
    show_report(report)
    print(
        f"\n  cost     ${report.cost.cost_usd:.5f}  "
        f"{report.cost.tokens_in} in / {report.cost.tokens_out} out  "
        f"{report.cost.model_calls} model call(s)"
    )
    print(f"  policy   {report.policy_version}  (tenant config {report.config_version})")

    rule("4. The run is waiting for a person")
    task = service.get_review_task(run_id)
    if task is None:
        print(f"  {RED}no pending review task{RESET}")
        return 1
    print(f"  tier     {BOLD}{task.tier}{RESET}")
    print(f"  asking   {task.headline()}")
    print(f"  evidence {len(task.evidence)} passage(s) attached")
    queue = service.get_queue(task.tier)
    print(f"\n  {DIM}queue for {task.tier}: {len(queue)} pending{RESET}")
    print(
        f"  {DIM}The system recommends; a human decides. There is no state in which it approves.{RESET}"
    )

    rule(f"5. Reviewer decides: {args.decision}")
    per_finding = []
    if args.disagree and report.findings:
        per_finding = [
            {
                "finding_id": report.findings[0].id,
                "verdict": "disagree",
                "note": "Defensible in context; the headline names the population.",
            }
        ]
        print(
            f"  {DIM}disagreeing with {report.findings[0].id} to exercise the eval flywheel{RESET}"
        )

    resumed = service.resume(
        run_id,
        {
            "decision": args.decision,
            "reviewer": "a.reviewer",
            "tier": task.tier,
            "note": "Reviewed via scripts/loop.py",
            "per_finding": per_finding,
        },
    )
    print(f"  status   {BOLD}{resumed.status}{RESET}")
    print(f"  {DIM}resumed from the checkpoint; nothing before the human step was re-run{RESET}")

    rule("6. Audit trail")
    events = service.get_audit(run_id)
    integrity = service.get_audit_integrity(run_id)
    print(
        f"  {len(events)} events   sequence complete: {integrity['sequence_complete']}   "
        f"duplicates: {integrity['duplicate_seqs'] or 'none'}"
    )
    print(f"\n  {'seq':>4}  {'node':22} {'type':8} {'model':20} {'prompt':24} {'ms':>6} {'$':>9}")
    for event in events:
        print(
            f"  {event['seq']:>4}  {event['node']:22} {event['event_type']:8} "
            f"{(event['model'] or '-'):20} {(event['prompt_version'] or '-'):24} "
            f"{event['latency_ms']:>6} {event['cost_usd']:>9.5f}"
        )

    rule("7. Replay: what the system knew at each step")
    history = service.get_history(run_id)
    print(f"  {len(history)} checkpoints")
    for entry in history[:: max(1, len(history) // 6)]:
        values = entry["values"]
        print(
            f"    step {entry['step']:>2}  next={','.join(entry['next']) or 'END':24} "
            f"claims={len(values.get('claims') or []):>2} "
            f"findings={len(values.get('final_findings') or values.get('findings') or []):>2} "
            f"route={values.get('route') or '-'}"
        )

    candidates = sorted((REPO_ROOT / "evals" / "candidates").glob(f"{run_id}-*.json"))
    if candidates:
        rule("8. Eval candidates from the reviewer's disagreement")
        for candidate in candidates:
            print(f"  {GREEN}wrote{RESET} evals/candidates/{candidate.name}")
            print(
                f"  {DIM}draft only: a human writes why_hard before it joins the golden set{RESET}"
            )

    record = store.get(run_id)
    rule("Done")
    print(
        f"  run {run_id}: {record.status}, routed to {record.route}, "
        f"{record.blockers} blocker(s) in {time.perf_counter() - started:.1f}s"
    )
    print(f"\n  {DIM}Inspect further:{RESET}")
    print(f"  {DIM}  make loop PIECE={args.piece} STEP=1        # step node by node{RESET}")
    print(
        f"  {DIM}  uv run python -m scripts.loop --piece zelvoran-hcp-isi-placement --tenant borealis_bio{RESET}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
