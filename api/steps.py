"""A run, step by step: what each node was given, what it produced, and how it went.

Deterministic code. No model involved. Everything here is read from two records the run already
keeps -- the audit log (one event per node) and the checkpoint history (the state after every
step) -- so this is a view over what happened, never a reconstruction of it.

Why both are needed: an audit event holds a *summary* of a node's output ("5 claims, 3 efficacy"),
which is right for a log. The claims themselves are in the state. A reader who clicks a step wants
the second, with the first beside it.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.encoders import jsonable_encoder

from api import submissions
from mlr import service
from mlr.graph import CORE_CHECKS, PIPELINE
from mlr.prompts.checks import NODE_FOR_CHECK_TYPE

# What each of the first nodes reads from the state. Listed by hand because a node's inputs
# are not recorded anywhere: the state is one shared object and a node simply reads what it needs.
# Nodes not listed show their output only.
READS: dict[str, tuple[str, ...]] = {
    "extract_claims": ("document",),
    "preapproved_filter": ("claims",),
}

# State keys that grow by appending, so "what this step added" is the new tail of the list.
APPENDED = ("findings", "gate_results", "errors")

# Bookkeeping a reader does not need to see as a step's output.
HIDDEN = {"run_id", "status", "cost"}


def _parsed(raw: str) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """What changed in the state between two checkpoints."""
    changed: dict[str, Any] = {}
    for key, value in after.items():
        if key in HIDDEN:
            continue
        previous = before.get(key)
        if key in APPENDED:
            added = list(value or [])[len(previous or []) :]
            if added:
                changed[key] = added
        elif value != previous:
            changed[key] = value
    return changed


def _findings_by_node(values: dict[str, Any]) -> dict[str, list[Any]]:
    """Attribute each raw finding to the reviewer that produced it.

    The reviewers run in one parallel step, so the checkpoint cannot say who added what. A finding
    can: one from a tenant rule names its rule, and the rule's kind names the node; any other is
    from the specialist check of its type.
    """
    rule_ids = {r["id"] for r in values.get("rules_applied") or []}
    grouped: dict[str, list[Any]] = {}
    for finding in values.get("findings") or []:
        if finding.get("rule_id"):
            # Every rule finding comes from the one rules node now, whatever kind the rule is.
            node = "house_rules" if finding["rule_id"] in rule_ids else ""
        else:
            node = NODE_FOR_CHECK_TYPE.get(finding["check_type"], "")
        if node:
            grouped.setdefault(node, []).append(finding)
    return grouped


def steps(run_id: str) -> dict[str, Any]:
    """Every node of the pipeline, in order, with its status, input and output for this run."""
    report = service.get_status(run_id)
    state = service.get_state(run_id)
    history = jsonable_encoder(service.get_history(run_id))
    values: dict[str, Any] = jsonable_encoder(state["values"] or {})
    pending = set(state["next"] or [])

    events: dict[str, dict[str, Any]] = {}
    for event in service.get_audit(run_id):
        events[event["node"]] = event

    # For each node: the state just before it ran, and what the step it ran in changed.
    before: dict[str, dict[str, Any]] = {}
    wrote: dict[str, dict[str, Any]] = {}
    for snapshot, following in zip(history, history[1:], strict=False):
        for node in snapshot["next"]:
            before[node] = snapshot["values"]
            wrote[node] = _delta(snapshot["values"], following["values"])

    by_reviewer = _findings_by_node(values)
    reviewers = set(by_reviewer) | set(CORE_CHECKS) | {"house_rules"}
    stored = submissions.load(run_id)
    last_done = max((i for i, n in enumerate(PIPELINE) if n in events), default=-1)
    live = report.status in ("queued", "running")

    out = []
    for index, node in enumerate(PIPELINE):
        event = events.get(node)
        if event and (event.get("error") or event.get("event_type") == "error"):
            status = "failed"
        elif node == "human_review" and report.status == "awaiting_human":
            status = "waiting"
        elif event:
            status = "done"
        elif node in pending and live:
            status = "running"
        elif index < last_done:
            # The run went past it without an event: nothing for it to do on this piece.
            status = "skipped"
        else:
            status = "pending"

        given = {k: before.get(node, {}).get(k) for k in READS.get(node, ())}
        if node == "ingest" and stored:
            given = {"submitted_text": stored["document"], "filename": stored.get("filename", "")}

        produced = wrote.get(node, {})
        if node == "ingest":
            # The façade cleans the text when the run is created, so the checkpoint before ingest
            # already holds it; what ingest hands on is the document and its section map.
            produced = {k: values.get(k) for k in ("document", "doc_meta") if values.get(k)}
        elif node in reviewers:
            produced = {"findings": by_reviewer.get(node, [])} if status == "done" else {}

        out.append(
            {
                "node": node,
                "status": status,
                "given": given,
                "produced": produced,
                "summary": _parsed(event.get("output_json", "")) if event else None,
                "model": event.get("model", "") if event else "",
                "prompt_version": event.get("prompt_version", "") if event else "",
                "tokens_in": event.get("tokens_in", 0) if event else 0,
                "tokens_out": event.get("tokens_out", 0) if event else 0,
                "cost_usd": event.get("cost_usd", 0.0) if event else 0.0,
                "latency_ms": event.get("latency_ms", 0) if event else 0,
                "error": event.get("error", "") if event else "",
            }
        )
    return {"run_id": run_id, "status": report.status, "steps": out}
