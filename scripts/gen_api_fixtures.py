"""Generate the API response fixtures under ``tests/fixtures/api/``.

Why generated rather than hand-written: a fixture is a promise about a response shape, and a
hand-written one can drift from the model it claims to illustrate without anything failing. These are
built through the Pydantic models in :mod:`mlr.examples`, so a fixture cannot be schema-invalid, and
``tests/test_contracts.py`` fails if the committed files no longer match what this script produces.

Run with ``make fixtures``. Deterministic: regenerating with no code change produces no diff.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlr import examples
from platform_core import tenancy
from platform_core.rules.schema import Rule

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "api"

RUN_ID = examples.RUN_ID


def _tenant_payload(tenant_id: str, *, detail: bool) -> dict[str, Any]:
    config = tenancy.load_tenant(tenant_id)
    payload: dict[str, Any] = {
        "tenant_id": config.tenant_id,
        "name": config.name,
        "version": config.version,
        "confidence_threshold": config.confidence_threshold,
        "products": [p.model_dump(mode="json") for p in config.products],
        "config_hash": config.config_hash(),
    }
    if detail:
        payload["severity_rules"] = [r.model_dump(mode="json") for r in config.severity_rules]
        payload["escalation"] = config.escalation.model_dump(mode="json")
        payload["policy_flags"] = config.policy_flags
        payload["active_rule_count"] = len(list(tenancy.tenant_rules_dir(tenant_id).glob("*.yaml")))
    return payload


def _rule_payload(tenant_id: str, filename: str) -> dict[str, Any]:
    import yaml

    path = tenancy.tenant_rules_dir(tenant_id) / filename
    rule = Rule.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return rule.model_dump(mode="json")


def build() -> dict[str, Any]:
    """Every fixture, keyed by the filename it is written to."""
    report = examples.example_report()
    complete = examples.example_report(status="complete")
    task = examples.example_review_task()

    running = report.model_copy(
        update={
            "status": "running",
            "route": "",
            "escalation": None,
            "findings": report.findings[:1],
            "gate_results": [],
        }
    )

    acme_rule = _rule_payload("acme_pharma", "acme-ban-001.yaml")
    draft_rule = {
        **acme_rule,
        "id": "ACME-BAN-003",
        "version": 1,
        "status": "draft",
        "title": "Prohibited hype language",
        "description": "Promotional material must not describe an Acme product as a "
        "breakthrough or as revolutionary, because neither term has a "
        "defined clinical meaning.",
        "params": {
            "type": "banned_terms",
            "terms": ["breakthrough", "revolutionary"],
            "regex": False,
        },
    }

    return {
        # --- Reviews (PROMPT.md section 8) ------------------------------------------------------
        "POST_reviews.json": {"run_id": RUN_ID, "status": "queued"},
        "GET_reviews_{run_id}.json": report.model_dump(mode="json"),
        "GET_reviews_{run_id}__running.json": running.model_dump(mode="json"),
        "GET_reviews_{run_id}__complete.json": complete.model_dump(mode="json"),
        "GET_reviews.json": {
            "runs": [
                {
                    "run_id": report.run_id,
                    "tenant_id": report.tenant_id,
                    "product_id": report.product_id,
                    "status": report.status,
                    "route": report.route,
                    "blockers": len(report.blockers()),
                    "advisories": len(report.advisories()),
                    "cost_usd": report.cost.cost_usd,
                }
            ],
            "total": 1,
        },
        "GET_queue.json": {"tier": "senior_regulatory", "tasks": [task.model_dump(mode="json")]},
        "POST_reviews_{run_id}_decision.json": complete.model_dump(mode="json"),
        "GET_reviews_{run_id}_audit.json": {
            "run_id": RUN_ID,
            "events": examples.example_audit_events(),
            "integrity": {
                "run_id": RUN_ID,
                "event_count": 12,
                "sequence_complete": True,
                "duplicate_seqs": [],
            },
        },
        # The .xlsx export is a binary download; the JSON export is one row per finding, matching
        # the spreadsheet's columns exactly so the two cannot disagree.
        "GET_reviews_{run_id}_report.json": {
            "run_id": RUN_ID,
            "tenant_id": report.tenant_id,
            "product_id": report.product_id,
            "policy_version": report.policy_version,
            "route": report.route,
            "rows": [
                {
                    "finding_id": f.id,
                    "location": next(
                        (c.location for c in examples.example_claims() if c.id == f.claim_id), ""
                    ),
                    "quote": f.quote,
                    "issue": f.issue,
                    "check": f.check_type,
                    "severity": f.severity,
                    "confidence": f.confidence,
                    "grounded_by": f.grounded_by,
                    "citation_source": f.citations[0].source_id if f.citations else "",
                    "citation_passage": f.citations[0].passage if f.citations else "",
                    "suggested_fix": f.suggested_fix,
                    "rule": f"{f.rule_id} v{f.rule_version}" if f.rule_id else "",
                    "severity_rule": f.severity_rule_id,
                }
                for f in report.findings
            ],
        },
        # --- Tenants ----------------------------------------------------------------------------
        "GET_tenants.json": {
            "tenants": [_tenant_payload(t, detail=False) for t in tenancy.tenant_ids()]
        },
        "GET_tenants_{id}.json": _tenant_payload("acme_pharma", detail=True),
        # --- Rules (PROMPT_RULES.md R5) ---------------------------------------------------------
        "GET_tenants_{id}_rules.json": {
            "tenant_id": "acme_pharma",
            "rules": [
                _rule_payload("acme_pharma", name)
                for name in ("acme-ban-001.yaml", "acme-llm-001.yaml", "acme-doc-001.yaml")
            ],
            "total": 8,
        },
        "GET_tenants_{id}_rules_{rule_id}.json": {
            "rule": acme_rule,
            "history": [
                {
                    "version": 1,
                    "status": "active",
                    "created_at": "2026-10-05T12:00:00+00:00",
                    "created_by": "seed",
                    "change_note": "seeded from acme-ban-001.yaml",
                }
            ],
        },
        "POST_tenants_{id}_rules.json": {
            "rule": draft_rule,
            "validation": {
                "rule_id": "ACME-BAN-003",
                "rule_version": 1,
                "errors": [],
                "warnings": [],
                "can_save": True,
                "can_activate": True,
                "accuracy": 1.0,
                "summary": "2/2 examples behave as labelled",
            },
        },
        "POST_tenants_{id}_rules_{rule_id}_test.json": {
            "rule_id": "ACME-BAN-003",
            "rule_version": 1,
            "source": "sample:zelvoran-hcp-launch-announcement",
            "violations": [
                {
                    "rule_id": "ACME-BAN-003",
                    "rule_version": 1,
                    "tenant_id": "acme_pharma",
                    "severity": "blocker",
                    "check_type": "business_rule",
                    "quote": "breakthrough",
                    "char_span": [214, 226],
                    "message": "Uses the prohibited term 'breakthrough'.",
                    "suggested_fix": "",
                    "citation_text": draft_rule["description"],
                    "citation_kind": "rule",
                    "citation_source_id": "ACME-BAN-003 v1",
                    "segment_id": "",
                    "confidence": 100,
                }
            ],
            "saved": False,
        },
        "POST_tenants_{id}_rules_{rule_id}_activate.json": {
            "rule_id": "ACME-BAN-003",
            "version": 1,
            "status": "active",
            "policy_version_before": "a6062d2216a1137f",
            "policy_version_after": "3f91c0d7b8e21a45",
            "regression_check": {
                "ran": True,
                "regressed": False,
                "movements": [
                    {
                        "metric": "blocker_recall.recall",
                        "baseline": 0.9231,
                        "current": 0.9231,
                        "delta": 0.0,
                    }
                ],
                "overridden": False,
                "override_reason": "",
            },
        },
        "POST_tenants_{id}_rules_{rule_id}_retire.json": {
            "rule_id": "ACME-BAN-003",
            "version": 1,
            "status": "retired",
            "policy_version_after": "a6062d2216a1137f",
        },
        "POST_tenants_{id}_rules_export.json": {
            "tenant_id": "acme_pharma",
            "written": [
                "corpus/tenants/acme_pharma/rules/acme-ban-001.yaml",
                "corpus/tenants/acme_pharma/rules/acme-ban-003.yaml",
            ],
        },
        # --- Evals ------------------------------------------------------------------------------
        "POST_evals_run.json": {
            "eval_run_id": "e7f8a9b0c1d2",
            "mode": "offline",
            "status": "queued",
        },
        "GET_evals_runs.json": {
            "runs": [
                {
                    "eval_run_id": "e7f8a9b0c1d2",
                    "mode": "offline",
                    "started_at": "2026-10-05T12:40:00+00:00",
                    "git_sha": "cf227f4",
                    "cases": 13,
                    "blocker_recall": 0.9231,
                    "false_escalation_rate": 0.0,
                }
            ]
        },
        "GET_evals_runs_{id}.json": {
            "eval_run_id": "e7f8a9b0c1d2",
            "mode": "offline",
            "started_at": "2026-10-05T12:40:00+00:00",
            "finished_at": "2026-10-05T12:41:12+00:00",
            "git_sha": "cf227f4",
            "backend": "fake",
            "model_strong": "claude-sonnet-5",
            "model_fast": "claude-haiku-4-5",
            "repeats": 1,
            "tenant_config_hashes": {
                "acme_pharma": "c754d98309699857",
                "borealis_bio": "1c8b9fada77781ce",
            },
            "metrics": {
                "blocker_recall": {"expected": 13, "matched": 12, "recall": 0.9231},
                "advisory_recall": {"expected": 4, "matched": 3, "recall": 0.75},
                "false_escalation_rate": {"standard_expected": 2, "escalated": 0, "rate": 0.0},
                "routing": {"scored": 13, "correct": 12, "accuracy": 0.9231},
                "grounding_rate": {"findings": 21, "grounded": 21, "rate": 1.0},
                "pass_rate": {"cases": 13, "passed": 12, "pass_rate": 0.9231},
                "cost_and_latency": {
                    "runs": 13,
                    "latency_p50_ms": 11840,
                    "latency_p95_ms": 18200,
                    "cost_usd_total": 0.8923,
                    "cost_usd_per_run": 0.0686,
                },
                "per_rule": {
                    "deterministic": {"rules": 10, "scored": 10, "all_perfect": True},
                    "not_self_testable": [{"rule_id": "ACME-LLM-001", "kind": "llm"}],
                },
            },
            "cases": [
                {
                    "case_id": "001-subgroup-overstatement",
                    "tenant_id": "acme_pharma",
                    "passed": True,
                    "expected_route": "senior_regulatory",
                    "actual_route": "senior_regulatory",
                    "route_correct": True,
                    "latency_ms": 11840,
                    "cost_usd": 0.0686,
                }
            ],
        },
        "POST_evals_sweep.json": {
            "thresholds": [50, 60, 70, 80, 90],
            "note": (
                "Routing recomputed from stored findings at each threshold. No model calls: the "
                "curve compares policy settings against one fixed set of findings, so a difference "
                "is the threshold's effect and not model nondeterminism."
            ),
            "points": [
                {"threshold": 50, "blocker_recall": 0.9231, "false_escalation_rate": 0.0},
                {"threshold": 60, "blocker_recall": 0.9231, "false_escalation_rate": 0.0},
                {"threshold": 70, "blocker_recall": 0.9231, "false_escalation_rate": 0.0},
                {"threshold": 80, "blocker_recall": 0.9231, "false_escalation_rate": 0.5},
                {"threshold": 90, "blocker_recall": 1.0, "false_escalation_rate": 1.0},
            ],
        },
        # --- Graph ------------------------------------------------------------------------------
        "GET_graph.json": {"format": "mermaid", "source": examples.__dict__ and None},
    }


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = build()

    # Filled in here rather than in build() so the mermaid source has one origin.
    from mlr.service import get_graph_mermaid

    fixtures["GET_graph.json"] = {"format": "mermaid", "source": get_graph_mermaid()}

    written = []
    for name, payload in sorted(fixtures.items()):
        path = FIXTURE_DIR / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        written.append(path.name)

    print(f"Wrote {len(written)} fixtures to {FIXTURE_DIR.relative_to(REPO_ROOT)}/")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
