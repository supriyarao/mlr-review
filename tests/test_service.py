"""The service façade, running the real graph.

These are the tests the API and UI depend on: the façade's job is to make the graph usable over HTTP,
which means a run has to be addressable the instant it is created, readable at every stage in one
shape, resumable by a later request, and never silently stuck.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mlr import service, store
from mlr.nodes.ingest import extract_text
from mlr.schemas import ReviewReport, ReviewTask
from platform_core.retrieval import Document, index_documents, reset_tenant_collection
from platform_core.rules.registry import RuleRegistry
from platform_core.rules.schema import Rule
from platform_core.tenancy import TenantConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "corpus"


@pytest.fixture
def seeded(monkeypatch):
    """The shipped corpus, indexed into this test's own data directory."""
    from platform_core import tenancy

    monkeypatch.setenv("MLR_TENANT_DIR", str(CORPUS / "tenants"))
    tenancy.clear_tenant_cache()

    for tenant_id in ("acme_pharma", "borealis_bio"):
        config = tenancy.load_tenant(tenant_id)
        reset_tenant_collection(tenant_id)

        documents: list[Document] = []
        for product in config.products:
            label = CORPUS / "labels" / f"{product.product_id}-v{product.label_version}.md"
            if label.is_file():
                documents.append(
                    Document(
                        source_id=f"{tenant_id}:label:{product.product_id}",
                        text=label.read_text(encoding="utf-8"),
                        tenant_id=tenant_id,
                        product_id=product.product_id,
                        label_version=product.label_version,
                        doc_type="label",
                        title=f"{product.name} label",
                    )
                )
            prefix = "zel" if product.product_id == "zelvoran" else "nex"
            for ref in sorted((CORPUS / "references").glob(f"{prefix}-*.md")):
                documents.append(
                    Document(
                        source_id=f"{tenant_id}:reference:{ref.stem}:1",
                        text=ref.read_text(encoding="utf-8"),
                        tenant_id=tenant_id,
                        product_id=product.product_id,
                        label_version=product.label_version,
                        doc_type="reference",
                        title=ref.stem.upper(),
                    )
                )

        preapproved = CORPUS / "preapproved_claims" / f"{tenant_id}.yaml"
        if preapproved.is_file():
            raw = yaml.safe_load(preapproved.read_text(encoding="utf-8")) or {}
            for entry in raw.get("claims", []) or []:
                documents.append(
                    Document(
                        source_id=f"{tenant_id}:preapproved:{entry['id']}",
                        text=entry["text"],
                        tenant_id=tenant_id,
                        product_id=str(entry.get("product_id", raw.get("product_id", ""))),
                        label_version=str(entry.get("label_version", raw.get("label_version", ""))),
                        doc_type="preapproved_claim",
                        title=f"Pre-approved claim {entry['id']}",
                    )
                )
        index_documents(tenant_id, documents)

        registry = RuleRegistry(tenant_id)
        for path in sorted(tenancy.tenant_rules_dir(tenant_id).glob("*.yaml")):
            rule = Rule.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            stored = registry.save_new_version(rule)
            registry.activate(
                stored.id,
                stored.version,
                reference_docs_dir=tenancy.tenant_reference_docs_dir(tenant_id),
            )
    yield
    tenancy.clear_tenant_cache()


def promo(stem: str) -> str:
    return extract_text(CORPUS / "promo" / f"{stem}.md")


def submit(stem: str, *, tenant_id: str = "acme_pharma", **overrides) -> str:
    kwargs = {
        "tenant_id": tenant_id,
        "product_id": "zelvoran",
        "document": promo(stem),
        "label_version": "1.0",
        "audience": "hcp",
        "channel": "sales_aid",
        "market": "us",
        # Blocking so a test gets a finished run without polling.
        "block": True,
    }
    kwargs.update(overrides)
    return service.start_run(**kwargs)


# --- The façade is no longer a stub --------------------------------------------------------------


def test_the_service_runs_the_real_graph():
    assert service.IS_STUB is False


def test_the_backend_note_says_where_findings_come_from():
    """The UI must never be ambiguous about which part was stood in for."""
    note = service.describe_backend_note()
    assert "fixture, not a model" in note
    assert "Everything else runs for real" in note


# --- Submitting ----------------------------------------------------------------------------------


def test_a_run_is_addressable_the_instant_it_is_created(seeded):
    """A UI that polls immediately after submitting must not get a 404 for the run it just made."""
    run_id = service.start_run(
        tenant_id="acme_pharma",
        product_id="zelvoran",
        document=promo("zelvoran-hcp-journal-ad"),
        audience="hcp",
        channel="journal_ad",
        block=False,
    )
    assert store.exists(run_id)
    report = service.get_status(run_id)
    assert isinstance(report, ReviewReport)
    assert report.status in ("queued", "running", "awaiting_human", "complete")


def test_an_unknown_tenant_fails_at_submission_not_as_a_failed_run(seeded):
    """Raised synchronously, so a bad request fails at the API boundary rather than by polling."""
    with pytest.raises(TenantConfigError):
        service.start_run(tenant_id="nope", product_id="zelvoran", document="copy", block=True)
    with pytest.raises(TenantConfigError):
        service.start_run(tenant_id="acme_pharma", product_id="nope", document="copy", block=True)


def test_an_unknown_run_is_not_found(seeded):
    for call in (service.get_status, service.get_audit, service.get_state, service.get_history):
        with pytest.raises(service.RunNotFoundError):
            call("nonexistent")


# --- A complete run ------------------------------------------------------------------------------


@pytest.fixture
def finished_run(seeded):
    return submit("zelvoran-hcp-subgroup-claim")


def test_a_run_reaches_the_human_step_and_reports_its_findings(finished_run):
    report = service.get_status(finished_run)
    assert report.status == "awaiting_human"
    assert report.route == "senior_regulatory"
    assert report.blockers(), "the subgroup case should produce blockers"
    assert report.claims_extracted > 0
    assert report.rules_applied, "Acme's rules should be in force"
    assert report.policy_version, "a report must carry the policy that produced it"


def test_every_finding_is_grounded_and_says_how(finished_run):
    for finding in service.get_status(finished_run).findings:
        assert finding.grounded, f"{finding.id} is ungrounded: {finding.issue[:60]}"
        assert finding.grounded_by in ("evidence", "rule", "doc_clause")


def test_a_rule_finding_and_a_core_finding_on_one_sentence_are_merged(finished_run):
    """The editor's job, and the reason the merge key is the quote rather than the claim id."""
    merged = [f for f in service.get_status(finished_run).findings if f.merged_from]
    assert merged, "two findings quoting the same sentence should have become one"
    survivor = merged[0]
    assert len(survivor.citations) >= 2, "a merge must keep every line of evidence"
    assert survivor.is_blocker, "a merge must never soften a blocker"
    kinds = {c.kind for c in survivor.citations}
    assert len(kinds) > 1, "the merged finding should carry both kinds of authority"


def test_the_report_shape_is_the_same_at_every_stage(seeded):
    """status is the only field that gates what is populated; a caller never branches on a payload."""
    run_id = submit("zelvoran-hcp-journal-ad", channel="journal_ad")
    report = service.get_status(run_id)
    assert isinstance(report, ReviewReport)
    assert report.status == "awaiting_human"

    resumed = service.resume(run_id, {"decision": "approve", "reviewer": "r", "tier": report.route})
    assert isinstance(resumed, ReviewReport)
    assert resumed.status == "complete"
    # Same fields present, different values.
    assert set(report.model_dump()) == set(resumed.model_dump())


# --- The human step ------------------------------------------------------------------------------


def test_a_pending_run_appears_in_its_tier_queue(finished_run):
    report = service.get_status(finished_run)
    assert [t.run_id for t in service.get_queue(report.route)] == [finished_run]
    # And not in another tier's queue: that is how the demo simulates logging in as someone else.
    assert finished_run not in [t.run_id for t in service.get_queue("legal")]


def test_the_review_task_tells_the_reviewer_why_it_reached_them(finished_run):
    task = service.get_review_task(finished_run)
    assert isinstance(task, ReviewTask)
    assert task.is_escalation
    assert "blocker" in task.reason_codes()
    assert task.evidence, "the reviewer must be given the evidence"
    assert task.report["findings"], "and the findings themselves"


def test_resuming_records_the_decision(finished_run):
    report = service.resume(
        finished_run,
        {
            "decision": "reject",
            "reviewer": "a.reviewer",
            "tier": "senior_regulatory",
            "note": "Rewrite the headline to name the subgroup.",
        },
    )
    assert report.status == "complete"
    assert store.get(finished_run).status == "complete"
    # And it leaves the queue.
    assert finished_run not in [t.run_id for t in service.get_queue()]


def test_an_invalid_decision_changes_nothing(finished_run):
    """It arrives over HTTP, so it is validated before anything is written."""
    with pytest.raises(ValidationError):
        service.resume(finished_run, {"decision": "approve"})  # no reviewer
    with pytest.raises(ValidationError):
        service.resume(finished_run, {"decision": "auto_approve", "reviewer": "r", "tier": "t"})
    assert store.get(finished_run).status == "awaiting_human", "the run must still be pending"


def test_a_completed_run_has_no_pending_task(finished_run):
    service.resume(
        finished_run, {"decision": "approve", "reviewer": "r", "tier": "senior_regulatory"}
    )
    assert service.get_review_task(finished_run) is None


# --- A clean piece -------------------------------------------------------------------------------


def test_a_clean_piece_still_goes_to_a_human(seeded):
    """There is no state in which the system approves anything."""
    run_id = submit("zelvoran-hcp-journal-ad", channel="journal_ad")
    report = service.get_status(run_id)
    assert report.status == "awaiting_human"
    assert report.route == "standard_reviewer"
    assert report.escalation is None
    assert report.blockers() == []

    task = service.get_review_task(run_id)
    assert task is not None and not task.is_escalation
    assert "no issues found" in task.report["summary"].lower()


# --- Multi-tenancy -------------------------------------------------------------------------------


def test_the_same_piece_routes_differently_under_the_two_tenants(seeded):
    """The headline demonstration, through the façade the UI actually calls."""
    outcomes = {}
    for tenant_id in ("acme_pharma", "borealis_bio"):
        run_id = submit("zelvoran-hcp-isi-placement", tenant_id=tenant_id, channel="leave_piece")
        report = service.get_status(run_id)
        outcomes[tenant_id] = {
            "route": report.route,
            "rules": [f.rule_id for f in report.blockers() if f.rule_id],
        }

    assert outcomes["acme_pharma"]["route"] == "standard_reviewer"
    assert outcomes["borealis_bio"]["route"] != "standard_reviewer"
    assert "BOR-ISI-001" in outcomes["borealis_bio"]["rules"]


def test_a_deterministic_rule_finding_needs_no_model_and_is_fully_confident(seeded):
    """Which is why the rules demo works with no API key at all."""
    run_id = submit("zelvoran-hcp-isi-placement", tenant_id="borealis_bio", channel="leave_piece")
    rule_findings = [f for f in service.get_status(run_id).findings if f.rule_id == "BOR-ISI-001"]
    assert rule_findings
    finding = rule_findings[0]
    assert finding.confidence == 100
    assert finding.model == "", "a deterministic rule must not involve a model"
    assert finding.grounded_by == "rule"


# --- Listing and audit ---------------------------------------------------------------------------


def test_runs_are_listable_and_filterable(seeded):
    acme = submit("zelvoran-hcp-journal-ad", channel="journal_ad")
    borealis = submit("zelvoran-hcp-isi-placement", tenant_id="borealis_bio", channel="leave_piece")

    all_ids = {r.run_id for r in service.list_runs()}
    assert {acme, borealis} <= all_ids
    assert {r.run_id for r in service.list_runs(tenant_id="borealis_bio")} == {borealis}
    assert all(r.status == "awaiting_human" for r in service.list_runs(status="awaiting_human"))


def test_the_audit_trail_is_ordered_and_complete(finished_run):
    events = service.get_audit(finished_run)
    assert events
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    nodes = {e["node"] for e in events}
    for expected in ("ingest", "extract_claims", "preapproved_filter", "consolidate", "route"):
        assert expected in nodes
    integrity = service.get_audit_integrity(finished_run)
    assert integrity["sequence_complete"] is True
    assert integrity["duplicate_seqs"] == []


def test_the_checkpoint_history_backs_the_replay_view(finished_run):
    history = service.get_history(finished_run)
    assert len(history) > 1
    assert [h["step"] for h in history] == list(range(len(history)))
    # State accumulates: the last checkpoint knows more than the first.
    assert len(history[-1]["values"].get("claims") or []) >= len(
        history[0]["values"].get("claims") or []
    )


# --- Step mode -----------------------------------------------------------------------------------


def test_a_stepped_run_advances_one_stage_at_a_time(seeded):
    """For the demo: watching findings accumulate is far clearer than a spinner."""
    run_id = submit("zelvoran-hcp-subgroup-claim", step=True)

    state = service.get_state(run_id)
    assert state["stepping"] is True
    assert state["next"], "a stepped run should be paused with something pending"

    seen_nodes: list[str] = []
    findings_over_time: list[int] = []
    for _ in range(len(state["steppable_nodes"]) + 2):
        state = service.get_state(run_id)
        if not state["next"]:
            break
        seen_nodes.extend(state["next"])
        report = service.advance(run_id)
        findings_over_time.append(len(report.findings))
        if report.status == "awaiting_human":
            break

    assert "ingest" in seen_nodes and "route" in seen_nodes
    assert findings_over_time == sorted(findings_over_time) or max(findings_over_time) > 0
    assert service.get_status(run_id).status == "awaiting_human"


def test_advancing_a_run_that_is_not_stepping_is_refused(seeded):
    run_id = submit("zelvoran-hcp-journal-ad", channel="journal_ad")
    with pytest.raises(ValueError, match="not started in step mode"):
        service.advance(run_id)


def test_a_stepped_run_at_the_human_step_must_be_resumed_not_advanced(seeded):
    """A stepped run still stops for a person, and that stop is not a step to advance past.

    Advancing it would resume the interrupt with no decision and fail validation deep inside the node,
    so the refusal happens here and says what to do instead.
    """
    run_id = submit("zelvoran-hcp-journal-ad", channel="journal_ad", step=True)
    for _ in range(20):
        if service.get_status(run_id).status == "awaiting_human":
            break
        if not service.get_state(run_id)["next"]:
            break
        service.advance(run_id)

    assert service.get_status(run_id).status == "awaiting_human"
    with pytest.raises(ValueError, match="waiting for a reviewer"):
        service.advance(run_id)


def test_a_stepped_run_can_still_be_resumed_by_a_reviewer(seeded):
    run_id = submit("zelvoran-hcp-subgroup-claim", step=True)
    for _ in range(20):
        if service.get_status(run_id).status == "awaiting_human":
            break
        if not service.get_state(run_id)["next"]:
            break
        service.advance(run_id)

    assert service.get_status(run_id).status == "awaiting_human"
    report = service.resume(
        run_id, {"decision": "reject", "reviewer": "r", "tier": "senior_regulatory"}
    )
    assert report.status == "complete"


# --- Failure handling ----------------------------------------------------------------------------


def test_a_run_that_crashes_is_recorded_as_failed(seeded, monkeypatch):
    """A crash in a worker thread must not leave a run stuck at queued forever."""

    def explode(*args, **kwargs):
        raise RuntimeError("checkpointer unavailable")

    # Patched on the service, which imported the name directly -- patching mlr.graph would not reach it.
    monkeypatch.setattr(service, "compile_with", explode)
    run_id = submit("zelvoran-hcp-journal-ad", channel="journal_ad")

    record = store.get(run_id)
    assert record.status == "failed"
    assert "checkpointer unavailable" in record.error
    # And the failure is visible through the façade, not only in the table.
    assert service.get_status(run_id).error
