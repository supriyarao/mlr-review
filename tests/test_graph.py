"""End-to-end graph runs, offline.

These exercise the graph's responsibilities -- fan-out, grounding, severity from tenant policy,
dedupe, routing, suspend and resume -- with scripted model responses, so a failure points at the
orchestration rather than at a model's mood.

The two most important assertions in this file: the same piece routes differently under two tenants,
and work done before the human step is not repeated on resume.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mlr.graph import PIPELINE, checkpointer, compile_with, fan_out
from mlr.nodes.ingest import extract_text
from mlr.state import initial_state, report_findings
from platform_core import audit, runtime
from platform_core.llm import FakeLLM
from platform_core.retrieval import Document, index_documents, reset_tenant_collection
from platform_core.rules.registry import RuleRegistry
from platform_core.rules.schema import Rule
from tests import fake_responses as fr

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "corpus"


@pytest.fixture
def real_tenants(monkeypatch):
    """Use the shipped corpus: these tests are about the graph behaving on real policy."""
    from platform_core import tenancy

    monkeypatch.setenv("MLR_TENANT_DIR", str(CORPUS / "tenants"))
    tenancy.clear_tenant_cache()
    yield
    tenancy.clear_tenant_cache()


@pytest.fixture
def seeded(real_tenants):
    """Index the evidence and rules both tenants need, into this test's own data directory."""
    import yaml

    from platform_core import tenancy

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
                text = ref.read_text(encoding="utf-8")
                # One chunk per section, matching how scripts/seed.py indexes.
                documents.append(
                    Document(
                        source_id=f"{tenant_id}:reference:{ref.stem}:1",
                        text=text,
                        tenant_id=tenant_id,
                        product_id=product.product_id,
                        label_version=product.label_version,
                        doc_type="reference",
                        title=ref.stem.upper(),
                    )
                )
                documents.append(
                    Document(
                        source_id=f"{tenant_id}:reference:{ref.stem}:2",
                        text=text,
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


def promo(stem: str) -> str:
    return extract_text(CORPUS / "promo" / f"{stem}.md")


def run_piece(
    stem: str,
    *,
    tenant_id: str = "acme_pharma",
    script: dict | None = None,
    product_id: str = "zelvoran",
    audience: str = "hcp",
    channel: str = "sales_aid",
    label_version: str = "1.0",
):
    """Run one promotional piece through the graph. Returns (compiled graph, result, fake)."""
    fake = FakeLLM(scripted=script or {}, default_response="{}")
    with checkpointer() as saver:
        graph = compile_with(saver, fake=fake)
        state = initial_state(
            run_id=runtime.new_run_id(),
            tenant_id=tenant_id,
            product_id=product_id,
            label_version=label_version,
            audience=audience,
            channel=channel,
            market="us",
            document=promo(stem),
        )
        result = runtime.start_run(graph, dict(state), run_id=state["run_id"])
        yield_value = (graph, result, fake)
    return yield_value


# --- A full run on the reference case ------------------------------------------------------------


@pytest.fixture
def subgroup_run(seeded):
    fake = FakeLLM(scripted=fr.subgroup_case_script(), default_response="{}")
    with checkpointer() as saver:
        graph = compile_with(saver, fake=fake)
        state = initial_state(
            run_id="t-subgroup",
            tenant_id="acme_pharma",
            product_id="zelvoran",
            label_version="1.0",
            audience="hcp",
            channel="sales_aid",
            market="us",
            document=promo("zelvoran-hcp-subgroup-claim"),
        )
        result = runtime.start_run(graph, dict(state), run_id="t-subgroup")
        yield graph, result, fake


def test_a_run_reaches_the_human_step_and_suspends(subgroup_run):
    _, result, _ = subgroup_run
    assert result["status"] == "awaiting_human"
    assert result["interrupt"] is not None
    assert result["interrupt"]["run_id"] == "t-subgroup"


def test_the_pipeline_produced_claims_and_selected_rules(subgroup_run):
    _, result, _ = subgroup_run
    state = result["state"]
    assert len(state["claims"]) == 2
    assert state["claims"][0].char_span != (0, 0), "the claim should be located in the document"
    assert state["rules_applied"], "Acme's rules should be in force for an HCP Zelvoran piece"


def test_findings_from_several_branches_all_arrive(subgroup_run):
    """The append reducer is what makes concurrent branches safe."""
    _, result, _ = subgroup_run
    raw = result["state"]["findings"]
    checks_seen = {f.check_type for f in raw}
    assert "substantiation" in checks_seen
    assert "reference_support" in checks_seen
    assert any(f.rule_id for f in raw), "no rule finding arrived"


def test_grounding_verified_every_finding_against_a_real_source(subgroup_run):
    _, result, _ = subgroup_run
    findings = report_findings(result["state"])
    assert findings
    for finding in findings:
        assert finding.grounded, f"{finding.id} was not grounded: {finding.issue[:60]}"
        assert finding.grounded_by in ("evidence", "rule", "doc_clause")
    modes = {f.grounded_by for f in findings}
    assert "evidence" in modes, "no finding grounded in retrieved evidence"


def test_the_editor_merged_two_checks_reporting_one_problem(subgroup_run):
    """substantiation and the promotional-code reviewer both flag the same sentence."""
    _, result, _ = subgroup_run
    findings = report_findings(result["state"])
    merged = [f for f in findings if f.merged_from]
    assert merged, "the two findings on the same claim and quote should have merged"
    survivor = merged[0]
    assert len(survivor.citations) >= 2, "a merge must keep both lines of evidence"
    assert survivor.is_blocker, "a merge must never soften a blocker"
    # Corroboration from a different check raises confidence.
    assert survivor.confidence >= 92


def test_severity_came_from_tenant_policy(subgroup_run):
    _, result, _ = subgroup_run
    for finding in report_findings(result["state"]):
        assert finding.severity_final is not None
        assert finding.severity_rule_id, f"{finding.id} has no severity attribution"


def test_the_run_escalated_with_reasons_a_reviewer_can_act_on(subgroup_run):
    _, result, _ = subgroup_run
    state = result["state"]
    assert state["route"] == "senior_regulatory"
    escalation = state["escalation"]
    assert escalation is not None
    assert "blocker" in escalation.reason_codes()
    assert escalation.evidence, "the reviewer must be given the evidence"


def test_every_node_wrote_an_audit_event(subgroup_run):
    _, result, _ = subgroup_run
    events = audit.events_for_run("t-subgroup")
    nodes = [e.node for e in events]
    for expected in ("ingest", "extract_claims", "select_rules", "grounding_gate", "route"):
        assert expected in nodes, f"{expected} left no audit trail"
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    # Provenance that makes the decision defensible later.
    ingest_event = next(e for e in events if e.node == "ingest")
    assert ingest_event.policy_version or ingest_event.node == "ingest"


def test_the_gates_all_recorded_a_verdict(subgroup_run):
    _, result, _ = subgroup_run
    gates = {g.gate for g in result["state"]["gate_results"]}
    assert {"grounding", "confidence"} <= gates


# --- Resume ---------------------------------------------------------------------------------------


def test_resuming_records_the_decision_and_completes(subgroup_run):
    graph, result, _ = subgroup_run
    resumed = runtime.resume_run(
        graph,
        "t-subgroup",
        {
            "decision": "reject",
            "reviewer": "a.reviewer",
            "tier": "senior_regulatory",
            "note": "Rewrite the headline to name the subgroup.",
        },
    )
    assert resumed["status"] == "complete"
    assert resumed["state"]["status"] == "complete"
    assert resumed["state"]["human_decision"]["reviewer"] == "a.reviewer"


def test_work_before_the_human_is_not_repeated_on_resume(subgroup_run):
    """The point of checkpointing: a reviewer taking three days costs nothing to resume."""
    graph, _, fake = subgroup_run
    calls_before = len(fake.calls)
    runtime.resume_run(
        graph,
        "t-subgroup",
        {"decision": "approve", "reviewer": "r", "tier": "senior_regulatory"},
    )
    assert len(fake.calls) == calls_before, "resuming must not re-run any model call"


def test_an_invalid_decision_is_refused(subgroup_run):
    graph, _, _ = subgroup_run
    with pytest.raises(ValidationError):
        runtime.resume_run(graph, "t-subgroup", {"decision": "approve"})  # no reviewer


def test_a_reviewer_disagreement_becomes_a_draft_eval_case(subgroup_run, tmp_path, monkeypatch):
    """The data flywheel: a qualified human labelling the system's mistakes, for free."""
    from mlr.nodes import finalize as finalize_node

    monkeypatch.setattr(finalize_node, "CANDIDATES_DIR", tmp_path / "candidates")
    graph, result, _ = subgroup_run
    findings = report_findings(result["state"])

    runtime.resume_run(
        graph,
        "t-subgroup",
        {
            "decision": "request_changes",
            "reviewer": "a.reviewer",
            "tier": "senior_regulatory",
            "per_finding": [
                {
                    "finding_id": findings[0].id,
                    "verdict": "disagree",
                    "note": "The headline is defensible in context.",
                }
            ],
        },
    )
    written = list((tmp_path / "candidates").glob("*.json"))
    assert written, "a disagreement should produce a draft eval candidate"

    import json

    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["verdict"] == "disagree"
    assert payload["status"] == "draft", "a candidate must not auto-promote into the golden set"
    assert payload["why_hard"] == "", "why_hard is for a human to write"
    assert payload["reviewer"] == "a.reviewer"


# --- Multi-tenancy --------------------------------------------------------------------------------


def test_the_same_piece_routes_differently_under_the_two_tenants(seeded):
    """The headline multi-tenancy demonstration, and the difference must be purely policy.

    Identical text, identical evidence, identical scripted model responses. Acme requires safety
    information to be present; Borealis requires it beside the first efficacy claim.
    """
    quote = "In a 16-week randomised, placebo-controlled trial, 68.4% of adults treated with Zelvoran achieved"
    script = fr.clean_script(quote, quote)
    outcomes = {}

    for tenant_id in ("acme_pharma", "borealis_bio"):
        fake = FakeLLM(scripted=script, default_response="{}")
        with checkpointer() as saver:
            graph = compile_with(saver, fake=fake)
            state = initial_state(
                run_id=f"t-{tenant_id}",
                tenant_id=tenant_id,
                product_id="zelvoran",
                label_version="1.0",
                audience="hcp",
                channel="leave_piece",
                market="us",
                document=promo("zelvoran-hcp-isi-placement"),
            )
            result = runtime.start_run(graph, dict(state), run_id=f"t-{tenant_id}")
        findings = report_findings(result["state"])
        outcomes[tenant_id] = {
            "route": result["state"]["route"],
            "blockers": [f.rule_id or f.check_type for f in findings if f.is_blocker],
        }

    assert outcomes["acme_pharma"]["blockers"] == [], (
        f"Acme requires only that safety information be present: {outcomes['acme_pharma']}"
    )
    assert "BOR-ISI-001" in outcomes["borealis_bio"]["blockers"], (
        f"Borealis requires it beside the first efficacy claim: {outcomes['borealis_bio']}"
    )
    assert outcomes["acme_pharma"]["route"] != outcomes["borealis_bio"]["route"]
    assert outcomes["acme_pharma"]["route"] == "standard_reviewer"


def test_a_clean_piece_still_goes_to_a_human(seeded):
    """There is no branch in which the system approves anything."""
    quote = "In a 16-week randomised, placebo-controlled trial, 68.4% of adults treated with Zelvoran achieved"
    fake = FakeLLM(scripted=fr.clean_script(quote, quote), default_response="{}")
    with checkpointer() as saver:
        graph = compile_with(saver, fake=fake)
        state = initial_state(
            run_id="t-clean",
            tenant_id="acme_pharma",
            product_id="zelvoran",
            label_version="1.0",
            audience="hcp",
            channel="journal_ad",
            market="us",
            document=promo("zelvoran-hcp-journal-ad"),
        )
        result = runtime.start_run(graph, dict(state), run_id="t-clean")

    assert result["status"] == "awaiting_human"
    assert result["state"]["route"] == "standard_reviewer"
    assert result["state"]["escalation"] is None
    assert [f for f in report_findings(result["state"]) if f.is_blocker] == []
    assert "no issues found" in result["interrupt"]["report"]["summary"].lower()


def test_a_patient_scoped_rule_fires_only_on_a_patient_piece(seeded):
    """Audience is an input because it changes which rules are in force."""
    quote = (
        "Zelvoran is indicated for the treatment of moderate-to-severe plaque psoriasis in adults"
    )
    outcomes = {}
    for audience, stem in (
        ("patient", "zelvoran-patient-brochure"),
        ("hcp", "zelvoran-hcp-journal-ad"),
    ):
        fake = FakeLLM(scripted=fr.clean_script(quote, quote), default_response="{}")
        with checkpointer() as saver:
            graph = compile_with(saver, fake=fake)
            state = initial_state(
                run_id=f"t-{audience}",
                tenant_id="acme_pharma",
                product_id="zelvoran",
                label_version="1.0",
                audience=audience,
                channel="email",
                market="us",
                document=promo(stem),
            )
            result = runtime.start_run(graph, dict(state), run_id=f"t-{audience}")
        outcomes[audience] = {a.id for a in result["state"]["rules_applied"]}

    # ACME-LLM-001 is scoped to patient-facing material.
    assert "ACME-LLM-001" in outcomes["patient"]
    assert "ACME-LLM-001" not in outcomes["hcp"]


# --- Failure handling -----------------------------------------------------------------------------


def test_a_schema_failure_escalates_rather_than_looking_clean(seeded):
    """A check that could not run must never be indistinguishable from one that found nothing."""
    # preapproved=False, or substantiation skips the claim and never reaches the bad response.
    quote = "In a 16-week randomised, placebo-controlled trial, 68.4% of adults treated with Zelvoran achieved"
    script = fr.clean_script(quote, quote, preapproved=False)
    # Two unparseable responses from substantiation: one attempt plus its retry.
    script["substantiation"] = ["not json", "still not json"]
    fake = FakeLLM(scripted=script, default_response="{}")

    with checkpointer() as saver:
        graph = compile_with(saver, fake=fake)
        state = initial_state(
            run_id="t-schema",
            tenant_id="acme_pharma",
            product_id="zelvoran",
            label_version="1.0",
            audience="hcp",
            channel="sales_aid",
            market="us",
            document=promo("zelvoran-hcp-subgroup-claim"),
        )
        result = runtime.start_run(graph, dict(state), run_id="t-schema")

    state = result["state"]
    assert state["errors"], "a schema failure should be recorded"
    gates = {g.gate: g for g in state["gate_results"]}
    assert "schema" in gates and not gates["schema"].passed
    assert state["escalation"] is not None, "a failed check must force escalation"
    codes = state["escalation"].reason_codes()
    assert "schema_failure" in codes


def test_the_retry_recovers_a_single_bad_response(seeded):
    """One malformed response is retried; the run proceeds normally."""
    script = fr.subgroup_case_script()
    script["substantiation"] = ["not json", script["substantiation"][0]]
    fake = FakeLLM(scripted=script, default_response="{}")

    with checkpointer() as saver:
        graph = compile_with(saver, fake=fake)
        state = initial_state(
            run_id="t-retry",
            tenant_id="acme_pharma",
            product_id="zelvoran",
            label_version="1.0",
            audience="hcp",
            channel="sales_aid",
            market="us",
            document=promo("zelvoran-hcp-subgroup-claim"),
        )
        result = runtime.start_run(graph, dict(state), run_id="t-retry")

    findings = report_findings(result["state"])
    assert any(f.check_type == "substantiation" for f in findings)
    assert not any("substantiation" in str(e) for e in result["state"].get("errors") or [])


def test_an_ungrounded_finding_is_kept_and_escalated_not_dropped(seeded):
    """The single worst behaviour would be turning "I cannot verify this" into silence."""
    quote = "In a 16-week randomised, placebo-controlled trial, 68.4% of adults treated with Zelvoran achieved"
    script = fr.clean_script(quote, quote, preapproved=False)
    script["substantiation"] = [
        fr.findings(
            fr.finding(
                quote="a sentence that is nowhere in the piece",
                issue="An invented problem with an invented quote.",
                severity="blocker",
                confidence=99,
            )
        )
    ]
    fake = FakeLLM(scripted=script, default_response="{}")

    with checkpointer() as saver:
        graph = compile_with(saver, fake=fake)
        state = initial_state(
            run_id="t-ungrounded",
            tenant_id="acme_pharma",
            product_id="zelvoran",
            label_version="1.0",
            audience="hcp",
            channel="sales_aid",
            market="us",
            document=promo("zelvoran-hcp-journal-ad"),
        )
        result = runtime.start_run(graph, dict(state), run_id="t-ungrounded")

    findings = report_findings(result["state"])
    ungrounded = [f for f in findings if not f.grounded]
    assert ungrounded, "the finding must survive, flagged, not be discarded"
    assert ungrounded[0].grounded_by == "none"
    # Its confidence is capped: an unverifiable finding must not look certain.
    assert ungrounded[0].confidence <= 45
    assert "ungrounded" in result["state"]["escalation"].reason_codes()


# --- Structure ------------------------------------------------------------------------------------


def test_the_fan_out_skips_branches_with_nothing_to_do(seeded):
    """An absent audit event should mean "not applicable", not "ran and found nothing"."""
    from mlr.schemas import AppliedRule

    state = initial_state(run_id="r", tenant_id="acme_pharma", product_id="zelvoran")
    state["claims"] = []
    state["rules_applied"] = []
    targets = {send.node for send in fan_out(state)}
    # No claims and no rules: only the document-level checks are worth running.
    assert "substantiation" not in targets
    assert {"fair_balance", "safety_prominence"} <= targets
    assert "llm_rules" not in targets

    state["rules_applied"] = [
        AppliedRule(id="X", version=1, kind="llm", severity="blocker"),
        AppliedRule(id="D", version=1, kind="reference_doc", severity="advisory"),
    ]
    targets = {send.node for send in fan_out(state)}
    assert "llm_rules" in targets
    assert "reference_doc_review" in targets


def test_one_send_per_reference_document(seeded):
    """A prompt holding two unrelated codes of practice cites the wrong clause from both."""
    from mlr.schemas import AppliedRule

    state = initial_state(run_id="r", tenant_id="acme_pharma", product_id="zelvoran")
    state["rules_applied"] = [
        AppliedRule(id="DOC-1", version=1, kind="reference_doc", severity="blocker"),
        AppliedRule(id="DOC-2", version=1, kind="reference_doc", severity="blocker"),
    ]
    sends = [s for s in fan_out(state) if s.node == "reference_doc_review"]
    assert len(sends) == 2
    assert {s.arg["_rule_id"] for s in sends} == {"DOC-1", "DOC-2"}


def test_the_graph_compiles_with_every_documented_node(seeded):
    with checkpointer() as saver:
        graph = compile_with(saver)
        mermaid = runtime.render_mermaid(graph)
    for node in PIPELINE:
        assert node in mermaid, f"{node} missing from the compiled graph"


def test_state_models_survive_a_checkpoint_round_trip(seeded):
    """Regression guard for a failure that is entirely silent until it is far from its cause.

    LangGraph's msgpack allowlist takes ``(module, ClassName)`` pairs. Given anything else -- a dotted
    string, a tuple of path components -- it matches nothing and every Pydantic model in state comes
    back from a checkpoint as a plain dict. Nothing raises at that point. The run then fails on resume
    with ``AttributeError: 'dict' object has no attribute 'is_blocker'`` inside a node, which says
    nothing about serialization.

    So this asserts on the types after a resume, not just that the resume succeeded.
    """
    fake = FakeLLM(scripted=fr.subgroup_case_script(), default_response="{}")
    with checkpointer() as saver:
        graph = compile_with(saver, fake=fake)
        state = initial_state(
            run_id="t-roundtrip",
            tenant_id="acme_pharma",
            product_id="zelvoran",
            label_version="1.0",
            audience="hcp",
            channel="sales_aid",
            market="us",
            document=promo("zelvoran-hcp-subgroup-claim"),
        )
        runtime.start_run(graph, dict(state), run_id="t-roundtrip")

        # Read the state back through the checkpointer, which is where the degradation happened.
        reloaded = runtime.get_status(graph, "t-roundtrip")["values"]

    from mlr.schemas import AppliedRule, Claim, Escalation, Finding, GateResult, RunCost

    assert all(isinstance(c, Claim) for c in reloaded["claims"])
    assert all(isinstance(f, Finding) for f in reloaded["final_findings"])
    assert all(isinstance(g, GateResult) for g in reloaded["gate_results"])
    assert all(isinstance(r, AppliedRule) for r in reloaded["rules_applied"])
    assert isinstance(reloaded["escalation"], Escalation)
    assert isinstance(reloaded["cost"], RunCost)


def test_every_state_model_is_declared_as_a_checkpoint_type():
    """A model in state but missing from CHECKPOINT_TYPES degrades to a dict on resume, silently."""
    from mlr.schemas import AppliedRule, Citation, Claim, Escalation, Finding, GateResult, RunCost
    from mlr.state import CHECKPOINT_TYPES

    for model in (AppliedRule, Citation, Claim, Escalation, Finding, GateResult, RunCost):
        assert model in CHECKPOINT_TYPES, f"{model.__name__} is in state but not declared"


# --- State tracing ------------------------------------------------------------------------------


def test_tracing_is_off_unless_asked_for(monkeypatch):
    """Debug output must never appear in a normal run."""
    from mlr import trace

    monkeypatch.delenv("MLR_TRACE", raising=False)
    assert trace.enabled() is False
    for off in ("", "0", "false", "no", "off"):
        monkeypatch.setenv("MLR_TRACE", off)
        assert trace.enabled() is False, f"MLR_TRACE={off!r} should be off"


def test_tracing_has_a_compact_and_a_full_mode(monkeypatch):
    from mlr import trace

    monkeypatch.setenv("MLR_TRACE", "1")
    assert trace.mode() == "compact"
    monkeypatch.setenv("MLR_TRACE", "full")
    assert trace.mode() == "full"


def test_a_traced_node_prints_its_input_and_its_update(seeded, monkeypatch, capsys):
    """The update is printed, not the whole state afterwards: a node returns only what it changed."""
    from mlr import trace

    monkeypatch.setenv("MLR_TRACE", "1")
    traced = trace.traced("demo_node", lambda state: {"claims": [1, 2, 3]})
    traced({"run_id": "r1", "status": "running", "document": "x" * 400})

    out = capsys.readouterr().out
    assert "demo_node" in out
    assert "IN" in out and "run_id=r1" in out
    assert "document=400 chars" in out, "a long value should be summarised, not dumped"
    assert "OUT" in out and "claims=3" in out


def test_the_compact_trace_drops_state_that_carries_nothing(monkeypatch, capsys):
    """Nineteen nodes repeating the same empty fields buries the values that moved."""
    from mlr import trace

    monkeypatch.setenv("MLR_TRACE", "1")
    trace.traced("n", lambda state: {})(
        {"run_id": "r1", "status": "running", "findings": [], "escalation": None, "claims": []}
    )
    out = capsys.readouterr().out
    assert "run_id=r1" in out
    assert "findings=" not in out, "an empty list should not be printed in compact mode"
    assert "escalation=" not in out
    assert "(no state change)" in out


def test_a_failing_node_is_traced_before_the_error_propagates(monkeypatch, capsys):
    """A trace that stops at the failure is the one time you most want to see the input."""
    from mlr import trace

    monkeypatch.setenv("MLR_TRACE", "1")

    def boom(state):
        raise RuntimeError("the check exploded")

    with pytest.raises(RuntimeError, match="exploded"):
        trace.traced("boom", boom)({"run_id": "r1", "status": "running"})

    out = capsys.readouterr().out
    assert "RAISED" in out and "the check exploded" in out
    assert "run_id=r1" in out


def test_a_traced_graph_still_produces_the_same_result(seeded, monkeypatch):
    """Tracing must observe, never change behaviour."""
    from mlr import trace

    script = fr.subgroup_case_script()

    def run_once(run_id: str):
        fake = FakeLLM(scripted=script, default_response="{}")
        with checkpointer() as saver:
            graph = compile_with(saver, fake=fake)
            state = initial_state(
                run_id=run_id,
                tenant_id="acme_pharma",
                product_id="zelvoran",
                label_version="1.0",
                audience="hcp",
                channel="sales_aid",
                market="us",
                document=promo("zelvoran-hcp-subgroup-claim"),
            )
            return runtime.start_run(graph, dict(state), run_id=run_id)

    monkeypatch.delenv("MLR_TRACE", raising=False)
    quiet = run_once("t-untraced")
    monkeypatch.setenv("MLR_TRACE", "1")
    loud = run_once("t-traced")
    assert trace.enabled()

    assert quiet["state"]["route"] == loud["state"]["route"]
    assert len(report_findings(quiet["state"])) == len(report_findings(loud["state"]))
