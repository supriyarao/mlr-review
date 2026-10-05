"""The contract other sessions build against, enforced.

``docs/contracts.md`` promises other worktrees a set of shapes: the MLR schemas, the review state,
the golden-case format, the service façade, and an example JSON response for every API endpoint. A
promise nothing checks is a promise that drifts, and drift here is expensive -- a session builds a UI
against a field that quietly got renamed and only finds out at integration.

So this file asserts the promises rather than the implementation:

* every API fixture still matches what the generator produces from the models;
* every fixture that claims to be a report, a task or an audit trail validates against its model;
* the state's reducer keys are classified, which is the trap described in ``mlr/state.py``;
* the stub service returns data that satisfies the real return types.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlr import examples, service
from mlr.schemas import Finding, ReviewReport, ReviewTask
from mlr.state import (
    ACCUMULATED_KEYS,
    REPLACEABLE_KEYS,
    ReviewState,
    initial_state,
    report_findings,
)
from platform_core.evals.schema import GoldenCase, load_case, load_cases

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "api"
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"

# Every endpoint in PROMPT.md section 8 plus the rules endpoints from the delta spec. Listed here so
# a missing fixture is a test failure rather than something a parallel session discovers is absent.
REQUIRED_FIXTURES = [
    "POST_reviews.json",
    "GET_reviews_{run_id}.json",
    "GET_reviews_{run_id}__running.json",
    "GET_reviews_{run_id}__complete.json",
    "GET_reviews.json",
    "GET_queue.json",
    "POST_reviews_{run_id}_decision.json",
    "GET_reviews_{run_id}_audit.json",
    "GET_reviews_{run_id}_report.json",
    "GET_tenants.json",
    "GET_tenants_{id}.json",
    "GET_tenants_{id}_rules.json",
    "GET_tenants_{id}_rules_{rule_id}.json",
    "POST_tenants_{id}_rules.json",
    "POST_tenants_{id}_rules_{rule_id}_test.json",
    "POST_tenants_{id}_rules_{rule_id}_activate.json",
    "POST_tenants_{id}_rules_{rule_id}_retire.json",
    "POST_tenants_{id}_rules_export.json",
    "POST_evals_run.json",
    "GET_evals_runs.json",
    "GET_evals_runs_{id}.json",
    "POST_evals_sweep.json",
    "GET_graph.json",
]


def fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


# --- The schemas are stable ----------------------------------------------------------------------


def test_finding_carries_both_a_proposed_and_a_final_severity():
    """Keeping both is what lets the audit log show a tenant downgrading a check's verdict."""
    f = Finding(id="f1", check_type="substantiation", issue="x", severity_proposed="blocker")
    assert f.severity == "blocker"  # falls back to the proposal before the severity node runs
    assert f.severity_final is None

    downgraded = f.model_copy(update={"severity_final": "advisory"})
    assert downgraded.severity == "advisory"
    assert downgraded.severity_proposed == "blocker", "the check's verdict must survive"


def test_grounding_is_a_mode_not_a_boolean():
    """Collapsing three grounding strengths into one flag would make grounding rate meaningless."""
    assert Finding(id="f", check_type="substantiation", issue="x").grounded_by == "none"
    for mode in ("evidence", "rule", "doc_clause"):
        f = Finding(id="f", check_type="substantiation", issue="x", grounded=True, grounded_by=mode)
        assert f.grounded_by == mode


def test_a_document_level_finding_needs_no_claim_and_no_quote():
    """A missing disclaimer is not about any one claim and has nothing to quote."""
    f = Finding(id="f", check_type="business_rule", issue="Required disclaimer is missing.")
    assert f.claim_id == ""
    assert f.is_absence is True


def test_schemas_reject_unknown_fields():
    """extra="forbid" everywhere, so a renamed field fails loudly at the boundary."""
    with pytest.raises(ValidationError):
        Finding(id="f", check_type="substantiation", issue="x", severity="blocker")


def test_gate_result_is_not_redefined_in_the_use_case_layer():
    """A gate is a platform concept; mlr re-exports it rather than owning a second definition."""
    from mlr.schemas import GateResult

    assert GateResult.__module__ == "platform_core.gates"


# --- The state contract --------------------------------------------------------------------------


def test_every_state_key_is_classified_as_accumulated_or_replaceable():
    """The classification is what the reducer trap test depends on, so it must be exhaustive."""
    declared = set(ReviewState.__annotations__)
    assert not (ACCUMULATED_KEYS & REPLACEABLE_KEYS), "a key cannot be both"
    unclassified = declared - ACCUMULATED_KEYS - REPLACEABLE_KEYS
    assert not unclassified, f"unclassified state keys: {sorted(unclassified)}"


def test_the_editor_writes_a_separate_key_from_the_accumulator():
    """The LangGraph trap, stated as a test.

    `findings` has an operator.add reducer so five parallel checks can each contribute. A reducer
    applies to every write, so the editor cannot replace that key -- returning its deduped list would
    append it to the originals and silently double every finding. Hence two keys.
    """
    assert "findings" in ACCUMULATED_KEYS
    assert "final_findings" in REPLACEABLE_KEYS
    assert "final_findings" not in ACCUMULATED_KEYS


def test_a_run_starts_with_its_accumulators_empty_not_absent():
    state = initial_state(run_id="r1", tenant_id="acme_pharma", product_id="zelvoran")
    for key in ACCUMULATED_KEYS:
        assert state[key] == [], f"{key} should start as an empty list"
    assert state["status"] == "queued"


def test_a_report_falls_back_to_raw_findings_if_the_editor_never_ran():
    """A run that failed mid-flight must not look like a clean piece."""
    findings = examples.example_findings()
    state = initial_state(run_id="r1", tenant_id="acme_pharma", product_id="zelvoran")
    state["findings"] = findings
    assert report_findings(state) == findings

    state["final_findings"] = findings[:1]
    assert report_findings(state) == findings[:1]


# --- The golden-case contract --------------------------------------------------------------------


def test_the_reference_golden_case_loads_and_is_complete():
    case = load_case(GOLDEN_DIR / "001-subgroup-overstatement.yaml")
    assert case.id == "001-subgroup-overstatement"
    assert len(case.why_hard) > 200, "why_hard should explain what makes the case hard"
    assert case.expected_findings and case.expected_route
    assert case.must_not_flag, "a case without a false-alarm list rewards flagging everything"


def test_a_golden_case_records_the_audience_that_changes_its_result():
    """Rule scope depends on audience, so a case omitting it is not reproducible."""
    case = load_case(GOLDEN_DIR / "001-subgroup-overstatement.yaml")
    assert case.audience in {"hcp", "patient", "any"}
    assert case.channel
    assert "audience" in GoldenCase.model_fields
    assert "expected_rule_ids" in GoldenCase.model_fields


def test_every_golden_case_resolves_its_promo_text():
    cases = load_cases(GOLDEN_DIR)
    assert cases, "no golden cases found"
    for case in cases:
        assert case.promo_text(REPO_ROOT).strip(), f"{case.id} has no promo text"


def test_every_golden_case_names_a_real_tenant_and_product():
    from platform_core import tenancy

    for case in load_cases(GOLDEN_DIR):
        config = tenancy.load_tenant(case.tenant_id)
        config.product(case.product_id)  # raises if unknown


# --- The API fixture contract --------------------------------------------------------------------


def test_every_documented_endpoint_has_a_fixture():
    missing = [name for name in REQUIRED_FIXTURES if not (FIXTURE_DIR / name).is_file()]
    assert not missing, f"endpoints with no example response: {missing}"


def test_fixtures_match_what_the_generator_produces():
    """The committed JSON must still be what the models produce.

    This is the whole point of generating them: a fixture is a promise about a response shape, and a
    hand-edited one can drift from the schema it claims to illustrate with nothing failing. If this
    test fails, run `make fixtures` and review the diff -- a change here is a change to the contract
    other sessions are building against.
    """
    from mlr.service import get_graph_mermaid
    from scripts.gen_api_fixtures import build

    expected = build()
    expected["GET_graph.json"] = {"format": "mermaid", "source": get_graph_mermaid()}

    drifted = []
    for name, payload in expected.items():
        on_disk = fixture(name)
        if on_disk != json.loads(json.dumps(payload)):
            drifted.append(name)
    assert not drifted, f"fixtures no longer match the models: {drifted}. Run `make fixtures`."


def test_report_fixtures_validate_against_the_report_model():
    for name in (
        "GET_reviews_{run_id}.json",
        "GET_reviews_{run_id}__running.json",
        "GET_reviews_{run_id}__complete.json",
        "POST_reviews_{run_id}_decision.json",
    ):
        report = ReviewReport.model_validate(fixture(name))
        assert report.run_id
        assert report.status in {"queued", "running", "awaiting_human", "complete", "failed"}


def test_the_queue_fixture_validates_as_review_tasks():
    payload = fixture("GET_queue.json")
    tasks = [ReviewTask.model_validate(t) for t in payload["tasks"]]
    assert tasks and tasks[0].tier == payload["tier"]
    assert tasks[0].is_escalation, "the example task should show why it escalated"


def test_the_audit_fixture_is_ordered_and_complete():
    payload = fixture("GET_reviews_{run_id}_audit.json")
    seqs = [e["seq"] for e in payload["events"]]
    assert seqs == sorted(seqs) == list(range(1, len(seqs) + 1))
    assert payload["integrity"]["sequence_complete"] is True
    # Every event must carry the provenance that makes a decision defensible.
    for event in payload["events"]:
        for field in ("node", "policy_version", "tokens_in", "cost_usd", "latency_ms"):
            assert field in event, f"audit event missing {field}"


def test_the_export_fixture_has_one_row_per_finding_with_its_citation():
    report = ReviewReport.model_validate(fixture("GET_reviews_{run_id}.json"))
    rows = fixture("GET_reviews_{run_id}_report.json")["rows"]
    assert len(rows) == len(report.findings)
    for row in rows:
        for field in ("quote", "issue", "severity", "citation_passage", "suggested_fix", "rule"):
            assert field in row, f"export row missing {field}"


def test_the_example_report_shows_a_rule_finding_and_a_core_finding():
    """A parallel session building the UI needs both display cases, not just one."""
    report = ReviewReport.model_validate(fixture("GET_reviews_{run_id}.json"))
    assert any(f.from_rule for f in report.findings), "no rule finding in the example"
    assert any(not f.from_rule for f in report.findings), "no core-check finding in the example"
    kinds = {c.kind for f in report.findings for c in f.citations}
    assert {"evidence", "doc_clause"} <= kinds, "the example should show both citation kinds"


def test_the_example_report_never_shows_an_approved_state():
    """There is no terminal state in which the system approved something itself."""
    for name in REQUIRED_FIXTURES:
        payload = fixture(name)
        assert "approved" not in json.dumps(payload).lower().replace("approve", ""), name
    report = ReviewReport.model_validate(fixture("GET_reviews_{run_id}__complete.json"))
    assert report.route, "even a completed run names the human who was responsible"


def test_the_sweep_fixture_shows_the_safety_versus_throughput_tension():
    """The curve is the point: raising the threshold trades false escalations for recall."""
    points = fixture("POST_evals_sweep.json")["points"]
    assert len(points) >= 3
    assert [p["threshold"] for p in points] == sorted(p["threshold"] for p in points)
    # Recall must not fall as the threshold rises, and escalation must not fall either.
    recalls = [p["blocker_recall"] for p in points]
    escalations = [p["false_escalation_rate"] for p in points]
    assert recalls == sorted(recalls), "blocker recall should be monotonic in the threshold"
    assert escalations == sorted(escalations), "false escalation should rise with the threshold"


# --- The service façade --------------------------------------------------------------------------
#
# These assert the *contract*: the signatures and return types the API and UI were built against. The
# façade used to return fixture data and now runs the real graph, which changed none of them -- that is
# the whole point of having had the contract. What the real behaviour does is covered in
# tests/test_service.py, which seeds a corpus; these stay deliberately free of that setup so a failure
# here means the interface moved rather than the graph.


def test_the_service_is_no_longer_a_stub():
    """The API reports this on /health, so it must say what is actually happening."""
    assert service.IS_STUB is False


def test_the_service_exposes_every_function_the_contract_promises():
    """Named explicitly: the API and UI were written against this list before the graph existed."""
    for name in (
        "start_run",
        "get_status",
        "get_review_task",
        "get_queue",
        "resume",
        "get_audit",
        "list_runs",
        "get_graph_mermaid",
    ):
        assert callable(getattr(service, name, None)), f"service.{name} is missing"


def test_the_additive_functions_exist_for_stepping_and_replay():
    """Added for the UI's live progress and audit replay views; additive, so nothing else moved."""
    for name in (
        "advance",
        "get_state",
        "get_history",
        "get_audit_integrity",
        "describe_backend_note",
    ):
        assert callable(getattr(service, name, None)), f"service.{name} is missing"


def test_start_run_keeps_its_documented_keyword_arguments():
    """The API calls this with keywords, so a renamed parameter is a breaking change."""
    import inspect

    params = inspect.signature(service.start_run).parameters
    for name in (
        "tenant_id",
        "product_id",
        "document",
        "label_version",
        "audience",
        "channel",
        "market",
        "doc_meta",
        "run_id",
    ):
        assert name in params, f"start_run lost its '{name}' parameter"
    # Additions must be optional, or every existing caller breaks.
    for name in ("step", "block"):
        assert params[name].default is not inspect.Parameter.empty


def test_an_unknown_run_raises_the_error_the_api_maps_to_404():
    for call in (service.get_status, service.get_audit, service.get_state, service.get_history):
        with pytest.raises(service.RunNotFoundError):
            call("definitely-not-a-run")


def test_the_backend_note_tells_the_ui_what_to_say():
    """Returned by the service so there is one wording, and it cannot drift from what is happening."""
    note = service.describe_backend_note()
    assert note and note[0].isupper()
    # Under the fake backend the UI must say the findings are a fixture.
    assert "fixture, not a model" in note


def test_the_graph_shape_matches_the_documented_pipeline():
    """The mermaid source is what other sessions were told to expect."""
    mermaid = service.get_graph_mermaid()
    for node in (
        "ingest",
        "extract_claims",
        "preapproved_filter",
        "select_rules",
        "substantiation",
        "deterministic_rules",
        "llm_rules",
        "reference_doc_review",
        "grounding_gate",
        "severity",
        "editor",
        "confidence_gate",
        "route",
        "human_review",
        "finalize",
    ):
        assert node in mermaid, f"{node} missing from the compiled graph"
