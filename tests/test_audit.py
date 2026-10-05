"""The audit log: append-only, ordered, and complete enough to defend a decision."""

from __future__ import annotations

import json

from platform_core import audit


def test_emit_records_the_conditions_that_produced_an_output():
    """The fields an auditor asks for must all survive a round trip."""
    ev = audit.emit(
        run_id="r1",
        tenant_id="acme_pharma",
        node="substantiation",
        input_value={"claim": "reduces risk"},
        output={"findings": 1},
        citations=[{"source_id": "label-1", "passage": "in a prespecified subgroup"}],
        model="claude-sonnet-5",
        prompt_version="substantiation-v1",
        policy_version="abc123",
        label_version="v1.0",
        tokens_in=1200,
        tokens_out=300,
        cost_usd=0.0054,
        latency_ms=820,
    )

    assert ev.id is not None
    assert ev.seq == 1
    assert ev.prompt_version == "substantiation-v1"
    assert ev.policy_version == "abc123"
    assert ev.label_version == "v1.0"
    assert json.loads(ev.citations_json)[0]["source_id"] == "label-1"


def test_inputs_are_hashed_not_stored():
    """A promotional document must not be duplicated into every audit row."""
    document = "Zelvoran reduces risk. " * 500
    ev = audit.emit(run_id="r1", tenant_id="t", node="ingest", input_value=document)
    assert document not in ev.input_hash
    assert len(ev.input_hash) == 16


def test_identical_inputs_hash_identically_regardless_of_key_order():
    a = audit.hash_input({"claim": "x", "tenant": "acme"})
    b = audit.hash_input({"tenant": "acme", "claim": "x"})
    assert a == b


def test_unserialisable_input_still_hashes():
    """Hashing must never be the reason a review run fails."""
    assert audit.hash_input(object()) != ""


def test_events_are_returned_in_execution_order():
    for node in ["ingest", "extract_claims", "substantiation", "editor", "route"]:
        audit.emit(run_id="r1", tenant_id="t", node=node)
    assert [e.node for e in audit.events_for_run("r1")] == [
        "ingest",
        "extract_claims",
        "substantiation",
        "editor",
        "route",
    ]
    assert [e.seq for e in audit.events_for_run("r1")] == [1, 2, 3, 4, 5]


def test_sequence_numbers_are_per_run():
    audit.emit(run_id="r1", tenant_id="t", node="ingest")
    audit.emit(run_id="r2", tenant_id="t", node="ingest")
    assert audit.events_for_run("r1")[0].seq == 1
    assert audit.events_for_run("r2")[0].seq == 1


def test_module_exposes_no_way_to_change_history():
    """Append-only by construction: there is no update or delete to call."""
    forbidden = [n for n in dir(audit) if any(w in n.lower() for w in ("update", "delete", "drop"))]
    assert forbidden == [], f"audit log must be append-only, found {forbidden}"


def test_human_decisions_share_the_log_with_model_steps():
    """An auditor asks 'who decided this'; splitting the record makes that harder."""
    audit.emit(run_id="r1", tenant_id="t", node="substantiation", event_type="node")
    audit.emit(
        run_id="r1",
        tenant_id="t",
        node="human_review",
        event_type="human",
        output={"decision": "reject", "reviewer": "a.reviewer"},
    )
    events = audit.events_for_run("r1")
    assert [e.event_type for e in events] == ["node", "human"]
    assert json.loads(events[1].output_json)["reviewer"] == "a.reviewer"


def test_errors_are_recorded_so_a_failed_run_is_still_auditable():
    ev = audit.emit(
        run_id="r1",
        tenant_id="t",
        node="extract_claims",
        event_type="error",
        error="schema failure",
    )
    assert ev.event_type == "error"
    assert audit.events_for_run("r1")[0].error == "schema failure"


def test_events_are_scoped_by_tenant():
    audit.emit(run_id="r1", tenant_id="acme_pharma", node="ingest")
    audit.emit(run_id="r2", tenant_id="borealis_bio", node="ingest")
    acme = audit.events_for_tenant("acme_pharma")
    assert len(acme) == 1
    assert acme[0].tenant_id == "acme_pharma"


def test_cost_summary_totals_a_run():
    audit.emit(
        run_id="r1",
        tenant_id="t",
        node="a",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.001,
        latency_ms=200,
    )
    audit.emit(
        run_id="r1",
        tenant_id="t",
        node="b",
        tokens_in=200,
        tokens_out=80,
        cost_usd=0.002,
        latency_ms=300,
    )
    summary = audit.run_cost_summary("r1")
    assert summary == {
        "events": 2,
        "tokens_in": 300,
        "tokens_out": 130,
        "cost_usd": 0.003,
        "latency_ms": 500,
    }


def test_integrity_check_detects_a_complete_sequence():
    for node in ["a", "b", "c"]:
        audit.emit(run_id="r1", tenant_id="t", node=node)
    check = audit.integrity_check("r1")
    assert check["event_count"] == 3
    assert check["sequence_complete"] is True
    assert check["duplicate_seqs"] == []


def test_wal_mode_is_actually_enabled():
    """A reader polling for status must never block the graph writing events."""
    audit.emit(run_id="r1", tenant_id="t", node="ingest")
    assert audit.journal_mode().lower() == "wal"
