"""Graph runtime: a run that stops at a human, survives, and resumes from its checkpoint.

Uses a deliberately tiny graph rather than the MLR one. The behaviour under test -- suspend,
persist, resume without re-running earlier work -- belongs to the platform, and testing it against
a two-node graph keeps the failure diagnosable when it breaks.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from platform_core import runtime


class ToyState(TypedDict):
    """Minimal state with an append reducer, mirroring how findings accumulate."""

    steps: Annotated[list[str], operator.add]
    decision: str
    side_effects: Annotated[list[str], operator.add]


def build_toy_graph(saver):
    """prepare -> human (interrupt) -> finalize."""

    def prepare(state: ToyState) -> dict:
        # Recorded so a test can prove resuming does not re-run this node.
        return {"steps": ["prepare"], "side_effects": ["prepare_ran"]}

    def human(state: ToyState) -> dict:
        answer = interrupt({"question": "approve?", "steps": state["steps"]})
        return {"steps": ["human"], "decision": answer["decision"]}

    def finalize(state: ToyState) -> dict:
        return {"steps": ["finalize"]}

    builder = StateGraph(ToyState)
    builder.add_node("prepare", prepare)
    builder.add_node("human", human)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "human")
    builder.add_edge("human", "finalize")
    builder.add_edge("finalize", END)
    return runtime.compile_graph(builder, saver=saver)


@pytest.fixture
def graph():
    with runtime.checkpointer() as saver:
        yield build_toy_graph(saver)


def test_run_ids_are_unique():
    assert runtime.new_run_id() != runtime.new_run_id()
    assert len(runtime.new_run_id()) == 12


def test_a_run_stops_at_the_human_and_says_what_it_is_asking(graph):
    result = runtime.start_run(graph, {"steps": [], "decision": "", "side_effects": []})
    assert result["status"] == "awaiting_human"
    assert result["interrupt"]["question"] == "approve?"
    # The work before the human actually happened.
    assert result["interrupt"]["steps"] == ["prepare"]


def test_a_suspended_run_has_not_reached_the_end(graph):
    started = runtime.start_run(graph, {"steps": [], "decision": "", "side_effects": []})
    status = runtime.get_status(graph, started["run_id"])
    assert status["status"] == "awaiting_human"
    assert "finalize" not in status["values"]["steps"]


def test_resuming_carries_the_human_decision_into_state(graph):
    started = runtime.start_run(graph, {"steps": [], "decision": "", "side_effects": []})
    resumed = runtime.resume_run(graph, started["run_id"], {"decision": "reject"})
    assert resumed["status"] == "complete"
    assert resumed["state"]["decision"] == "reject"
    assert resumed["state"]["steps"] == ["prepare", "human", "finalize"]


def test_resuming_does_not_re_run_work_done_before_the_human(graph):
    """The point of checkpointing: a reviewer taking three days costs nothing to resume."""
    started = runtime.start_run(graph, {"steps": [], "decision": "", "side_effects": []})
    resumed = runtime.resume_run(graph, started["run_id"], {"decision": "approve"})
    assert resumed["state"]["side_effects"] == ["prepare_ran"], (
        "prepare must run exactly once across suspend and resume"
    )


def test_status_is_reportable_after_completion(graph):
    started = runtime.start_run(graph, {"steps": [], "decision": "", "side_effects": []})
    runtime.resume_run(graph, started["run_id"], {"decision": "approve"})
    status = runtime.get_status(graph, started["run_id"])
    assert status["status"] == "complete"
    assert status["interrupt"] is None
    assert status["next"] == []


def test_an_unknown_run_is_unknown_not_an_error(graph):
    """The API polls by id; a bad id must not raise a 500."""
    assert runtime.get_status(graph, "nonexistent")["status"] == "unknown"


def test_two_runs_do_not_share_state(graph):
    """thread_id scoping: concurrent reviews must not see each other's findings."""
    a = runtime.start_run(graph, {"steps": ["a"], "decision": "", "side_effects": []})
    b = runtime.start_run(graph, {"steps": ["b"], "decision": "", "side_effects": []})
    assert a["run_id"] != b["run_id"]
    runtime.resume_run(graph, a["run_id"], {"decision": "approve"})
    runtime.resume_run(graph, b["run_id"], {"decision": "reject"})
    assert runtime.get_status(graph, a["run_id"])["values"]["decision"] == "approve"
    assert runtime.get_status(graph, b["run_id"])["values"]["decision"] == "reject"


def test_history_shows_what_the_system_knew_at_each_step(graph):
    """Backs the audit page's replay view."""
    started = runtime.start_run(graph, {"steps": [], "decision": "", "side_effects": []})
    runtime.resume_run(graph, started["run_id"], {"decision": "approve"})
    steps = runtime.history(graph, started["run_id"])
    assert len(steps) > 1
    assert [s["step"] for s in steps] == list(range(len(steps)))
    # Oldest first: state accumulates as the run progresses.
    assert len(steps[-1]["values"]["steps"]) >= len(steps[0].get("values", {}).get("steps", []))


def test_a_run_id_can_be_supplied_so_audit_and_checkpoint_share_one_identity(graph):
    result = runtime.start_run(
        graph, {"steps": [], "decision": "", "side_effects": []}, run_id="fixed-id"
    )
    assert result["run_id"] == "fixed-id"
    assert runtime.get_status(graph, "fixed-id")["status"] == "awaiting_human"


def test_checkpoints_persist_across_separate_connections(tmp_path, monkeypatch):
    """A reviewer may answer days later, in a different process."""
    monkeypatch.setenv("MLR_DATA_DIR", str(tmp_path))

    with runtime.checkpointer() as saver:
        first = build_toy_graph(saver)
        started = runtime.start_run(first, {"steps": [], "decision": "", "side_effects": []})
        run_id = started["run_id"]

    # New connection, new compiled graph, same database.
    with runtime.checkpointer() as saver:
        second = build_toy_graph(saver)
        assert runtime.get_status(second, run_id)["status"] == "awaiting_human"
        resumed = runtime.resume_run(second, run_id, {"decision": "approve"})

    assert resumed["state"]["steps"] == ["prepare", "human", "finalize"]
    assert resumed["state"]["side_effects"] == ["prepare_ran"]


def test_checkpointer_uses_wal(tmp_path, monkeypatch):
    """So a status poll never blocks the graph writing a checkpoint."""
    import sqlite3

    monkeypatch.setenv("MLR_DATA_DIR", str(tmp_path))
    with runtime.checkpointer():
        pass
    conn = sqlite3.connect(str(runtime.checkpoint_path()))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_mermaid_is_rendered_from_the_compiled_graph(graph):
    """Generated, not hand-drawn, so the README diagram cannot drift from the code."""
    mermaid = runtime.render_mermaid(graph)
    assert "prepare" in mermaid
    assert "human" in mermaid
    assert "finalize" in mermaid
