"""Eval endpoints.

The API reaches the runner through a two-function seam documented in ``api/evals.py``: it serves
real runs when ``scripts/run_evals.py`` exposes both ``run`` and ``sweep``, and committed contract
examples when it does not. It exposed only ``sweep`` for a long time, so the Evals page showed an
"Example data" banner over numbers nobody had produced.

The first group checks the seam against the real runner. The rest use a stand-in, so the endpoints'
own behaviour -- storing a result, surfacing a failure, comparing to the baseline -- is tested
without paying for an eval.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from api import evals, stubs
from platform_core.evals import harness
from platform_core.evals.schema import CaseResult, EvalResult
from tests.api.conftest import assert_same_shape, fixture


def make_result(eval_run_id: str = "aaaaaaaaaaaa", *, recall: float = 1.0, passed: bool = True):
    return EvalResult(
        eval_run_id=eval_run_id,
        started_at="2026-10-05T12:40:00+00:00",
        finished_at="2026-10-05T12:41:00+00:00",
        mode="offline",
        git_sha="abc1234",
        backend="fake",
        cases=[
            CaseResult(
                case_id="001-subgroup-overstatement",
                tenant_id="acme_pharma",
                passed=passed,
                expected_route="senior_regulatory",
                actual_route="senior_regulatory" if passed else "standard_reviewer",
                route_correct=passed,
            )
        ],
        metrics={
            "blocker_recall": {"expected": 2, "matched": int(2 * recall), "recall": recall},
            "false_escalation_rate": {"standard_expected": 1, "escalated": 0, "rate": 0.0},
        },
    )


@pytest.fixture
def wired(monkeypatch):
    """A runner providing the seam, recording how it was called."""
    calls: list[dict] = []

    def run(*, mode, tenant_id, repeats):
        calls.append({"mode": mode, "tenant_id": tenant_id, "repeats": repeats})
        return make_result("runner-own-id")

    def sweep(result, thresholds):
        return [
            {"threshold": t, "blocker_recall": 1.0, "false_escalation_rate": t / 100}
            for t in thresholds
        ]

    monkeypatch.setattr(evals, "runner", lambda: SimpleNamespace(run=run, sweep=sweep))
    return calls


# --- The seam ------------------------------------------------------------------------------------


def test_the_runner_is_connected_so_nothing_is_served_as_an_example(client):
    """The runner seam is satisfied, so these endpoints answer with real runs.

    api/evals.py treats the runner as available only when scripts/run_evals.py exposes BOTH
    ``run`` and ``sweep`` at module level with the documented signatures. It exposed only ``sweep``
    for a long time, so every eval endpoint served a committed contract example and the UI showed
    an "Example data" banner over numbers nobody had produced. This asserts the seam holds; invert
    it and the banner comes back.
    """
    assert evals.is_stub() is False

    started = client.post("/evals/run", json={})
    assert started.status_code == 202
    assert stubs.STUB_HEADER not in started.headers

    for response in (client.get("/evals/runs"), client.post("/evals/sweep", json={})):
        assert stubs.STUB_HEADER not in response.headers


def test_the_runner_seam_is_exactly_what_the_api_documents(client):
    """Both halves, by name and by shape, because one missing name silently stubs the page."""
    import scripts.run_evals as runner

    assert callable(runner.run)
    assert callable(runner.sweep)

    result = runner.run(mode="offline", tenant_id="acme_pharma", repeats=1)
    assert result.cases, "run() must return an EvalResult carrying its cases"

    points = runner.sweep(result, [50, 90])
    assert isinstance(points, list) and len(points) == 2
    assert {"threshold", "blocker_recall", "false_escalation_rate"} <= set(points[0])


def test_reading_a_stored_result_never_depends_on_the_runner(client):
    """Reading results is pure file IO, so it works whatever state the runner is in."""
    harness.save_result(make_result(), directory=evals.RESULTS_DIR)
    listed = client.get("/evals/runs")
    assert stubs.STUB_HEADER not in listed.headers
    assert [r["eval_run_id"] for r in listed.json()["runs"]] == ["aaaaaaaaaaaa"]


def test_an_unknown_eval_run_is_a_404(client):
    assert client.get("/evals/runs/nope").status_code == 404
    assert client.get("/evals/runs/nope/comparison").status_code == 404
    assert client.get("/evals/baseline").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "production"},
        {"repeats": 0},
        {"repeats": 500},
        {"tenant": "acme_pharma"},  # unknown field, not silently ignored
    ],
)
def test_an_invalid_eval_request_is_rejected(client, body):
    assert client.post("/evals/run", json=body).status_code == 422


def test_an_eval_for_an_unknown_tenant_is_a_404(client):
    assert client.post("/evals/run", json={"tenant_id": "nobody"}).status_code == 404


@pytest.mark.parametrize("thresholds", [[], [150], [-1, 50]])
def test_a_sweep_with_impossible_thresholds_is_rejected(client, thresholds):
    assert client.post("/evals/sweep", json={"thresholds": thresholds}).status_code == 422


# --- Endpoint behaviour, against a stand-in runner -----------------------------------------------


def test_a_run_is_executed_stored_and_readable_under_the_id_that_was_returned(client, wired):
    started = client.post(
        "/evals/run", json={"mode": "offline", "tenant_id": "acme_pharma", "repeats": 2}
    )
    assert started.status_code == 202
    assert stubs.STUB_HEADER not in started.headers
    assert_same_shape(started.json(), fixture("POST_evals_run.json"))
    eval_run_id = started.json()["eval_run_id"]
    assert wired == [{"mode": "offline", "tenant_id": "acme_pharma", "repeats": 2}]

    listed = client.get("/evals/runs").json()
    assert_same_shape(listed, fixture("GET_evals_runs.json"))
    assert [r["eval_run_id"] for r in listed["runs"]] == [eval_run_id]
    assert listed["runs"][0]["blocker_recall"] == 1.0

    detail = client.get(f"/evals/runs/{eval_run_id}").json()
    assert_same_shape(detail, fixture("GET_evals_runs_{id}.json"))
    assert (detail["eval_run_id"], detail["status"]) == (eval_run_id, "complete")
    # The full per-case record is served, so the UI can show expected against actual.
    assert "matches" in detail["cases"][0] and "false_alarms" in detail["cases"][0]


def test_a_run_that_fails_is_reported_as_failed_not_left_running(client, wired, monkeypatch):
    def broken(**_kwargs):
        raise RuntimeError("cassette missing for case 004")

    monkeypatch.setattr(evals, "runner", lambda: SimpleNamespace(run=broken, sweep=lambda *a: []))
    eval_run_id = client.post("/evals/run", json={}).json()["eval_run_id"]

    detail = client.get(f"/evals/runs/{eval_run_id}").json()
    assert detail["status"] == "failed"
    assert "cassette missing" in detail["error"]
    # Same shape as a finished run, so a polling client only ever branches on status.
    assert_same_shape(detail, fixture("GET_evals_runs_{id}.json") | {"cases": []})
    assert client.get("/evals/runs").json() == {"runs": []}


def test_a_queued_run_has_the_same_shape_as_a_finished_one(client):
    eval_run_id = evals.new_job("offline")
    detail = client.get(f"/evals/runs/{eval_run_id}").json()
    assert (detail["status"], detail["cases"], detail["metrics"]) == ("queued", [], {})


def test_comparison_names_the_regressed_metric_and_the_case_that_broke(client, wired):
    evals.BASELINE_PATH.parent.mkdir(parents=True)
    evals.BASELINE_PATH.write_text(make_result("baseline0000").model_dump_json(), encoding="utf-8")
    harness.save_result(
        make_result("worse0000000", recall=0.5, passed=False), directory=evals.RESULTS_DIR
    )

    assert client.get("/evals/baseline").json()["eval_run_id"] == "baseline0000"

    body = client.get("/evals/runs/worse0000000/comparison").json()
    assert body["baseline_available"] is True
    assert body["regressed"] is True
    assert body["regressions"][0]["metric"] == "blocker_recall.recall"
    assert body["regressions"][0]["delta"] == -0.5
    assert body["case_changes"] == [
        {"case_id": "001-subgroup-overstatement", "change": "broke", "passed": False}
    ]


def test_comparison_without_a_baseline_says_so(client):
    harness.save_result(make_result(), directory=evals.RESULTS_DIR)
    assert client.get("/evals/runs/aaaaaaaaaaaa/comparison").json() == {"baseline_available": False}


def test_the_sweep_is_computed_from_a_stored_result(client, wired):
    assert client.post("/evals/sweep", json={}).status_code == 404  # nothing stored yet

    harness.save_result(make_result(), directory=evals.RESULTS_DIR)
    body = client.post("/evals/sweep", json={"thresholds": [90, 50, 70, 70]}).json()
    assert_same_shape(body, fixture("POST_evals_sweep.json"))
    assert body["thresholds"] == [50, 70, 90]
    assert [p["threshold"] for p in body["points"]] == [50, 70, 90]
    assert body["eval_run_id"] == "aaaaaaaaaaaa"


def test_the_regression_check_runs_the_tenants_offline_set_against_the_baseline(wired):
    """What rule activation calls. With no baseline it must report that it did not run."""
    assert evals.regression_check("acme_pharma") == {
        "ran": False,
        "regressed": False,
        "movements": [],
    }

    evals.BASELINE_PATH.parent.mkdir(parents=True)
    evals.BASELINE_PATH.write_text(make_result("baseline0000").model_dump_json(), encoding="utf-8")
    check = evals.regression_check("acme_pharma")
    assert (check["ran"], check["regressed"]) == (True, False)
    assert wired[-1] == {"mode": "offline", "tenant_id": "acme_pharma", "repeats": 1}
