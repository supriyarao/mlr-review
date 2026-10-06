"""Every Streamlit page renders against the real API without raising.

The pages are run headlessly with Streamlit's ``AppTest``, and their HTTP client is swapped for a
``TestClient`` bound to the FastAPI app -- so this exercises the same requests the browser demo
makes, with no server, no network and no model.

It lives beside the API tests because that is what it checks: that the UI and the API agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"
PAGES = ["Home.py", *sorted(f"pages/{p.name}" for p in (UI_DIR / "pages").glob("*.py"))]


@pytest.fixture
def ui(monkeypatch, seeded):
    """Run a page by filename against the in-process API."""
    from api.main import app

    # `streamlit run ui/Home.py` puts ui/ on the path; the pages import their helpers from there.
    monkeypatch.syspath_prepend(str(UI_DIR))
    for name in ("api_client", "view"):
        sys.modules.pop(name, None)
    import api_client

    client = TestClient(app)
    monkeypatch.setattr(api_client, "_client", lambda: client)
    for cached in (
        api_client.health,
        api_client.tenants,
        api_client.tenant,
        api_client.tenant_rules,
        api_client.samples,
        api_client.sample,
        api_client.graph_source,
        api_client.prompts,
    ):
        cached.clear()

    def run(page: str, **session) -> AppTest:
        at = AppTest.from_file(str(UI_DIR / page), default_timeout=30)
        for key, value in session.items():
            at.session_state[key] = value
        return at.run()

    yield run
    for name in ("api_client", "view"):
        sys.modules.pop(name, None)


def test_there_is_a_page_for_every_screen_in_the_brief():
    assert [Path(p).stem.split("_", 1)[-1] for p in PAGES] == [
        "Home",
        "Submit_Review",
        "Run_Detail",
        "Reviewer_Queue",
        "Audit_Trail",
        "Evals",
        "Tenants",
    ]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_an_exception(ui, page):
    at = ui(page)
    assert not at.exception, [e.value for e in at.exception]
    # A red callout is also how an escalation is shown, so only a failed API call counts here.
    assert not [e.value for e in at.error if "(HTTP " in e.value]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_carries_the_disclaimer(ui, page):
    text = " ".join(m.value for m in ui(page).markdown)
    assert "not medical, legal or regulatory advice" in text


def test_the_home_page_explains_every_node_in_the_graph(ui, client):
    """The node guide is hand-written, so it is checked against the graph it describes."""
    import view

    at = ui("Home.py")
    described = " ".join(c.value for c in at.caption)
    nodes = view.graph_nodes(client.get("/graph").json()["source"])
    assert len(nodes) >= 15, f"could not read the graph's nodes: {nodes}"
    missing = [n for n in nodes if f"`{n}`" not in described]
    assert not missing, f"nodes in the graph with no description on the home page: {missing}"


def test_every_model_backed_node_serves_its_prompt(client):
    prompts = {p["node"]: p for p in client.get("/prompts").json()["prompts"]}
    assert set(prompts) == {
        "extract_claims",
        "preapproved_filter",
        "substantiation",
        "reference_support",
        "off_label",
        "fair_balance",
        "safety_prominence",
        "llm_rules",
        "reference_doc_review",
    }
    for prompt in prompts.values():
        assert prompt["version"] and prompt["file"].endswith(".py") and len(prompt["text"]) > 100
        assert prompt["tier"] in {"strong", "fast"}


def test_the_run_pages_invite_a_submission_when_there_are_no_runs(ui):
    at = ui("pages/2_Run_Detail.py")
    assert any("No reviews yet" in i.value for i in at.info)


def test_a_reviewer_decision_is_sent_and_confirmed(ui, client, submitted):
    tier = client.get(f"/reviews/{submitted}").json()["route"]
    at = ui("pages/3_Reviewer_Queue.py")
    at.selectbox[0].select(tier).run()
    assert not at.exception, [e.value for e in at.exception]

    at.button(key="decide-request_changes").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Recorded: request changes by a.reviewer" in s.value for s in at.success)
    assert client.get(f"/reviews/{submitted}").json()["status"] == "complete"


def test_the_queue_is_empty_for_a_tier_nothing_was_routed_to(ui):
    at = ui("pages/3_Reviewer_Queue.py")
    at.selectbox[0].select("legal").run()
    assert any("Nothing is waiting for legal" in i.value for i in at.info)


def test_run_detail_shows_a_finished_review(ui, submitted):
    at = ui("pages/2_Run_Detail.py", run_id=submitted)
    assert not at.exception, [e.value for e in at.exception]
    assert [s.value for s in at.subheader][:2] == ["Progress", "Findings"]


def test_run_detail_shows_two_runs_side_by_side_for_a_tenant_comparison(ui, submitted):
    compare = [("Acme Pharma", submitted), ("Borealis Bio", submitted)]
    at = ui("pages/2_Run_Detail.py", compare=compare)
    assert not at.exception, [e.value for e in at.exception]
    assert [h.value for h in at.header] == ["Acme Pharma", "Borealis Bio"]
    # Identical run ids mean the columns are the same review; the page has to say so.
    assert any("same run id" in w.value for w in at.warning)


def test_the_tenants_page_marks_the_settings_that_differ(ui):
    markdown = ui("pages/6_Tenants.py").markdown
    table = next(m.value for m in markdown if '<table class="mlr-diff"' in m.value)
    differing = [row for row in table.split("<tr") if 'class="differs"' in row]
    assert any("Confidence threshold" in row for row in differing)
    assert any("Severity: Fair balance" in row for row in differing)
    assert not any("Severity: Off-label" in row for row in differing)
