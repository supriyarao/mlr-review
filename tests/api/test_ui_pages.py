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
def ui(monkeypatch, seeded_rules):
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
def test_every_page_carries_the_disclaimer_and_says_when_data_is_an_example(ui, page):
    text = " ".join(m.value for m in ui(page).markdown)
    assert "not medical, legal or regulatory advice" in text
    assert "Example data." in text  # the review service is a stub today


def test_the_reviewer_queue_shows_the_humans_decide_banner(ui):
    at = ui("pages/3_Reviewer_Queue.py")
    assert any("The system recommends; a human decides." in m.value for m in at.markdown)


def test_a_reviewer_decision_is_sent_and_confirmed(ui):
    """Senior tier is the second option; the example run is escalated to it."""
    at = ui("pages/3_Reviewer_Queue.py")
    at.selectbox[0].select("senior_regulatory").run()
    assert not at.exception, [e.value for e in at.exception]

    at.button(key="decide-request_changes").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Recorded: request changes by a.reviewer" in s.value for s in at.success)


def test_the_queue_is_empty_for_a_tier_nothing_was_routed_to(ui):
    at = ui("pages/3_Reviewer_Queue.py")
    at.selectbox[0].select("legal").run()
    assert any("Nothing is waiting for legal" in i.value for i in at.info)


def test_run_detail_shows_two_runs_side_by_side_for_a_tenant_comparison(ui):
    compare = [("Acme Pharma", "a1b2c3d4e5f6"), ("Borealis Bio", "a1b2c3d4e5f6")]
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
