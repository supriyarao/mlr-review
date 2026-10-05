"""Reviewer Queue: take a pending review, give a verdict on each finding, and decide the piece."""

from __future__ import annotations

from typing import Any

import api_client as api
import streamlit as st
import view

view.page("Reviewer queue")
view.human_banner()

VERDICTS = {"agree": "Agree", "disagree": "Disagree", "edit": "Edit"}
DECISIONS = [
    ("approve", "Approve", "The piece can be used as submitted."),
    ("request_changes", "Request changes", "Send it back to the author with the findings."),
    ("reject", "Reject", "The piece cannot be used."),
]


def all_tiers() -> list[str]:
    """Every reviewer tier any tenant's policy can route to, standard tier first."""
    tiers: list[str] = []
    for summary in api.tenants():
        escalation = api.tenant(summary["tenant_id"])["escalation"]
        names = [escalation["standard_tier"], escalation["default_escalation_tier"]]
        names += [rule["tier"] for rule in escalation["rules"]]
        for name in names:
            if name not in tiers:
                tiers.append(name)
    return tiers


who, tier_col = st.columns(2)
reviewer = who.text_input("Your name", value=st.session_state.get("reviewer", "a.reviewer"))
st.session_state["reviewer"] = reviewer
tier = tier_col.selectbox(
    "Signed in as",
    all_tiers(),
    format_func=view.tier_label,
    help="Stands in for a login. A real deployment takes the tier from the authenticated user.",
)

tasks: list[dict[str, Any]] = api.get_json("/queue", tier=tier)["tasks"]
if not tasks:
    st.info(
        f"Nothing is waiting for {view.tier_label(tier).lower()}. Reviews are routed by tenant "
        "policy, so try another tier or submit a piece."
    )
    view.footer()
    st.stop()

task = st.selectbox(
    f"Pending reviews ({len(tasks)})",
    tasks,
    format_func=lambda t: f"{t['run_id']}  ({t['tenant_id']}): {t['report'].get('summary', '')}",
)
run_id = task["run_id"]
findings: list[dict[str, Any]] = task["report"].get("findings", [])
document = api.document(run_id)

if task["reasons"]:
    st.error(f"Escalated to **{view.tier_label(task['tier'])}**.")
    for reason in task["reasons"]:
        st.markdown(
            f"- **{reason['code'].replace('_', ' ').capitalize()}**: {reason['explanation']}"
        )
else:
    st.success("Routine review: nothing called for a higher tier.")

if task.get("evidence"):
    with st.expander(f"Supporting evidence ({len(task['evidence'])})"):
        for item in task["evidence"]:
            st.markdown(
                f'<div class="mlr-proof source">{view.escape(item.get("passage", ""))}</div>',
                unsafe_allow_html=True,
            )
            st.caption(item.get("title") or item.get("source_id", ""))

st.subheader("Your verdict on each finding")
st.caption(
    "A disagreement is the most useful thing a reviewer can give the system: it is a labelled "
    "example of the system being wrong, and it is kept."
)

per_finding: list[dict[str, Any]] = []
for finding in findings:
    with st.container(border=True):
        st.markdown(f"**{view.finding_title(finding)}**")
        with st.expander("Evidence"):
            view.finding_body(finding, document)
        prefix = f"{run_id}-{finding['id']}"
        verdict = st.radio(
            "Verdict",
            list(VERDICTS),
            format_func=VERDICTS.get,
            horizontal=True,
            key=f"{prefix}-verdict",
            label_visibility="collapsed",
        )
        review: dict[str, Any] = {"finding_id": finding["id"], "verdict": verdict}
        if verdict == "edit":
            issue_col, severity_col = st.columns([3, 1])
            review["edited_issue"] = issue_col.text_area(
                "Corrected issue", value=finding["issue"], key=f"{prefix}-issue"
            )
            severities = ["blocker", "advisory"]
            review["edited_severity"] = severity_col.selectbox(
                "Corrected severity",
                severities,
                index=severities.index(view.severity_of(finding)),
                format_func=str.capitalize,
                key=f"{prefix}-severity",
            )
        if verdict != "agree":
            review["note"] = st.text_input(
                "Why?", key=f"{prefix}-note", placeholder="What did the system get wrong?"
            )
        per_finding.append(review)

st.subheader("Your decision")
note = st.text_area("Note for the record", placeholder="Optional. Stored with the decision.")
if not reviewer.strip():
    st.warning("Enter your name: an unattributed decision is not an accountable one.")

for column, (value, label, description) in zip(st.columns(3), DECISIONS, strict=True):
    clicked = column.button(
        label,
        key=f"decide-{value}",
        type="primary" if value == "approve" else "secondary",
        disabled=not reviewer.strip(),
        width="stretch",
    )
    column.caption(description)
    if clicked:
        try:
            result = api.post_json(
                f"/reviews/{run_id}/decision",
                {
                    "decision": value,
                    "reviewer": reviewer.strip(),
                    "tier": tier,
                    "per_finding": per_finding,
                    "note": note,
                },
            )
        except api.ApiError as exc:
            view.show_error(exc)
        else:
            disagreed = sum(1 for r in per_finding if r["verdict"] != "agree")
            st.success(
                f"Recorded: {label.lower()} by {reviewer.strip()} "
                f"({view.tier_label(tier).lower()}). Run {run_id} is now "
                f"{result['status'].replace('_', ' ')}; {disagreed} finding(s) disputed."
            )
            st.session_state["run_id"] = run_id
            st.caption("Open the Audit Trail page to see this decision in the run's record.")

view.footer()
