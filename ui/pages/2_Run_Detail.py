"""Run Detail: live progress, then the findings beside their evidence, the gates and the routing."""

from __future__ import annotations

from typing import Any

import api_client as api
import streamlit as st
import view

view.page("Run detail")

LIVE = ("queued", "running")


@st.fragment(run_every=1)
def live_progress(run_id: str) -> None:
    """Poll once a second while the review runs, then hand over to the full page."""
    report = api.review(run_id)
    if report["status"] not in LIVE:
        st.rerun()
    st.markdown(view.status_chip(report["status"]), unsafe_allow_html=True)
    view.timeline(api.graph_source(), api.audit(run_id)["events"], report["status"])
    found = len(report.get("findings", []))
    st.caption(f"{found} finding(s) so far. This page updates every second.")


def findings_section(report: dict[str, Any], document: str, key: str) -> None:
    findings = report.get("findings", [])
    if not findings:
        st.success(
            "No issues found. The piece still goes to a reviewer: a clean result is a "
            "recommendation, not a clearance."
        )
        return

    severities = sorted({view.severity_of(f) for f in findings}, reverse=True)
    checks = sorted({f["check_type"] for f in findings})
    sev_col, check_col = st.columns(2)
    chosen_sev = sev_col.multiselect(
        "Severity", severities, default=severities, format_func=str.capitalize, key=f"{key}-sev"
    )
    chosen_checks = check_col.multiselect(
        "Check",
        checks,
        default=checks,
        format_func=lambda c: view.CHECK_LABELS.get(c, c),
        key=f"{key}-check",
    )
    shown = [
        f
        for f in findings
        if view.severity_of(f) in chosen_sev and f["check_type"] in chosen_checks
    ]

    st.dataframe(
        [
            {
                "Severity": view.severity_of(f).capitalize(),
                "Check": view.CHECK_LABELS.get(f["check_type"], f["check_type"]),
                "Quote": f.get("quote") or "(something required is missing)",
                "Issue": f["issue"],
                "Confidence": f["confidence"],
                "Grounded in": view.GROUNDING_SHORT.get(f["grounded_by"], "Nothing"),
                "Rule": f"{f['rule_id']} v{f['rule_version']}" if f.get("rule_id") else "",
            }
            for f in shown
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            "Issue": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(f"Showing {len(shown)} of {len(findings)} finding(s). Open one for its evidence.")
    for finding in shown:
        with st.expander(view.finding_title(finding)):
            view.finding_body(finding, document)


def render_run(run_id: str, *, key: str, compact: bool = False) -> None:
    try:
        report = api.review(run_id)
    except api.ApiError as exc:
        view.show_error(exc)
        return

    st.markdown(
        f"{view.status_chip(report['status'])} "
        f"{view.chip(report['tenant_id'], 'quiet')} {view.chip(report['product_id'], 'quiet')} "
        f"{view.chip('audience: ' + report['audience'], 'quiet')} "
        f"{view.chip('policy ' + (report['policy_version'] or 'pending'), 'quiet')}",
        unsafe_allow_html=True,
    )

    if report["status"] in LIVE:
        live_progress(run_id)
        return
    if report["status"] == "failed":
        st.error(f"This review failed: {report.get('error') or 'no error message was recorded'}")

    audit = api.audit(run_id)
    document = api.document(run_id)

    view.summary_metrics(report)
    if report["status"] == "awaiting_human":
        st.info(
            f"Waiting for a decision from **{view.tier_label(report['route'])}**. "
            "Open the Reviewer Queue to decide it."
        )
    elif report["status"] == "complete":
        st.success(f"Decided by a reviewer at **{view.tier_label(report['route'])}**.")

    st.subheader("Progress")
    view.timeline(api.graph_source(), audit["events"], report["status"])

    st.subheader("Findings")
    findings_section(report, document, key)

    st.subheader("Why it was routed here")
    view.routing(report)

    st.subheader("Validation gates")
    view.gates(report.get("gate_results", []))

    if compact:
        return

    applied = report.get("rules_applied", [])
    with st.expander(f"Tenant rules in force for this run ({len(applied)})"):
        st.dataframe(
            [
                {
                    "Rule": f"{r['id']} v{r['version']}",
                    "Title": r["title"],
                    "Kind": r["kind"].replace("_", " "),
                    "Severity": r["severity"].capitalize(),
                }
                for r in applied
            ],
            hide_index=True,
            width="stretch",
        )

    st.subheader("Export")
    xlsx_col, json_col, _ = st.columns([1, 1, 3])
    xlsx_col.download_button(
        "Download findings (.xlsx)",
        api.get(f"/reviews/{run_id}/report.xlsx").content,
        file_name=f"mlr-review-{run_id}.xlsx",
        key=f"{key}-xlsx",
    )
    json_col.download_button(
        "Download findings (.json)",
        api.get(f"/reviews/{run_id}/report.json").content,
        file_name=f"mlr-review-{run_id}.json",
        mime="application/json",
        key=f"{key}-json",
    )
    with st.expander("Raw report"):
        st.json(report, expanded=False)


compare = st.session_state.get("compare")
if compare:
    st.markdown(
        "The same piece, reviewed under two tenants' policies. Differences in severity and routing "
        "come from configuration, not from different code."
    )
    if st.button("Close comparison"):
        st.session_state.pop("compare")
        st.rerun()
    if len({run_id for _, run_id in compare}) < len(compare):
        st.warning(
            "Both submissions were given the same run id, so the two columns show the same review. "
            "This happens while the review graph is not connected."
        )
    for column, (tenant_name, run_id) in zip(st.columns(len(compare)), compare, strict=True):
        with column:
            st.header(tenant_name)
            render_run(run_id, key=f"cmp-{tenant_name}", compact=True)
else:
    run_id = view.pick_run()
    if run_id:
        render_run(run_id, key="single")

view.footer()
