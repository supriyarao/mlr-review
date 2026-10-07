"""Run Detail: one review, step by step, then what it found and who should look at it."""

from __future__ import annotations

from typing import Any

import api_client as api
import nodes
import streamlit as st
import view

view.page("Run detail")

LIVE = ("queued", "running")

STEP_ICONS = {
    "done": "✓",
    "running": "⏳",
    "waiting": "⏸",
    "failed": "✕",
    "skipped": "–",
    "pending": "·",
}
STEP_WORDS = {
    "done": "Done",
    "running": "Running now",
    "waiting": "Waiting for a person",
    "failed": "Failed",
    "skipped": "Skipped: nothing for it to do on this piece",
    "pending": "Not reached yet",
}

# The step buttons echo the picture on the Home page: rounded asks a model, square is plain code.
# Streamlit adds a class named after each widget's key, which is what these selectors match.
STRIP_CSS = f"""
<style>
div[class*="-kind-model-"] button {{ border-radius: 999px; }}
div[class*="-kind-code-"] button {{ border-radius: 3px; }}
div[class*="-is-done-"] button {{ border-color: {view.PASS}; }}
div[class*="-is-failed-"] button {{ border-color: {view.BLOCKER}; color: {view.BLOCKER}; }}
div[class*="-is-running-"] button, div[class*="-is-waiting-"] button {{
  border-color: {view.ADVISORY};
}}
div[class*="-kind-"] button p {{ font-size: 0.84rem; }}
</style>
"""


# --- A step's panel ------------------------------------------------------------------------------


def text_box(text: str, *, source: bool = False) -> None:
    css = "mlr-proof source" if source else "mlr-proof"
    st.markdown(
        f'<div class="{css}" style="max-height: 320px; overflow-y: auto;">{view.block(text)}</div>',
        unsafe_allow_html=True,
    )


def reader_panel(given: dict[str, Any], produced: dict[str, Any]) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**What it was given**")
        if given.get("submitted_text"):
            st.caption(given.get("filename") or "The text as it was submitted")
            text_box(given["submitted_text"])
        else:
            st.caption("The submitted text isn't available for this run.")
    with right:
        st.markdown("**What it produced**")
        st.caption("Clean text. Every later step quotes from this.")
        text_box(produced.get("document", ""), source=True)

    meta = produced.get("doc_meta") or {}
    sections = meta.get("sections") or {}
    if sections:
        st.markdown("**The section map**")
        st.dataframe(
            [
                {
                    "Section": name.replace("_", " ").capitalize(),
                    "Starts at": span["start"],
                    "Ends at": span["end"],
                    "Length": span["end"] - span["start"],
                }
                for name, span in sections.items()
            ],
            hide_index=True,
            width="stretch",
        )
        a, b, c = st.columns(3)
        a.metric("Benefit section", f"{meta.get('efficacy_block_chars', 0):,} characters")
        b.metric("Safety section", f"{meta.get('safety_block_chars', 0):,} characters")
        gap = meta.get("safety_distance_chars")
        c.metric(
            "Safety starts",
            f"{gap:,} characters in" if isinstance(gap, int) else "n/a",
            help="How far the safety section starts after the benefit section begins.",
        )


def claims_table(claims: list[dict[str, Any]]) -> None:
    st.dataframe(
        [
            {
                "Claim": c["original_text"],
                "Type": c["claim_type"].capitalize(),
                "Section": c.get("location", ""),
                "References": ", ".join(c.get("reference_markers") or []) or "none",
                "Found in text": "Yes" if c["char_span"][1] else "No",
            }
            for c in claims
        ],
        hide_index=True,
        width="stretch",
    )


def claim_spotter_panel(given: dict[str, Any], produced: dict[str, Any]) -> None:
    claims = produced.get("claims") or []
    st.markdown("**What it was given**")
    st.caption(f"The clean text from the reader, {len(given.get('document') or ''):,} characters.")
    with st.expander("Show the text"):
        text_box(given.get("document") or "", source=True)
    st.markdown("**What it produced**")
    if not claims:
        st.info("It didn't find any claims in this piece.")
        return
    st.caption(f'{len(claims)} claim(s). "Found in text" means code located the exact words.')
    claims_table(claims)


def matcher_panel(given: dict[str, Any], produced: dict[str, Any], summary: Any) -> None:
    claims = given.get("claims") or []
    st.markdown("**What it was given**")
    st.caption(f"The {len(claims)} claim(s) from the claim spotter.")
    if claims:
        with st.expander("Show the claims"):
            claims_table(claims)
    st.markdown("**What it produced**")
    matches = produced.get("preapproved_matches") or {}
    considered = summary.get("candidates_considered") if isinstance(summary, dict) else None
    if not matches:
        looked = f" It looked at {considered} cleared claim(s)." if considered else ""
        st.info(f"No claim matched wording the company has already cleared.{looked}")
        return
    by_id = {c["id"]: c["original_text"] for c in claims}
    st.dataframe(
        [
            {"Claim": by_id.get(claim_id, claim_id), "Matches cleared claim": cleared}
            for claim_id, cleared in matches.items()
        ],
        hide_index=True,
        width="stretch",
    )


def tool_calls(tools: list[dict[str, Any]], step: dict[str, Any]) -> None:
    """The company's own checkers this step ran, in order, before it asked the model anything."""
    st.markdown("**First, the company's own checkers**")
    st.caption(
        "Code the company already uses and trusts. It ran before the model was asked anything, "
        "and it only measures. It doesn't judge."
    )
    for column, tool in zip(st.columns(len(tools)), tools, strict=True):
        with column.container(border=True):
            name = tool["tool"].replace("_", " ").capitalize()
            version = view.chip(f"v{tool['version']}", "quiet")
            st.markdown(f"**{view.escape(name)}** &nbsp;{version}", unsafe_allow_html=True)
            if not tool.get("ok", True):
                st.markdown(view.chip("Didn't run", "blocker"), unsafe_allow_html=True)
                st.caption(tool.get("summary") or "No reason was recorded.")
                continue
            failed, unknown = tool.get("failed") or [], tool.get("inconclusive") or []
            # Three different answers, three different looks. "Couldn't measure" is not a pass.
            chips = [view.chip(f"Outside the limit: {m.replace('_', ' ')}") for m in failed]
            chips += [
                view.chip(f"Couldn't measure: {m.replace('_', ' ')}", "unknown") for m in unknown
            ]
            if not chips:
                chips = [view.chip("Everything within limits", "pass")]
            st.markdown(" ".join(chips), unsafe_allow_html=True)
            st.caption(
                f"{tool.get('summary', '')}. Took {view.duration(tool.get('latency_ms', 0))}."
            )

    st.markdown("**Then, the model**")
    asked = f"`{step['model']}` was" if step.get("model") else "A model was"
    st.caption(
        f"{asked} given those measurements and asked what they mean for this piece. A finding "
        "that quotes a number no checker reported gets marked as unverified."
    )


def generic_panel(step: dict[str, Any], document: str) -> None:
    """For steps without a purpose-built view yet: the step's own summary, and any findings."""
    summary = step.get("summary")
    tools = summary.get("tools") if isinstance(summary, dict) else None
    if tools:
        tool_calls(tools, step)
    st.markdown("**What it produced**")
    if isinstance(summary, dict) and summary:
        st.dataframe(
            [
                {"": k.replace("_", " ").capitalize(), "Value": view.plain(v)}
                for k, v in summary.items()
                if k != "tools" and not (k == "by_concern" and tools)
            ],
            hide_index=True,
        )
        if tools and summary.get("by_concern"):
            st.caption(
                "Findings by type: "
                + ", ".join(
                    f"{view.CHECK_LABELS.get(concern, concern)} {count}"
                    for concern, count in summary["by_concern"].items()
                )
                + "."
            )
    findings = (step.get("produced") or {}).get("findings") or []
    for finding in findings:
        with st.expander(view.finding_title(finding)):
            view.finding_body(finding, document)
    if not summary and not findings:
        st.caption("Nothing was recorded for this step.")


def step_panel(step: dict[str, Any], document: str) -> None:
    node = nodes.ALL_NODES.get(step["node"])
    name = node.name if node else step["node"]
    with st.container(border=True):
        if step.get("model"):
            how = view.chip(f"Asked {step['model']}")
        elif node and node.model:
            how = view.chip("Asks a model")
        else:
            how = view.chip("Plain code, no model", "quiet")
        st.markdown(
            f"#### {name} &nbsp;{how} {view.chip(STEP_WORDS[step['status']], 'quiet')}",
            unsafe_allow_html=True,
        )
        if node:
            st.markdown(node.does)
        if step.get("error"):
            st.error(step["error"])
        if step["status"] in ("pending", "skipped", "running"):
            st.caption("Nothing to show for this step yet." if step["status"] != "skipped" else "")
            return

        given, produced = step.get("given") or {}, step.get("produced") or {}
        if step["node"] == "ingest":
            reader_panel(given, produced)
        elif step["node"] == "extract_claims":
            claim_spotter_panel(given, produced)
        elif step["node"] == "preapproved_filter":
            matcher_panel(given, produced, step.get("summary"))
        else:
            generic_panel(step, document)

        facts = [f"Took {view.duration(step.get('latency_ms', 0))}"]
        if step.get("model"):
            facts.append(f"prompt version {step.get('prompt_version') or 'n/a'}")
            facts.append(f"{step['tokens_in']:,} tokens in, {step['tokens_out']:,} out")
            facts.append(f"cost {view.money(step.get('cost_usd', 0.0))}")
        st.caption(", ".join(facts) + f". Called `{step['node']}` in the audit trail.")
        with st.expander("Raw data for this step"):
            st.json(
                {"given": given, "produced": produced, "summary": step.get("summary")},
                expanded=False,
            )


# --- The strip -----------------------------------------------------------------------------------


def step_strip(run_id: str, key: str, document: str) -> None:
    """The four stages from the Home page, for this run. Click a step to open it."""
    try:
        steps = {s["node"]: s for s in api.get_json(f"/reviews/{run_id}/steps")["steps"]}
    except api.ApiError as exc:
        view.show_error(exc)
        return

    chosen_key = f"{key}-step-{run_id}"
    chosen = st.session_state.get(chosen_key, "ingest")
    st.markdown(STRIP_CSS, unsafe_allow_html=True)

    for column, (title, stage, in_sequence) in zip(
        st.columns([1, 1.9, 1, 1]), nodes.STAGES, strict=True
    ):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            slots = None if in_sequence else st.columns(2)
            for index, node in enumerate(stage):
                step = steps.get(node.key, {"status": "pending"})
                kind = "model" if node.model else "code"
                holder = slots[index % 2] if slots else st
                clicked = holder.button(
                    f"{STEP_ICONS[step['status']]} {node.name}",
                    key=f"{key}-kind-{kind}-is-{step['status']}-{node.key}",
                    type="primary" if node.key == chosen else "secondary",
                    width="stretch",
                    help=STEP_WORDS[step["status"]],
                )
                if clicked:
                    st.session_state[chosen_key] = node.key
                    st.rerun()
    st.caption(
        "Same steps as on the Home page. Rounded asks a model, square is plain code. "
        "Click a step to see what it was given and what it produced."
    )
    if chosen in steps:
        step_panel(steps[chosen], document)


@st.fragment(run_every=1)
def live_steps(run_id: str, key: str) -> None:
    """Refresh once a second while the review runs, then hand over to the full page."""
    report = api.review(run_id)
    if report["status"] not in LIVE:
        st.rerun()
    st.caption("Still running. This updates every second.")
    step_strip(run_id, key, "")


# --- What it found -------------------------------------------------------------------------------


def findings_section(report: dict[str, Any], document: str, key: str) -> None:
    findings = report.get("findings", [])
    if not findings:
        st.success("Nothing flagged. That isn't the same as approved, so it's still worth a read.")
        return

    severities = sorted({view.severity_of(f) for f in findings}, reverse=True)
    checks = sorted({f["check_type"] for f in findings})
    sev_col, check_col = st.columns(2)
    chosen_sev = sev_col.multiselect(
        "Severity", severities, default=severities, format_func=str.capitalize, key=f"{key}-sev"
    )
    chosen_checks = check_col.multiselect(
        "Raised by",
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
                "Raised by": view.CHECK_LABELS.get(f["check_type"], f["check_type"]),
                "The words in question": f.get("quote") or "(something required is missing)",
                "What's wrong": f["issue"],
                "How sure": f["confidence"],
                "Backed by": view.GROUNDING_SHORT.get(f["grounded_by"], "Nothing"),
                "Rule": f"{f['rule_id']} v{f['rule_version']}" if f.get("rule_id") else "",
            }
            for f in shown
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "How sure": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            "What's wrong": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(f"Showing {len(shown)} of {len(findings)}. Open one to see the evidence.")
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
        f"{view.chip('for: ' + report['audience'], 'quiet')} "
        f"{view.chip('rulebook ' + (report['policy_version'] or 'pending'), 'quiet')}",
        unsafe_allow_html=True,
    )

    if report["status"] in LIVE:
        st.subheader("Steps")
        live_steps(run_id, key)
        return
    if report["status"] == "failed":
        st.error(f"This review stopped early: {report.get('error') or 'no reason was recorded'}")

    document = api.document(run_id)

    view.summary_metrics(report)
    if report["status"] == "awaiting_human":
        st.info("All done. These are recommendations: have a read and make the call.")
    elif report["status"] == "complete":
        st.success("Closed. A reviewer has made the call on this one.")

    if not compact:
        # Side by side there isn't room for the strip; the full view of either run has it.
        st.subheader("Steps")
        step_strip(run_id, key, document)

    st.subheader("What it found")
    findings_section(report, document, key)

    st.subheader("Who should look at this, and why")
    view.routing(report)

    st.subheader("Checks on the findings")
    view.gates(report.get("gate_results", []))

    if compact:
        return

    applied = report.get("rules_applied", [])
    with st.expander(f"Company rules that applied to this run ({len(applied)})"):
        st.dataframe(
            [
                {
                    "Rule": f"{r['id']} v{r['version']}",
                    "What it's about": r["title"],
                    "Kind": r["kind"].replace("_", " "),
                    "Severity": r["severity"].capitalize(),
                }
                for r in applied
            ],
            hide_index=True,
            width="stretch",
        )

    st.subheader("Download")
    xlsx_col, json_col, _ = st.columns([1, 1, 3])
    xlsx_col.download_button(
        "Findings as a spreadsheet",
        api.get(f"/reviews/{run_id}/report.xlsx").content,
        file_name=f"mlr-review-{run_id}.xlsx",
        key=f"{key}-xlsx",
    )
    json_col.download_button(
        "Findings as JSON",
        api.get(f"/reviews/{run_id}/report.json").content,
        file_name=f"mlr-review-{run_id}.json",
        mime="application/json",
        key=f"{key}-json",
    )
    with st.expander("Raw data for the whole run"):
        st.json(report, expanded=False)


compare = st.session_state.get("compare")
if compare:
    st.markdown(
        "Same piece, two companies. Any difference below comes from their settings, not from "
        "different code."
    )
    if st.button("Close comparison"):
        st.session_state.pop("compare")
        st.rerun()
    if len({run_id for _, run_id in compare}) < len(compare):
        st.warning(
            "Both submissions were given the same run id, so the two columns show the same review."
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
