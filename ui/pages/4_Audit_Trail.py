"""Audit Trail: every step of a run in order, with the model, prompt and policy behind it."""

from __future__ import annotations

import json
from typing import Any

import api_client as api
import streamlit as st
import view

view.page(
    "Audit trail",
    "One event per node, gate and human decision. Each records what produced it, so a decision can "
    "be explained months later: which model, which prompt version, which policy.",
)

EVENT_ICONS = {"node": "⚙️", "gate": "🚧", "human": "🧑", "error": "❌"}

run_id = view.pick_run()
if not run_id:
    view.footer()
    st.stop()

try:
    trail = api.audit(run_id)
except api.ApiError as exc:
    view.show_error(exc)
    st.stop()

events: list[dict[str, Any]] = trail["events"]
integrity = trail["integrity"]

if not events:
    st.info("No events have been recorded for this run yet.")
    view.footer()
    st.stop()

if integrity["sequence_complete"] and not integrity["duplicate_seqs"]:
    st.success(
        f"{integrity['event_count']} events, numbered without gaps or duplicates. "
        "Ordering uses the sequence number, not the clock, because parallel checks can finish in "
        "the same millisecond."
    )
else:
    st.error(
        f"The trail is incomplete: {integrity['event_count']} events, "
        f"duplicate sequence numbers {integrity['duplicate_seqs'] or 'none'}. "
        "An event was dropped or written twice."
    )

policies = sorted({e["policy_version"] for e in events if e.get("policy_version")})
total_a, total_b, total_c, total_d = st.columns(4)
total_a.metric("Model calls", sum(1 for e in events if e.get("model")))
total_b.metric(
    "Tokens in / out",
    f"{sum(e['tokens_in'] for e in events):,} / {sum(e['tokens_out'] for e in events):,}",
)
total_c.metric("Cost", view.money(sum(e["cost_usd"] for e in events)))
total_d.metric("Policy version", policies[0] if len(policies) == 1 else f"{len(policies)} versions")
if len(policies) > 1:
    st.warning("The policy changed while this run was in flight: " + ", ".join(policies))

st.subheader("Events")
st.dataframe(
    [
        {
            "#": e["seq"],
            "Step": f"{EVENT_ICONS.get(e['event_type'], '')} {e['node'].replace('_', ' ')}",
            "Kind": e["event_type"],
            "Model": e.get("model") or "none (code)",
            "Prompt": e.get("prompt_version", ""),
            "Policy": e.get("policy_version", ""),
            "Tokens in": e["tokens_in"],
            "Tokens out": e["tokens_out"],
            "Latency (ms)": e["latency_ms"],
            "Cost ($)": e["cost_usd"],
            "Time": e["ts"],
        }
        for e in events
    ],
    hide_index=True,
    width="stretch",
    column_config={"Cost ($)": st.column_config.NumberColumn(format="%.5f")},
)


def parsed(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


st.caption("Open an event for what went in and what came out.")
for event in events:
    icon = EVENT_ICONS.get(event["event_type"], "")
    title = f"{event['seq']}. {icon} {event['node'].replace('_', ' ')}"
    if event.get("error"):
        title += " (failed)"
    with st.expander(title):
        if event.get("error"):
            st.error(event["error"])
        left, right = st.columns(2)
        with left:
            st.markdown("**Input**")
            st.code(event.get("input_hash") or "(none)", language=None)
            st.caption(
                "A hash, not the input itself: the log proves what a step saw without keeping a "
                "second copy of the submitted material."
            )
        with right:
            st.markdown("**Output**")
            if event.get("output_json"):
                st.json(parsed(event["output_json"]), expanded=False)
            else:
                st.caption("No output was recorded for this step.")
            if event.get("citations_json"):
                st.markdown("**Citations**")
                st.json(parsed(event["citations_json"]), expanded=False)
        st.caption(
            f"Label version {event.get('label_version') or 'n/a'}, "
            f"policy {event.get('policy_version') or 'n/a'}, recorded {event['ts']}."
        )

st.subheader("Replay: what the system knew at a step")
step = st.slider("After step", 1, len(events), len(events)) if len(events) > 1 else 1
so_far = events[:step]
current = so_far[-1]
st.markdown(
    f"After step **{step}** ({current['node'].replace('_', ' ')}), the run had completed "
    f"**{len(so_far)}** of {len(events)} recorded steps."
)
view.timeline(api.graph_source(), so_far, "running" if step < len(events) else "replayed")
knew_a, knew_b, knew_c = st.columns(3)
knew_a.metric("Spent so far", view.money(sum(e["cost_usd"] for e in so_far)))
knew_b.metric("Model calls so far", sum(1 for e in so_far if e.get("model")))
knew_c.metric("Gates passed through", sum(1 for e in so_far if e["event_type"] == "gate"))
if current.get("output_json"):
    st.markdown("**What this step produced**")
    st.json(parsed(current["output_json"]), expanded=False)
st.caption(
    "Rebuilt from the audit events up to the chosen step. The full state at each checkpoint is "
    "kept by the graph runtime but is not yet exposed through the service, so this view shows "
    "what each step recorded rather than the whole state."
)

view.footer()
