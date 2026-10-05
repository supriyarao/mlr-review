"""Evals: run the golden set, read the metrics, and see what the confidence threshold trades."""

from __future__ import annotations

from typing import Any

import altair as alt
import api_client as api
import streamlit as st
import view

view.page(
    "Evals",
    "A golden set of deliberately hard pieces, each with the findings a reviewer would expect and "
    "the things that must not be flagged. Blocker recall is the headline: a missed blocker is the "
    "expensive mistake.",
    eval_page=True,
)

# Categorical series colours, fixed per measure. Validated for colour-blind separation and for
# contrast against the chart surface; deliberately not the red/amber/green reserved for severity.
RECALL_COLOR = "#2a78d6"
ESCALATION_COLOR = "#eb6834"
RECALL = "Blocker recall"
ESCALATION = "False escalation rate"


def pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, int | float) else "n/a"


def metric_value(metrics: dict[str, Any], group: str, key: str) -> Any:
    block = metrics.get(group)
    return block.get(key) if isinstance(block, dict) else None


# --- Start a run ---------------------------------------------------------------------------------

with st.container(border=True):
    st.markdown("**Run the golden set**")
    mode_col, repeat_col, tenant_col, button_col = st.columns([2, 1, 2, 1])
    mode = mode_col.radio(
        "Mode",
        ["offline", "live"],
        format_func={"offline": "Offline replay", "live": "Live models"}.get,
        horizontal=True,
        help="Offline replays recorded model responses: free, deterministic, and what CI runs. "
        "Live calls the configured model backend and costs money.",
    )
    repeats = repeat_col.number_input(
        "Repeats",
        1,
        10,
        1,
        help="With more than one repeat the worst outcome per case is kept: catching a blocker "
        "four times in five is a miss.",
    )
    tenant_options = ["", *[t["tenant_id"] for t in api.tenants()]]
    tenant_filter = tenant_col.selectbox(
        "Tenant", tenant_options, format_func=lambda t: t or "All tenants"
    )
    button_col.write("")
    if button_col.button("Run eval", type="primary", width="stretch"):
        try:
            started = api.post_json(
                "/evals/run",
                {"mode": mode, "tenant_id": tenant_filter, "repeats": int(repeats)},
            )
        except api.ApiError as exc:
            view.show_error(exc)
        else:
            st.session_state["eval_run_id"] = started["eval_run_id"]


@st.fragment(run_every=2)
def wait_for(eval_run_id: str) -> None:
    detail = api.get_json(f"/evals/runs/{eval_run_id}")
    if detail.get("status") not in ("queued", "running"):
        st.rerun()
    st.info(f"Eval {eval_run_id} is {detail.get('status')}. This updates every two seconds.")


# --- Choose a run --------------------------------------------------------------------------------

runs: list[dict[str, Any]] = api.get_json("/evals/runs")["runs"]
run_ids = [r["eval_run_id"] for r in runs]
started_id = st.session_state.get("eval_run_id")
if started_id and started_id not in run_ids:
    run_ids.insert(0, started_id)
if not run_ids:
    st.info("No eval has been run yet. Run the golden set above to see results here.")
    view.footer()
    st.stop()

by_id = {r["eval_run_id"]: r for r in runs}


def run_name(eval_run_id: str) -> str:
    run = by_id.get(eval_run_id)
    if not run:
        return f"{eval_run_id} (in progress)"
    return f"{eval_run_id}  ({run['mode']}, {run['started_at'][:16].replace('T', ' ')})"


selected = st.selectbox(
    "Eval run",
    run_ids,
    index=run_ids.index(started_id) if started_id in run_ids else 0,
    format_func=run_name,
)
try:
    detail = api.get_json(f"/evals/runs/{selected}")
except api.ApiError as exc:
    view.show_error(exc)
    st.stop()

if detail.get("status") in ("queued", "running"):
    wait_for(selected)
    view.footer()
    st.stop()
if detail.get("status") == "failed":
    st.error(f"This eval failed: {detail.get('error')}")
    view.footer()
    st.stop()

metrics: dict[str, Any] = detail.get("metrics", {})
cases: list[dict[str, Any]] = detail.get("cases", [])

try:
    comparison = api.get_json(f"/evals/runs/{selected}/comparison")
except api.ApiError:
    comparison = {"baseline_available": False}

# --- Metrics -------------------------------------------------------------------------------------

recall = metric_value(metrics, "blocker_recall", "recall")
recall_delta = None
if comparison.get("baseline_available"):
    for movement in comparison.get("movements", []):
        if movement["metric"] == "blocker_recall.recall":
            recall_delta = movement["delta"]

headline, detail_col = st.columns([1, 2])
with headline.container(border=True):
    st.metric(
        "Blocker recall",
        pct(recall),
        delta=f"{recall_delta * 100:+.1f} pts vs baseline" if recall_delta is not None else None,
    )
    matched = metric_value(metrics, "blocker_recall", "matched")
    expected = metric_value(metrics, "blocker_recall", "expected")
    if expected is not None:
        st.caption(f"{matched} of {expected} expected blockers were caught.")
with detail_col:
    row_one = st.columns(3)
    row_one[0].metric(
        "False escalation rate",
        pct(metric_value(metrics, "false_escalation_rate", "rate")),
        help="Pieces sent to a senior tier that should have gone to a standard reviewer.",
    )
    row_one[1].metric("Routing accuracy", pct(metric_value(metrics, "routing", "accuracy")))
    row_one[2].metric("Grounding rate", pct(metric_value(metrics, "grounding_rate", "rate")))
    row_two = st.columns(3)
    row_two[0].metric("Advisory recall", pct(metric_value(metrics, "advisory_recall", "recall")))
    cost = metric_value(metrics, "cost_and_latency", "cost_usd_per_run")
    row_two[1].metric("Cost per review", view.money(cost) if cost is not None else "n/a")
    p50 = metric_value(metrics, "cost_and_latency", "latency_p50_ms")
    p95 = metric_value(metrics, "cost_and_latency", "latency_p95_ms")
    row_two[2].metric(
        "Latency p50 / p95",
        f"{view.duration(p50)} / {view.duration(p95)}" if p50 is not None else "n/a",
    )

st.caption(
    f"Run {detail['eval_run_id']}: {detail.get('mode')} mode, backend "
    f"{detail.get('backend') or 'n/a'}, commit {detail.get('git_sha') or 'n/a'}, "
    f"{detail.get('repeats', 1)} repeat(s), models {detail.get('model_strong') or 'n/a'} and "
    f"{detail.get('model_fast') or 'n/a'}."
)

# --- Cases ---------------------------------------------------------------------------------------

st.subheader("Cases")
passed = sum(1 for c in cases if c["passed"])
st.caption(f"{passed} of {len(cases)} case(s) passed.")
st.dataframe(
    [
        {
            "Result": "✅ Pass" if c["passed"] else "❌ Fail",
            "Case": c["case_id"],
            "Tenant": c["tenant_id"],
            "Expected route": view.tier_label(c.get("expected_route", "")),
            "Actual route": view.tier_label(c.get("actual_route", "")),
            "Latency": view.duration(c.get("latency_ms", 0)),
            "Cost": view.money(c.get("cost_usd", 0.0)),
        }
        for c in cases
    ],
    hide_index=True,
    width="stretch",
)

for case in cases:
    icon = "✅" if case["passed"] else "❌"
    with st.expander(f"{icon} {case['case_id']}: expected against actual"):
        if case.get("error"):
            st.error(case["error"])
        route_ok = case.get("route_correct")
        st.markdown(
            f"**Route.** Expected {view.tier_label(case.get('expected_route', ''))}, got "
            f"{view.tier_label(case.get('actual_route', ''))} "
            + (view.chip("match", "pass") if route_ok else view.chip("mismatch", "blocker")),
            unsafe_allow_html=True,
        )
        if "matches" not in case:
            st.caption("Per-finding detail was not recorded for this run.")
            continue
        rows = []
        for match in case["matches"]:
            wanted = match["expected"]
            if not match["matched"]:
                outcome = "❌ Missed"
            elif match.get("severity_mismatch"):
                outcome = f"⚠️ Found, but as {match.get('actual_severity')}"
            else:
                outcome = "✅ Found"
            rows.append(
                {
                    "Expected finding": wanted["claim_contains"],
                    "Check": view.CHECK_LABELS.get(wanted["check_type"], wanted["check_type"]),
                    "Expected severity": wanted.get("severity", ""),
                    "Outcome": outcome,
                }
            )
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        for alarm in case.get("false_alarms", []):
            st.markdown(f"- ❌ Flagged something it must not flag: {alarm}")
        unexpected = case.get("unexpected_findings", [])
        if unexpected:
            st.markdown(f"**{len(unexpected)} finding(s) nobody asked for**")
            st.json(unexpected, expanded=False)

# --- Threshold sweep -----------------------------------------------------------------------------

st.subheader("What the confidence threshold trades")
st.markdown(
    "Findings below a tenant's confidence threshold are escalated. Raising it catches more, and "
    "sends more clean pieces to senior reviewers. The curve is recomputed from stored findings "
    "with no model calls, so the only thing that changes along it is the threshold."
)
try:
    sweep = api.post_json("/evals/sweep", {"eval_run_id": selected if selected in by_id else ""})
except api.ApiError as exc:
    sweep = None
    st.info(f"No sweep is available for this run: {exc.message}")

if sweep and sweep["points"]:
    points = sweep["points"]
    long_form = [
        {"threshold": p["threshold"], "measure": name, "rate": p[key]}
        for p in points
        for name, key in ((RECALL, "blocker_recall"), (ESCALATION, "false_escalation_rate"))
    ]
    tenant_marks = [
        {"threshold": t["confidence_threshold"], "tenant": t["name"]} for t in api.tenants()
    ]

    color = alt.Color(
        "measure:N",
        scale=alt.Scale(domain=[RECALL, ESCALATION], range=[RECALL_COLOR, ESCALATION_COLOR]),
        legend=alt.Legend(title=None, orient="top"),
    )
    x = alt.X(
        "threshold:Q",
        title="Confidence threshold",
        scale=alt.Scale(
            domain=[min(p["threshold"] for p in points) - 5, 100], zero=False, nice=False
        ),
    )
    y = alt.Y("rate:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1.05]))
    base = alt.Chart(alt.Data(values=long_form))
    lines = base.mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    dots = base.mark_circle(size=80, opacity=1).encode(
        x=x,
        y=y,
        color=color,
        tooltip=[
            alt.Tooltip("measure:N", title="Measure"),
            alt.Tooltip("threshold:Q", title="Threshold"),
            alt.Tooltip("rate:Q", title="Rate", format=".1%"),
        ],
    )
    # Where each tenant has actually set its threshold: the point on the curve it has chosen.
    marks = alt.Chart(alt.Data(values=tenant_marks))
    rules = marks.mark_rule(strokeDash=[4, 4], color="#898781").encode(x="threshold:Q")
    labels = marks.mark_text(align="left", dx=5, dy=-4, color="#898781").encode(
        x="threshold:Q", y=alt.value(12), text="tenant:N"
    )
    st.altair_chart((rules + labels + lines + dots).properties(height=340), width="stretch")
    st.caption(
        "Dashed lines mark the threshold each tenant runs at today. " + sweep.get("note", "")
    )
    with st.expander("The same numbers as a table"):
        st.dataframe(
            [
                {
                    "Threshold": p["threshold"],
                    RECALL: pct(p["blocker_recall"]),
                    ESCALATION: pct(p["false_escalation_rate"]),
                }
                for p in points
            ],
            hide_index=True,
        )

# --- Baseline and history ------------------------------------------------------------------------

st.subheader("Against the baseline and earlier runs")
if comparison.get("baseline_available"):
    if comparison["regressed"]:
        st.error(
            "Blocker recall is below the committed baseline. In CI this fails the build, so a "
            "policy or prompt change cannot reduce recall unnoticed."
        )
    else:
        st.success("No regression against the committed baseline.")
    st.dataframe(
        [
            {
                "Metric": m["metric"],
                "Baseline": pct(m["baseline"]),
                "This run": pct(m["current"]),
                "Change": f"{m['delta'] * 100:+.1f} pts",
            }
            for m in comparison["movements"]
        ],
        hide_index=True,
    )
    for change in comparison.get("case_changes", []):
        st.markdown(f"- `{change['case_id']}`: {change['change']}")
else:
    st.caption("No committed baseline is available to compare this run with yet.")

if runs:
    st.dataframe(
        [
            {
                "Run": r["eval_run_id"],
                "Started": r["started_at"][:16].replace("T", " "),
                "Mode": r["mode"],
                "Commit": r["git_sha"],
                "Cases": r["cases"],
                RECALL: pct(r["blocker_recall"]),
                ESCALATION: pct(r["false_escalation_rate"]),
            }
            for r in runs
        ],
        hide_index=True,
        width="stretch",
    )

with st.expander("Raw metrics"):
    st.json(metrics, expanded=False)

view.footer()
