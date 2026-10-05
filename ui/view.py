"""Shared presentation for every page: the frame, the severity colours, and how a finding is shown.

One module so that a blocker is the same red and a finding has the same anatomy wherever it
appears. Everything that comes from a submission or a model is escaped before it is placed in
HTML -- the quote column is, by definition, text somebody else wrote.
"""

from __future__ import annotations

import re
from html import escape
from typing import Any

import api_client as api
import streamlit as st

DISCLAIMER = (
    "Synthetic data and fictional products. A demonstration only: not medical, legal or regulatory "
    "advice, and not for review of real promotional material."
)
HUMAN_BANNER = "The system recommends; a human decides."

# Severity colours are status colours: reserved for blocker / advisory / pass and never reused for
# anything else on a page, so red always means the same thing.
BLOCKER = "#B42318"
ADVISORY = "#B25E09"
PASS = "#067647"
NEUTRAL = "#5B6472"
INK = "#1F3A5F"

CHECK_LABELS = {
    "substantiation": "Substantiation",
    "reference_support": "Reference support",
    "off_label": "Off-label",
    "fair_balance": "Fair balance",
    "safety_prominence": "Safety prominence",
    "business_rule": "Business rule",
}
GROUNDING_LABELS = {
    "evidence": "Retrieved evidence",
    "rule": "The rule's own wording",
    "doc_clause": "A clause of a tenant reference document",
    "none": "Not grounded",
}
GROUNDING_SHORT = {"evidence": "Evidence", "rule": "Rule text", "doc_clause": "Document clause"}
CITATION_LABELS = {"evidence": "Evidence", "rule": "Tenant rule", "doc_clause": "Document clause"}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,650&display=swap');

h1, h2, h3 {{
  font-family: 'Source Serif 4', Georgia, serif !important; letter-spacing: -0.01em;
}}
[data-testid="stMainBlockContainer"] {{ padding-top: 3rem; max-width: 1240px; }}
[data-testid="stMainBlockContainer"] p {{ max-width: 78ch; }}

/* Streamlit's default accent is red. Red is reserved for blockers here, so the accent on primary
   buttons and selected filter tags is restated as a neutral ink. */
button[kind="primary"] {{
  background-color: {INK} !important; border-color: {INK} !important; color: #fff !important;
}}
button[kind="primary"]:disabled {{ opacity: 0.45; }}
span[data-baseweb="tag"] {{ background-color: {NEUTRAL} !important; }}

.mlr-chip {{
  display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600; line-height: 1.5; white-space: nowrap;
  color: #fff; background: {NEUTRAL};
}}
.mlr-chip.blocker {{ background: {BLOCKER}; }}
.mlr-chip.advisory {{ background: {ADVISORY}; }}
.mlr-chip.pass {{ background: {PASS}; }}
.mlr-chip.quiet {{
  background: transparent; color: inherit; border: 1px solid rgba(128,128,128,.45);
  font-weight: 500;
}}

.mlr-human {{
  border-left: 4px solid {PASS}; background: rgba(6,118,71,.09);
  padding: 0.7rem 1rem; border-radius: 0 6px 6px 0; margin: 0.2rem 0 1.1rem 0;
  font-family: 'Source Serif 4', Georgia, serif; font-size: 1.12rem;
}}
.mlr-stub {{
  border: 1px dashed rgba(178,94,9,.7); background: rgba(178,94,9,.08);
  padding: 0.55rem 0.9rem; border-radius: 6px; margin-bottom: 1rem; font-size: 0.9rem;
}}
.mlr-footer {{
  margin-top: 3rem; padding-top: 0.9rem; border-top: 1px solid rgba(128,128,128,.3);
  font-size: 0.8rem; opacity: 0.75;
}}

/* The piece under review, shown as a proof with the offending words marked up. */
.mlr-proof {{
  font-family: 'Source Serif 4', Georgia, serif; font-size: 0.98rem; line-height: 1.65;
  padding: 0.9rem 1.1rem; border-radius: 4px; white-space: pre-wrap;
  background: rgba(128,128,128,.08); border-left: 3px solid rgba(128,128,128,.4);
}}
.mlr-proof mark {{
  background: rgba(255,214,10,.45); color: inherit; padding: 0.05rem 0.1rem;
  text-decoration: underline wavy; text-underline-offset: 4px; text-decoration-thickness: 1.5px;
}}
.mlr-proof mark.blocker {{ text-decoration-color: {BLOCKER}; }}
.mlr-proof mark.advisory {{ text-decoration-color: {ADVISORY}; }}
.mlr-proof.source {{ border-left-color: {PASS}; }}
.mlr-caption {{ font-size: 0.8rem; opacity: 0.75; margin: 0.15rem 0 0.4rem 0; }}

.mlr-steps {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.3rem 0 1rem 0; }}
.mlr-step {{
  border: 1px solid rgba(128,128,128,.35); border-radius: 6px; padding: 0.3rem 0.6rem;
  font-size: 0.82rem; line-height: 1.35;
}}
.mlr-step small {{ display: block; opacity: 0.7; font-size: 0.72rem; }}
.mlr-step.done {{ border-color: rgba(6,118,71,.55); }}
.mlr-step.now {{ border-color: {ADVISORY}; background: rgba(178,94,9,.1); }}
.mlr-step.error {{ border-color: {BLOCKER}; background: rgba(180,35,24,.1); }}
.mlr-step.todo {{ opacity: 0.55; }}

.mlr-diff td, .mlr-diff th {{
  padding: 0.4rem 0.7rem; border-bottom: 1px solid rgba(128,128,128,.25);
  vertical-align: top; text-align: left; font-size: 0.9rem;
}}
.mlr-diff {{ border-collapse: collapse; width: 100%; }}
.mlr-diff tr.differs td {{ background: rgba(255,214,10,.16); }}
.mlr-diff tr.differs td:first-child {{ border-left: 3px solid {ADVISORY}; font-weight: 600; }}
</style>
"""


# --- Page frame ----------------------------------------------------------------------------------


def page(title: str, intro: str = "", *, eval_page: bool = False) -> None:
    """Start a page: config, styles, the stub notice when one applies, and the heading."""
    st.set_page_config(page_title=f"{title} | MLR Review", page_icon="🔎", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    stub_notice(eval_page=eval_page)
    st.title(title)
    if intro:
        st.markdown(intro)


def stub_notice(*, eval_page: bool = False) -> None:
    """Say so, on screen, when what is shown is example data rather than a real run."""
    stubs = api.health().get("stubs", {})
    parts = []
    if stubs.get("review_service"):
        parts.append(
            "the review graph is not connected yet, so every run shows the same example review "
            "whatever is submitted"
        )
    if eval_page and stubs.get("eval_runner"):
        parts.append("the eval runner is not connected yet, so eval results are example data")
    if parts:
        st.markdown(
            f'<div class="mlr-stub"><b>Example data.</b> {escape("; ".join(parts).capitalize())}.'
            "</div>",
            unsafe_allow_html=True,
        )


def human_banner() -> None:
    st.markdown(f'<div class="mlr-human">{HUMAN_BANNER}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="mlr-footer">{DISCLAIMER}</div>', unsafe_allow_html=True)


def show_error(exc: api.ApiError) -> None:
    st.error(f"{exc.message} (HTTP {exc.status_code})")


# --- Small pieces --------------------------------------------------------------------------------


def chip(text: str, kind: str = "") -> str:
    return f'<span class="mlr-chip {kind}">{escape(str(text))}</span>'


def severity_chip(severity: str) -> str:
    return chip(severity.capitalize(), severity if severity in ("blocker", "advisory") else "")


def severity_of(finding: dict[str, Any]) -> str:
    """The severity that counts: the tenant's decision, falling back to the check's proposal."""
    return finding.get("severity_final") or finding.get("severity_proposed") or "advisory"


def tier_label(tier: str) -> str:
    return tier.replace("_", " ").capitalize() if tier else "Not routed yet"


def money(value: float) -> str:
    return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"


def duration(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{int(ms)} ms"


def status_chip(status: str) -> str:
    kind = {"failed": "blocker", "awaiting_human": "advisory", "complete": "pass"}.get(status, "")
    return chip(status.replace("_", " ").capitalize(), kind)


def run_label(run: dict[str, Any]) -> str:
    return (
        f"{run['run_id']}  ({run['tenant_id']}, {run['product_id']}, "
        f"{run['status'].replace('_', ' ')})"
    )


def pick_run(key: str = "run_id") -> str | None:
    """A run selector shared by the pages that show one run.

    The selection follows the user between pages (session state) and survives a reload or a shared
    link (query parameter).
    """
    runs = api.runs()
    known = [r["run_id"] for r in runs]
    wanted = st.query_params.get(key) or st.session_state.get(key)
    if wanted and wanted not in known:
        known.insert(0, wanted)
    if not known:
        st.info("No reviews yet. Submit a piece on the Submit Review page to start one.")
        return None
    labels = {r["run_id"]: run_label(r) for r in runs}
    chosen = st.selectbox(
        "Review",
        known,
        index=known.index(wanted) if wanted in known else 0,
        format_func=lambda run_id: labels.get(run_id, run_id),
    )
    st.session_state[key] = chosen
    st.query_params[key] = chosen
    return chosen


# --- The node timeline ---------------------------------------------------------------------------

_EDGE = re.compile(r"^\s*(\w+)(?:[(\[].*?[)\]])?\s*-\.?-?>\s*(?:\|.*?\|\s*)?(\w+)")


def graph_nodes(mermaid: str) -> list[str]:
    """Node names in the order the graph source first mentions them."""
    nodes: list[str] = []
    for line in mermaid.splitlines():
        match = _EDGE.match(line)
        if not match:
            continue
        for name in match.groups():
            if not name.startswith("__") and name not in nodes:
                nodes.append(name)
    return nodes


def timeline(mermaid: str, events: list[dict[str, Any]], status: str) -> None:
    """Every node of the graph, marked by what the audit trail says happened to it."""
    by_node: dict[str, dict[str, Any]] = {}
    for event in events:
        by_node[event["node"]] = event
    nodes = graph_nodes(mermaid) or list(by_node)
    for name in by_node:
        if name not in nodes:
            nodes.append(name)

    last_done = max((i for i, n in enumerate(nodes) if n in by_node), default=-1)
    running = status in ("queued", "running")
    steps = []
    for index, name in enumerate(nodes):
        event = by_node.get(name)
        label = escape(name.replace("_", " "))
        if event and (event.get("error") or event.get("event_type") == "error"):
            steps.append(f'<div class="mlr-step error">❌ {label}<small>failed</small></div>')
        elif event and event.get("event_type") == "human" and status == "awaiting_human":
            steps.append(f'<div class="mlr-step now">⏳ {label}<small>waiting</small></div>')
        elif event:
            detail = duration(event.get("latency_ms", 0))
            steps.append(f'<div class="mlr-step done">✅ {label}<small>{detail}</small></div>')
        elif running and index == last_done + 1:
            steps.append(f'<div class="mlr-step now">⏳ {label}<small>running</small></div>')
        elif index < last_done:
            steps.append(f'<div class="mlr-step todo">{label}<small>no event</small></div>')
        else:
            steps.append(f'<div class="mlr-step todo">{label}<small>pending</small></div>')
    st.markdown(f'<div class="mlr-steps">{"".join(steps)}</div>', unsafe_allow_html=True)


# --- Findings ------------------------------------------------------------------------------------


def excerpt_html(document: str, quote: str, char_span: list[int], severity: str) -> str | None:
    """The quote marked up in its surrounding text, or None if it cannot be found there.

    The span is used only if the text at that span really is the quote; otherwise the quote is
    searched for. Nothing is highlighted on trust.
    """
    if not document or not quote.strip():
        return None
    start, end = int(char_span[0]), int(char_span[1])
    if document[start:end] != quote:
        start = document.find(quote)
        if start < 0:
            return None
        end = start + len(quote)
    before = document[max(0, start - 260) : start]
    after = document[end : end + 260]
    lead = "…" if start > 260 else ""
    tail = "…" if end + 260 < len(document) else ""
    return (
        f'<div class="mlr-proof">{lead}{escape(before)}'
        f'<mark class="{escape(severity)}">{escape(quote)}</mark>{escape(after)}{tail}</div>'
    )


def finding_title(finding: dict[str, Any]) -> str:
    severity = severity_of(finding)
    icon = "🔴" if severity == "blocker" else "🟠"
    check = CHECK_LABELS.get(finding["check_type"], finding["check_type"])
    quote = finding.get("quote") or "Something required is missing"
    if len(quote) > 70:
        quote = quote[:67] + "…"
    return f"{icon} {severity.capitalize()}: {check}. “{quote}”"


def finding_body(finding: dict[str, Any], document: str) -> None:
    """One finding in full: the words on the page beside the passage it is judged against."""
    severity = severity_of(finding)
    chips = [severity_chip(severity), chip(CHECK_LABELS.get(finding["check_type"], ""), "quiet")]
    if finding.get("rule_id"):
        chips.append(chip(f"{finding['rule_id']} v{finding['rule_version']}", "quiet"))
    if finding.get("low_confidence"):
        chips.append(chip("Low confidence", "advisory"))
    if not finding.get("grounded"):
        chips.append(chip("Not grounded", "blocker"))
    st.markdown(" ".join(chips), unsafe_allow_html=True)
    st.markdown(f"**{finding['issue']}**")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="mlr-caption">In the piece</div>', unsafe_allow_html=True)
        marked = excerpt_html(document, finding.get("quote", ""), finding["char_span"], severity)
        if marked:
            st.markdown(marked, unsafe_allow_html=True)
        elif finding.get("quote"):
            st.markdown(
                f'<div class="mlr-proof"><mark class="{severity}">{escape(finding["quote"])}'
                "</mark></div>",
                unsafe_allow_html=True,
            )
            st.caption("Shown without its surrounding text: the submitted piece is not available.")
        else:
            st.info("Nothing to quote: this finding is about something missing from the piece.")
    with right:
        st.markdown('<div class="mlr-caption">Judged against</div>', unsafe_allow_html=True)
        if not finding.get("citations"):
            st.warning("No citation. A finding without one is escalated, never dropped.")
        for citation in finding.get("citations", []):
            st.markdown(
                f'<div class="mlr-proof source">{escape(citation["passage"])}</div>',
                unsafe_allow_html=True,
            )
            kind = CITATION_LABELS.get(citation["kind"], citation["kind"])
            st.caption(f"{kind}: {citation.get('title') or citation['source_id']}")

    if finding.get("suggested_fix"):
        st.markdown(f"**Suggested fix.** {finding['suggested_fix']}")

    confidence, sev_col, ground, origin = st.columns(4)
    confidence.metric("Confidence", f"{finding['confidence']} / 100")
    proposed = finding.get("severity_proposed", "")
    sev_col.metric(
        "Severity",
        severity.capitalize(),
        help=(
            f"The check proposed {proposed}; tenant policy "
            f"{finding.get('severity_rule_id') or '(not applied yet)'} decided the final severity."
        ),
    )
    ground.metric("Grounded in", GROUNDING_SHORT.get(finding["grounded_by"], "Nothing"))
    origin.metric("Produced by", "Model" if finding.get("model") else "Code")
    provenance = (
        f"{finding['model']}, prompt {finding['prompt_version']}"
        if finding.get("model")
        else "Deterministic code, no model involved"
    )
    st.caption(
        f"Finding {finding['id']}. Grounding: "
        f"{GROUNDING_LABELS.get(finding['grounded_by'], finding['grounded_by'])}. "
        f"Check proposed {proposed}; severity rule {finding.get('severity_rule_id') or 'n/a'}. "
        f"{provenance}."
    )


def gates(gate_results: list[dict[str, Any]]) -> None:
    """The validation gates, each with its verdict. A pass says what was checked, too."""
    if not gate_results:
        st.caption("No gate has run yet.")
        return
    for column, gate in zip(st.columns(len(gate_results)), gate_results, strict=True):
        with column.container(border=True):
            verdict = chip("Passed", "pass") if gate["passed"] else chip("Failed", "blocker")
            st.markdown(
                f"**{escape(gate['gate'].capitalize())} gate** {verdict}", unsafe_allow_html=True
            )
            st.caption(gate["details"])
            if gate["failed_items"]:
                st.caption(f"Tripped by: {', '.join(gate['failed_items'])}")


def routing(report: dict[str, Any]) -> None:
    """Who this went to, and why -- in the tenant policy's terms."""
    if not report.get("route"):
        st.caption("Not routed yet.")
        return
    escalation = report.get("escalation")
    if not escalation or not escalation.get("reasons"):
        st.success(
            f"Routed to **{tier_label(report['route'])}** for a routine review. "
            "Nothing in the findings or the gates called for a higher tier."
        )
        return
    st.error(f"Escalated to **{tier_label(escalation['tier'])}**.")
    for reason in escalation["reasons"]:
        items = f" (findings {', '.join(reason['items'])})" if reason.get("items") else ""
        st.markdown(
            f"- **{reason['code'].replace('_', ' ').capitalize()}**{items}: {reason['explanation']}"
        )


def summary_metrics(report: dict[str, Any]) -> None:
    findings = report.get("findings", [])
    blockers = sum(1 for f in findings if severity_of(f) == "blocker")
    cost = report.get("cost", {})
    a, b, c, d, e = st.columns(5)
    a.metric("Blockers", blockers)
    b.metric("Advisories", len(findings) - blockers)
    c.metric("Routed to", tier_label(report.get("route", "")))
    d.metric("Cost", money(cost.get("cost_usd", 0.0)))
    e.metric("Model time", duration(cost.get("latency_ms", 0)))
