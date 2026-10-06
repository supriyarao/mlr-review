"""Home: what this is, how a review flows node by node, and where the platform ends."""

from __future__ import annotations

from dataclasses import dataclass

import api_client as api
import streamlit as st
import view
from streamlit_mermaid import st_mermaid

view.page("MLR Review Agent")

st.markdown(
    """
Drug ads and brochures need a medical, legal and regulatory check before anyone sees them.
It is slow, careful work.

This app does the first pass. Give it a piece and it will:

- pick out the claims the piece makes
- check each one against the approved label, the studies it cites, and the company's own rules
- show what it found, next to the exact sentence it is relying on

What comes back is a list of recommendations. You read them and make the call.

Two made-up companies share the app, each with its own rulebook. Same piece, different rulebook,
different result.
"""
)

st.subheader("Where to go")
links = [
    ("pages/1_Submit_Review.py", "Submit a review", "Upload a piece or pick a sample, and run it."),
    ("pages/2_Run_Detail.py", "Run detail", "Watch a review run; read findings beside evidence."),
    ("pages/3_Reviewer_Queue.py", "Reviewer queue", "Take a pending review and decide it."),
    ("pages/4_Audit_Trail.py", "Audit trail", "Every step, with the model and policy behind it."),
    ("pages/5_Evals.py", "Evals", "Blocker recall, false escalations and the threshold curve."),
    ("pages/6_Tenants.py", "Tenants", "Two policies side by side, and one piece under both."),
]
for row in (links[:3], links[3:]):
    for column, (target, label, description) in zip(st.columns(3), row, strict=True):
        with column.container(border=True):
            st.page_link(target, label=f"**{label}**")
            st.caption(description)


# --- The graph, node by node ---------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """One step of the review, described for someone who has never seen the code."""

    key: str  # the node's name in the graph and the audit trail
    name: str  # what we call it on this page
    does: str
    how: str  # the logic, for a code step; what the model is asked, for a model step
    uses: str  # anything it reads from outside the run
    saves: str
    model: bool = False


GETTING_READY = [
    Node(
        "ingest",
        "The reader",
        "Opens the file and turns it into clean text. Everything later quotes from this text, so "
        "it has to be exact.",
        "Strips the header block and comments, removes bold and italic markers, then maps where "
        "each section starts and ends. It also notes where the safety section sits compared to "
        "the first benefit claim.",
        "Nothing outside the file you submitted.",
        "The cleaned text and a section map, in the run's state.",
    ),
    Node(
        "extract_claims",
        "The claim spotter",
        "Reads the piece and lists every promotional claim: a headline, a statistic, a "
        "comparison. Deciding what counts as a claim takes judgement, so a model does it.",
        "The model returns the claims. Plain code then finds each one in the text, because "
        "models are bad at counting characters. A claim that can't be found word for word gets "
        "flagged.",
        "The model.",
        "The list of claims, each with its position in the text.",
        model=True,
    ),
    Node(
        "preapproved_filter",
        "The already-cleared matcher",
        "Checks whether a claim is wording the company has signed off before. If so, it skips "
        "the evidence check. It still gets checked for balance and off-label use, because the "
        "wording was cleared, not the page it now sits on.",
        "A search pulls likely matches from the company's cleared claims. A small model then "
        "confirms the two really say the same thing.",
        "The company's cleared-claims list (local search index) and the model.",
        "Which claims matched which cleared claim.",
        model=True,
    ),
    Node(
        "select_rules",
        "The rulebook picker",
        "Works out which of the company's own rules apply to this piece.",
        "A rule applies if it belongs to this company, is switched on, is inside its date range, "
        "and covers this audience, channel, product and market. A patient-only rule is left out "
        "of a piece written for doctors.",
        "The company's rules (local database).",
        "The exact rules and versions used, so you can later see why a rule did or didn't fire.",
    ),
]

REVIEWERS = [
    Node(
        "substantiation",
        "The evidence reviewer",
        "Asks: does the evidence really back this claim, as strongly as the piece says it?",
        "Looks for a subgroup result passed off as a general one, a bigger result than was "
        "measured, a dropped comparison, or a claim with no support at all.",
        "The approved label and the studies (local search index) and the model.",
        "Findings, each with the quote and the passage it relies on.",
        model=True,
    ),
    Node(
        "reference_support",
        "The citation checker",
        "Asks: does the study in the footnote actually say this?",
        "Compares the claim with the cited study: same result measured, same patients, same "
        "point in time.",
        "The cited studies (local search index) and the model.",
        "Findings with the passage from the study.",
        model=True,
    ),
    Node(
        "off_label",
        "The label reviewer",
        "Asks: does the piece promote a use the medicine isn't approved for, outright or by "
        "hinting?",
        "Compares what the piece says or implies with the approved use and patient group.",
        "The approved label (local search index) and the model.",
        "Findings with the label passage.",
        model=True,
    ),
    Node(
        "fair_balance",
        "The balance reviewer",
        "Asks: do the risks get a fair share of the page next to the benefits?",
        "Mostly reasons over the layout numbers the reader worked out, so a small model is enough.",
        "The section map from the reader, and the model.",
        "Findings about prominence and placement.",
        model=True,
    ),
    Node(
        "safety_prominence",
        "The safety reviewer",
        "Asks: is the required safety information there, and easy to find?",
        "Checks the safety section exists and isn't tucked away. Runs even when no claims were "
        "found, since a piece can still be missing its safety text.",
        "The section map, the label, and the model.",
        "Findings about missing or buried safety information.",
        model=True,
    ),
    Node(
        "deterministic_rules",
        "The house-rule checker",
        "Applies the company rules that have a yes-or-no answer.",
        "Pattern matching: banned words, required wording, required sections, a trademark "
        "symbol on first mention, too many claims without a reference. No model is called, so "
        "the same piece always gets the same answer.",
        "The rules chosen by the rulebook picker.",
        "Findings that name the rule and its version.",
    ),
    Node(
        "llm_rules",
        "The house-rule reader",
        "Applies the company rules written in plain English that need judgement.",
        "A small model reads a handful of rules at once. Rule text is treated as data, not "
        "instructions. A finding that names a rule it wasn't given is thrown away.",
        "The rules chosen by the rulebook picker, and the model.",
        "Findings that name the rule and its version.",
        model=True,
    ),
    Node(
        "reference_doc_review",
        "The code-of-practice reviewer",
        "Reads the piece against one of the company's own documents, such as a promotional "
        "code. One reviewer per document.",
        "The model must quote the clause it is relying on. Code checks that clause really is in "
        "the document before the finding is kept.",
        "The company's reference documents (files) and the model.",
        "Findings with the clause quoted.",
        model=True,
    ),
]

CHECKING = [
    Node(
        "grounding_gate",
        "The fact-checker",
        "Checks the reviewers' work. Nothing a model said is trusted until it has been looked up.",
        "Two tests. The quote must appear word for word in the piece. The passage it cites must "
        "appear word for word in this company's own sources. A finding that fails is kept and "
        "marked as unverified. It is never quietly dropped.",
        "The piece, plus the company's label, studies, rules and documents.",
        "Each finding marked with how it was verified, and the gate's pass or fail.",
    ),
    Node(
        "severity",
        "The severity setter",
        "Decides whether each finding is a blocker or an advisory.",
        "A lookup in the company's settings, top to bottom, first match wins. One company can "
        "treat a balance problem as a blocker and the other as an advisory. A finding from a "
        "company rule keeps the severity that rule was written with.",
        "The company's settings file.",
        "The final severity, and which setting decided it. The reviewer's first suggestion is "
        "kept too.",
    ),
    Node(
        "editor",
        "The editor",
        "Tidies up. If two reviewers flagged the same words, you see one finding, not two.",
        "Findings that quote exactly the same text are merged and keep all their evidence. "
        "Confidence goes up when two reviewers agree and is capped low for anything unverified. "
        "It is arithmetic, not a model, so it can never merge away a real problem.",
        "Nothing outside the run.",
        "The final list of findings shown in the report.",
    ),
    Node(
        "confidence_gate",
        "The confidence check",
        "Flags findings the system isn't sure about.",
        "Compares each finding's confidence with the company's threshold: 70 for one company, 80 "
        "for the other. Anything below is flagged, not removed.",
        "The company's threshold.",
        "A low-confidence flag on each finding, and the gate's pass or fail.",
    ),
    Node(
        "route",
        "The router",
        "Works out who should look at the result, and lists every reason why.",
        "Blockers, unverified findings, low confidence or a reviewer that failed to run all send "
        "it to a more senior reviewer, picked from the company's settings. A clean piece goes to "
        "a standard reviewer. No path ends with the system approving anything.",
        "The company's escalation settings.",
        "The recommended reviewer level, the reasons, and the run's total cost.",
    ),
]

WRAPPING_UP = [
    Node(
        "human_review",
        "The pause",
        "The run stops here and waits. The findings are ready to read as recommendations.",
        "The run is saved and paused. When a decision comes in it picks up from this exact "
        "point, so nothing before it is run or paid for twice.",
        "Nothing outside the run.",
        "The paused run, and later the decision with who made it.",
    ),
    Node(
        "finalize",
        "The record keeper",
        "Closes the run and keeps what was learned.",
        "Records the outcome. If someone disagreed with a finding, that is saved as a draft test "
        "case, so the system can be checked against it later.",
        "Nothing outside the run.",
        "The final status, and draft test cases in evals/candidates/.",
    ),
]


def node_card(node: Node, prompts: dict[str, dict]) -> None:
    prompt = prompts.get(node.key)
    with st.container(border=True):
        if node.model:
            size = {"strong": "the larger model", "fast": "the smaller model"}
            which = size.get(prompt["tier"], "a model") if prompt else "a model"
            kind = view.chip(f"Asks {which}")
        else:
            kind = view.chip("Plain code, no model", "quiet")
        st.markdown(f"**{node.name}** &nbsp;{kind}", unsafe_allow_html=True)
        st.markdown(node.does)
        st.markdown(
            f"- **How it decides.** {node.how}\n"
            f"- **What it reads.** {node.uses}\n"
            f"- **What it saves.** {node.saves}"
        )
        footer, link = st.columns([3, 2])
        footer.caption(f"Called `{node.key}` in the graph and the audit trail.")
        if prompt:
            with link.popover("Read the prompt"):
                st.caption(f"`{prompt['file']}`, version `{prompt['version']}`")
                # Plain escaped HTML rather than st.code: the code block loads its highlighter
                # from the server on first use, and that fetch fails if the server has restarted.
                st.markdown(
                    f'<pre class="prompt-text">{view.escape(prompt["text"])}</pre>',
                    unsafe_allow_html=True,
                )


def node_grid(nodes: list[Node], prompts: dict[str, dict]) -> None:
    for start in range(0, len(nodes), 2):
        for column, node in zip(st.columns(2), nodes[start : start + 2], strict=False):
            with column:
                node_card(node, prompts)


STAGES = [
    ("1. Getting ready", GETTING_READY, True),
    ("2. The reviewers, side by side", REVIEWERS, False),
    ("3. Checking the reviewers' work", CHECKING, True),
    ("4. Wrapping up", WRAPPING_UP, True),
]


FLOW_CSS = """
<style>
.flow { display: flex; align-items: stretch; gap: 0.5rem; margin: 0.4rem 0 0.3rem 0; }
.flow-stage {
  flex: 1 1 0; border: 1px solid rgba(128,128,128,.35); border-radius: 8px;
  padding: 0.7rem 0.75rem 0.8rem 0.75rem; background: rgba(128,128,128,.05);
}
.flow-stage.wide { flex: 1.9 1 0; }
.flow-stage h5 { margin: 0 0 0.55rem 0; padding: 0; font-size: 0.86rem; font-weight: 600; }
.flow-nodes { display: flex; flex-direction: column; gap: 0.3rem; }
.flow-stage.wide .flow-nodes { display: grid; grid-template-columns: 1fr 1fr; gap: 0.35rem;
  align-items: start; }
.flow-node {
  border: 1px solid rgba(128,128,128,.5); border-radius: 3px; padding: 0.28rem 0.55rem;
  font-size: 0.82rem; line-height: 1.3; background: rgba(128,128,128,.07);
}
.flow-node.model { border-radius: 999px; border-color: #1F3A5F; background: rgba(31,58,95,.1); }
.flow-down { text-align: center; font-size: 0.7rem; line-height: 0.6; opacity: 0.55; }
.prompt-text {
  white-space: pre-wrap; font-size: 0.8rem; line-height: 1.5; max-width: 640px;
  max-height: 420px; overflow-y: auto; padding: 0.7rem 0.85rem; border-radius: 4px;
  background: rgba(128,128,128,.1);
}
.flow-next { align-self: center; font-size: 1.1rem; opacity: 0.55; }
@media (max-width: 900px) { .flow { flex-direction: column; } .flow-next { transform: rotate(90deg); } }
</style>
"""


def flow_diagram() -> str:
    """The review as four stages, drawn from the same node list the guide below is written from.

    The graph's own diagram uses the code's node names and is too tall to read at a glance, so this
    one uses the names on this page. A test checks the list against the real graph, so a node added
    to the graph cannot go missing here.
    """
    stages = []
    for title, nodes, in_sequence in STAGES:
        pills = []
        for index, node in enumerate(nodes):
            if in_sequence and index:
                pills.append('<div class="flow-down">&#9660;</div>')
            css = "flow-node model" if node.model else "flow-node"
            pills.append(f'<div class="{css}">{view.escape(node.name)}</div>')
        wide = "" if in_sequence else " wide"
        stages.append(
            f'<div class="flow-stage{wide}"><h5>{view.escape(title)}</h5>'
            f'<div class="flow-nodes">{"".join(pills)}</div></div>'
        )
    arrow = '<div class="flow-next">&#9654;</div>'
    return f'{FLOW_CSS}<div class="flow">{arrow.join(stages)}</div>'


st.subheader("How a review flows")
st.markdown(flow_diagram(), unsafe_allow_html=True)
st.caption("Rounded boxes ask a model. Square boxes are plain code. Read left to right.")

st.markdown(
    """
Each box in the picture is one step. Some steps ask a model, because the question needs
judgement. The rest are plain code, because the answer should be the same every time. Here is
what each one does, in the order it runs.
"""
)
with st.container(border=True):
    st.markdown(
        """
**Three things that hold for every step**

- **Where things are kept.** The run's state is saved after every step in `data/checkpoints.db`.
  Every step also writes one line to the audit log in `data/app.db`: what ran, which model and
  prompt version, how long it took and what it cost.
- **What leaves your machine.** Only the model calls. With Ollama the model runs locally. With
  Claude the text is sent to Anthropic. Evidence search runs locally, over an index built from the
  files in `corpus/`.
- **Tools and MCP.** No step calls a tool or an MCP server. Each model step is one prompt in and
  one structured answer out. Looking up evidence is ordinary code that runs before the prompt is
  built.
"""
    )

prompts = api.prompts()

st.markdown("#### 1. Getting ready")
node_grid(GETTING_READY, prompts)

st.markdown("#### 2. The reviewers")
st.caption(
    "These run side by side, and each one asks a different question of the same piece. A reviewer "
    "with nothing to do is skipped: no claims, or no rules of its kind."
)
node_grid(REVIEWERS, prompts)

st.markdown("#### 3. Checking the reviewers' work")
st.caption("All plain code, in a fixed order. Given the same findings, the result is the same.")
node_grid(CHECKING, prompts)

st.markdown("#### 4. Wrapping up")
node_grid(WRAPPING_UP, prompts)

st.subheader("Platform and use case")
st.caption(
    "MLR review is the first use case on the platform, not the platform itself. The reusable layer "
    "knows nothing about claims or severities, and a test fails the build if it ever imports them."
)
st_mermaid(
    """
graph TB
    subgraph surfaces["Delivery surfaces"]
        ui["Streamlit UI"] -->|HTTP only| apisvc["FastAPI"]
    end
    subgraph usecase["mlr: the MLR use case"]
        graphdef["Review graph and nodes"]
        prompts["Versioned prompts"]
        schemas["Claims, findings, report"]
    end
    subgraph platform["platform_core: reusable agent platform"]
        tenancy["Tenancy and policy as data"]
        gates["Validation gates"]
        hitl["Human review and escalation"]
        audit["Audit log"]
        retrieval["Tenant-scoped retrieval"]
        rules["Business rules registry"]
        evals["Eval harness"]
    end
    apisvc --> usecase
    usecase --> platform
    platform -. "never imports" .-> usecase
""",
    height="520px",
)

view.footer()
