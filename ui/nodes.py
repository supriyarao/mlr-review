"""The review's steps, described once for every page that shows them.

The Home page explains each step; the Run Detail page shows the same steps for one run. Both read
this list, so a step has one name and one description wherever it appears. A test checks the list
against the real graph, so a node added to the graph cannot go undescribed.
"""

from __future__ import annotations

from dataclasses import dataclass

import view


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
    # A decision model is still a model, but a different kind: it is asked a question with set
    # answers and gives back how sure it is. It writes no text, so it can't invent a finding.
    decision: bool = False


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
        "A search pulls likely matches from the company's cleared claims. A decision model then "
        "says how sure it is that each one means the same thing. It only answers that one "
        "question and writes no text of its own. The best match wins if it is at least 85% sure. "
        "Anything less and the claim counts as new.",
        "The company's cleared-claims list (local search index) and a decision model.",
        "Which claims matched which cleared claim.",
        model=True,
        decision=True,
    ),
]

REVIEWERS = [
    Node(
        "substantiation",
        "The evidence reviewer",
        "Asks: does the evidence really back this claim, as strongly as the piece says it? And "
        "does the study in the footnote actually say it?",
        "Looks for a subgroup result passed off as a general one, a bigger result than was "
        "measured, a dropped comparison, a claim with no support at all, or a footnote pointing "
        "at a study that measured something else.",
        "The approved label and the studies (local search index) and the model.",
        "Findings, each with the quote and the passage it relies on.",
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
        "presentation",
        "The balance and safety reviewer",
        "Asks three things. Is the safety information easy to find? Does it cover everything the "
        "product's label says it must, such as a boxed warning? And do the risks get a fair "
        "share of the page next to the benefits?",
        "Runs the company's own two checkers first, the ones it already trusts. They measure "
        "the page: where the safety section sits, how long it is, what it leaves out. Then a "
        "small model reads those numbers and says what they mean. It runs even when no claims "
        "were found, since a piece can still be missing its safety text.",
        "The claims, the section map from the reader, the product's label, the company's two "
        "checkers, and the model.",
        "Findings tagged as prominence, completeness or balance, each backed by something a "
        "checker measured. The tags stay separate because the companies grade them differently.",
        model=True,
    ),
    Node(
        "house_rules",
        "The house-rule reviewer",
        "Applies the company's own rules: the yes-or-no ones, the ones that need judgement, and "
        "the company's code of practice.",
        "The yes-or-no rules run first, as plain code: banned words, required wording, required "
        "sections, a trademark symbol on first mention. Only what a pattern can't answer goes to "
        "a model. A small model reads the plain-English rules. A larger one reads the piece "
        "against each company document and must quote the clause it relies on. Rule text is "
        "treated as data, not instructions, and a finding that names a rule it wasn't given is "
        "thrown away.",
        "The company rules and reference documents that cover this piece, and the models.",
        "Findings that name the rule and its version, or quote the clause.",
        model=True,
    ),
]

CHECKING = [
    Node(
        "consolidate",
        "The checker",
        "Checks the reviewers' work, then tidies it into the list you read. Nothing a model "
        "said is trusted until it has been looked up.",
        "Four things, always in this order, and no model is involved in any of them. It looks "
        "up every quote and every source word for word, and marks anything it can't find as "
        "unverified. It never quietly drops one. It uses the company's settings to decide "
        "blocker or advisory. It merges two reviewers who flagged the same words into one "
        "finding. Then it flags anything below the company's confidence threshold.",
        "The piece, plus the company's label, studies, rules, documents and settings.",
        "The final list of findings. Each one says how it was verified and which setting "
        "decided its severity.",
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


# Steps that make more than one model call, and what the API calls each one. The prompts and the
# audit trail are listed under the call's name, not the step's.
MODEL_CALLS: dict[str, tuple[str, ...]] = {"house_rules": ("llm_rules", "reference_doc_review")}

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
.flow-node.decision { border-style: dashed; }
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
            css += " decision" if node.decision else ""
            pills.append(f'<div class="{css}">{view.escape(node.name)}</div>')
        wide = "" if in_sequence else " wide"
        stages.append(
            f'<div class="flow-stage{wide}"><h5>{view.escape(title)}</h5>'
            f'<div class="flow-nodes">{"".join(pills)}</div></div>'
        )
    arrow = '<div class="flow-next">&#9654;</div>'
    return f'{FLOW_CSS}<div class="flow">{arrow.join(stages)}</div>'


ALL_NODES: dict[str, Node] = {node.key: node for _, nodes, _ in STAGES for node in nodes}
