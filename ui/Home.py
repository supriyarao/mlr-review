"""Home: what this is, how a review flows node by node, and where the platform ends."""

from __future__ import annotations

import api_client as api
import nodes
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


def node_card(node: nodes.Node, prompts: dict[str, dict]) -> None:
    # Most model steps make one call, named after the step. The house-rule step makes two.
    asked = [prompts[key] for key in nodes.MODEL_CALLS.get(node.key, (node.key,)) if key in prompts]
    with st.container(border=True):
        if node.model:
            size = {"strong": "the larger model", "fast": "the smaller model"}
            if len(asked) == 1:
                kind = view.chip(f"Asks {size.get(asked[0]['tier'], 'a model')}")
            elif asked:
                kind = view.chip("Asks both models")
            else:
                kind = view.chip("Asks a model")
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
        if asked:
            with link.popover("Read the prompt" if len(asked) == 1 else "Read the prompts"):
                for prompt in asked:
                    st.caption(f"`{prompt['file']}`, version `{prompt['version']}`")
                    # Plain escaped HTML rather than st.code: the code block loads its highlighter
                    # from the server on first use, and that fetch fails if the server has
                    # restarted.
                    st.markdown(
                        f'<pre class="prompt-text">{view.block(prompt["text"])}</pre>',
                        unsafe_allow_html=True,
                    )


def node_grid(stage: list[nodes.Node], prompts: dict[str, dict]) -> None:
    for start in range(0, len(stage), 2):
        for column, node in zip(st.columns(2), stage[start : start + 2], strict=False):
            with column:
                node_card(node, prompts)


st.subheader("How a review flows")
st.markdown(nodes.flow_diagram(), unsafe_allow_html=True)
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
- **Tools and MCP.** There is no MCP server, and a model never picks a tool to call. One step,
  the balance and safety reviewer, runs the company's own checkers as plain code and hands the
  model what they measured. Each model step is still one prompt in and one structured answer
  out. Looking up evidence is ordinary code that runs before the prompt is built.
"""
    )

prompts = api.prompts()

st.markdown("#### 1. Getting ready")
node_grid(nodes.GETTING_READY, prompts)

st.markdown("#### 2. The reviewers")
st.caption(
    "These run side by side, and each one asks a different question of the same piece. A reviewer "
    "with nothing to do is skipped: no claims, or no rules of its kind."
)
node_grid(nodes.REVIEWERS, prompts)

st.markdown("#### 3. Checking the reviewers' work")
st.caption("All plain code, in a fixed order. Given the same findings, the result is the same.")
node_grid(nodes.CHECKING, prompts)

st.markdown("#### 4. Wrapping up")
node_grid(nodes.WRAPPING_UP, prompts)

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
