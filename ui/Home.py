"""Home: what this is, how a review flows, and where the platform ends and the use case begins."""

from __future__ import annotations

import api_client as api
import streamlit as st
import view
from streamlit_mermaid import st_mermaid

view.page("MLR Review Agent")

st.markdown(
    """
Promotional material for a medicine has to clear medical, legal and regulatory review before it is
used. This demo reads a piece, pulls out the claims it makes, and checks each one against the
approved label, the cited studies and the tenant's own promotional rules. It reports what it found
with the exact passage each finding rests on, then hands the piece to a reviewer of the right
seniority. Two fictional companies run on the same platform with different policies, so the same
piece can be routed differently for each.
"""
)
view.human_banner()

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

st.subheader("How a review flows")
st.caption(
    "Drawn from the compiled graph, so this picture cannot drift from the code. Checks and tenant "
    "rules run in parallel; two gates sit between them and the reviewer; every path ends at a "
    "person."
)
st_mermaid(api.graph_source(), height="620px")

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
