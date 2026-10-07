"""Submit Review: pick a company and product, give it a piece, and start a review."""

from __future__ import annotations

import api_client as api
import streamlit as st
import view

view.page(
    "Submit a review",
    "Pick a company and a product, give it a piece, and run it. It takes three short steps.",
)

AUDIENCES = {"hcp": "Healthcare professionals", "patient": "Patients", "any": "Not specified"}
NOTHING_TO_REVIEW = (
    "There's nothing to review yet. Pick a sample, upload a file or paste some text."
)


def channel_options(tenant_id: str) -> list[str]:
    """Every channel the system already knows: the samples', plus any a company rule is scoped to.

    There is no fixed list of channels. A rule only applies when the channel matches its scope
    exactly, so a pick-list of the known ones is safer than a box to type in.
    """
    known = {s["channel"] for s in api.samples()}
    for rule in api.tenant_rules(tenant_id):
        known.update((rule.get("scope") or {}).get("channels") or [])
    return ["", *sorted(c for c in known if c)]


def channel_label(channel: str) -> str:
    return channel.replace("_", " ").capitalize() if channel else "Not specified"


def preview(text: str, label: str) -> None:
    with st.expander(label):
        st.markdown(
            f'<div class="mlr-proof" style="max-height: 320px; overflow-y: auto;">'
            f"{view.block(text)}</div>",
            unsafe_allow_html=True,
        )


def lock() -> None:
    # Runs before the page redraws, so the button is already locked when it does.
    st.session_state["submitting"] = True


# --- 1. Whose rules ------------------------------------------------------------------------------

st.markdown("#### 1. Whose rules")
tenants = api.tenants()
tenant_col, product_col = st.columns(2)
tenant = tenant_col.selectbox("Company", tenants, format_func=lambda t: t["name"])
product = product_col.selectbox(
    "Product",
    tenant["products"],
    format_func=lambda p: f"{p['name']} (label v{p['label_version']})",
)
st.caption(f"Approved for: {product['indication']}.")

# --- 2. The piece --------------------------------------------------------------------------------

st.markdown("#### 2. The piece")
source = st.radio(
    "Where is it?",
    ["Pick a sample", "Upload a file", "Paste text"],
    horizontal=True,
    label_visibility="collapsed",
)

text = ""
upload: tuple[str, bytes] | None = None
suggested: dict[str, str] = {}
sample_name = ""

# The sample picker sits outside the form because choosing one changes what the form shows.
if source == "Pick a sample":
    samples = [s for s in api.samples() if s["product_id"] == product["product_id"]]
    if not samples:
        st.info(f"No samples for {product['name']} yet. Upload or paste one instead.")
    else:
        chosen = st.selectbox("Sample", samples, format_func=lambda s: s["title"])
        loaded = api.sample(chosen["name"])
        sample_name = chosen["name"]
        text = loaded["text"]
        suggested = {"audience": loaded["audience"], "channel": loaded["channel"]}
        if loaded["notes"]:
            st.caption(f"What it's for: {loaded['notes']}")

submitting = st.session_state.get("submitting", False)

# One form for everything else, so the button reads what is in the boxes at the moment it is
# clicked. Outside a form, a text box only reports what was typed once you click away from it.
with st.form("submit", border=False):
    if source == "Pick a sample" and text:
        preview(text, "Read the sample")
    elif source == "Upload a file":
        uploaded = st.file_uploader("File", type=["pdf", "md", "txt"])
        if uploaded is not None:
            upload = (uploaded.name, uploaded.getvalue())
            st.caption(f"Got {uploaded.name}, {len(upload[1]) / 1024:,.0f} KB.")
    elif source == "Paste text":
        text = st.text_area("Your text", height=260, placeholder="Paste the piece here.")

    st.markdown("#### 3. Who it's for")
    st.caption("These decide which of the company's rules apply, so they're asked, not guessed.")
    audience_col, channel_col, market_col = st.columns(3)
    audience_options = list(AUDIENCES)
    channels = channel_options(tenant["tenant_id"])
    # Keyed by the sample, so picking a different sample brings its own audience and channel.
    audience = audience_col.selectbox(
        "Audience",
        audience_options,
        index=audience_options.index(suggested["audience"])
        if suggested.get("audience") in audience_options
        else 0,
        format_func=AUDIENCES.get,
        key=f"audience-{sample_name}",
    )
    channel = channel_col.selectbox(
        "Channel",
        channels,
        index=channels.index(suggested["channel"]) if suggested.get("channel") in channels else 0,
        format_func=channel_label,
        key=f"channel-{sample_name}",
    )
    market = market_col.text_input("Market", value="us")

    st.form_submit_button(
        "Starting…" if submitting else "Run review",
        type="primary",
        disabled=submitting,
        on_click=lock,
    )

if submitting:
    # The click locked the button; this is the run that does the work, then unlocks it.
    st.session_state["submitting"] = False
    if not (text.strip() or upload):
        st.session_state["submit_problem"] = NOTHING_TO_REVIEW
        st.rerun()
    with st.spinner("Starting the review…"):
        try:
            run_id = api.submit_review(
                tenant_id=tenant["tenant_id"],
                product_id=product["product_id"],
                audience=audience,
                channel=channel,
                market=market.strip(),
                text="" if upload else text,
                upload=upload,
            )
        except api.ApiError as exc:
            st.session_state["submit_problem"] = f"{exc.message} (HTTP {exc.status_code})"
            st.rerun()
    st.session_state["run_id"] = run_id
    st.session_state.pop("compare", None)
    st.switch_page("pages/2_Run_Detail.py")

problem = st.session_state.pop("submit_problem", None)
if problem:
    st.warning(problem)

view.footer()
