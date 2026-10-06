"""Submit Review: choose a tenant and product, supply the piece, and start a review."""

from __future__ import annotations

import api_client as api
import streamlit as st
import view

view.page(
    "Submit a review",
    "Choose whose policy applies, supply the piece, and say who it is for. Audience and channel are "
    "asked for rather than guessed, because they decide which of the tenant's rules are in scope.",
)

AUDIENCES = {"hcp": "Healthcare professionals", "patient": "Patients", "any": "Not specified"}

tenants = api.tenants()
tenant_col, product_col = st.columns(2)
tenant = tenant_col.selectbox("Tenant", tenants, format_func=lambda t: t["name"])
product = product_col.selectbox(
    "Product",
    tenant["products"],
    format_func=lambda p: f"{p['name']} (label v{p['label_version']})",
)
st.caption(f"Approved indication: {product['indication']}.")

source = st.radio(
    "Where is the piece?", ["Pick a sample", "Upload a file", "Paste text"], horizontal=True
)

text = ""
upload: tuple[str, bytes] | None = None
suggested: dict[str, str] = {}

if source == "Pick a sample":
    samples = [s for s in api.samples() if s["product_id"] == product["product_id"]]
    if not samples:
        st.info(f"There are no sample pieces for {product['name']}. Upload or paste one instead.")
    else:
        chosen = st.selectbox("Sample piece", samples, format_func=lambda s: s["title"])
        loaded = api.sample(chosen["name"])
        text = loaded["text"]
        suggested = {"audience": loaded["audience"], "channel": loaded["channel"]}
        if loaded["notes"]:
            with st.expander("What this sample is designed to test"):
                st.write(loaded["notes"])
elif source == "Upload a file":
    uploaded = st.file_uploader("Promotional piece", type=["pdf", "md", "txt"])
    if uploaded is not None:
        upload = (uploaded.name, uploaded.getvalue())
        if not uploaded.name.lower().endswith(".pdf"):
            text = uploaded.getvalue().decode("utf-8", errors="replace")
else:
    text = st.text_area("Promotional copy", height=260, placeholder="Paste the piece here.")

audience_col, channel_col, market_col = st.columns(3)
audience_options = list(AUDIENCES)
audience = audience_col.selectbox(
    "Audience",
    audience_options,
    index=audience_options.index(suggested.get("audience", "hcp"))
    if suggested.get("audience") in audience_options
    else 0,
    format_func=AUDIENCES.get,
)
channel = channel_col.text_input(
    "Channel", value=suggested.get("channel", ""), placeholder="journal_ad, sales_aid, email…"
)
market = market_col.text_input("Market", value="us")

if text:
    with st.expander("Preview of the text that will be reviewed", expanded=True):
        st.markdown(f'<div class="mlr-proof">{view.block(text)}</div>', unsafe_allow_html=True)
elif upload:
    st.caption(f"{upload[0]} will be read as text when the review starts.")

ready = bool(text.strip() or upload)
if st.button("Run review", type="primary", disabled=not ready):
    try:
        run_id = api.submit_review(
            tenant_id=tenant["tenant_id"],
            product_id=product["product_id"],
            audience=audience,
            channel=channel.strip(),
            market=market.strip(),
            text="" if upload else text,
            upload=upload,
        )
    except api.ApiError as exc:
        view.show_error(exc)
    else:
        st.session_state["run_id"] = run_id
        st.session_state.pop("compare", None)
        st.switch_page("pages/2_Run_Detail.py")
if not ready:
    st.caption("Supply a piece to enable the review.")

view.footer()
