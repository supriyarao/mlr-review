"""Tenants: each tenant's policy side by side, and the same piece reviewed under each."""

from __future__ import annotations

from typing import Any

import api_client as api
import streamlit as st
import view

view.page(
    "Tenants",
    "Each tenant's policy is a configuration file, not code. The rows marked below are where the "
    "tenants differ, and those differences are why the same piece can be routed differently.",
)

details = [api.tenant(t["tenant_id"]) for t in api.tenants()]
if not details:
    st.info("No tenants are configured.")
    st.stop()


def text(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value in (None, ""):
        return "Not set"
    return str(value)


def policy_rows(tenant: dict[str, Any]) -> dict[str, str]:
    """One tenant's policy flattened to setting -> value, in display order."""
    rows: dict[str, str] = {
        "Config version": tenant["version"],
        "Config hash": tenant["config_hash"],
        "Confidence threshold": str(tenant["confidence_threshold"]),
        "Products": ", ".join(p["name"] for p in tenant["products"]),
        "Active business rules": str(tenant["active_rule_count"]),
    }
    for rule in tenant["severity_rules"]:
        check = view.CHECK_LABELS.get(rule["check_type"], "Anything else")
        rows[f"Severity: {check}"] = rule["severity"].capitalize()
    escalation = tenant["escalation"]
    rows["Standard reviewer tier"] = view.tier_label(escalation["standard_tier"])
    rows["Default escalation tier"] = view.tier_label(escalation["default_escalation_tier"])
    for rule in escalation["rules"]:
        reason = rule["when_reason"].replace("_", " ")
        rows[f"Escalate {reason} to"] = view.tier_label(rule["tier"])
    for flag, value in tenant["policy_flags"].items():
        rows[f"Switch: {flag.replace('_', ' ')}"] = text(value)
    return rows


# Identity rows always differ and say nothing about policy, so they are never marked.
IDENTITY = {"Config version", "Config hash", "Products"}

per_tenant = [policy_rows(t) for t in details]
settings: list[str] = []
for rows in per_tenant:
    for setting in rows:
        if setting not in settings:
            settings.append(setting)

only_differences = st.toggle("Show only the differences")
header = "".join(f"<th>{view.escape(t['name'])}</th>" for t in details)
body = []
differing = 0
for setting in settings:
    values = [rows.get(setting, "Not set") for rows in per_tenant]
    differs = len(set(values)) > 1 and setting not in IDENTITY
    differing += differs
    if only_differences and not differs:
        continue
    cells = "".join(f"<td>{view.escape(v)}</td>" for v in values)
    css = ' class="differs"' if differs else ""
    body.append(f"<tr{css}><td>{view.escape(setting)}</td>{cells}</tr>")
st.markdown(
    f'<table class="mlr-diff"><tr><th>Setting</th>{header}</tr>{"".join(body)}</table>',
    unsafe_allow_html=True,
)
st.caption(f"{differing} policy setting(s) differ between tenants.")

st.subheader("Business rules")
st.caption(
    "Authored by each tenant's compliance owners, versioned, and recorded against every run they "
    "applied to."
)
for column, tenant in zip(st.columns(len(details)), details, strict=True):
    with column:
        st.markdown(f"**{tenant['name']}**")
        rules = api.tenant_rules(tenant["tenant_id"])
        if not rules:
            st.info("No rules are loaded. Run `make seed` to load the shipped rule set.")
            continue
        st.dataframe(
            [
                {
                    "Rule": f"{r['id']} v{r['version']}",
                    "Title": r["title"],
                    "Kind": r["kind"].replace("_", " "),
                    "Severity": r["severity"].capitalize(),
                    "Audience": r["scope"]["audience"],
                    "Status": r["status"],
                }
                for r in rules
            ],
            hide_index=True,
            width="stretch",
        )

st.subheader("Review the same piece under each tenant")
shared_products = set.intersection(*[{p["product_id"] for p in t["products"]} for t in details])
shared_samples = [s for s in api.samples() if s["product_id"] in shared_products]
if len(details) < 2 or not shared_samples:
    st.info("This needs at least two tenants that both promote a product with a sample piece.")
else:
    chosen = st.selectbox("Sample piece", shared_samples, format_func=lambda s: s["title"])
    names = " and ".join(t["name"] for t in details)
    if st.button(f"Run under {names}", type="primary"):
        piece = api.sample(chosen["name"])
        compare: list[tuple[str, str]] = []
        try:
            for tenant in details:
                run_id = api.submit_review(
                    tenant_id=tenant["tenant_id"],
                    product_id=piece["product_id"],
                    audience=piece["audience"],
                    channel=piece["channel"],
                    market="us",
                    text=piece["text"],
                )
                compare.append((tenant["name"], run_id))
        except api.ApiError as exc:
            view.show_error(exc)
        else:
            st.session_state["compare"] = compare
            st.switch_page("pages/2_Run_Detail.py")

view.footer()
