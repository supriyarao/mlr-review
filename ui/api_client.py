"""The UI's only route to the system: HTTP calls to the API.

The UI never imports the graph, the service or the database (``tests/test_import_boundary.py``
enforces it). In production it is a separate process on a separate machine, so everything it shows
has to be something the API can say.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("MLR_API_URL", "http://localhost:8000").rstrip("/")
STUB_HEADER = "x-mlr-stub"


class ApiError(Exception):
    """The API answered with an error. ``detail`` is whatever it said, for display."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)

    @property
    def message(self) -> str:
        detail = self.detail
        if isinstance(detail, dict):
            return str(detail.get("message", detail))
        if isinstance(detail, list):
            # FastAPI validation errors: say which field and why.
            return "; ".join(
                f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}" for e in detail
            )
        return str(detail)


@st.cache_resource
def _client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=30.0)


def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    try:
        response = _client().request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        st.error(
            f"Can't reach the API at {API_URL}. Start it with `make api`, or run both together "
            f"with `make demo`.\n\n`{type(exc).__name__}: {exc}`"
        )
        st.stop()
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(response.status_code, detail)
    return response


def get(path: str, **params: Any) -> Any:
    return _request("GET", path, params={k: v for k, v in params.items() if v not in ("", None)})


def get_json(path: str, **params: Any) -> Any:
    return get(path, **params).json()


def post_json(path: str, body: dict[str, Any] | None = None) -> Any:
    return _request("POST", path, json=body or {}).json()


def is_stubbed(response: httpx.Response) -> bool:
    return STUB_HEADER in response.headers


# --- Calls the pages share -----------------------------------------------------------------------


@st.cache_data(ttl=10, show_spinner=False)
def health() -> dict[str, Any]:
    return get_json("/health")


@st.cache_data(ttl=60, show_spinner=False)
def tenants() -> list[dict[str, Any]]:
    return get_json("/tenants")["tenants"]


@st.cache_data(ttl=60, show_spinner=False)
def tenant(tenant_id: str) -> dict[str, Any]:
    return get_json(f"/tenants/{tenant_id}")


@st.cache_data(ttl=60, show_spinner=False)
def tenant_rules(tenant_id: str) -> list[dict[str, Any]]:
    return get_json(f"/tenants/{tenant_id}/rules")["rules"]


@st.cache_data(ttl=300, show_spinner=False)
def markets(product_id: str) -> list[str]:
    """The markets the approval records know for a product. Empty when there are none."""
    return [m["market"] for m in get_json("/markets", product_id=product_id)["markets"]]


@st.cache_data(ttl=300, show_spinner=False)
def samples() -> list[dict[str, Any]]:
    return get_json("/samples")["samples"]


@st.cache_data(ttl=300, show_spinner=False)
def sample(name: str) -> dict[str, Any]:
    return get_json(f"/samples/{name}")


@st.cache_data(ttl=300, show_spinner=False)
def sample_image(name: str) -> bytes:
    """The artwork for a sample that is an image. Fetched here because the browser can't reach
    the API directly; only this process can."""
    return get(f"/samples/{name}/image").content


@st.cache_data(ttl=300, show_spinner=False)
def graph_source() -> str:
    return get_json("/graph")["source"]


@st.cache_data(ttl=300, show_spinner=False)
def prompts() -> dict[str, dict[str, Any]]:
    """Each model-backed node's prompt, keyed by node name."""
    return {p["node"]: p for p in get_json("/prompts")["prompts"]}


def runs(tenant_id: str = "", status: str = "") -> list[dict[str, Any]]:
    return get_json("/reviews", tenant_id=tenant_id, status=status)["runs"]


def review(run_id: str) -> dict[str, Any]:
    return get_json(f"/reviews/{run_id}")


def audit(run_id: str) -> dict[str, Any]:
    return get_json(f"/reviews/{run_id}/audit")


def document(run_id: str) -> str:
    """The submitted text, or "" when this API instance did not receive the submission."""
    try:
        return get_json(f"/reviews/{run_id}/document")["document"]
    except ApiError as exc:
        if exc.status_code == 404:
            return ""
        raise


def submit_review(
    *,
    tenant_id: str,
    product_id: str,
    audience: str,
    channel: str,
    market: str,
    text: str = "",
    upload: tuple[str, bytes] | None = None,
    sample_name: str = "",
) -> str:
    """Submit a piece; returns the run id."""
    data = {
        "sample_name": sample_name,
        "tenant_id": tenant_id,
        "product_id": product_id,
        "audience": audience,
        "channel": channel,
        "market": market,
    }
    if upload is not None:
        response = _request("POST", "/reviews", data=data, files={"file": upload})
    else:
        response = _request("POST", "/reviews", data={**data, "text": text})
    return response.json()["run_id"]
