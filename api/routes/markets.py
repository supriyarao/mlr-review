"""Which markets a product has an approval record for.

Exists so a submission form can offer a pick-list instead of free text. It reads from the same
placeholder regulatory lookup the off_label check calls, deliberately: duplicating the market list
into tenant configuration would create a second source that drifts, and the first symptom would be
a picker offering a market the check has never heard of.

Deterministic code. No model involved.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from mlr.tools.approval_status import AS_OF, SOURCE, markets_for

router = APIRouter(tags=["markets"])


@router.get("/markets")
def list_markets(product_id: str = "") -> dict[str, Any]:
    """Markets with an approval record for this product.

    A market with no record is absent from the list rather than listed as unapproved: the record is
    missing, which is not the same as a refusal, and a form has no business asserting the stronger
    claim. An unknown or empty ``product_id`` therefore returns an empty list, not an error.
    """
    return {
        "product_id": product_id.strip().lower(),
        "markets": markets_for(product_id),
        "source": SOURCE,
        "as_of": AS_OF,
    }
