"""**Placeholder** for a lookup against the company's regulatory information system.

The third tool, and the one that is a *lookup* rather than a measurement. The sibling validators
compute something about the piece in front of them; this one answers a question the piece cannot
answer about itself: is this product approved in the market this material is for, and for what.

Why a tool and not tenant configuration. Approval status is not a marketing decision and does not
live in a marketing config. It lives in a regulatory information system, it changes on dates nobody
in the review loop controls, and it is the sort of thing an auditor will ask you to evidence from
source. Modelling it as an external lookup -- with an ``as_of`` date and a named source on every
answer -- is both the more honest shape and a better demonstration of why the tool registry exists.
A real deployment replaces the dictionary below with an API client and changes nothing else.

**This does not emit findings.** Like its siblings, its output is input to the model's judgement.
The off_label check decides what an unapproved status means for a given claim.
"""

from __future__ import annotations

from typing import Any

from platform_core.tools import ToolSpec, register

VERSION = "1.0.0"

# Stands in for the RIM system. Keyed by (product_id, market).
#
# The interesting row is declasal/eu. Under `us` Declasal is approved for skin and skin-structure
# infections, so promoting it for UTI is off-label -- serious, and correctable by removing the
# indication. Under `eu` it is not approved at all, so the same piece promotes an unapproved
# product, which no edit to the claim can fix. One file, two markets, two different findings.
_APPROVALS: dict[tuple[str, str], dict[str, Any]] = {
    ("zelvoran", "us"): {
        "status": "approved",
        "indications": ["moderate-to-severe plaque psoriasis in adults"],
        "approved_on": "2025-03-14",
    },
    ("zelvoran", "eu"): {
        "status": "approved",
        "indications": ["moderate-to-severe plaque psoriasis in adults"],
        "approved_on": "2025-09-02",
    },
    ("nexapril", "us"): {
        "status": "approved",
        "indications": [
            "adults with type 2 diabetes mellitus and established atherosclerotic "
            "cardiovascular disease"
        ],
        "approved_on": "2024-11-08",
    },
    ("declasal", "us"): {
        "status": "approved",
        "indications": [
            "skin and skin-structure infections in adults caused by designated susceptible bacteria"
        ],
        "approved_on": "2026-01-20",
    },
    ("declasal", "eu"): {
        "status": "not_approved",
        "indications": [],
        "submitted_on": "2026-04-02",
    },
}

SOURCE = "acme-rim (placeholder)"
AS_OF = "2026-10-01"


def lookup_approval_status(
    *,
    product_id: str = "",
    market: str = "",
    **_: Any,
) -> dict[str, Any]:
    """Approval status for one product in one market.

    An unknown pair returns ``status: "unknown"`` rather than guessing either way. Treating a
    missing record as "approved" would let an unapproved product through silently, and treating it
    as "not approved" would block every product this placeholder has never heard of. Unknown is the
    only honest answer, and the check is told to escalate rather than decide on it.
    """
    product = (product_id or "").strip().lower()
    place = (market or "").strip().lower()

    if not product or not place:
        return {
            "source": SOURCE,
            "as_of": AS_OF,
            "product_id": product,
            "market": place,
            "status": "unknown",
            "why": "a product and a market are both needed to look up an approval",
        }

    record = _APPROVALS.get((product, place))
    if record is None:
        return {
            "source": SOURCE,
            "as_of": AS_OF,
            "product_id": product,
            "market": place,
            "status": "unknown",
            "why": (
                f"no approval record for {product!r} in {place!r}. This is not a statement that it "
                f"is unapproved -- the record is missing, and the difference matters."
            ),
            "known_markets": sorted({m for p, m in _APPROVALS if p == product}),
        }

    return {
        "source": SOURCE,
        "as_of": AS_OF,
        "product_id": product,
        "market": place,
        **record,
    }


SPEC = register(
    ToolSpec(
        name="approval_status",
        version=VERSION,
        description=(
            "Looks up whether a product is approved in a given market, and for which indications, "
            "against the company's regulatory information system. Returns a status of approved, "
            "not_approved or unknown with the date the answer is current as of. Returns a record, "
            "not a finding."
        ),
        fn=lookup_approval_status,
        requires=("product_id", "market"),
    ),
    replace=True,
)
