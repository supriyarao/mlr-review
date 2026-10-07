"""Model responses for running the system with no API key.

Why this is needed: the demo has to be clickable by someone who cloned the repo, and three of the
backends cannot help. ``anthropic`` needs a key. ``ollama`` needs a local server and takes about two
minutes per model call on a CPU-only machine, so a reviewer watching the UI would wait a quarter of an
hour. ``fake`` has no responses of its own.

So when the backend is ``fake``, a submitted piece is matched against the golden set and the eval
oracle supplies that case's findings. That makes the whole pipeline run for real -- ingest, retrieval,
grounding against the actual corpus, tenant severity rules, dedupe, routing, deterministic business
rules, the human interrupt -- with only the *LLM findings* synthesised.

**What the UI must say when this is active:** findings came from a fixture, not a model. The honest
framing is that the demo shows the orchestration, and the model's contribution is the one part being
stood in for. ``mlr.service.describe_backend_note()`` returns that sentence so the UI does not have to
invent it, and ``platform_core.llm.describe_backend()`` already reports ``backend: fake``.

A piece that matches no golden case gets a no-findings script, so it still runs end to end and still
reaches a human -- it simply has nothing to report.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from platform_core.evals.schema import GoldenCase, load_cases
from platform_core.rules.deterministic import normalize_whitespace

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"

# How much of a document to compare when matching a submission to a golden case. Enough to be
# distinctive, short enough that a trailing edit does not stop it matching.
# Kept only to describe the old behaviour in the comment below; nothing reads it.
_OLD_MATCH_PREFIX_CHARS = 400

NO_FINDINGS = json.dumps({"findings": []})


@lru_cache(maxsize=1)
def _cases() -> tuple[GoldenCase, ...]:
    try:
        return tuple(load_cases(GOLDEN_DIR))
    except Exception:  # pragma: no cover - a missing golden set must not break the demo
        return ()


def _fingerprint(text: str) -> str:
    """The whole piece, whitespace-normalised. Not a prefix.

    This used to hash the first 400 characters, on the reasoning that ingest strips front matter
    and emphasis and an exact match would fail for reasons unrelated to the content. The reasoning
    was sound and the consequence was not: an edit below character 400 left the fingerprint
    unchanged, so a modified piece silently matched the sample it was derived from and was handed
    that sample's recorded findings.

    That is the worst answer available. Someone pastes a sample in, adds a paediatric claim to a
    medicine approved in adults, and gets back a confident review that never saw the edit -- not
    "I cannot review this offline", but a specific clean verdict on a document nothing read.

    Normalising still absorbs the whitespace and emphasis differences the prefix was there to
    tolerate. What it no longer does is tolerate a different document.
    """
    return normalize_whitespace(text).lower()


def match_case(document: str, *, tenant_id: str = "") -> GoldenCase | None:
    """The golden case this submission corresponds to, if any.

    Matched on the whole normalised text: see :func:`_fingerprint` for why a prefix was worse than
    no match at all. A piece that does not match any sample gets the no-findings script, which is
    honest but useless -- an edited piece has to be reviewed by a real backend.
    """
    target = _fingerprint(document)
    if not target:
        return None

    candidates = [c for c in _cases() if not tenant_id or c.tenant_id == tenant_id]
    for case in candidates:
        try:
            body = case.promo_text(REPO_ROOT)
        except Exception:  # pragma: no cover
            continue
        from mlr.nodes.ingest import normalize_document, parse_front_matter

        _, stripped = parse_front_matter(body)
        if _fingerprint(normalize_document(stripped)) == target:
            return case
    return None


def script_for(document: str, *, tenant_id: str = "") -> dict[str, list[str]]:
    """Scripted model responses for one submission.

    Returns the oracle's script when the piece is a known golden case, and a no-findings script
    otherwise -- so an arbitrary upload still runs the full pipeline and still reaches a reviewer.
    """
    case = match_case(document, tenant_id=tenant_id)
    if case is not None:
        from mlr import eval_oracle

        return eval_oracle.build_script(case, document)

    return {
        "extract_claims": [json.dumps({"claims": []})],
        "preapproved_filter": [json.dumps({"equivalent": False, "reason": "No fixture match."})],
        "substantiation": [NO_FINDINGS],
        "off_label": [NO_FINDINGS],
        "presentation": [NO_FINDINGS],
        # Keyed by the model call inside house_rules, not by the node.
        "llm_rules": [NO_FINDINGS],
        "reference_doc_review": [NO_FINDINGS],
    }


def build_fake(document: str, *, tenant_id: str = "") -> Any:
    """A :class:`~platform_core.llm.FakeLLM` primed for this submission."""
    from platform_core.llm import FakeLLM

    return FakeLLM(scripted=script_for(document, tenant_id=tenant_id), default_response="{}")


def describes_a_known_case(document: str, *, tenant_id: str = "") -> str:
    """The id of the golden case this submission matches, or "" -- for the UI to show."""
    case = match_case(document, tenant_id=tenant_id)
    return case.id if case else ""
