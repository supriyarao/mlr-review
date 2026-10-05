"""LLM-as-judge helper.

Why this exists: some eval questions have no deterministic answer. "Did the model extract the same
claims a human would?" and "does this cited passage actually support this finding?" are judgement
calls, and hand-writing exact expected strings for them would measure phrasing rather than
substance.

An LLM decision -- and therefore used sparingly and with guardrails:

* **Judges a sample, not everything.** A judge costs money and can be wrong; it is used to spot
  systemic problems, not to grade every finding.
* **Returns a verdict plus a reason.** A score with no reason cannot be audited, and a judge that
  cannot be audited is just another opaque model in the loop.
* **Never judges whether a finding is correct.** That is what the golden set's deterministic
  matching is for. The judge is confined to questions the golden set structurally cannot answer.
* **Replayable.** Judge calls go through the same cassette mechanism as everything else, so the
  offline eval reproduces exactly.

Trade-off: a judge shares a family, and therefore some blind spots, with the model being judged.
Reported as a limitation in the README rather than papered over; the honest fix is periodic human
spot-checks of the judge itself.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JUDGE_PROMPT_VERSION = "judge-v1"

_FAITHFULNESS_PROMPT = """You are auditing a citation in an automated content review.

A reviewer system produced this finding:
  issue: {issue}

It cited this passage as evidence:
  passage: {passage}

Question: does the cited passage actually support the stated issue?

Answer with JSON only, no other text:
{{"supports": true or false, "confidence": 0-100, "reason": "one sentence"}}

Judge only whether the passage supports the issue as stated. Do not judge whether the issue is a
real problem, and do not use any knowledge beyond the passage shown."""

_EXTRACTION_PROMPT = """You are auditing claim extraction in an automated content review.

Expected claim (written by a human):
  {expected}

Claim the system extracted:
  {actual}

Question: do these refer to the same underlying claim? Wording may differ.

Answer with JSON only, no other text:
{{"equivalent": true or false, "confidence": 0-100, "reason": "one sentence"}}"""


class JudgeVerdict(BaseModel):
    """One judge decision. ``reason`` is required so every verdict is inspectable."""

    model_config = ConfigDict(extra="forbid")

    verdict: bool
    confidence: int = Field(ge=0, le=100, default=0)
    reason: str = ""
    # Set when the judge's own output could not be parsed, which is itself a signal worth seeing
    # rather than hiding behind a default verdict.
    parse_error: str = ""

    @property
    def usable(self) -> bool:
        return not self.parse_error


def parse_verdict(text: str, *, key: Literal["supports", "equivalent"]) -> JudgeVerdict:
    """Parse a judge response.

    Tolerates a model wrapping JSON in prose or a code fence, because a judge that fails on
    formatting would make the metric about formatting. A genuinely unparseable response is
    recorded as a parse error rather than silently becoming ``False`` -- a false negative from the
    judge would look identical to a real finding of unfaithfulness.
    """
    raw = text.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return JudgeVerdict(verdict=False, parse_error=f"no JSON object in response: {raw[:120]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return JudgeVerdict(verdict=False, parse_error=f"invalid JSON: {exc}")

    if key not in data:
        return JudgeVerdict(verdict=False, parse_error=f"response missing {key!r} field")

    return JudgeVerdict(
        verdict=bool(data[key]),
        confidence=int(data.get("confidence") or 0),
        reason=str(data.get("reason") or ""),
    )


def faithfulness_prompt(*, issue: str, passage: str) -> str:
    """Prompt asking whether a cited passage supports a finding."""
    return _FAITHFULNESS_PROMPT.format(issue=issue.strip(), passage=passage.strip())


def extraction_prompt(*, expected: str, actual: str) -> str:
    """Prompt asking whether two phrasings describe the same claim."""
    return _EXTRACTION_PROMPT.format(expected=expected.strip(), actual=actual.strip())


def sample_for_judging(items: list[Any], *, limit: int, seed: int = 1234) -> list[Any]:
    """Deterministic sample of items to judge.

    Seeded so the offline eval judges the same items every time -- otherwise the judged metric
    would drift run to run even with identical findings, and the baseline could never be
    reproduced exactly.
    """
    if limit <= 0 or len(items) <= limit:
        return list(items)
    import random

    rng = random.Random(seed)
    return rng.sample(list(items), limit)


def summarize(verdicts: list[JudgeVerdict]) -> dict[str, Any]:
    """Aggregate judge verdicts, keeping unusable ones visible.

    ``rate`` is computed over usable verdicts only, with the unusable count reported alongside, so
    a judge that silently started failing shows up as a shrinking denominator instead of a
    mysteriously moving rate.
    """
    usable = [v for v in verdicts if v.usable]
    positive = sum(1 for v in usable if v.verdict)
    return {
        "judged": len(verdicts),
        "usable": len(usable),
        "unusable": len(verdicts) - len(usable),
        "positive": positive,
        "rate": round(positive / len(usable), 4) if usable else 0.0,
        "mean_confidence": (
            round(sum(v.confidence for v in usable) / len(usable), 1) if usable else 0.0
        ),
        "failures": [
            {"reason": v.reason, "confidence": v.confidence} for v in usable if not v.verdict
        ],
    }
