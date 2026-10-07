"""Apply the tenant's business rules: the yes-or-no ones, the judged ones, and the documents.

One node, :func:`house_rules`, with three parts inside it. They were three nodes until the
timeline made the cost obvious: three boxes labelled some variation of "the company's rules" told a
reader nothing that the finding's own citation does not already say, and every one of these
findings names its rule by id and version.

The three parts remain genuinely different in cost and failure mode, and the order they run in is
the repo's standing rule about when a model earns its place, made visible in one step:

* :func:`_deterministic_part` makes **no model call at all**. A banned-term list is not a judgement
  call, and routing it through a model would add cost, latency and nondeterminism to a question
  regex answers exactly. This is also what makes the rules demo work with no API key.
* :func:`_llm_part` batches several rules into one call. Rules are short and a model can hold a
  handful at once, so one call per rule would multiply cost for no accuracy gain -- up to a point.
* :func:`_reference_doc_part` puts a whole tenant document in context, one call per document.

Each part contributes independently: a schema failure in one escalates on its own and leaves the
others' findings intact. Three separate nodes gave that for free; one node has to preserve it
deliberately, which is why the parts return their errors rather than raising them.

All three convert a platform :class:`~platform_core.rules.schema.RuleViolation` into an MLR
:class:`~mlr.schemas.Finding`. That conversion is the seam that lets the rules engine stay in
``platform_core`` without knowing what a finding is.
"""

from __future__ import annotations

from typing import Any

from mlr.nodes.base import node_run
from mlr.prompts import rules as rule_prompts
from mlr.schemas import Citation, Finding
from platform_core import llm
from platform_core.rules import deterministic, prompting
from platform_core.rules.registry import RuleRegistry
from platform_core.rules.schema import Rule, RuleInput, RuleViolation, Segment
from platform_core.tenancy import tenant_reference_docs_dir

# Trade-off: how many rules go into one model call. Four is a deliberate middle.
#
# One rule per call gives the model a single question and the best per-rule accuracy, but costs one
# call per rule -- with eight rules that is eight round trips on a path a reviewer is waiting on, and
# the system prompt and the piece are re-sent every time.
#
# All rules in one call is cheapest but measurably worse: attention is split, and the failure mode is
# that the model reports the two most obvious violations and quietly stops looking.
#
# Four keeps the prompt focused enough that each rule gets stated and evaluated, while cutting calls
# by three quarters. The right value is an empirical question and the eval harness can measure it;
# this is the starting point, not a tuned result.
LLM_RULE_BATCH_SIZE = 4


def _rule_findings(violations: list[RuleViolation], *, prefix: str) -> list[Finding]:
    """Convert platform rule violations into MLR findings.

    The seam that keeps ``platform_core.rules`` free of MLR concepts: the engine returns neutral
    violations and this is the only place that knows they become findings.
    """
    findings: list[Finding] = []
    for i, v in enumerate(violations):
        findings.append(
            Finding(
                id=f"{prefix}-{i + 1}",
                # A rule names the check it belongs to, which is how a rule finding merges with a
                # core finding about the same problem instead of double-reporting it.
                check_type=v.check_type if v.check_type in _VALID_CHECKS else "business_rule",
                claim_id=v.segment_id,
                quote=v.quote,
                char_span=v.char_span,
                issue=v.message,
                suggested_fix=v.suggested_fix,
                citations=[
                    Citation(
                        source_id=v.citation_source_id,
                        passage=v.citation_text,
                        kind=v.citation_kind,
                        title=f"{v.rule_id} v{v.rule_version}",
                        rule_id=v.rule_id,
                        rule_version=v.rule_version,
                    )
                ],
                # A rule's severity is the tenant's own decision about its own rule, so it is taken
                # as the proposal and the severity node does not re-derive it from check_type.
                severity_proposed=v.severity,
                confidence=v.confidence,
                rule_id=v.rule_id,
                rule_version=v.rule_version,
            )
        )
    return findings


_VALID_CHECKS = {
    "substantiation",
    "off_label",
    "fair_balance",
    "safety_prominence",
    "business_rule",
}


def _rule_input(state: dict[str, Any]) -> RuleInput:
    """Build the generic input a rule evaluator sees.

    Claims become ``Segment`` objects: that is the seam that lets a claim-level rule such as
    ``max_claims_without_reference`` live in ``platform_core`` without the platform knowing what a
    claim is. Only efficacy-type claims are passed, because "every efficacy claim needs a reference"
    should not fire on a dosing statement.
    """
    claims = state.get("claims") or []
    segments = [
        Segment(id=c.id, text=c.original_text, char_span=c.char_span)
        for c in claims
        if getattr(c, "claim_type", "other") in ("efficacy", "comparative", "safety")
    ]
    return RuleInput(
        text=str(state.get("document") or ""),
        doc_meta=dict(state.get("doc_meta") or {}),
        segments=segments,
        audience=str(state.get("audience") or "any"),
        channel=str(state.get("channel") or ""),
        product_id=str(state.get("product_id") or ""),
        market=str(state.get("market") or ""),
    )


def _rules_of_kind(state: dict[str, Any], kind: str) -> list[Rule]:
    """Re-read the selected rules at full fidelity.

    ``state["rules_applied"]`` holds ids and versions, not params -- it is a record, not a payload.
    Reading the exact versions back from the registry means this node applies precisely what
    ``ingest`` selected, even if a rule was edited in between.
    """
    applied = state.get("rules_applied") or []
    wanted = {(a.id, a.version) for a in applied if a.kind == kind}
    if not wanted:
        return []
    registry = RuleRegistry(str(state["tenant_id"]))
    return [registry.get(rule_id, version=version) for rule_id, version in sorted(wanted)]


def _deterministic_part(state: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    """Every applicable deterministic rule. **No model call, ever.**

    Not routed through the tool registry, unlike the validators the presentation check calls. That
    registry is for code the platform did not write -- a customer's own qualified scripts, reached
    across a JSON boundary. This is the platform's own rules engine, it returns typed violations
    carrying rule ids and versions, and serialising those through a tool call would lose exactly the
    provenance that makes a rule finding citable.

    A misconfigured rule yields a violation describing the misconfiguration rather than raising, so
    one broken rule cannot take down the review and a broken blocker still reaches a human.
    """
    rules = _rules_of_kind(state, "deterministic")
    if not rules:
        return [], {"rules": 0, "violations": 0}

    violations = deterministic.evaluate_all(rules, _rule_input(state))
    return _rule_findings(violations, prefix="dr"), {
        "rules": len(rules),
        "violations": len(violations),
        "by_rule": sorted({v.rule_id for v in violations}),
    }


def _llm_part(
    state: dict[str, Any], run: Any, *, fake: Any = None
) -> tuple[list[Finding], dict[str, Any], list[str], list[Any]]:
    """Judge the plain-English rules with a model, in batches.

    Two defences against a tenant's rule text being used as an instruction channel, beyond the
    delimiting and neutralising done in :mod:`platform_core.rules.prompting`:

    * any returned finding naming a rule that was not in the batch is **discarded**, so injected text
      cannot invent a rule or attribute a finding to one it prefers;
    * the quote still has to survive the grounding gate, so no finding can be fabricated.
    """
    rules = _rules_of_kind(state, "llm")
    if not rules:
        return [], {"rules": 0, "violations": 0}, [], []

    document = str(state.get("document") or "")
    claims = state.get("claims") or []
    findings: list[Finding] = []
    discarded = 0

    for batch_index in range(0, len(rules), LLM_RULE_BATCH_SIZE):
        batch = rules[batch_index : batch_index + LLM_RULE_BATCH_SIZE]
        allowed = {r.id: r for r in batch}
        prompt = prompting.build_rule_check_prompt(batch, content=document)
        prompt += rule_prompts.rule_response_instructions(claims)

        try:
            result = llm.call_structured(
                "fast",
                node="llm_rules",
                prompt=prompt,
                schema=rule_prompts.RuleCheckResponse,
                prompt_version=prompting.PROMPT_VERSION,
                fake=fake,
            )
        except llm.SchemaFailure as exc:
            # The batch is lost, the rest of the node is not. Before these three were one node a
            # failure here could not touch the deterministic findings; keeping that true is the
            # point of returning the error rather than raising it.
            return (
                findings,
                {"rules": len(rules), "violations": len(findings), "failed": True},
                [f"llm_rules: {exc}"],
                [rule_prompts.schema_failure_gate("llm_rules", exc)],
            )

        run.record(result)
        for i, reported in enumerate(result.parsed.findings):
            rule = allowed.get(reported.rule_id)
            if rule is None:
                # Either a hallucinated rule id or a successful injection. Either way it is not
                # something this batch was asked about, so it does not become a finding.
                discarded += 1
                continue
            findings.append(
                Finding(
                    id=f"lr-{batch_index + i + 1}",
                    check_type=(
                        rule.check_type if rule.check_type in _VALID_CHECKS else "business_rule"
                    ),
                    claim_id=reported.claim_id,
                    quote=reported.quote,
                    issue=reported.issue,
                    suggested_fix=reported.suggested_fix,
                    citations=[
                        Citation(
                            source_id=rule.label(),
                            passage=rule.description,
                            kind="rule",
                            title=rule.title,
                            rule_id=rule.id,
                            rule_version=rule.version,
                        )
                    ],
                    severity_proposed=rule.severity,
                    confidence=reported.confidence,
                    rule_id=rule.id,
                    rule_version=rule.version,
                    model=result.usage.model,
                    prompt_version=result.prompt_version,
                )
            )

    return (
        findings,
        {
            "rules": len(rules),
            "batches": (len(rules) + LLM_RULE_BATCH_SIZE - 1) // LLM_RULE_BATCH_SIZE,
            "violations": len(findings),
            "discarded_unknown_rule_id": discarded,
        },
        [],
        [],
    )


def _reference_doc_part(
    state: dict[str, Any], run: Any, rule: Rule, *, fake: Any = None
) -> tuple[list[Finding], dict[str, Any], list[str], list[Any]]:
    """Review the piece against one tenant reference document.

    One call per document, never several codes of practice in one prompt: a reviewer prompt
    carrying two unrelated documents does measurably worse at citing the right clause from either.
    The loop over documents is sequential inside this node now rather than a parallel fan-out --
    each tenant has one such document today, so the concurrency was theoretical.

    A cited clause is verified to exist in the document before it becomes a finding. A fabricated
    clause would otherwise show a reviewer an authority they cannot look up, which is worse than no
    citation at all.
    """
    docs_dir = tenant_reference_docs_dir(str(state["tenant_id"]))
    doc_path = docs_dir / str(rule.params.get("doc_path", ""))
    if not doc_path.is_file():
        return (
            [],
            {"rule_id": rule.id, "missing_document": doc_path.name},
            [f"reference_doc_review: {rule.label()} cites a missing document"],
            [],
        )

    document_text = doc_path.read_text(encoding="utf-8")
    prompt = prompting.build_reference_doc_prompt(
        rule,
        document=document_text,
        content=str(state.get("document") or ""),
        clauses=[str(c) for c in rule.params.get("clauses", []) or []],
    )
    prompt += rule_prompts.rule_response_instructions(state.get("claims") or [])

    try:
        result = llm.call_structured(
            "strong",
            node="reference_doc_review",
            prompt=prompt,
            schema=rule_prompts.RuleCheckResponse,
            prompt_version=prompting.PROMPT_VERSION,
            fake=fake,
        )
    except llm.SchemaFailure as exc:
        return (
            [],
            {"rule_id": rule.id, "failed": True},
            [f"reference_doc_review: {exc}"],
            [rule_prompts.schema_failure_gate("reference_doc_review", exc)],
        )

    run.record(result)
    flat_document = deterministic.normalize_whitespace(document_text)
    findings: list[Finding] = []
    unverifiable = 0

    for i, reported in enumerate(result.parsed.findings):
        if reported.rule_id and reported.rule_id != rule.id:
            continue
        clause = (reported.clause or "").strip()
        clause_exists = bool(clause) and (
            deterministic.normalize_whitespace(clause) in flat_document
        )
        if clause and not clause_exists:
            # The model quoted a clause the document does not contain. Reported with no clause
            # citation and low confidence rather than dropped: the underlying observation may
            # still be sound, and the grounding gate will escalate it.
            unverifiable += 1

        citation = Citation(
            source_id=f"{rule.params.get('doc_path', '')}",
            passage=clause if clause_exists else rule.description,
            kind="doc_clause" if clause_exists else "rule",
            title=f"{rule.title} ({rule.label()})",
            rule_id=rule.id,
            rule_version=rule.version,
        )
        findings.append(
            Finding(
                id=f"rd-{rule.id}-{i + 1}",
                check_type=(
                    rule.check_type if rule.check_type in _VALID_CHECKS else "business_rule"
                ),
                claim_id=reported.claim_id,
                quote=reported.quote,
                issue=reported.issue,
                suggested_fix=reported.suggested_fix,
                citations=[citation],
                severity_proposed=rule.severity,
                confidence=min(reported.confidence, 40)
                if clause and not clause_exists
                else reported.confidence,
                rule_id=rule.id,
                rule_version=rule.version,
                model=result.usage.model,
                prompt_version=result.prompt_version,
            )
        )

    return (
        findings,
        {
            "rule_id": rule.id,
            "document": doc_path.name,
            "violations": len(findings),
            "clauses_not_found_in_document": unverifiable,
        },
        [],
        [],
    )


def house_rules(state: dict[str, Any], *, fake: Any = None) -> dict[str, Any]:
    """Apply the tenant's own rules: the yes-or-no ones, the judged ones, and the documents.

    **Mixed: deterministic first, then LLM decisions.** One node rather than three, because three
    boxes in a timeline for "the company's rules" is noise to everyone except the person who wrote
    the graph. The distinction that matters to a reviewer is not which node fired but what the
    finding cites, and every one of these cites the rule or the clause by id and version.

    The order is the point. The deterministic rules run first and cost nothing -- a banned-term list
    is not a judgement call, and routing it through a model would add cost, latency and
    nondeterminism to a question regex answers exactly. Only what regex cannot answer reaches a
    model. That ordering is the repo's standing rule about when a model earns its place, made
    visible in a single step.

    Each part contributes independently: a schema failure in the judged rules escalates on its own
    and leaves the deterministic findings intact, which is what the three separate nodes gave for
    free and what this node has to preserve deliberately.
    """
    with node_run(state, "house_rules") as run:
        findings: list[Finding] = []
        errors: list[str] = []
        gates: list[Any] = []
        summary: dict[str, Any] = {}

        got, summary["deterministic"] = _deterministic_part(state)
        findings += got

        got, summary["judged"], part_errors, part_gates = _llm_part(state, run, fake=fake)
        findings += got
        errors += part_errors
        gates += part_gates

        reviewed: list[dict[str, Any]] = []
        for rule in _rules_of_kind(state, "reference_doc"):
            got, detail, part_errors, part_gates = _reference_doc_part(state, run, rule, fake=fake)
            findings += got
            reviewed.append(detail)
            errors += part_errors
            gates += part_gates
        summary["documents"] = reviewed

        summary["findings"] = len(findings)
        run.output = summary
        run.citations = [{"source_id": c.source_id} for f in findings for c in f.citations]

        out: dict[str, Any] = {"findings": findings}
        if errors:
            out["errors"] = errors
        if gates:
            out["gate_results"] = gates
        return out
