"""Tenant business rules: list, author, test, activate, retire, export.

Deterministic code. No model is called from this module -- which is also why only deterministic
rules can be tried out here; an ``llm`` or ``reference_doc`` rule is exercised by running a review.

The storage rules are the registry's, not this module's: an edit is always a new version, and
validation is enforced inside ``RuleRegistry.activate``. What this layer adds is the part that is
about the *request*: who asked, the policy version before and after, the regression check, and an
audit event for every transition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api import evals
from api.deps import known_tenant
from api.routes.samples import load_sample
from platform_core import audit, tenancy
from platform_core.policy import policy_version
from platform_core.rules import deterministic
from platform_core.rules.registry import RuleActivationError, RuleRegistry
from platform_core.rules.schema import Rule, RuleInput
from platform_core.rules.validation import ValidationReport, validate_rule

router = APIRouter(tags=["rules"])

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# --- Request bodies ------------------------------------------------------------------------------


class RuleWrite(BaseModel):
    """A rule to store. ``author`` is free text because the demo has no login."""

    model_config = ConfigDict(extra="forbid")

    rule: dict[str, Any]
    author: str = ""
    change_note: str = ""


class RuleTest(BaseModel):
    """What to try a rule against: pasted text, or a sample piece by name.

    ``rule`` lets an author test content that has not been saved yet, which is the point of a test
    button -- finding out a rule is wrong should not require storing it first.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    sample: str = ""
    doc_meta: dict[str, Any] = Field(default_factory=dict)
    version: int | None = None
    rule: dict[str, Any] | None = None


class RuleActivate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int | None = None
    actor: str = ""
    note: str = ""
    # Activate although the rule fails its own examples. The registry requires a note.
    force: bool = False
    # Activate although blocker recall dropped against the baseline. Requires a reason.
    override_regression: bool = False
    override_reason: str = ""


class RuleRetire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int | None = None
    actor: str = ""
    note: str = ""


# --- Helpers -------------------------------------------------------------------------------------


def _registry(tenant_id: str) -> RuleRegistry:
    known_tenant(tenant_id)
    return RuleRegistry(tenant_id)


def _parse_rule(tenant_id: str, raw: dict[str, Any], *, rule_id: str = "") -> Rule:
    """Validate submitted rule content, pinned to the tenant and rule named in the URL.

    A body naming a different tenant or rule is rejected rather than corrected: silently rewriting
    it would store something other than what the author sent.
    """
    declared_tenant = raw.get("tenant_id", tenant_id)
    if declared_tenant != tenant_id:
        raise HTTPException(422, f"rule declares tenant {declared_tenant!r}, not {tenant_id!r}")
    if rule_id and raw.get("id", rule_id) != rule_id:
        raise HTTPException(422, f"rule declares id {raw.get('id')!r}, not {rule_id!r}")
    payload = {**raw, "tenant_id": tenant_id}
    if rule_id:
        payload["id"] = rule_id
    return Rule.model_validate(payload)


def _validate(rule: Rule) -> ValidationReport:
    return validate_rule(rule, reference_docs_dir=tenancy.tenant_reference_docs_dir(rule.tenant_id))


def validation_payload(report: ValidationReport) -> dict[str, Any]:
    return {
        "rule_id": report.rule_id,
        "rule_version": report.rule_version,
        "errors": report.errors,
        "warnings": report.warnings,
        "can_save": report.can_save,
        "can_activate": report.can_activate,
        "accuracy": report.accuracy(),
        "summary": report.summary(),
        "examples": [e.model_dump(mode="json") for e in report.examples],
    }


def _version_or_latest(registry: RuleRegistry, rule_id: str, version: int | None) -> int:
    latest = registry.latest_version_number(rule_id)
    if not latest:
        raise HTTPException(404, f"tenant {registry.tenant_id!r} has no rule {rule_id!r}")
    if version is not None and not 1 <= version <= latest:
        raise HTTPException(404, f"{rule_id} has no version {version}")
    return version or latest


def _record(tenant_id: str, action: str, rule: Rule, *, actor: str, **output: Any) -> None:
    """Write a rule transition to the audit log.

    The registry's ``status`` column is mutable, and this event is what makes that acceptable: the
    history of who changed the policy, and what the policy version became, lives here.
    """
    audit.emit(
        run_id=f"policy:{tenant_id}",
        tenant_id=tenant_id,
        node=f"rule_{action}",
        event_type="human",
        input_value={"rule_id": rule.id, "version": rule.version},
        output={"rule_id": rule.id, "version": rule.version, "actor": actor, **output},
        policy_version=policy_version(tenant_id),
    )


def _save(registry: RuleRegistry, rule: Rule, body: RuleWrite) -> dict[str, Any]:
    """Validate, then store as a new draft version. A structurally invalid rule is not stored."""
    candidate = rule.model_copy(
        update={"version": registry.latest_version_number(rule.id) + 1, "status": "draft"}
    )
    report = _validate(candidate)
    if not report.can_save:
        raise HTTPException(
            422, {"message": report.summary(), "validation": validation_payload(report)}
        )
    stored = registry.save_new_version(
        candidate, created_by=body.author, change_note=body.change_note
    )
    _record(registry.tenant_id, "save", stored, actor=body.author, change_note=body.change_note)
    return {"rule": stored.model_dump(mode="json"), "validation": validation_payload(report)}


# --- Reads ---------------------------------------------------------------------------------------


@router.get("/tenants/{tenant_id}/rules")
def list_rules(tenant_id: str, status: str = "", kind: str = "") -> dict[str, Any]:
    """The current state of each of a tenant's rules: one entry per rule, not per version."""
    rules = _registry(tenant_id).list_rules(status=status, kind=kind)
    return {
        "tenant_id": tenant_id,
        "rules": [r.model_dump(mode="json") for r in rules],
        "total": len(rules),
    }


@router.get("/tenants/{tenant_id}/rules/{rule_id}")
def get_rule(tenant_id: str, rule_id: str, version: int | None = None) -> dict[str, Any]:
    """One rule with its version history. Defaults to the active version, else the latest."""
    registry = _registry(tenant_id)
    if version is not None:
        _version_or_latest(registry, rule_id, version)
    elif not registry.latest_version_number(rule_id):
        raise HTTPException(404, f"tenant {tenant_id!r} has no rule {rule_id!r}")
    history = [
        {k: v for k, v in entry.items() if k != "rule"} for entry in registry.history(rule_id)
    ]
    return {
        "rule": registry.get(rule_id, version=version).model_dump(mode="json"),
        "history": history,
    }


# --- Writes --------------------------------------------------------------------------------------


@router.post("/tenants/{tenant_id}/rules", status_code=201)
def create_rule(tenant_id: str, body: RuleWrite) -> dict[str, Any]:
    """Create a rule as a draft. Nothing takes effect until it is activated."""
    registry = _registry(tenant_id)
    rule = _parse_rule(tenant_id, body.rule)
    if registry.latest_version_number(rule.id):
        raise HTTPException(409, f"{rule.id} already exists; PUT to it to add a new version")
    return _save(registry, rule, body)


@router.put("/tenants/{tenant_id}/rules/{rule_id}")
def update_rule(tenant_id: str, rule_id: str, body: RuleWrite) -> dict[str, Any]:
    """Store an edit as a new draft version. The active version stays in force until this one is
    activated, so an edit can be tested without changing what reviews run under."""
    registry = _registry(tenant_id)
    _version_or_latest(registry, rule_id, None)
    return _save(registry, _parse_rule(tenant_id, body.rule, rule_id=rule_id), body)


@router.post("/tenants/{tenant_id}/rules/{rule_id}/test")
def test_rule(tenant_id: str, rule_id: str, body: RuleTest) -> dict[str, Any]:
    """Try a rule against some text. Saves nothing and changes nothing."""
    registry = _registry(tenant_id)
    if body.rule is not None:
        rule = _parse_rule(tenant_id, body.rule, rule_id=rule_id)
    else:
        rule = registry.get(rule_id, version=_version_or_latest(registry, rule_id, body.version))

    if rule.kind != "deterministic":
        raise HTTPException(
            422,
            f"{rule.label()} is a {rule.kind} rule: it needs a model, so it is exercised by "
            f"running a review rather than tested here",
        )
    if bool(body.text.strip()) == bool(body.sample):
        raise HTTPException(422, "provide exactly one of: text, or a sample name")

    data = RuleInput(text=body.text, doc_meta=body.doc_meta)
    source = "inline"
    if body.sample:
        sample = load_sample(body.sample)
        data = RuleInput(
            text=sample["text"],
            doc_meta=body.doc_meta,
            audience=sample["audience"],
            channel=sample["channel"],
            product_id=sample["product_id"],
        )
        source = f"sample:{body.sample}"

    try:
        violations = deterministic.evaluate(rule, data)
    except deterministic.RuleEvaluationError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "rule_id": rule.id,
        "rule_version": rule.version,
        "source": source,
        "violations": [v.model_dump(mode="json") for v in violations],
        "saved": False,
    }


@router.post("/tenants/{tenant_id}/rules/{rule_id}/activate")
def activate_rule(tenant_id: str, rule_id: str, body: RuleActivate) -> dict[str, Any]:
    """Put one version of a rule into force.

    Two gates, in order. The registry refuses a rule that fails its own examples. Then, because a
    rule that behaves as written can still change what gets caught, the offline golden set is
    replayed under the new policy and compared with the baseline; a drop in blocker recall rolls the
    activation back unless the caller overrides it with a reason. Both overrides are recorded.
    """
    registry = _registry(tenant_id)
    version = _version_or_latest(registry, rule_id, body.version)
    if body.override_regression and not body.override_reason.strip():
        raise HTTPException(422, "overriding the regression check requires override_reason")

    before = policy_version(tenant_id)
    previous = next((r for r in registry.all_versions(rule_id) if r.status == "active"), None)
    try:
        rule = registry.activate(
            rule_id,
            version,
            note=body.note,
            reference_docs_dir=tenancy.tenant_reference_docs_dir(tenant_id),
            force=body.force,
        )
    except RuleActivationError as exc:
        detail: dict[str, Any] = {"message": str(exc)}
        if exc.report is not None:
            detail["validation"] = validation_payload(exc.report)
        raise HTTPException(409, detail) from exc

    check = {
        **evals.regression_check(tenant_id),
        "overridden": False,
        "override_reason": "",
    }
    if check["regressed"]:
        if not body.override_regression:
            # Roll back, so a refused activation leaves the previous policy in force. A rule
            # with no earlier active version is retired rather than left live.
            if previous is not None and previous.version != version:
                registry.activate(
                    rule_id, previous.version, note="rollback: regression", force=True
                )
            elif previous is None:
                registry.retire(rule_id, version, note="rollback: regression")
            _record(tenant_id, "activate_refused", rule, actor=body.actor, regression_check=check)
            raise HTTPException(
                409,
                {
                    "message": f"activating {rule.label()} reduces blocker recall",
                    "regression_check": check,
                },
            )
        check.update(overridden=True, override_reason=body.override_reason)

    after = policy_version(tenant_id)
    _record(
        tenant_id,
        "activate",
        rule,
        actor=body.actor,
        note=body.note,
        forced=body.force,
        policy_version_before=before,
        policy_version_after=after,
        regression_check=check,
    )
    return {
        "rule_id": rule.id,
        "version": rule.version,
        "status": rule.status,
        "policy_version_before": before,
        "policy_version_after": after,
        "regression_check": check,
    }


@router.post("/tenants/{tenant_id}/rules/{rule_id}/retire")
def retire_rule(tenant_id: str, rule_id: str, body: RuleRetire) -> dict[str, Any]:
    """Stop applying a rule. It stays readable, so past decisions that cite it remain auditable."""
    registry = _registry(tenant_id)
    version = _version_or_latest(registry, rule_id, body.version)
    before = policy_version(tenant_id)
    rule = registry.retire(rule_id, version, note=body.note)
    after = policy_version(tenant_id)
    _record(
        tenant_id,
        "retire",
        rule,
        actor=body.actor,
        note=body.note,
        policy_version_before=before,
        policy_version_after=after,
    )
    return {
        "rule_id": rule.id,
        "version": rule.version,
        "status": rule.status,
        "policy_version_after": after,
    }


@router.post("/tenants/{tenant_id}/rules/export")
def export_rules(tenant_id: str) -> dict[str, Any]:
    """Write the active rule set back to the tenant's YAML files, so a change made through the API
    becomes a reviewable diff in the repository."""
    registry = _registry(tenant_id)
    written = registry.export_to_yaml(tenancy.tenant_rules_dir(tenant_id))

    def shown(path: Path) -> str:
        resolved = path.resolve()
        return (
            str(resolved.relative_to(REPO_ROOT))
            if resolved.is_relative_to(REPO_ROOT)
            else str(resolved)
        )

    return {"tenant_id": tenant_id, "written": [shown(p) for p in written]}
