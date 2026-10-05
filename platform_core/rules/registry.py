"""Rule storage: versioned, tenant-scoped, and never overwritten.

Why SQLite rather than YAML alone: rules are edited through a web form by people who do not use git,
and every edit has to be retained. YAML files remain the *shipped* state -- they are what a clone of
the repo gets, and what a change is reviewed as -- but the live registry needs row-level versioning
and queries, so rules are seeded from YAML into SQLite and exported back when someone wants to commit
a change.

Deterministic code. No model involved.

The versioning contract:

* One row per ``(tenant_id, rule_id, version)``. An edit writes a **new version**; content is never
  modified in place. This mirrors the audit log's philosophy -- if you can silently change the rule
  that produced last month's decision, that decision is no longer defensible.
* ``status`` is the one mutable column, because activation is a state change rather than a content
  change. Activating version N retires whichever version of that rule was previously active.
* Every transition is written to the audit log by the caller, which is what makes the mutable column
  acceptable: the history lives there rather than in the table.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlmodel import Field, Session, SQLModel, select

from platform_core.audit import get_engine, session_scope
from platform_core.rules.schema import Rule, RuleScope


class RuleStorageError(Exception):
    """Raised on a versioning or tenant-scoping violation."""


class RuleActivationError(Exception):
    """Raised when a rule fails validation and activation was not explicitly overridden.

    Carries the validation report so a caller -- the API, the UI, or ``make seed`` -- can show the
    author exactly which examples disagree with the rule rather than just refusing.
    """

    def __init__(self, message: str, report: Any = None) -> None:
        super().__init__(message)
        self.report = report


class RuleRow(SQLModel, table=True):
    """One version of one rule.

    Content is stored as JSON rather than as columns: a rule's ``params`` shape depends on its kind,
    and modelling five param shapes as nullable columns would be worse than storing the document and
    validating it with Pydantic on the way out.
    """

    __tablename__ = "rules"

    # Composite identity. A rule id is unique within a tenant, not globally, so tenant_id is part of
    # the key -- two tenants may both have a rule called BAN-001 without colliding.
    tenant_id: str = Field(primary_key=True)
    rule_id: str = Field(primary_key=True)
    version: int = Field(primary_key=True)

    # The one mutable column. See the module docstring.
    status: str = Field(index=True)

    payload_json: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Who made this version. Free text, since this demo has no auth; a real deployment would carry
    # an authenticated identity here.
    created_by: str = ""
    # Why. Required for an override, so a decision to activate despite a regression is on record.
    change_note: str = ""

    def to_rule(self) -> Rule:
        data = json.loads(self.payload_json)
        data["status"] = self.status
        return Rule.model_validate(data)


def _ensure_tables() -> None:
    """Create the rules table if it does not exist. Shares the engine with the audit log."""
    SQLModel.metadata.create_all(get_engine())


def _dump(rule: Rule) -> str:
    return json.dumps(rule.model_dump(mode="json"), sort_keys=True, default=str)


class RuleRegistry:
    """Read and write rules for one tenant.

    Bound to a tenant for its lifetime, for the same reason
    :class:`platform_core.retrieval.TenantRetriever` is: a caller cannot forget to scope a query, and
    an instance cannot be reused across tenants by accident.
    """

    def __init__(self, tenant_id: str) -> None:
        if not tenant_id or not str(tenant_id).strip():
            raise RuleStorageError("RuleRegistry requires a non-empty tenant_id")
        self.tenant_id = tenant_id
        _ensure_tables()

    # --- reads ----------------------------------------------------------------------------------

    def _rows(self, session: Session) -> Sequence[RuleRow]:
        return session.exec(
            select(RuleRow)
            .where(RuleRow.tenant_id == self.tenant_id)
            .order_by(RuleRow.rule_id, RuleRow.version)
        ).all()

    def all_versions(self, rule_id: str = "") -> list[Rule]:
        """Every stored version, oldest first. Optionally for one rule."""
        with session_scope() as session:
            rows = [r for r in self._rows(session) if not rule_id or r.rule_id == rule_id]
            return [r.to_rule() for r in rows]

    def history(self, rule_id: str) -> list[dict[str, Any]]:
        """Version history for the UI's diff view."""
        with session_scope() as session:
            rows = [r for r in self._rows(session) if r.rule_id == rule_id]
            return [
                {
                    "version": r.version,
                    "status": r.status,
                    "created_at": r.created_at.isoformat(),
                    "created_by": r.created_by,
                    "change_note": r.change_note,
                    "rule": r.to_rule().model_dump(mode="json"),
                }
                for r in rows
            ]

    def get(self, rule_id: str, version: int | None = None) -> Rule:
        """One rule: a specific version, or the active one, or else the latest."""
        versions = self.all_versions(rule_id)
        if not versions:
            raise RuleStorageError(f"tenant {self.tenant_id!r} has no rule {rule_id!r}")
        if version is not None:
            for rule in versions:
                if rule.version == version:
                    return rule
            raise RuleStorageError(f"{rule_id} has no version {version}")
        for rule in versions:
            if rule.status == "active":
                return rule
        return versions[-1]

    def latest_version_number(self, rule_id: str) -> int:
        versions = self.all_versions(rule_id)
        return max((r.version for r in versions), default=0)

    def list_rules(self, *, status: str = "", kind: str = "") -> list[Rule]:
        """The current state of each rule -- one entry per rule id, not per version."""
        by_id: dict[str, Rule] = {}
        for rule in self.all_versions():
            current = by_id.get(rule.id)
            # Prefer the active version; otherwise the highest version number.
            if (
                current is None
                or rule.status == "active"
                or (current.status != "active" and rule.version > current.version)
            ):
                by_id[rule.id] = rule
        out = [r for r in by_id.values() if (not status or r.status == status)]
        if kind:
            out = [r for r in out if r.kind == kind]
        return sorted(out, key=lambda r: r.id)

    def active_rules(self) -> list[Rule]:
        return self.list_rules(status="active")

    def applicable_rules(
        self,
        *,
        audience: str = "any",
        channel: str = "",
        product_id: str = "",
        market: str = "",
        on: date | None = None,
    ) -> list[Rule]:
        """The rules that apply to one submission.

        Four filters, all of which must pass: the rule is active, it is inside its effective window
        on the review date, and its scope covers this piece. Evaluated per run rather than cached,
        because the effective window means the answer legitimately changes with the date.
        """
        when = on or date.today()
        return [
            rule
            for rule in self.active_rules()
            if rule.is_effective_on(when)
            and rule.scope.applies_to(
                audience=audience, channel=channel, product_id=product_id, market=market
            )
        ]

    # --- writes ---------------------------------------------------------------------------------

    def save_new_version(self, rule: Rule, *, created_by: str = "", change_note: str = "") -> Rule:
        """Store a rule as a new version. Never overwrites an existing one.

        The version number is assigned here rather than taken from the caller, so a client cannot
        overwrite history by resubmitting a version number that already exists.
        """
        if rule.tenant_id != self.tenant_id:
            raise RuleStorageError(
                f"refusing to store a rule for {rule.tenant_id!r} in tenant {self.tenant_id!r}"
            )

        next_version = self.latest_version_number(rule.id) + 1
        stored = rule.model_copy(update={"version": next_version, "status": "draft"})

        with session_scope() as session:
            session.add(
                RuleRow(
                    tenant_id=self.tenant_id,
                    rule_id=stored.id,
                    version=next_version,
                    status="draft",
                    payload_json=_dump(stored),
                    created_by=created_by,
                    change_note=change_note,
                )
            )
        return stored

    def _set_status(self, rule_id: str, version: int, status: str, *, note: str = "") -> Rule:
        with session_scope() as session:
            row = session.exec(
                select(RuleRow).where(
                    RuleRow.tenant_id == self.tenant_id,
                    RuleRow.rule_id == rule_id,
                    RuleRow.version == version,
                )
            ).first()
            if row is None:
                raise RuleStorageError(f"{rule_id} v{version} not found for {self.tenant_id}")
            row.status = status
            if note:
                row.change_note = note
            session.add(row)
            session.flush()
            return row.to_rule()

    def activate(
        self,
        rule_id: str,
        version: int,
        *,
        note: str = "",
        reference_docs_dir: Path | None = None,
        force: bool = False,
    ) -> Rule:
        """Make one version the active one, retiring whichever version was active before.

        Exactly one version of a rule can be active at a time; otherwise "which rule produced this
        finding" would have more than one answer.

        **Validation is enforced here, not in the callers.** Activation is the moment a rule starts
        affecting real reviews, and it is reachable from seeding, the API and the UI. Putting the bar
        in each of those paths means it only holds in the ones someone remembered -- which is exactly
        what happened: ``seed_from_yaml`` activated every rule file without validating, so a blocker
        whose own examples contradicted it went live and ``make seed`` printed success.

        ``force`` exists because a rule can legitimately need to go live against the checker's
        advice. It requires a ``note``, so the override is recorded rather than silent.
        """
        from platform_core.rules.validation import validate_rule

        rule = self.get(rule_id, version=version)
        report = validate_rule(rule, reference_docs_dir=reference_docs_dir)

        if not report.can_activate:
            if not force:
                raise RuleActivationError(
                    f"{rule.label()} cannot be activated: {report.summary()}", report
                )
            if not note.strip():
                # An override with no reason is indistinguishable from a mistake six months later.
                raise RuleActivationError(
                    f"{rule.label()} fails validation; overriding requires a note explaining why",
                    report,
                )

        for existing in self.all_versions(rule_id):
            if existing.status == "active" and existing.version != version:
                self._set_status(rule_id, existing.version, "retired")
        return self._set_status(rule_id, version, "active", note=note)

    def retire(self, rule_id: str, version: int, *, note: str = "") -> Rule:
        """Stop applying a rule, keeping it readable for audits of past decisions."""
        return self._set_status(rule_id, version, "retired", note=note)

    # --- provenance -----------------------------------------------------------------------------

    def active_set_hash(self) -> str:
        """Content hash of this tenant's active rule set.

        Feeds ``policy_version`` in audit events. Hashes each active rule's full content, not just
        its id and version, so an edit that somehow reused a version number still changes the hash.
        """
        active = self.active_rules()
        canonical = json.dumps(
            [[r.id, r.version, _dump(r)] for r in sorted(active, key=lambda r: r.id)],
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def applied_set(self, rules: Iterable[Rule]) -> list[dict[str, Any]]:
        """Compact record of which rules a run applied, for state and the audit log."""
        return [
            {"id": r.id, "version": r.version, "kind": r.kind, "severity": r.severity}
            for r in sorted(rules, key=lambda r: r.id)
        ]

    # --- YAML seed and export -------------------------------------------------------------------

    def seed_from_yaml(
        self,
        rules_dir: Path,
        *,
        activate: bool = True,
        reference_docs_dir: Path | None = None,
    ) -> list[Rule]:
        """Load rule YAML files into the registry.

        Idempotent in the sense that matters: re-seeding identical content does not create a new
        version, because a developer running ``make seed`` twice should not inflate the version
        history. Changed content does create one.

        A rule that fails validation is still **stored** -- losing authored work would be worse --
        but is left as a draft, and :attr:`rejected` records why. Seeding must not be able to put a
        rule into force that the checker would refuse.
        """
        self.rejected: list[tuple[str, Any]] = []
        if not rules_dir.is_dir():
            return []

        loaded: list[Rule] = []
        for path in sorted(rules_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise RuleStorageError(f"{path} must contain a YAML mapping")
            raw.setdefault("tenant_id", self.tenant_id)
            rule = Rule.model_validate(raw)

            if rule.tenant_id != self.tenant_id:
                raise RuleStorageError(
                    f"{path} declares tenant {rule.tenant_id!r} but sits in {self.tenant_id!r}"
                )

            existing_versions = self.all_versions(rule.id)
            unchanged = next(
                (
                    v
                    for v in existing_versions
                    if _dump(v.model_copy(update={"version": 0, "status": "draft"}))
                    == _dump(rule.model_copy(update={"version": 0, "status": "draft"}))
                ),
                None,
            )
            if unchanged is not None:
                if activate and unchanged.status != "active":
                    self._try_activate(unchanged.id, unchanged.version, path, reference_docs_dir)
                loaded.append(self.get(rule.id))
                continue

            stored = self.save_new_version(
                rule, created_by="seed", change_note=f"seeded from {path.name}"
            )
            if activate:
                stored = (
                    self._try_activate(stored.id, stored.version, path, reference_docs_dir)
                    or stored
                )
            loaded.append(stored)
        return loaded

    def _try_activate(
        self, rule_id: str, version: int, path: Path, reference_docs_dir: Path | None
    ) -> Rule | None:
        """Activate a seeded rule, recording a refusal instead of raising.

        One invalid rule file must not abort the whole seed: the other rules are still worth
        loading, and the operator needs to see every problem at once rather than one per run.
        """
        try:
            return self.activate(
                rule_id,
                version,
                note=f"seeded from {path.name}",
                reference_docs_dir=reference_docs_dir,
            )
        except RuleActivationError as exc:
            self.rejected.append((f"{rule_id} v{version} ({path.name})", exc.report))
            return None

    def export_to_yaml(self, rules_dir: Path) -> list[Path]:
        """Write the active rule set back to YAML so a UI change can be committed.

        This is what keeps the repo the source of truth for the *shipped* rules even though the UI
        writes to SQLite: a compliance owner edits in the UI, exports, and the diff is reviewable.
        """
        rules_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for rule in self.active_rules():
            data = rule.model_dump(mode="json")
            # Version and status are properties of the registry, not of the shipped rule content.
            data.pop("version", None)
            data.pop("status", None)
            path = rules_dir / f"{rule.id.lower().replace('_', '-')}.yaml"
            path.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
            written.append(path)
        return written


def scope_from_dict(data: dict[str, Any]) -> RuleScope:
    """Build a scope from loose input, for the API's create/edit endpoints."""
    return RuleScope.model_validate(data or {})
