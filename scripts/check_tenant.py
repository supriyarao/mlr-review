"""Say what is missing or still a stub for a tenant, before seeding hides it.

Run with ``make check-tenant ID=<tenant_id>``. Reports and fixes nothing.

Why this exists rather than letting ``make seed`` fail: most of the ways a half-built tenant goes
wrong do not fail at all. A missing study file means retrieval returns nothing, every claim comes
back unverifiable, and it reads as a model problem days later. A left-in TODO means the reviewer
is checking claims against the word "TODO". A rule stuck in draft means the client's own policy
quietly does not apply. All three seed cleanly and all three are wrong.

Three severities, and the distinction matters:

* **blocking** - seeding will fail, or the tenant cannot be reviewed at all.
* **will mislead** - it works and gives answers you should not trust. The dangerous category.
* **worth doing** - real but not urgent.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from platform_core.rules.schema import Rule
from platform_core.rules.validation import validate_rule
from platform_core.tenancy import TenantConfigError, load_tenant

REPO_ROOT = Path(__file__).resolve().parent.parent
TODO = re.compile(r"\bTODO\b")


class Report:
    def __init__(self) -> None:
        self.blocking: list[str] = []
        self.misleading: list[str] = []
        self.worth_doing: list[str] = []

    def block(self, msg: str) -> None:
        self.blocking.append(msg)

    def mislead(self, msg: str) -> None:
        self.misleading.append(msg)

    def minor(self, msg: str) -> None:
        self.worth_doing.append(msg)

    def print(self, tenant_id: str) -> int:
        groups = [
            ("BLOCKING - seeding or reviewing will fail", self.blocking),
            ("WILL MISLEAD - it works, and you should not trust the answers", self.misleading),
            ("WORTH DOING", self.worth_doing),
        ]
        for title, items in groups:
            if not items:
                continue
            print(f"\n{title}")
            for item in items:
                print(f"  - {item}")

        print()
        if self.blocking:
            print(f"{tenant_id}: not ready. Fix the blocking items, then run this again.")
            return 1
        if self.misleading:
            print(
                f"{tenant_id}: it will seed and run, but the answers would be misleading.\n"
                f"Nothing here fails loudly, which is why this check exists."
            )
            return 1
        if self.worth_doing:
            print(f"{tenant_id}: ready. {len(self.worth_doing)} thing(s) worth doing.")
            return 0
        print(f"{tenant_id}: ready.")
        return 0


def _todo_lines(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if TODO.search(line))
    except OSError:
        return 0


def check(tenant_id: str) -> int:
    r = Report()
    tenant_root = REPO_ROOT / "corpus" / "tenants" / tenant_id
    config_path = tenant_root / "tenant.yaml"

    if not config_path.is_file():
        print(f"\nNo tenant at {config_path.relative_to(REPO_ROOT)}.")
        print(f"Create one with:  make new-tenant ID={tenant_id} PRODUCT=<product_id>\n")
        return 1

    # The config has to load through the real validator, not a hand-rolled parse: if it does not
    # validate here it will not validate in the graph either.
    try:
        config = load_tenant(tenant_id)
    except TenantConfigError as exc:
        r.block(f"tenant.yaml does not validate: {exc}")
        return r.print(tenant_id)

    todos = _todo_lines(config_path)
    if todos:
        r.mislead(
            f"tenant.yaml has {todos} line(s) still marked TODO - the indication is read by the "
            f"off-label reviewer, so a TODO there means it is comparing claims against the word TODO"
        )

    # Products: the label is required, the studies are what make a claim verifiable.
    for product in config.products:
        label = (
            REPO_ROOT / "corpus" / "labels" / f"{product.product_id}-v{product.label_version}.md"
        )
        if not label.is_file():
            r.block(
                f"no label for {product.product_id} v{product.label_version} at "
                f"{label.relative_to(REPO_ROOT)} - seeding raises rather than warning, "
                f"deliberately"
            )
        else:
            count = _todo_lines(label)
            if count:
                r.mislead(
                    f"{label.relative_to(REPO_ROOT)} has {count} TODO line(s) - this file is the "
                    f"evidence every claim is checked against"
                )

        prefixes = product.reference_prefixes()
        studies = sorted(
            s
            for prefix in prefixes
            for s in (REPO_ROOT / "corpus" / "references").glob(f"{prefix}-*.md")
        )
        if not studies:
            looked = ", ".join(f"{p}-*.md" for p in prefixes)
            r.mislead(
                f"no studies found for {product.product_id} (looked for {looked} in "
                f"corpus/references/) - retrieval will return nothing and every claim will come "
                f"back unverifiable, which reads as a model problem"
            )
        else:
            stubbed = [s.name for s in studies if _todo_lines(s)]
            if stubbed:
                r.mislead(f"study stubs still have TODOs: {', '.join(stubbed)}")

    # Rules: a draft rule does not apply, silently.
    rules_dir = tenant_root / "rules"
    rule_files = sorted(rules_dir.glob("*.yaml")) if rules_dir.is_dir() else []
    not_self_testable: list[str] = []
    if not rule_files:
        r.minor("no rules yet - the tenant works, but none of its own policy applies")
    for path in rule_files:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            r.block(f"{path.relative_to(REPO_ROOT)} is not valid YAML: {exc}")
            continue
        # Checked before anything mutates raw: a rule file may omit tenant_id and inherit it from
        # the directory it sits in, but one that names a different tenant is a copy-paste error
        # that seeding refuses outright.
        declared = raw.get("tenant_id")
        if declared is not None and declared != tenant_id:
            r.block(f"{path.name} says tenant_id {declared!r}, not {tenant_id!r}")
            continue

        if _todo_lines(path):
            r.minor(f"{path.name} has TODO lines (owner, source_ref)")

        doc_path = (raw.get("params") or {}).get("doc_path")
        if raw.get("kind") == "reference_doc" and doc_path:
            target = tenant_root / "reference_docs" / str(doc_path)
            if not target.is_file():
                r.block(f"{path.name} cites a missing document: reference_docs/{doc_path}")

        if raw.get("status") != "active":
            r.mislead(
                f"{path.name} has status {raw.get('status', 'missing')!r}, so this rule does not "
                f"apply - a draft rule is indistinguishable from no rule at review time"
            )
            continue

        # Declaring `status: active` is a request, not an outcome. Seeding runs each deterministic
        # rule against its own examples and refuses activation if they do not behave as labelled,
        # leaving the rule stored as a draft that never fires. The YAML still says active, so
        # reading the file cannot tell you -- the same validation has to run here. Found by
        # scaffolding a tenant and watching its own starter rule get refused while this check
        # reported nothing at all.
        raw.setdefault("tenant_id", tenant_id)
        try:
            report = validate_rule(
                Rule.model_validate(raw), reference_docs_dir=tenant_root / "reference_docs"
            )
        except Exception as exc:
            r.block(f"{path.name} will not load: {exc}")
            continue

        if not report.can_activate:
            r.mislead(
                f"{path.name} says active but seeding will refuse it and leave it a draft "
                f"({report.summary()}) - the file keeps saying active, so only running the "
                f"rule tells you"
            )
        for warning in report.warnings:
            # "examples cannot be verified without a model call" is a permanent property of an
            # llm or reference_doc rule, not something anyone can fix, so it is counted and
            # reported once rather than repeated per file -- a checklist that always has the same
            # three un-actionable items on it stops being read.
            if "without a model call" in warning:
                not_self_testable.append(path.name)
            else:
                r.minor(f"{path.name}: {warning}")

    if not_self_testable:
        r.minor(
            f"{len(not_self_testable)} rule(s) are model-backed, so activating them checks their "
            f"structure but cannot run their examples: {', '.join(not_self_testable)}"
        )

    # Reference documents: cited clauses are verified against these, so a stub means no finding
    # can ever be grounded by a clause.
    docs_dir = tenant_root / "reference_docs"
    doc_files = sorted(docs_dir.glob("*.md")) if docs_dir.is_dir() else []
    if not doc_files:
        r.minor("no reference documents - fine unless a rule cites one")
    for path in doc_files:
        if _todo_lines(path):
            r.minor(f"{path.name} has TODO clauses - a clause must be real to be quotable")

    # Cleared claims.
    preapproved = REPO_ROOT / "corpus" / "preapproved_claims" / f"{tenant_id}.yaml"
    if preapproved.is_file():
        raw = yaml.safe_load(preapproved.read_text(encoding="utf-8")) or {}
        if not (raw.get("claims") or []):
            r.minor("no cleared claims yet - optional, and the list is empty rather than wrong")

    # Something to review. A promo piece does not name a tenant -- it names a product, and the
    # tenant owns the product. Matching on product_id is how the app itself decides which pieces to
    # offer a tenant, so checking it any other way would report something the demo does not do.
    product_ids = {p.product_id for p in config.products}
    mine: list[Path] = []
    for path in sorted((REPO_ROOT / "corpus" / "promo").glob("*.md")):
        front = yaml.safe_load(path.read_text(encoding="utf-8").split("---")[1] or "") or {}
        if str(front.get("product_id", "")) in product_ids:
            mine.append(path)
    if not mine:
        r.minor(
            f"nothing in corpus/promo/ is for {' or '.join(sorted(product_ids))} - you can still "
            f"paste text in by hand, but the samples list will be empty for this tenant"
        )
    else:
        stubbed = [p.name for p in mine if _todo_lines(p)]
        if stubbed:
            r.minor(f"promo pieces still stubs: {', '.join(stubbed)}")

    # Eval coverage: not required, but a tenant outside the golden set is outside the gate.
    golden = sorted((REPO_ROOT / "evals" / "golden").glob("*.yaml"))
    needle = f"tenant_id: {tenant_id}"
    covered = any(needle in p.read_text(encoding="utf-8") for p in golden)
    if not covered:
        r.minor(
            "no golden case covers this tenant, so the regression gate does not protect it - "
            "copy the shape of evals/golden/001-subgroup-overstatement.yaml"
        )

    return r.print(tenant_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report what a tenant is still missing.")
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    return check(args.id.strip().lower())


if __name__ == "__main__":
    raise SystemExit(main())
