"""Seed the corpus: index evidence per tenant, and load each tenant's business rules.

Run with ``make seed``. Idempotent: re-running rebuilds the vector store from scratch and re-seeds
rules without inflating their version history.

Why evidence is indexed **per tenant** rather than once: each tenant gets its own physical
collection, which is the outermost of the three tenant-isolation mechanisms in
:mod:`platform_core.retrieval`. A co-promoted product therefore appears in both tenants'
collections, as two separately held copies -- which is how affiliate structures actually work, and
which keeps the isolation guarantee structural rather than a filter someone has to remember.

Deterministic code. No model calls: embeddings come from Chroma's bundled local model, so seeding
works with no API key.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from platform_core import tenancy
from platform_core.env import load_env
from platform_core.retrieval import Document, index_documents, reset_tenant_collection
from platform_core.rules.registry import RuleRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = REPO_ROOT / "corpus" / "labels"
REFERENCES_DIR = REPO_ROOT / "corpus" / "references"
PREAPPROVED_DIR = REPO_ROOT / "corpus" / "preapproved_claims"

# Which documents belong to which product. Both come from the data rather than from this file, so
# adding a product is a matter of dropping files in and naming them in the tenant config.
_LABEL_NAME = re.compile(r"^(?P<product>[a-z0-9]+)-v(?P<version>[\d.]+)$")

# Chunking: split on markdown headings, so a retrieved passage is a section a human would
# recognise -- "Indications and Usage", "ZEL-302 endpoints" -- rather than an arbitrary window.
# Trade-off: heading-based chunks vary in size and a very long section stays whole, which costs
# retrieval precision on big documents. For a corpus of label extracts and study summaries that is
# the right trade: a citation that lands mid-sentence is worse than one that returns extra context,
# because the reviewer has to be able to read the passage and judge it.
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)

# A paragraph is a run of non-blank lines. Matched rather than split, so each chunk carries its true
# offset in the document instead of one accumulated from assumed separator lengths.
_PARAGRAPH = re.compile(r"[^\n]+(?:\n[^\n]+)*")


def _chunks(text: str, *, max_chars: int = 2400) -> list[tuple[str, str, tuple[int, int]]]:
    """Split a markdown document into (heading, body, char_span) chunks.

    ``char_span`` is the chunk's exact offset in the source document, which is what lets the UI
    highlight a cited passage inside the original rather than showing it as a floating quote. So the
    spans are derived from real match positions throughout -- an earlier version advanced an offset by
    ``len(paragraph) + 2``, assuming the separator was always exactly one blank line, and the spans
    drifted further out of alignment with every paragraph.

    A section longer than ``max_chars`` is split on paragraph boundaries, because an oversized chunk
    both dilutes its embedding and makes a citation hard to check by eye.
    """
    out: list[tuple[str, str, tuple[int, int]]] = []
    matches = list(_HEADING.finditer(text))

    if not matches:
        return [("", text.strip(), (0, len(text)))]

    # Text before the first heading is still content. Dropping it silently would mean a document
    # with a preamble -- a disclaimer above the title, say -- had that text missing from retrieval
    # with nothing reporting it.
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        out.append(("", text[: matches[0].start()].strip(), (0, matches[0].start())))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        body = text[start:end]

        if len(body) <= max_chars:
            out.append((heading, body.strip(), (start, end)))
            continue

        # Paragraph spans come from finditer on the section itself, so every offset is real.
        for para in _PARAGRAPH.finditer(body):
            chunk = para.group(0).strip()
            if chunk:
                out.append((heading, chunk, (start + para.start(), start + para.end())))
    return out


def _label_documents(tenant_id: str, product_id: str, label_version: str) -> list[Document]:
    path = LABELS_DIR / f"{product_id}-v{label_version}.md"
    if not path.is_file():
        # Raised rather than warned. A tenant seeded with no label evidence does not fail visibly --
        # it fails as "cannot verify" on every single claim, days later, looking like a model problem.
        raise FileNotFoundError(
            f"no label extract for {product_id} v{label_version} at "
            f"{path.relative_to(REPO_ROOT)}; check the tenant's label_version"
        )

    text = path.read_text(encoding="utf-8")
    return [
        Document(
            source_id=f"{tenant_id}:label:{product_id}:v{label_version}:{i}",
            text=body,
            tenant_id=tenant_id,
            product_id=product_id,
            label_version=label_version,
            doc_type="label",
            title=f"{product_id.title()} label v{label_version} — {heading}"
            if heading
            else "label",
            char_span=span,
        )
        for i, (heading, body, span) in enumerate(_chunks(text))
    ]


def _reference_documents(
    tenant_id: str, product: tenancy.Product, label_version: str
) -> list[Document]:
    """Study summaries for a product, matched by the filename prefix the product declares."""
    product_id = product.product_id
    prefixes = [product.studies_prefix()]
    docs: list[Document] = []
    for path in sorted(REFERENCES_DIR.glob("*.md")):
        if not any(path.stem.startswith(p) for p in prefixes):
            continue
        text = path.read_text(encoding="utf-8")
        for i, (heading, body, span) in enumerate(_chunks(text)):
            docs.append(
                Document(
                    source_id=f"{tenant_id}:reference:{path.stem}:{i}",
                    text=body,
                    tenant_id=tenant_id,
                    product_id=product_id,
                    label_version=label_version,
                    doc_type="reference",
                    title=f"{path.stem.upper()} — {heading}" if heading else path.stem.upper(),
                    char_span=span,
                )
            )
    return docs


def _preapproved_documents(tenant_id: str) -> list[Document]:
    """Pre-approved claims, indexed so the filter node can match a submitted claim against them."""
    path = PREAPPROVED_DIR / f"{tenant_id}.yaml"
    if not path.is_file():
        return []

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    default_product = str(raw.get("product_id", ""))
    default_version = str(raw.get("label_version", ""))

    docs: list[Document] = []
    for claim in raw.get("claims", []) or []:
        text = str(claim.get("text", "")).strip()
        if not text:
            continue
        docs.append(
            Document(
                source_id=f"{tenant_id}:preapproved:{claim['id']}",
                text=text,
                tenant_id=tenant_id,
                product_id=str(claim.get("product_id", default_product)),
                label_version=str(claim.get("label_version", default_version)),
                doc_type="preapproved_claim",
                title=f"Pre-approved claim {claim['id']}",
                char_span=(0, len(text)),
            )
        )
    return docs


def seed_evidence(tenant_id: str) -> dict[str, int]:
    """Rebuild one tenant's evidence collection from the corpus."""
    config = tenancy.load_tenant(tenant_id)
    reset_tenant_collection(tenant_id)

    documents: list[Document] = []
    for product in config.products:
        documents.extend(_label_documents(tenant_id, product.product_id, product.label_version))
        documents.extend(_reference_documents(tenant_id, product, product.label_version))
    documents.extend(_preapproved_documents(tenant_id))

    index_documents(tenant_id, documents)

    counts: dict[str, int] = {}
    for doc in documents:
        counts[doc.doc_type] = counts.get(doc.doc_type, 0) + 1
    return counts


def seed_rules(tenant_id: str, *, activate: bool = True) -> tuple[list[str], list[str]]:
    """Load one tenant's business rules from YAML into the registry.

    Returns the loaded rules and any that were refused activation. A refused rule is stored as a
    draft rather than discarded, so the author's work survives, but it does not take effect.
    """
    registry = RuleRegistry(tenant_id)
    rules = registry.seed_from_yaml(
        tenancy.tenant_rules_dir(tenant_id),
        activate=activate,
        reference_docs_dir=tenancy.tenant_reference_docs_dir(tenant_id),
    )
    refused = [
        f"{label}: {report.summary() if report else 'validation failed'}"
        for label, report in getattr(registry, "rejected", [])
    ]
    return [f"{r.id} v{r.version} ({r.status})" for r in rules], refused


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser(description="Seed the synthetic corpus and tenant rules.")
    parser.add_argument("--tenant", default="", help="seed only this tenant")
    parser.add_argument(
        "--rules-only", action="store_true", help="skip the vector store, seed rules only"
    )
    parser.add_argument(
        "--draft", action="store_true", help="load rules as drafts instead of activating them"
    )
    args = parser.parse_args()

    tenant_ids = [args.tenant] if args.tenant else tenancy.tenant_ids()
    if not tenant_ids:
        print("No tenants found under corpus/tenants/.", file=sys.stderr)
        return 1

    # Collected rather than raised immediately, so one bad tenant does not hide problems in another.
    problems: list[str] = []

    for tenant_id in tenant_ids:
        config = tenancy.load_tenant(tenant_id)
        print(f"\n{config.name} ({tenant_id})  config v{config.version}")

        if not args.rules_only:
            counts = seed_evidence(tenant_id)
            total = sum(counts.values())
            detail = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
            print(f"  evidence: {total} passages indexed ({detail})")

        rules, refused = seed_rules(tenant_id, activate=not args.draft)
        print(f"  rules:    {len(rules)} loaded")
        for line in rules:
            print(f"            {line}")
        if refused:
            problems.append(f"{tenant_id}: {len(refused)} rule(s) refused activation")
            print(f"  REFUSED:  {len(refused)} rule(s) left as drafts and NOT in force")
            for line in refused:
                print(f"            {line}", file=sys.stderr)

    # Printed last because it is the value an auditor will be asked about: it identifies the
    # complete policy -- config plus rules -- that any run after this seed will have applied.
    from platform_core.policy import policy_parts

    print("\nPolicy in force:")
    for tenant_id in tenant_ids:
        parts = policy_parts(tenant_id)
        print(
            f"  {tenant_id:14} policy_version={parts['policy_version']}  "
            f"config={parts['config_hash']}  rules={parts['rules_hash']}  "
            f"active_rules={parts['active_rule_count']}"
        )

    if problems:
        print("\nSeeding completed with problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
