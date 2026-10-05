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
from platform_core.retrieval import Document, index_documents, reset_tenant_collection
from platform_core.rules.registry import RuleRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = REPO_ROOT / "corpus" / "labels"
REFERENCES_DIR = REPO_ROOT / "corpus" / "references"
PREAPPROVED_DIR = REPO_ROOT / "corpus" / "preapproved_claims"

# Which documents belong to which product. Taken from the filename so adding a label or a study is
# a matter of dropping a file in, not editing this script.
_LABEL_NAME = re.compile(r"^(?P<product>[a-z0-9]+)-v(?P<version>[\d.]+)$")
_STUDY_PREFIX = {"zel": "zelvoran", "nex": "nexapril"}

# Chunking: split on markdown headings, so a retrieved passage is a section a human would
# recognise -- "Indications and Usage", "ZEL-302 endpoints" -- rather than an arbitrary window.
# Trade-off: heading-based chunks vary in size and a very long section stays whole, which costs
# retrieval precision on big documents. For a corpus of label extracts and study summaries that is
# the right trade: a citation that lands mid-sentence is worse than one that returns extra context,
# because the reviewer has to be able to read the passage and judge it.
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def _chunks(text: str, *, max_chars: int = 2400) -> list[tuple[str, str, tuple[int, int]]]:
    """Split a markdown document into (heading, body, char_span) chunks.

    A section longer than ``max_chars`` is split on paragraph boundaries, because an oversized chunk
    both dilutes its embedding and makes a citation hard to check by eye.
    """
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text.strip(), (0, len(text)))]

    out: list[tuple[str, str, tuple[int, int]]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        body = text[start:end]

        if len(body) <= max_chars:
            out.append((heading, body.strip(), (start, end)))
            continue

        offset = start
        for para in re.split(r"\n\s*\n", body):
            if para.strip():
                out.append((heading, para.strip(), (offset, offset + len(para))))
            offset += len(para) + 2
    return out


def _label_documents(tenant_id: str, product_id: str, label_version: str) -> list[Document]:
    path = LABELS_DIR / f"{product_id}-v{label_version}.md"
    if not path.is_file():
        print(f"  ! no label extract at {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return []

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


def _reference_documents(tenant_id: str, product_id: str, label_version: str) -> list[Document]:
    """Study summaries for a product, matched by filename prefix."""
    prefixes = [p for p, product in _STUDY_PREFIX.items() if product == product_id]
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
        documents.extend(_reference_documents(tenant_id, product.product_id, product.label_version))
    documents.extend(_preapproved_documents(tenant_id))

    index_documents(tenant_id, documents)

    counts: dict[str, int] = {}
    for doc in documents:
        counts[doc.doc_type] = counts.get(doc.doc_type, 0) + 1
    return counts


def seed_rules(tenant_id: str, *, activate: bool = True) -> list[str]:
    """Load one tenant's business rules from YAML into the registry."""
    rules = RuleRegistry(tenant_id).seed_from_yaml(
        tenancy.tenant_rules_dir(tenant_id), activate=activate
    )
    return [f"{r.id} v{r.version} ({r.status})" for r in rules]


def main() -> int:
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

    for tenant_id in tenant_ids:
        config = tenancy.load_tenant(tenant_id)
        print(f"\n{config.name} ({tenant_id})  config v{config.version}")

        if not args.rules_only:
            counts = seed_evidence(tenant_id)
            total = sum(counts.values())
            detail = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
            print(f"  evidence: {total} passages indexed ({detail})")

        rules = seed_rules(tenant_id, activate=not args.draft)
        print(f"  rules:    {len(rules)} loaded")
        for line in rules:
            print(f"            {line}")

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
