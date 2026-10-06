"""Tenant-scoped hybrid retrieval.

Why this exists: in a multi-tenant review platform, retrieval is the boundary that matters most.
If one client's evidence can be used to justify a statement about another client's product, the
system has produced a confidential-data leak *and* a false approval in the same step. So tenant
scoping here is structural rather than a filter the caller is trusted to remember:

* each tenant gets its own physical vector-store collection, ``{tenant_id}__evidence``;
* the retriever cannot be constructed without a ``tenant_id``;
* the tenant filter is applied inside the query, before ranking, not to the results afterwards;
* every returned passage carries its tenant, and a final assertion drops anything foreign --
  belt and braces, because the cost of being wrong here is much higher than one extra loop.

Deterministic code. No model decisions; the embedding model is a fixed local one.

Hybrid search: BM25 for exact terminology (endpoint names, dose strings, "PASI-75" -- where
lexical overlap is the signal) fused with dense vectors for paraphrase ("works better than" vs
"superiority"). Reciprocal rank fusion combines them without needing the two score scales to be
comparable, which they are not.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[a-z0-9][a-z0-9\-/.]*")


class RetrievalError(Exception):
    """Raised on a tenant-scoping violation or a missing collection."""


class Passage(BaseModel):
    """One retrieved chunk of source evidence.

    ``char_span`` is the chunk's offset inside its source document, which is what lets the UI
    show a citation highlighted in the original rather than as a floating quote.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    text: str
    tenant_id: str
    product_id: str = ""
    label_version: str = ""
    # "label" | "reference" | "preapproved_claim" -- the use case decides what these mean.
    doc_type: str = ""
    title: str = ""
    char_span: tuple[int, int] = (0, 0)
    score: float = 0.0
    # Which retrievers surfaced this passage. Useful in the UI and when debugging why an obvious
    # passage was missed: "bm25 only" and "vector only" are different failure modes.
    retrievers: list[str] = Field(default_factory=list)


@dataclass
class Document:
    """A source document to index, pre-split into chunks by the caller."""

    source_id: str
    text: str
    tenant_id: str
    product_id: str = ""
    label_version: str = ""
    doc_type: str = ""
    title: str = ""
    char_span: tuple[int, int] = (0, 0)
    extra: dict[str, Any] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping clinical tokens like ``pasi-75`` and ``12.5mg`` intact."""
    return _WORD.findall(text.lower())


def chroma_path() -> Path:
    path = Path(os.environ.get("MLR_DATA_DIR", "data")) / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return path


def collection_name(tenant_id: str) -> str:
    """Physical collection for a tenant. One collection per tenant, never a shared one."""
    if not tenant_id:
        raise RetrievalError("tenant_id is required to name a collection")
    return f"{tenant_id}__evidence"


def _client() -> Any:
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_path()))


def reset_tenant_collection(tenant_id: str) -> None:
    """Drop and recreate a tenant's collection. Used by the seeding script."""
    client = _client()
    name = collection_name(tenant_id)
    # Chroma raises a provider-specific error when the collection does not exist; nothing to
    # delete is the expected state on a first seed.
    with contextlib.suppress(Exception):
        client.delete_collection(name)
    client.get_or_create_collection(name)


def index_documents(tenant_id: str, documents: list[Document]) -> int:
    """Index documents into one tenant's collection.

    Every document is tagged with its tenant twice over -- the collection it lives in, and a
    ``tenant_id`` metadata field -- so the filter at query time has something to assert against
    even if a document somehow reached the wrong collection.
    """
    if not tenant_id:
        raise RetrievalError("tenant_id is required to index documents")

    foreign = sorted({d.tenant_id for d in documents if d.tenant_id != tenant_id})
    if foreign:
        raise RetrievalError(
            f"refusing to index documents belonging to {foreign} into tenant {tenant_id!r}"
        )
    if not documents:
        return 0

    collection = _client().get_or_create_collection(collection_name(tenant_id))
    collection.upsert(
        ids=[d.source_id for d in documents],
        documents=[d.text for d in documents],
        metadatas=[
            {
                "tenant_id": d.tenant_id,
                "product_id": d.product_id,
                "label_version": d.label_version,
                "doc_type": d.doc_type,
                "title": d.title,
                "char_start": d.char_span[0],
                "char_end": d.char_span[1],
            }
            for d in documents
        ],
    )
    return len(documents)


def _rrf(rankings: dict[str, list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion.

    Score an item by the sum of ``1/(k + rank)`` across the rankings that contain it. Uses rank,
    not score, so BM25's unbounded scores and cosine distances never have to be put on a common
    scale -- the usual source of silent weighting bugs in hybrid search. ``k=60`` is the value
    from the original paper and is not tuned here; tuning it would need a retrieval-specific eval
    set, which is noted in the README as a next step.
    """
    scores: dict[str, float] = {}
    for ids in rankings.values():
        for rank, source_id in enumerate(ids, start=1):
            scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (k + rank)
    return scores


class TenantRetriever:
    """Hybrid retriever bound to exactly one tenant for its whole lifetime.

    The tenant is a constructor argument rather than a query argument on purpose: a caller cannot
    forget to pass it, and a retriever instance can never be reused across tenants by accident.
    """

    def __init__(self, tenant_id: str) -> None:
        if not tenant_id or not str(tenant_id).strip():
            # The most important line in this module. A missing tenant must be a loud failure,
            # never an unfiltered query across every tenant's evidence.
            raise RetrievalError("TenantRetriever requires a non-empty tenant_id")
        self.tenant_id = tenant_id

    def _collection(self) -> Any:
        return _client().get_or_create_collection(collection_name(self.tenant_id))

    def _where(
        self,
        product_id: str = "",
        label_version: str = "",
        doc_type: str = "",
        exclude_doc_types: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Metadata filter applied inside the query, before ranking.

        Filtering after ranking would mean the top-k was chosen from the wrong candidate pool: a
        tenant with more data could crowd out the correct passages before the filter ever ran.

        ``doc_type`` narrows to one kind of source. The pre-approved claims filter needs it: asking
        "does this claim match an already-cleared claim?" against a pool that also contains the label
        and the studies would match the underlying evidence instead, and wrongly report the claim as
        pre-approved.

        ``exclude_doc_types`` is the inverse, and exists for the opposite mistake: a caller asking
        "does the evidence support this claim?" must not be shown the cleared-claim library, or it
        can answer yes because similar wording was approved once before. That is circular, and the
        caller cannot defend against it after the fact -- a returned passage carries its doc_type,
        but by then it has already taken a slot from real evidence.
        """
        clauses: list[dict[str, Any]] = [{"tenant_id": {"$eq": self.tenant_id}}]
        if product_id:
            clauses.append({"product_id": {"$eq": product_id}})
        if label_version:
            clauses.append({"label_version": {"$eq": label_version}})
        if doc_type:
            clauses.append({"doc_type": {"$eq": doc_type}})
        if exclude_doc_types:
            clauses.append({"doc_type": {"$nin": sorted(set(exclude_doc_types))}})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _passage(self, source_id: str, text: str, meta: dict[str, Any]) -> Passage:
        return Passage(
            source_id=source_id,
            text=text,
            tenant_id=str(meta.get("tenant_id", "")),
            product_id=str(meta.get("product_id", "")),
            label_version=str(meta.get("label_version", "")),
            doc_type=str(meta.get("doc_type", "")),
            title=str(meta.get("title", "")),
            char_span=(int(meta.get("char_start", 0)), int(meta.get("char_end", 0))),
        )

    def _candidates(
        self,
        product_id: str,
        label_version: str,
        doc_type: str = "",
        exclude_doc_types: Sequence[str] = (),
    ) -> list[Passage]:
        """Every passage in scope, for BM25 to rank.

        Trade-off: BM25 is built over the filtered candidate set on each query rather than kept as
        a persistent index. For a corpus of this size that costs microseconds and keeps the index
        from going stale after a reseed. A production corpus would need a persistent lexical index
        (Elasticsearch, or SQLite FTS5) instead.
        """
        got = self._collection().get(
            where=self._where(product_id, label_version, doc_type, exclude_doc_types),
            include=["documents", "metadatas"],
        )
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        return [
            self._passage(i, d or "", m or {}) for i, d, m in zip(ids, docs, metas, strict=False)
        ]

    def search(
        self,
        query: str,
        *,
        product_id: str = "",
        label_version: str = "",
        doc_type: str = "",
        exclude_doc_types: Sequence[str] = (),
        k: int = 6,
    ) -> list[Passage]:
        """Hybrid search within this tenant's evidence.

        Returns up to ``k`` passages, ranked by reciprocal rank fusion of BM25 and dense vectors.
        ``doc_type`` narrows to one kind of source; ``exclude_doc_types`` rules kinds out. Both are
        applied before ranking, so an excluded kind never costs a slot.
        """
        if not query.strip():
            return []

        candidates = self._candidates(product_id, label_version, doc_type, exclude_doc_types)
        if not candidates:
            return []

        by_id = {p.source_id: p for p in candidates}
        rankings: dict[str, list[str]] = {}
        surfaced_by: dict[str, set[str]] = {}

        # Lexical: exact clinical terminology, dose strings, endpoint names.
        query_tokens = tokenize(query)
        tokenized = [tokenize(p.text) for p in candidates]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query_tokens)

        # Eligibility is term overlap, not a positive BM25 score. BM25's idf term degenerates on
        # small corpora -- with N documents, a term appearing in exactly half of them scores
        # log((N-n+0.5)/(n+0.5)+1) which collapses toward zero, and rank_bm25 floors non-positive
        # idf values to epsilon * average_idf, which is itself zero when no term has positive idf.
        # Filtering on score > 0 therefore throws away the entire lexical ranking on a small
        # tenant corpus, silently turning hybrid search into vector-only search. Ranking by score
        # with overlap as the eligibility rule and the tiebreaker keeps lexical signal usable at
        # any corpus size.
        wanted = set(query_tokens)
        overlap = [len(wanted & set(tokens)) for tokens in tokenized]
        lexical = [
            candidates[i].source_id
            for i in sorted(
                (i for i in range(len(candidates)) if overlap[i] > 0),
                key=lambda i: (scores[i], overlap[i]),
                reverse=True,
            )
        ][: k * 3]
        rankings["bm25"] = lexical
        for sid in lexical:
            surfaced_by.setdefault(sid, set()).add("bm25")

        # Dense: paraphrase and synonymy. The tenant filter goes into the query itself.
        try:
            result = self._collection().query(
                query_texts=[query],
                n_results=min(k * 3, max(len(candidates), 1)),
                where=self._where(product_id, label_version, doc_type, exclude_doc_types),
                include=["documents", "metadatas"],
            )
            dense_ids = (result.get("ids") or [[]])[0]
            dense_docs = (result.get("documents") or [[]])[0]
            dense_metas = (result.get("metadatas") or [[]])[0]
            for sid, doc, meta in zip(dense_ids, dense_docs, dense_metas, strict=False):
                by_id.setdefault(sid, self._passage(sid, doc or "", meta or {}))
                surfaced_by.setdefault(sid, set()).add("vector")
            rankings["vector"] = list(dense_ids)
        except Exception as exc:  # pragma: no cover - depends on the local vector store
            # Degrade to lexical-only rather than failing the review. A missing dense ranking
            # makes retrieval worse; an exception here would make the whole run fail.
            rankings["vector"] = []
            surfaced_by.setdefault("__vector_error__", set())
            self.last_vector_error = str(exc)

        fused = _rrf({name: ids for name, ids in rankings.items() if ids})
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        out: list[Passage] = []
        for source_id, score in ordered:
            passage = by_id.get(source_id)
            if passage is None:
                continue
            # Final assertion. Nothing from another tenant leaves this method, whatever the
            # collection or the filter did. This is the line the isolation test exercises.
            if passage.tenant_id != self.tenant_id:
                raise RetrievalError(
                    f"tenant isolation violation: passage {source_id!r} belongs to "
                    f"{passage.tenant_id!r}, queried as {self.tenant_id!r}"
                )
            passage.score = round(score, 6)
            passage.retrievers = sorted(surfaced_by.get(source_id, set()))
            out.append(passage)
            if len(out) >= k:
                break
        return out

    def contains_verbatim(
        self, needle: str, *, product_id: str = "", label_version: str = "", doc_type: str = ""
    ) -> bool:
        """Is this exact text present in this tenant's evidence, ignoring whitespace?

        The deterministic half of the grounding gate: a model may claim a citation, but only this
        can confirm the quoted passage actually exists in a source belonging to this tenant.
        """
        target = normalize_whitespace(needle)
        if not target:
            return False
        return any(
            target in normalize_whitespace(p.text)
            for p in self._candidates(product_id, label_version, doc_type)
        )

    def all_passages(
        self, *, product_id: str = "", label_version: str = "", doc_type: str = ""
    ) -> list[Passage]:
        """Every in-scope passage. Used by the grounding gate and by the seeding report."""
        return self._candidates(product_id, label_version, doc_type)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip.

    Used for verbatim comparison, because a model quoting a source will reliably differ in line
    breaks and indentation while reproducing the words exactly. Normalising whitespace is the
    difference between a grounding check that works and one that rejects correct citations.
    """
    return " ".join(text.split())
