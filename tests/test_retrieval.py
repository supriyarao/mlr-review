"""Tenant-scoped retrieval, and the isolation guarantee the whole multi-tenant claim rests on.

If one tenant's evidence can ground a finding about another tenant's product, the system has
produced a confidential-data leak and a false justification in the same step. These tests are the
reason to believe that cannot happen.
"""

from __future__ import annotations

import pytest

from platform_core.retrieval import (
    Document,
    RetrievalError,
    TenantRetriever,
    collection_name,
    index_documents,
    normalize_whitespace,
    reset_tenant_collection,
    tokenize,
)

ACME_LABEL = (
    "Zelvoran is indicated for the treatment of moderate-to-severe plaque psoriasis in adults "
    "who are candidates for systemic therapy."
)
ACME_STUDY = (
    "In study ZEL-302, PASI-75 was achieved at week 16 in the overall randomised population. "
    "A prespecified subgroup analysis in patients with prior biologic exposure showed a larger "
    "treatment difference."
)
BOREALIS_STUDY = (
    "In study NEX-114, Nexapril reduced major adverse cardiovascular events by 22 percent over "
    "36 months in adults with established cardiometabolic disease."
)


@pytest.fixture
def seeded():
    """Two tenants, each with their own evidence. Nothing shared."""
    for tenant in ("acme_pharma", "borealis_bio"):
        reset_tenant_collection(tenant)

    index_documents(
        "acme_pharma",
        [
            Document(
                source_id="acme-label-1",
                text=ACME_LABEL,
                tenant_id="acme_pharma",
                product_id="zelvoran",
                label_version="v1.0",
                doc_type="label",
                title="Zelvoran label: indications",
                char_span=(0, len(ACME_LABEL)),
            ),
            Document(
                source_id="acme-ref-3",
                text=ACME_STUDY,
                tenant_id="acme_pharma",
                product_id="zelvoran",
                label_version="v1.0",
                doc_type="reference",
                title="Study ZEL-302",
            ),
        ],
    )
    index_documents(
        "borealis_bio",
        [
            Document(
                source_id="borealis-ref-1",
                text=BOREALIS_STUDY,
                tenant_id="borealis_bio",
                product_id="nexapril",
                label_version="v2.1",
                doc_type="reference",
                title="Study NEX-114",
            )
        ],
    )


# --- The isolation guarantee -------------------------------------------------------------------


def test_a_retriever_cannot_exist_without_a_tenant():
    """A missing tenant must be a loud failure, never an unscoped search across everyone."""
    for bad in ("", "   ", None):
        with pytest.raises(RetrievalError, match="non-empty tenant_id"):
            TenantRetriever(bad)  # type: ignore[arg-type]


def test_tenant_a_cannot_see_tenant_b_evidence(seeded):
    """The headline isolation test: Acme's retriever must never surface Borealis's study."""
    results = TenantRetriever("acme_pharma").search(
        "reduced major adverse cardiovascular events by 22 percent", k=10
    )
    assert all(p.tenant_id == "acme_pharma" for p in results)
    assert all("Nexapril" not in p.text for p in results)
    assert all(p.source_id != "borealis-ref-1" for p in results)


def test_evidence_that_only_exists_for_another_tenant_cannot_be_grounded(seeded):
    """The cross-tenant leakage trap: the correct answer is 'cannot verify', not 'supported'."""
    acme = TenantRetriever("acme_pharma")
    assert (
        acme.contains_verbatim("reduced major adverse cardiovascular events by 22 percent") is False
    )
    # The same text IS verifiable for the tenant that actually owns it.
    borealis = TenantRetriever("borealis_bio")
    assert borealis.contains_verbatim("reduced major adverse cardiovascular events by 22 percent")


def test_each_tenant_gets_its_own_physical_collection():
    assert collection_name("acme_pharma") == "acme_pharma__evidence"
    assert collection_name("borealis_bio") == "borealis_bio__evidence"
    with pytest.raises(RetrievalError):
        collection_name("")


def test_indexing_refuses_documents_belonging_to_another_tenant():
    """Defence in depth: a seeding bug must fail loudly rather than quietly cross-contaminate."""
    reset_tenant_collection("acme_pharma")
    with pytest.raises(RetrievalError, match="refusing to index"):
        index_documents(
            "acme_pharma",
            [Document(source_id="x", text="t", tenant_id="borealis_bio")],
        )


def test_every_returned_passage_carries_its_tenant(seeded):
    """So the grounding gate can assert on provenance rather than trusting the caller."""
    results = TenantRetriever("acme_pharma").search("plaque psoriasis in adults", k=5)
    assert results
    assert all(p.tenant_id == "acme_pharma" for p in results)


# --- Filtering and ranking ---------------------------------------------------------------------


def test_results_are_filtered_by_product_and_label_version(seeded):
    retriever = TenantRetriever("acme_pharma")
    assert retriever.search("psoriasis", product_id="zelvoran", label_version="v1.0", k=5)
    # Evidence approved under a different label version is out of scope, not merely ranked lower.
    assert retriever.search("psoriasis", product_id="zelvoran", label_version="v9.9", k=5) == []
    assert retriever.search("psoriasis", product_id="not_a_product", k=5) == []


def test_lexical_search_finds_exact_clinical_terminology(seeded):
    """BM25's job: endpoint names and dose strings, where exact overlap is the signal."""
    results = TenantRetriever("acme_pharma").search("PASI-75 week 16", k=3)
    assert results[0].source_id == "acme-ref-3"
    assert "bm25" in results[0].retrievers


def test_hybrid_search_reports_which_retrievers_surfaced_a_passage(seeded):
    """'bm25 only' and 'vector only' are different failure modes when debugging a miss."""
    results = TenantRetriever("acme_pharma").search("candidates for systemic therapy", k=5)
    assert results
    assert any(r in {"bm25", "vector"} for p in results for r in p.retrievers)


def test_an_empty_query_returns_nothing_rather_than_everything(seeded):
    assert TenantRetriever("acme_pharma").search("   ") == []


def test_search_on_an_empty_collection_is_empty_not_an_error():
    reset_tenant_collection("empty_tenant")
    assert TenantRetriever("empty_tenant").search("anything", k=5) == []


def test_results_are_capped_at_k(seeded):
    assert len(TenantRetriever("acme_pharma").search("psoriasis adults study", k=1)) == 1


# --- Verbatim grounding ------------------------------------------------------------------------


def test_verbatim_check_ignores_whitespace_differences(seeded):
    """A model quoting a source reproduces the words but not the line breaks."""
    retriever = TenantRetriever("acme_pharma")
    assert retriever.contains_verbatim("moderate-to-severe   plaque\n  psoriasis in adults")


def test_verbatim_check_rejects_text_that_is_not_in_any_source(seeded):
    """An invented statistic must not be groundable. This is the hallucination backstop."""
    retriever = TenantRetriever("acme_pharma")
    assert retriever.contains_verbatim("PASI-90 was achieved at week 12 in 81 percent") is False


def test_verbatim_check_rejects_empty_text(seeded):
    assert TenantRetriever("acme_pharma").contains_verbatim("   ") is False


def test_verbatim_check_is_scoped_by_product(seeded):
    retriever = TenantRetriever("acme_pharma")
    assert retriever.contains_verbatim("plaque psoriasis in adults", product_id="zelvoran")
    assert not retriever.contains_verbatim("plaque psoriasis in adults", product_id="other")


# --- Helpers ----------------------------------------------------------------------------------


def test_tokenizer_keeps_clinical_tokens_intact():
    assert "pasi-75" in tokenize("PASI-75 response")
    assert "12.5mg" in tokenize("Dose: 12.5mg daily")


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("  a\n\tb   c  ") == "a b c"


def test_lexical_ranking_survives_bm25_idf_collapse(seeded):
    """Regression guard for a silent hybrid-search failure.

    BM25's idf term degenerates on a small corpus: with two documents, a term appearing in one of
    them scores log(1.5) - log(1.5) = 0, and rank_bm25 floors non-positive idf to
    epsilon * average_idf, which is also zero when no term scores positively. Filtering lexical
    hits on "score > 0" therefore discards the whole lexical ranking and turns hybrid search into
    vector-only search without raising anything. A tenant corpus can legitimately be this small.
    """
    results = TenantRetriever("acme_pharma").search("PASI-75 week 16", k=3)
    assert results, "lexical ranking must still return hits when every idf collapses to zero"
    assert "bm25" in results[0].retrievers
    assert results[0].source_id == "acme-ref-3"


# --- Defence in depth: each isolation mechanism tested on its own -------------------------------
#
# Tenant isolation here rests on three independent mechanisms, and a test that only exercises the
# outermost one would pass even if the inner two were broken. So each is tested in isolation by
# defeating the layers above it:
#
#   1. a physical per-tenant collection,
#   2. a tenant_id metadata filter applied inside the query,
#   3. a final assertion on every returned passage.


def test_a_retriever_only_ever_reads_its_own_collection(seeded):
    """Mechanism 1: the collection a retriever reads is derived from its bound tenant."""
    retriever = TenantRetriever("acme_pharma")
    assert retriever._collection().name == "acme_pharma__evidence"
    assert TenantRetriever("borealis_bio")._collection().name == "borealis_bio__evidence"


def _poison(tenant_collection: str, *, foreign_tenant: str, text: str) -> None:
    """Write a foreign-tenant document directly into a tenant's collection.

    Deliberately bypasses ``index_documents``, whose guard would reject this. Simulates the
    realistic failure: a migration, a manual fix, or a future seeding path that writes to Chroma
    without going through the guarded helper.
    """
    import chromadb

    from platform_core.retrieval import chroma_path

    chromadb.PersistentClient(path=str(chroma_path())).get_or_create_collection(
        tenant_collection
    ).upsert(
        ids=["poisoned-doc"],
        documents=[text],
        metadatas=[
            {
                "tenant_id": foreign_tenant,
                "product_id": "nexapril",
                "label_version": "v2.1",
                "doc_type": "reference",
                "title": "foreign evidence",
                "char_start": 0,
                "char_end": len(text),
            }
        ],
    )


def test_the_metadata_filter_excludes_a_foreign_doc_inside_the_right_collection(seeded):
    """Mechanism 2, with mechanism 1 defeated.

    A foreign document is placed inside Acme's own collection, so the collection boundary cannot
    help. The metadata filter must still keep it out of Acme's results.
    """
    _poison("acme_pharma__evidence", foreign_tenant="borealis_bio", text=BOREALIS_STUDY)

    results = TenantRetriever("acme_pharma").search(
        "reduced major adverse cardiovascular events", k=10
    )
    assert all(p.tenant_id == "acme_pharma" for p in results)
    assert all(p.source_id != "poisoned-doc" for p in results)


def test_the_poisoned_doc_also_cannot_be_used_as_verbatim_grounding(seeded):
    """The grounding path is filtered by the same rule, not just the ranked search path."""
    _poison("acme_pharma__evidence", foreign_tenant="borealis_bio", text=BOREALIS_STUDY)
    acme = TenantRetriever("acme_pharma")
    assert (
        acme.contains_verbatim("reduced major adverse cardiovascular events by 22 percent") is False
    )


def test_the_final_assertion_catches_a_passage_that_slips_past_the_filter(seeded, monkeypatch):
    """Mechanism 3, with mechanisms 1 and 2 both defeated.

    The filter is neutralised and a foreign document sits in Acme's collection. Nothing is left but
    the final per-passage assertion, which must refuse loudly rather than return the passage.
    """
    _poison("acme_pharma__evidence", foreign_tenant="borealis_bio", text=BOREALIS_STUDY)
    retriever = TenantRetriever("acme_pharma")
    # Neutralise the metadata filter the way a real regression would: a syntactically valid
    # filter that happens to match every tenant's documents. (Chroma rejects an empty filter
    # outright, so an empty dict would not reproduce the bug being guarded against.)
    monkeypatch.setattr(
        type(retriever), "_where", lambda self, *a, **kw: {"tenant_id": {"$ne": "__nobody__"}}
    )

    with pytest.raises(RetrievalError, match="tenant isolation violation"):
        retriever.search("reduced major adverse cardiovascular events", k=10)
