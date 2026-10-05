"""Reusable agent platform.

Nothing in this package may know about any specific use case. It provides tenancy,
validation gates, human-in-the-loop escalation, an append-only audit log, tenant-scoped
retrieval, graph runtime/checkpointing, and a use-case-agnostic eval harness.

Enforced by ``tests/test_import_boundary.py``: this package must not import from ``mlr``,
``api`` or ``ui``.
"""
